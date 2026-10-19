"""tools/mimo/bake.py - bake a refusal direction into a MiMo GGUF in place (weight orthogonalisation).

Every matrix that WRITES the residual stream gets the direction r projected out of its output: W <- W - r (r^T W)
for attn_output (o_proj) and the dense/MTP ffn_down, E <- E - (E r) r^T for token_embd, and with --experts every
routed expert's down projection.  The float source is the ORIGINAL checkpoint (BF16 o_proj/embed, FP8 dense down,
MXFP4 experts), not the GGUF's own bytes, so nothing is rounded twice; the result is quantized to the tensor's
existing GGUF type with the calibration imatrix (tools/mimo/calib.py stats) and written over the same bytes (same
size).  The bytes it replaces go to --backup first; --restore puts them back.

python bake.py --gguf FIRST_SHARD --dir dir.npy [--experts] [--layers 0-72] [--scale 1.0]
python bake.py --gguf FIRST_SHARD --restore
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calib as C          # noqa: E402  (Checkpoint: MXFP4 / FP8 / BF16 tensors as float)
import quantize as Q       # noqa: E402  (Stats, Imx: ggml with the imatrix)

gguf = Q.gguf
QT = Q.QT


def shards(first: pathlib.Path) -> list[pathlib.Path]:
    m = re.match(r"(.*)-00001-of-(\d+)\.gguf$", first.name)
    if not m:
        return [first]
    return [first.parent / f"{m.group(1)}-{i:05d}-of-{m.group(2)}.gguf" for i in range(1, int(m.group(2)) + 1)]


def source_name(gname: str, n_text: int) -> tuple[str, str] | None:
    """GGUF tensor -> (checkpoint name, kind) for the residual writers; None for everything else"""
    if gname == "token_embd.weight":
        return "model.embed_tokens.weight", "embed"
    m = re.match(r"blk\.(\d+)\.(attn_output|ffn_down|ffn_down_exps)\.weight$", gname)
    if not m:
        return None
    il, part = int(m.group(1)), m.group(2)
    pre = f"model.layers.{il}." if il < n_text else f"model.mtp.layers.{il - n_text}."
    if part == "attn_output":
        return pre + "self_attn.o_proj.weight", "out"
    if part == "ffn_down":
        return pre + "mlp.down_proj.weight", "out"
    return pre + "mlp.experts.{e}.down_proj.weight", "experts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--model", default="/data/models/MiMo-V2.6-Pro-MOPD-heretic")
    ap.add_argument("--stats", default="/data/mimo/calib/stats")
    ap.add_argument("--dir", default="/data/mimo/refusal/dir.npy")
    ap.add_argument("--scale", type=float, default=1.0, help="1.0 = remove the component entirely")
    ap.add_argument("--experts", action="store_true")
    ap.add_argument("--layers", default=None, help="a-b: only these blocks (embed always unless --no-embed)")
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--backup", default="/data/mimo/bake-backup")
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--threads", type=int, default=24)
    a = ap.parse_args()
    files = shards(pathlib.Path(a.gguf))
    bk = pathlib.Path(a.backup)
    if a.restore:
        n = 0
        for f in files:
            r = gguf.GGUFReader(str(f), "r+")
            for t in r.tensors:
                p = bk / (t.name + ".bin")
                if p.exists():
                    t.data.reshape(-1).view(np.uint8)[:] = np.fromfile(p, dtype=np.uint8)
                    n += 1
            r.data.flush()
        print(f"restored {n} tensors from {bk}")
        return
    bk.mkdir(parents=True, exist_ok=True)
    r_vec = np.load(a.dir).astype(np.float32).reshape(-1)
    r_vec /= np.linalg.norm(r_vec)
    rt = torch.from_numpy(r_vec)
    ck = C.Checkpoint(a.model)
    n_text = int(ck.cfg["num_hidden_layers"])
    lo, hi = (0, 10 ** 6) if a.layers is None else tuple(int(x) for x in a.layers.split("-"))
    stats = Q.Stats(a.stats)
    imx = Q.Imx(Q.GGML_LIB, a.threads, stats, name_of=lambda: None)
    t_all = time.time()
    n_done = 0
    for f in files:
        rd = gguf.GGUFReader(str(f), "r+")
        for t in rd.tensors:
            src = source_name(t.name, n_text)
            if src is None:
                continue
            cname, kind = src
            if kind == "embed" and a.no_embed:
                continue
            if kind != "embed":
                il = int(t.name.split(".")[1])
                if not (lo <= il <= hi):
                    continue
            if kind == "experts" and not a.experts:
                continue
            t0 = time.time()
            qtype = QT(int(t.tensor_type))
            shape = tuple(int(x) for x in reversed([int(v) for v in t.shape]))   # numpy order [.., rows, n]
            if kind == "experts":
                E = shape[0]
                w = torch.stack([ck.get(cname.format(e=e), dtype=torch.float32) for e in range(E)])
                w = w - a.scale * torch.einsum("o,eoi->ei", rt, w)[:, None, :] * rt[None, :, None]
            else:
                w = ck.get(cname, dtype=torch.float32)
                if kind == "embed":
                    w = w - a.scale * (w @ rt)[:, None] * rt[None, :]
                else:
                    w = w - a.scale * rt[:, None] * (rt @ w)[None, :]
            arr = w.numpy()
            if tuple(arr.shape) != shape:
                raise SystemExit(f"{t.name}: source {arr.shape} vs gguf {shape}")
            if qtype in (QT.F32, QT.F16, QT.BF16):
                new = arr.astype({QT.F32: np.float32, QT.F16: np.float16}.get(qtype, np.float32))
                if qtype == QT.BF16:
                    new = (torch.from_numpy(arr).to(torch.bfloat16).view(torch.uint16).numpy())
                new = new.view(np.uint8)
            else:
                im = stats.get(t.name, shape)
                new = imx.eager_im(arr, qtype, im) if im is not None else imx.eager(arr, qtype)
            dst = t.data.reshape(-1).view(np.uint8)
            if new.nbytes != dst.nbytes:
                raise SystemExit(f"{t.name}: new {new.nbytes} B vs file {dst.nbytes} B")
            p = bk / (t.name + ".bin")
            if not p.exists():
                np.asarray(dst).tofile(p)
            dst[:] = new.reshape(-1)
            n_done += 1
            print(f"{t.name:34s} {qtype.name:8s} {new.nbytes / 1e6:9.1f} MB {time.time() - t0:6.1f} s", flush=True)
        rd.data.flush()
        del rd
    print(f"BAKE DONE: {n_done} tensors in {(time.time() - t_all) / 60:.1f} min (backup {bk})", flush=True)


if __name__ == "__main__":
    main()
