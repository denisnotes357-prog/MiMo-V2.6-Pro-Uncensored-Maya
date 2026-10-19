"""tools/mimo/fix_mtp_qkv.py - rewrite the MTP layers' attn_qkv of a MiMo GGUF with the TP-aware dequant, in place.

llama.cpp's MimoV2Model.dequant_model matches only ^model.layers.N.self_attn.qkv_proj.weight_scale_inv, so the 3 MTP
layers (model.mtp.layers.N -> blk.70-72) were converted with the plain block-FP8 dequant: rows still TP8
rank-interleaved ([q|k|v] per rank) and the padded scale blocks misapplied - the MTP attention reads scrambled Q/K/V
(check_mtp_qkv.py: rel err 1.41 vs the right weight).  Here the right weight (calib.Checkpoint = llama.cpp's own
_tp_aware_qkv_dequant) is quantized to the tensor's existing type and written over the same bytes; the old bytes go
to --backup first (bake.py --restore --backup DIR puts them back).

python fix_mtp_qkv.py --gguf FIRST_SHARD
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bake as B          # noqa: E402
import calib as C          # noqa: E402
import quantize as Q       # noqa: E402

gguf, QT = Q.gguf, Q.QT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--model", default="/data/models/MiMo-V2.6-Pro-MOPD-heretic")
    ap.add_argument("--stats", default="/data/mimo/calib/stats")
    ap.add_argument("--backup", default="/data/mimo/mtp-qkv-backup")
    ap.add_argument("--threads", type=int, default=24)
    a = ap.parse_args()
    ck = C.Checkpoint(a.model)
    n_text = int(ck.cfg["num_hidden_layers"])
    n_mtp = int(ck.cfg.get("num_nextn_predict_layers", 3) or 3)
    want = {f"blk.{n_text + i}.attn_qkv.weight": f"model.mtp.layers.{i}.self_attn.qkv_proj.weight" for i in range(n_mtp)}
    bk = pathlib.Path(a.backup)
    bk.mkdir(parents=True, exist_ok=True)
    stats = Q.Stats(a.stats)
    imx = Q.Imx(Q.GGML_LIB, a.threads, stats, name_of=lambda: None)
    n = 0
    for f in B.shards(pathlib.Path(a.gguf)):
        rd = gguf.GGUFReader(str(f), "r+")
        for t in rd.tensors:
            if t.name not in want:
                continue
            t0 = time.time()
            qtype = QT(int(t.tensor_type))
            shape = tuple(int(x) for x in reversed([int(v) for v in t.shape]))
            arr = ck.get(want[t.name], dtype=torch.float32).numpy()
            if tuple(arr.shape) != shape:
                raise SystemExit(f"{t.name}: source {arr.shape} vs gguf {shape}")
            im = stats.get(t.name, shape)
            new = imx.eager_im(arr, qtype, im) if im is not None else imx.eager(arr, qtype)
            dst = t.data.reshape(-1).view(np.uint8)
            if new.nbytes != dst.nbytes:
                raise SystemExit(f"{t.name}: new {new.nbytes} B vs file {dst.nbytes} B")
            p = bk / (t.name + ".bin")
            if not p.exists():
                np.asarray(dst).tofile(p)
            dst[:] = new.reshape(-1)
            n += 1
            print(f"{t.name} {qtype.name} imatrix={'yes' if im is not None else 'no'} {time.time() - t0:.1f} s", flush=True)
        rd.data.flush()
        del rd
    print(f"FIX DONE: {n} tensors (backup {bk})", flush=True)


if __name__ == "__main__":
    main()
