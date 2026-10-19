"""tools/mimo/check_mtp_qkv.py - is the GGUF's attn_qkv of a layer the TP-aware [Q|K|V] dequant of the checkpoint?

llama.cpp's MimoV2Model.dequant_model matches only ^model.layers.N.self_attn.qkv_proj.weight_scale_inv, so the MTP
layers (model.mtp.layers.N) get the plain block-FP8 dequant: TP8 rank-interleaved rows, padded scale blocks.  This
dequantizes blk.L.attn_qkv.weight from the quant and compares it with both readings of the checkpoint (CPU).

python check_mtp_qkv.py --gguf FIRST_SHARD [--layers 1,70,71,72]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calib as C  # noqa: E402

sys.path.insert(0, C.LLAMA + "/gguf-py")
import gguf  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/MiMo-V2.6-Pro-MOPD-heretic")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--layers", default="1,70,71,72")
    a = ap.parse_args()
    ck = C.Checkpoint(a.model)
    first = pathlib.Path(a.gguf)
    shards = sorted(first.parent.glob(first.name.split("-00001-of-")[0] + "-0000*-of-*.gguf"))
    want = {f"blk.{l}.attn_qkv.weight": int(l) for l in a.layers.split(",")}
    n_txt = int(ck.cfg["num_hidden_layers"])
    for sh in shards:
        r = gguf.GGUFReader(str(sh))
        for t in r.tensors:
            if t.name not in want:
                continue
            il = want[t.name]
            g = torch.from_numpy(gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32))
            hf = (f"model.layers.{il}." if il < n_txt else f"model.mtp.layers.{il - n_txt}.") + "self_attn.qkv_proj.weight"
            tp = ck.get(hf, dtype=torch.float32)
            w = ck.raw(hf).float()
            s = ck.raw(hf + "_scale_inv").float()
            s = s.repeat_interleave(128, 0)[: w.shape[0]].repeat_interleave(128, 1)[:, : w.shape[1]]
            naive = w * s
            g = g.reshape(tp.shape)
            rel = lambda x: float((g - x).norm() / x.norm())
            print(f"{t.name} {t.tensor_type.name} {tuple(g.shape)}: rel err vs TP-aware {rel(tp):.3f}, vs plain {rel(naive):.3f}",
                  flush=True)


if __name__ == "__main__":
    main()
