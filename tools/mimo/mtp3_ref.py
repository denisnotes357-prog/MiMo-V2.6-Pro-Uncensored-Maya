#!/usr/bin/env python3
"""tools/mimo/mtp3_ref.py - which drafting scheme MiMo-V2.6-Pro's three MTP layers were trained for, measured: teacher-
forced top-1 over the trunk's hidden states calib.py left in its streams (the same inputs as mtp_ref.py), on the CPU.

For a position i (trunk hidden h_i, pre-final-norm) and depth k = 0, 1, 2 the block predicts token t_{i+2+k}:
  native     block k:  x_i = eh_k(cat(enorm_k(emb(t_{i+1+k})), hnorm_k(h_i)))           (non-chained, layer k at depth k)
  block0     block 0:  the same inputs with layer 0's weights at every depth           (non-chained, one layer)
  chained    block 0:  depth 0 as above; depth k feeds back final_norm(block-0 output of depth k-1) in place of h_i
                       with emb(t_{i+1+k}) - what the Maya engine does today (fast_mtp's head_x)
Each scheme's block attends (SWA 128 + sink, its own K/V) over its own inputs at positions 0..i.  Teacher forcing uses
the real tokens, so a depth's rate is its acceptance given the drafts before it stood.

python mtp3_ref.py [--streams DIR] [--eval eval.npy] [--seqs 1,2] [--T 1024] [--threads 32]
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


def rms(x, w, eps):
    return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)) * w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/MiMo-V2.6-Pro-MOPD-heretic")
    ap.add_argument("--streams", default="/data/mimo/calib/sanity")
    ap.add_argument("--calib", default="/data/mimo/calib/calib.npy")
    ap.add_argument("--eval", default="/data/mimo/calib/eval.npy")
    ap.add_argument("--n-cal", type=int, default=1)
    ap.add_argument("--n-eval", type=int, default=2)
    ap.add_argument("--seqs", default="1,2")
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sys.path.insert(0, C.HF)
    from configuration_mimo_v2 import MiMoV2Config
    from modeling_mimo_v2 import MiMoV2Attention, MiMoV2MLP, MiMoV2RotaryEmbedding

    dev = torch.device("cpu")
    ck = C.Checkpoint(a.model)
    cfg = MiMoV2Config.from_json_file(str(pathlib.Path(a.model) / "config.json"))
    cfg._attn_implementation = "eager"
    eps = float(cfg.layernorm_epsilon)
    H = int(cfg.hidden_size)
    seqs = np.concatenate([np.load(a.calib)[: a.n_cal], np.load(a.eval)[: a.n_eval]], 0).astype(np.int64)
    n_seq, T_all = seqs.shape
    mm = np.memmap(pathlib.Path(a.streams) / "streams_a.bin", dtype=np.uint16, mode="r", shape=(n_seq, T_all, H))
    emb = ck.get("model.embed_tokens.weight", dtype=torch.bfloat16)
    lm = ck.get("lm_head.weight", dtype=torch.float32)

    blocks = []
    for k in range(3):
        P = f"model.mtp.layers.{k}."
        W = lambda n, dt=torch.float32: ck.get(P + n, dev=dev, dtype=dt)  # noqa: E731
        with torch.device("meta"):
            attn = MiMoV2Attention(cfg, True, 1, projection_layout=cfg.attention_projection_layout)
            mlp = MiMoV2MLP(cfg)
        attn = attn.to_empty(device=dev).to(torch.float32)
        mlp = mlp.to_empty(device=dev).to(torch.float32)
        with torch.no_grad():
            for n, p in attn.named_parameters():
                p.copy_(ck.get(P + "self_attn." + n, dtype=torch.float32))
            for n, p in mlp.named_parameters():
                p.copy_(ck.get(P + "mlp." + n, dtype=torch.float32))
        attn.eval(); mlp.eval()
        blocks.append(dict(enorm=W("enorm.weight"), hnorm=W("hnorm.weight"), eh=W("eh_proj.weight"),
                           in_norm=W("input_layernorm.weight"), pre_mlp=W("pre_mlp_layernorm.weight"),
                           fin=W("final_layernorm.weight"), attn=attn, mlp=mlp))
        print(f"loaded MTP layer {k}", flush=True)
    rot = MiMoV2RotaryEmbedding(cfg, is_swa=True)
    T = min(a.T, T_all) - 4
    pos = torch.arange(T).unsqueeze(0)
    rope = rot(torch.empty(1, 1, H, dtype=torch.float32), pos)
    i_ = torch.arange(T)
    d = i_[:, None] - i_[None, :]
    mask = torch.where((d >= 0) & (d < int(cfg.sliding_window)), 0.0, torch.finfo(torch.float32).min)

    def run(b, e, h):
        """block b over inputs (token embeddings e, previous hiddens h) -> (final-normed output, logits)"""
        x = torch.cat([rms(e, b["enorm"], eps), rms(h, b["hnorm"], eps)], -1) @ b["eh"].t()
        y, _ = b["attn"](rms(x, b["in_norm"], eps)[None], position_embeddings=rope, attention_mask=mask[None, None],
                         position_ids=pos)
        r = x + y[0]
        r = r + b["mlp"](rms(r, b["pre_mlp"], eps))
        o = rms(r, b["fin"], eps)
        return o, o @ lm.t()

    hits = {s: [0, 0, 0] for s in ("native", "block0", "chained")}
    tot = 0
    with torch.no_grad():
        for sq in [int(x) for x in a.seqs.split(",")]:
            ids = torch.from_numpy(seqs[sq])
            h = torch.from_numpy(np.asarray(mm[sq][:T])).view(torch.bfloat16).float()
            prev = None
            for k in range(3):
                e = emb[ids[1 + k:T + 1 + k]].float()
                tgt = ids[2 + k:T + 2 + k]
                _, lg = run(blocks[k], e, h)
                hits["native"][k] += int((lg.argmax(-1) == tgt).sum())
                _, lg0 = run(blocks[0], e, h)
                if k > 0:
                    hits["block0"][k] += int((lg0.argmax(-1) == tgt).sum())
                o, lgc = run(blocks[0], e, h if prev is None else prev)
                hits["chained"][k] += int((lgc.argmax(-1) == tgt).sum())
                prev = o
                print(f"  seq {sq} depth {k + 1} done", flush=True)
            tot += T
    hits["block0"][0] = hits["chained"][0]
    print(f"\nteacher-forced top-1 over {tot} positions (depth d predicts t_(i+1+d)):")
    for s, v in hits.items():
        print(f"  {s:8s} " + "  ".join(f"d{k + 1} {100 * v[k] / tot:5.1f}%" for k in range(3)))


if __name__ == "__main__":
    main()
