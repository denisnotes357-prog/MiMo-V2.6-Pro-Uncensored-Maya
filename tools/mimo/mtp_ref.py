"""tools/mimo/mtp_ref.py - a torch reference of MiMo-V2.6-Pro's first MTP layer (vLLM mimo_v2_mtp.py's forward) over the
trunk's hidden states that calib.py left in its streams: how often the block's argmax is the real token two ahead.

    h_i  = trunk hidden at position i (final layer output; --normed applies the model's final norm first)
    x_i  = eh_proj(cat(enorm(embed(t_{i+1})), hnorm(h_i)))
    y    = final_layernorm(MLP-block(attention-block(x)))   (SWA window + sink, swa rope base, dense MLP)
    pred = argmax(lm_head(y_i))  vs  t_{i+2}

python mtp_ref.py --streams /data/mimo/calib/sanity --seqs 1,2 [--normed]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calib as C  # noqa: E402


def rms(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)) * w.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/MiMo-V2.6-Pro-MOPD-heretic")
    ap.add_argument("--streams", default="/data/mimo/calib/sanity")
    ap.add_argument("--calib", default="/data/mimo/calib/calib.npy")
    ap.add_argument("--eval", default="/data/mimo/calib/eval.npy")
    ap.add_argument("--n-cal", type=int, default=1, help="how the streams were made: n_cal + n_eval sequences")
    ap.add_argument("--n-eval", type=int, default=2)
    ap.add_argument("--seqs", default="1,2")
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--ablate", default=None, help="the quant's subspace: projected out of the embedding and of the MTP"
                    " block's residual writers (eh_proj, o_proj, down) as quantize.py --ablate bakes it")
    a = ap.parse_args()
    sys.path.insert(0, C.HF)
    from configuration_mimo_v2 import MiMoV2Config
    from modeling_mimo_v2 import MiMoV2Attention, MiMoV2MLP, MiMoV2RotaryEmbedding

    dev = torch.device("cuda")
    ck = C.Checkpoint(a.model)
    cfg = MiMoV2Config.from_json_file(str(pathlib.Path(a.model) / "config.json"))
    cfg._attn_implementation = "eager"
    eps = float(cfg.layernorm_epsilon)
    H = int(cfg.hidden_size)
    seqs = np.concatenate([np.load(a.calib)[: a.n_cal], np.load(a.eval)[: a.n_eval]], 0).astype(np.int64)
    n_seq, T_all = seqs.shape
    mm = np.memmap(pathlib.Path(a.streams) / "streams_a.bin", dtype=np.uint16, mode="r", shape=(n_seq, T_all, H))
    P = "model.mtp.layers.0."
    W = lambda n, dt=torch.float32: ck.get(P + n, dev=dev, dtype=dt)
    emb = ck.get("model.embed_tokens.weight", dtype=torch.bfloat16)
    lm = ck.get("lm_head.weight", dev=dev, dtype=torch.bfloat16)
    norm_w = ck.get("model.norm.weight", dev=dev, dtype=torch.float32)
    enorm, hnorm = W("enorm.weight"), W("hnorm.weight")
    eh = W("eh_proj.weight", torch.bfloat16)
    in_norm, pre_mlp, fin = W("input_layernorm.weight"), W("pre_mlp_layernorm.weight"), W("final_layernorm.weight")
    with torch.device("meta"):
        attn = MiMoV2Attention(cfg, True, 1, projection_layout=cfg.attention_projection_layout)
        mlp = MiMoV2MLP(cfg)
    attn = attn.to_empty(device=dev).to(torch.bfloat16)
    mlp = mlp.to_empty(device=dev).to(torch.bfloat16)
    with torch.no_grad():
        for n, p in attn.named_parameters():
            p.copy_(ck.get(P + "self_attn." + n, dev=dev, dtype=p.dtype))
        for n, p in mlp.named_parameters():
            p.copy_(ck.get(P + "mlp." + n, dev=dev, dtype=p.dtype))
    attn.eval(); mlp.eval()
    rot = MiMoV2RotaryEmbedding(cfg, is_swa=True).to(dev)
    T = min(a.T, T_all) - 2
    pos = torch.arange(T, device=dev).unsqueeze(0)
    rope = rot(torch.empty(1, 1, H, device=dev, dtype=torch.bfloat16), pos)
    i_ = torch.arange(T, device=dev)
    d = i_[:, None] - i_[None, :]
    mask = torch.where((d >= 0) & (d < int(cfg.sliding_window)), 0.0, torch.finfo(torch.bfloat16).min).to(torch.bfloat16)
    Qa = None
    if a.ablate:
        Qa = torch.from_numpy(np.load(a.ablate).astype(np.float32)).to(dev)
    def pr(v):
        return v if Qa is None else v - (v.float() @ Qa) @ Qa.t()
    for normed in (False, True):
        hits = tot = 0
        for s in [int(x) for x in a.seqs.split(",")]:
            ids = torch.from_numpy(seqs[s])
            h = torch.from_numpy(np.asarray(mm[s][:T])).view(torch.bfloat16).to(dev).float()
            if normed:
                h = rms(h, norm_w, eps)
            e = pr(emb[ids[1:T + 1]].to(dev).float())
            x = pr((torch.cat([rms(e, enorm, eps), rms(h, hnorm, eps)], -1).to(torch.bfloat16) @ eh.t()).float())
            with torch.no_grad():
                r = x.float()
                y, _ = attn(rms(r, in_norm, eps).to(torch.bfloat16)[None], position_embeddings=rope,
                            attention_mask=mask[None, None], position_ids=pos)
                r = r + pr(y[0].float())
                r = r + pr(mlp(rms(r, pre_mlp, eps).to(torch.bfloat16)).float())
                lg = rms(r, fin, eps).to(torch.bfloat16) @ lm.t()
            pred = lg.float().argmax(-1).cpu()
            tgt = ids[2:T + 2]
            hits += int((pred == tgt).sum())
            tot += T
        print(f"MTP layer 0, previous hidden {'NORMED (final norm)' if normed else 'RAW (pre-norm)'}: "
              f"top-1 = t+2 on {hits}/{tot} = {100 * hits / tot:.1f}%", flush=True)


if __name__ == "__main__":
    main()
