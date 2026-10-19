"""tools/mimo/capture.py - residual streams of MiMo-V2.6-Pro on harmful vs harmless prompts, for the refusal direction.

Runs the same layer-by-layer pass as calib.py (its Checkpoint / load_layer / StatMoE, statistics off) over short chat
prompts, left-padded to one length per batch (pad keys masked), and keeps the residual stream at the LAST prompt
position (the end of the generation prompt) after the embedding and after every layer: caps.npy [n, 71, hidden] fp32
plus tags.json.  directions = mean(H) - mean(B) per layer is computed by derive.py from it.

python capture.py [--prompts derive-prompts.jsonl] [--out DIR] [--batch 16] [--think on|off]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calib as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/MiMo-V2.6-Pro-MOPD-heretic")
    ap.add_argument("--prompts", default="/data/glm53/uncensored/derive-prompts.jsonl")
    ap.add_argument("--out", default="/data/mimo/refusal")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--think", default="on", choices=["on", "off"])
    ap.add_argument("--ablate", default=None, help="unit direction .npy projected out of the residual after the"
                    " embedding and after every layer (= orthogonalising every writer)")
    ap.add_argument("--ablate-layers", default=None, help="comma list: use these rows of a [L, hidden] dirs file as"
                    " a subspace (orthonormalised) instead of one direction")
    ap.add_argument("--ablate-scale", type=float, default=1.0, help="1 = project out, 2 = reflect")
    ap.add_argument("--first", default=None, help="also write the next-token top-10 at the prompt end to this json")
    a = ap.parse_args()
    sys.path.insert(0, C.HF)
    from configuration_mimo_v2 import MiMoV2Config
    from modeling_mimo_v2 import MiMoV2RotaryEmbedding
    from transformers import AutoTokenizer

    dev = torch.device("cuda")
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=False)
    rows = [json.loads(l) for l in open(a.prompts)]
    ids = []
    for r in rows:
        kw = {} if a.think == "on" else {"enable_thinking": False}
        t = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], add_generation_prompt=True,
                                    tokenize=False, **kw)
        ids.append(tok(t, add_special_tokens=False)["input_ids"])
    tags = [r["tag"] for r in rows]
    json.dump({"tags": tags, "think": a.think, "example": tok.decode(ids[0])}, open(out / "tags.json", "w"))
    print(f"{len(ids)} prompts ({tags.count('H')} H), lengths {min(map(len, ids))}-{max(map(len, ids))}; "
          f"example: {tok.decode(ids[0])!r}", flush=True)

    ck = C.Checkpoint(a.model)
    cfg = MiMoV2Config.from_json_file(str(pathlib.Path(a.model) / "config.json"))
    cfg._attn_implementation = "eager"
    H, n_layers = int(cfg.hidden_size), int(cfg.num_hidden_layers)
    pat = list(cfg.hybrid_layer_pattern)
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))           # similar lengths share a batch
    batches = [order[i:i + a.batch] for i in range(0, len(order), a.batch)]
    caps = np.zeros((len(ids), n_layers + 1, H), dtype=np.float32)
    emb = ck.get(C.P + "embed_tokens.weight", dtype=C.DT)
    states, metas = [], []
    rot = {0: MiMoV2RotaryEmbedding(cfg, is_swa=False).to(dev), 1: MiMoV2RotaryEmbedding(cfg, is_swa=True).to(dev)}
    neg = torch.finfo(C.DT).min
    for b in batches:
        T = max(len(ids[i]) for i in b)
        x = torch.zeros(len(b), T, H, dtype=C.DT)
        valid = torch.zeros(len(b), T, dtype=torch.bool)
        pos = torch.zeros(len(b), T, dtype=torch.long)
        for j, i in enumerate(b):
            n = len(ids[i])
            x[j, T - n:] = emb[torch.tensor(ids[i])]
            valid[j, T - n:] = True
            pos[j, T - n:] = torch.arange(n)
            caps[i, 0] = x[j, -1].float().numpy()
        q = torch.arange(T)
        d = q[:, None] - q[None, :]
        m = {}
        for kind in (0, 1):
            ok = d >= 0
            if kind == 1:
                ok = ok & (d < int(cfg.sliding_window))
            ok = ok[None] & valid[:, None, :]
            ok = ok | torch.eye(T, dtype=torch.bool)[None]                    # a pad query sees itself (no NaN)
            m[kind] = torch.where(ok, 0.0, neg).to(C.DT)[:, None].to(dev)
        pos = pos.to(dev)
        rope = {k: rot[k](torch.empty(1, 1, H, device=dev, dtype=C.DT), pos) for k in (0, 1)}
        states.append(x)
        metas.append((b, m, pos, rope))
    del emb
    rv = None
    if a.ablate:
        m = np.load(a.ablate).astype(np.float32)
        if a.ablate_layers:
            m = m[[int(x) for x in a.ablate_layers.split(",")]]
        m = m.reshape(-1, m.shape[-1])
        q, _ = np.linalg.qr(m.T)                        # [hidden, k] orthonormal basis of the subspace
        rv = torch.from_numpy(q.astype(np.float32))
        print(f"ablating a {rv.shape[1]}-dim subspace, scale {a.ablate_scale}", flush=True)

    def proj(t: torch.Tensor) -> torch.Tensor:
        if rv is None:
            return t
        r = rv.to(t.device)
        tf = t.float()
        return (tf - a.ablate_scale * (tf @ r) @ r.t()).to(t.dtype)

    states = [proj(x) for x in states]
    for il in range(n_layers):
        t0 = time.time()
        layer = C.load_layer(ck, cfg, il, dev)
        moe = None
        if cfg.moe_layer_freq[il]:
            moe = C.StatMoE(ck, cfg, il, dev)
            moe.collect = False
            layer.mlp = moe
        kind = pat[il]
        for k, (b, m, pos, rope) in enumerate(metas):
            with torch.no_grad():
                y = layer(states[k].to(dev), attention_mask=m[kind], position_ids=pos, past_key_values=None,
                          use_cache=False, position_embeddings=rope[kind])
            y = proj((y[0] if isinstance(y, tuple) else y)).cpu()
            states[k] = y
            for j, i in enumerate(b):
                caps[i, il + 1] = y[j, -1].float().numpy()
        del layer, moe
        torch.cuda.empty_cache()
        print(f"layer {il:2d}: {time.time() - t0:5.1f} s", flush=True)
    if a.first:
        norm_w = ck.get(C.P + "norm.weight", dtype=torch.float32).to(dev)
        lm = ck.get("lm_head.weight", dev=dev)
        res = []
        for k, (b, m, pos, rope) in enumerate(metas):
            h = states[k][:, -1].to(dev).float()
            h = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + float(cfg.layernorm_epsilon)) * norm_w
            pr = torch.softmax((h.to(C.DT) @ lm.t()).float(), -1)
            v, i_ = torch.topk(pr, 10, -1)
            for j, i in enumerate(b):
                res.append({"i": i, "tag": tags[i], "top": [[tok.decode([int(t)]), round(float(p), 4)]
                                                           for t, p in zip(i_[j].tolist(), v[j].tolist())]})
        res.sort(key=lambda r: r["i"])
        json.dump(res, open(a.first, "w"), ensure_ascii=False)
        print(f"first-token top-10 -> {a.first}", flush=True)
    if a.ablate is None:
        np.save(out / f"caps-think{a.think}.npy", caps)
    print("CAPTURE DONE", flush=True)


if __name__ == "__main__":
    main()
