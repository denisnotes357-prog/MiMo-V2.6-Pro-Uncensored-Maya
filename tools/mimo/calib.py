"""tools/mimo/calib.py - calibration statistics (imatrix) and the reference for MiMo-V2.6-Pro (mimo_v2), the stage the
quants are built on.  Port of tools/glm53/calib.py; the format of stats_XX.pt / ffn_in_XX.bin / ref_topk.npz is kept.

The shipped checkpoint (/data/models/MiMo-V2.6-Pro-MOPD-heretic) runs layer by layer on ONE GPU: each layer is
the model's own `MiMoV2DecoderLayer` (modeling_mimo_v2.py from the checkpoint, local copy patched for transformers
5.19's mask API), eager attention (the SWA layers' sink bias is eager-only), with the routed experts replaced by a
module that records per expert:
    n          tokens routed to it
    imat_gu    sum over its tokens of x^2 per input channel (imatrix of gate/up)
    imat_d     the same for down's input h = silu(gate) * up
    sal        sum of routing weight * ||expert output||
and every nn.Linear of the layer (qkv_proj, o_proj, layer 0's dense MLP) gets its imatrix by module name.

MiMo-V2.6-Pro facts this pass encodes (docs/MIMO2-ENGINE-PORT.md):
  * 70 layers, layer 0 dense (16384), 1..69 MoE: 384 experts top-8, sigmoid + noaux_tc bias (n_group 1), norm top-k,
    routed_scaling_factor None -> 1.0, plain SwiGLU, no shared expert.  The 3 MTP layers are not part of this pass.
  * Attention per hybrid_layer_pattern: 0 = full (rope_theta), 1 = SWA window 128 (swa_rope_theta) with sink bias.
  * Weights: routed experts MXFP4 (U8 nibbles, E2M1, low nibble first; U8 E8M0 scale per 32 inputs), qkv_proj and the
    dense MLP block-FP8 128x128 (qkv's scales are TP=8 sharded with per-rank padding - dequantised with llama.cpp's
    own MimoV2Model._tp_aware_qkv_dequant), the rest BF16.

python calib.py [--model DIR] [--calib calib.npy] [--eval eval.npy] [--out DIR] [--layers 0-69] [--batch 2]
                [--ffn-in-seqs 32] [--n-cal N] [--n-eval N]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_tf32 = False

P = "model."
DT = torch.bfloat16
HF = str(pathlib.Path(__file__).resolve().parent / "hf")   # MiMo-V2 modeling/configuration (Xiaomi, Apache-2.0; see hf/NOTICE)
LLAMA = "/data/glm53/llama.cpp"
FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def mxfp4_dequant(blocks: torch.Tensor, scales: torch.Tensor, dtype=DT) -> torch.Tensor:
    """[rows, cols/2] U8 nibbles + [rows, cols/32] U8 E8M0 -> [rows, cols]; low nibble = even column."""
    lut = FP4.to(blocks.device)
    lo = lut[(blocks & 0x0F).long()]
    hi = lut[(blocks >> 4).long()]
    w = torch.stack([lo, hi], -1).reshape(blocks.shape[0], -1)              # [rows, cols]
    e = torch.pow(2.0, scales.float() - 127.0)                               # E8M0
    w = (w.view(blocks.shape[0], -1, 32) * e[:, :, None]).reshape(blocks.shape[0], -1)
    return w.to(dtype)


class Checkpoint:
    """The MXFP4/FP8/BF16 MiMo checkpoint, tensor by tensor (no index json: the headers are scanned once)."""

    def __init__(self, root: str):
        self.root = pathlib.Path(root)
        self.cfg = json.load(open(self.root / "config.json"))
        self.map: dict[str, str] = {}
        for f in sorted(self.root.glob("*.safetensors")):
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                h = json.loads(fh.read(n))
            h.pop("__metadata__", None)
            for k in h:
                self.map[k] = f.name
        self._open: dict[str, object] = {}
        sys.path.insert(0, LLAMA)
        sys.path.insert(0, LLAMA + "/gguf-py")
        from conversion.mimo import MimoV2Model                               # noqa: E402
        self._qkv = MimoV2Model._tp_aware_qkv_dequant

    def has(self, name: str) -> bool:
        return name in self.map

    def raw(self, name: str) -> torch.Tensor:
        shard = self.map[name]
        f = self._open.get(shard)
        if f is None:
            f = safe_open(str(self.root / shard), framework="pt")
            self._open[shard] = f
        return f.get_tensor(name)

    def get(self, name: str, dev=None, dtype: torch.dtype = DT) -> torch.Tensor:
        t = self.raw(name)
        if dev is not None:
            t = t.to(dev)
        if t.dtype == torch.uint8 and self.has(name + "_scale"):               # MXFP4 routed expert
            s = self.raw(name + "_scale")
            s = s.to(dev) if dev is not None else s
            return mxfp4_dequant(t, s, dtype)
        if t.dtype == torch.float8_e4m3fn:
            s = self.raw(name + "_scale_inv")
            s = s.to(dev) if dev is not None else s
            if name.endswith("self_attn.qkv_proj.weight"):
                c = self.cfg
                w = self._qkv(t, s, int(c["num_attention_heads"]), int(c["num_key_value_heads"]),
                              int(c["head_dim"]), int(c["v_head_dim"]))
                return w.to(dtype)
            w = t.float()
            s = s.float().repeat_interleave(128, 0)[: w.shape[0]].repeat_interleave(128, 1)[:, : w.shape[1]]
            return (w * s).to(dtype)
        return t.to(dtype)


def load_layer(ck: Checkpoint, cfg, il: int, dev) -> torch.nn.Module:
    """One decoder layer's own weights on the GPU (bf16); a MoE layer's mlp is replaced by Identity."""
    from modeling_mimo_v2 import MiMoV2DecoderLayer
    with torch.device("meta"):
        layer = MiMoV2DecoderLayer(cfg, il, attention_projection_layout=cfg.attention_projection_layout)
    moe = bool(cfg.moe_layer_freq[il])
    if moe:
        layer.mlp = torch.nn.Identity()
    layer = layer.to_empty(device=dev).to(DT)
    pref = P + f"layers.{il}."
    with torch.no_grad():
        for name, p in layer.named_parameters():
            k = pref + name
            if not ck.has(k):
                raise KeyError(f"checkpoint has no {k}")
            p.copy_(ck.get(k, dev=dev, dtype=p.dtype))
    layer.eval()
    return layer


class Stat:
    def __init__(self, n_in: int, dev):
        self.ss = torch.zeros(n_in, dtype=torch.float64, device=dev)
        self.n = 0

    def add(self, x: torch.Tensor):
        self.ss += x.double().square().sum(0)
        self.n += x.shape[0]


class StatMoE(torch.nn.Module):
    """MiMoV2MoE (sigmoid + noaux_tc with one group, norm top-k, x routed_scaling) recording its statistics."""

    def __init__(self, ck: Checkpoint, cfg, il: int, dev):
        super().__init__()
        L = P + f"layers.{il}.mlp."
        self.E, self.k = int(cfg.n_routed_experts), int(cfg.num_experts_per_tok)
        sf = cfg.routed_scaling_factor
        self.scale = 1.0 if sf is None else float(sf)
        self.norm = bool(cfg.norm_topk_prob)
        if int(cfg.n_group or 1) != 1:
            raise SystemExit("n_group > 1 is not implemented (MiMo-V2.6-Pro has 1)")
        H, I = int(cfg.hidden_size), int(cfg.moe_intermediate_size)
        self.H, self.I = H, I
        self.router = ck.get(L + "gate.weight", dev=dev, dtype=torch.float32)
        self.bias = ck.get(L + "gate.e_score_correction_bias", dev=dev, dtype=torch.float32)
        # 384 x 3 x 2048 x 6144 x 2 B = 29 GB bf16 on the device
        self.wgu = torch.empty(self.E, 2 * I, H, dtype=DT, device=dev)
        self.wd = torch.empty(self.E, H, I, dtype=DT, device=dev)
        for e in range(self.E):
            self.wgu[e, :I].copy_(ck.get(f"{L}experts.{e}.gate_proj.weight", dev=dev))
            self.wgu[e, I:].copy_(ck.get(f"{L}experts.{e}.up_proj.weight", dev=dev))
            self.wd[e].copy_(ck.get(f"{L}experts.{e}.down_proj.weight", dev=dev))
        self.collect = True
        self.ffn = None
        self.n = torch.zeros(self.E, dtype=torch.float64, device=dev)
        self.sal = torch.zeros(self.E, dtype=torch.float64, device=dev)
        self.imat_gu = torch.zeros(self.E, H, dtype=torch.float64, device=dev)
        self.imat_d = torch.zeros(self.E, I, dtype=torch.float64, device=dev)
        self.router_in = torch.zeros(H, dtype=torch.float64, device=dev)
        self.rin_n = 0

    def forward(self, hs: torch.Tensor) -> torch.Tensor:
        shape = hs.shape
        xb = hs.reshape(-1, shape[-1]).contiguous()
        if self.ffn is not None:
            mm, rows = self.ffn
            xnp = xb.view(-1, shape[1], shape[2]).view(torch.uint16).cpu().numpy()
            for i, s in enumerate(rows):
                if s >= 0:
                    mm[s] = xnp[i]
        xf = xb.float()
        scores = F.linear(xf, self.router).sigmoid()
        idx = torch.topk(scores + self.bias, self.k, dim=-1, sorted=False)[1]
        w = scores.gather(1, idx)
        if self.norm and self.k > 1:
            w = w / (w.sum(-1, keepdim=True) + 1e-20)
        w = w * self.scale
        flat = idx.reshape(-1)
        order = torch.argsort(flat)
        counts = torch.bincount(flat, minlength=self.E).tolist()
        tok = order // self.k
        wk = w.reshape(-1)[order]
        out = torch.zeros_like(xf)
        pos = 0
        for e in range(self.E):
            c = counts[e]
            if c == 0:
                continue
            t = tok[pos:pos + c]
            we = wk[pos:pos + c]
            pos += c
            xe = xb[t]
            gu = (xe @ self.wgu[e].t()).float()
            h = F.silu(gu[:, : self.I]) * gu[:, self.I:]
            y = (h.to(DT) @ self.wd[e].t()).float()
            if self.collect:
                self.n[e] += c
                self.imat_gu[e] += xe.double().square().sum(0)
                self.imat_d[e] += h.double().square().sum(0)
                self.sal[e] += (we.double() * y.double().norm(dim=1)).sum()
            out.index_add_(0, t, y * we[:, None])
        if self.collect:
            self.router_in += xf.double().square().sum(0)
            self.rin_n += xf.shape[0]
        return out.reshape(shape).to(hs.dtype)

    def stats(self):
        return {"n": self.n.cpu(), "sal": self.sal.cpu(), "imat_gu": self.imat_gu.float().cpu(),
                "imat_d": self.imat_d.float().cpu(), "router_in": self.router_in.float().cpu(), "rin_n": self.rin_n}


def hook_linear_stats(layer, collect):
    stats, hooks = {}, []
    for name, m in layer.named_modules():
        if isinstance(m, torch.nn.Linear):
            def h(mod, inp, out, name=name):
                if not collect[0]:
                    return
                x = inp[0].reshape(-1, inp[0].shape[-1])
                s = stats.get(name)
                if s is None:
                    s = stats[name] = Stat(x.shape[-1], x.device)
                s.add(x)
            hooks.append(m.register_forward_hook(h))
    return stats, hooks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/models/MiMo-V2.6-Pro-MOPD-heretic")
    ap.add_argument("--calib", default="/data/mimo/calib/calib.npy")
    ap.add_argument("--eval", default="/data/mimo/calib/eval.npy")
    ap.add_argument("--out", default="/data/mimo/calib/stats")
    ap.add_argument("--layers", default=None, help="a-b (default: all, resuming)")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--ffn-in-seqs", type=int, default=32)
    ap.add_argument("--n-cal", type=int, default=None)
    ap.add_argument("--n-eval", type=int, default=None)
    ap.add_argument("--ablate", default=None, help="[hidden, k] orthonormal basis projected out of the residual stream after"
                    " the embedding and after every layer (what quantize.py --ablate bakes into the writers)")
    ap.add_argument("--ablate-scale", type=float, default=1.0, help="fraction of the subspace component removed (1 = all)")
    a = ap.parse_args()
    sys.path.insert(0, HF)
    from configuration_mimo_v2 import MiMoV2Config
    from modeling_mimo_v2 import MiMoV2RotaryEmbedding

    dev = torch.device("cuda")
    ck = Checkpoint(a.model)
    cfg = MiMoV2Config.from_json_file(str(pathlib.Path(a.model) / "config.json"))
    cfg._attn_implementation = "eager"
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    calib, evals = np.load(a.calib), np.load(a.eval)
    if a.n_cal is not None:
        calib = calib[: a.n_cal]
    if a.n_eval is not None:
        evals = evals[: a.n_eval]
    seqs = np.concatenate([calib, evals], 0).astype(np.int64)
    n_cal, n_seq, T = calib.shape[0], seqs.shape[0], seqs.shape[1]
    H = int(cfg.hidden_size)
    n_layers = int(cfg.num_hidden_layers)
    n_ffn = min(a.ffn_in_seqs, n_cal)
    first, last = 0, n_layers - 1
    if a.layers:
        first, last = (int(v) for v in a.layers.split("-"))
    while first <= last and (out / f"stats_{first:02d}.pt").exists():
        first += 1
    pat = list(cfg.hybrid_layer_pattern)
    print(f"MiMo-V2.6-Pro calibration: layers {first}-{last} of 0-{n_layers - 1}, {n_cal} cal + {n_seq - n_cal}"
          f" eval x {T} tokens, batch {a.batch}; full layers {[i for i, p in enumerate(pat) if p == 0]},"
          f" SWA window {cfg.sliding_window}", flush=True)
    bufs = [out / "streams_a.bin", out / "streams_b.bin"]
    nbytes = n_seq * T * H * 2
    for b in bufs:
        if b.exists() and b.stat().st_size != nbytes:
            raise SystemExit(f"{b} holds {b.stat().st_size} bytes, expected {nbytes} - use an empty --out")
    if first > 0 and not bufs[0].exists():
        raise SystemExit("resuming but the hidden streams are gone - use an empty --out instead")
    mm = [np.memmap(b, dtype=np.uint16, mode="r+" if b.exists() else "w+", shape=(n_seq, T, H)) for b in bufs]
    Qab = None
    if a.ablate:
        Qab = torch.from_numpy(np.load(a.ablate).astype(np.float32))
        print(f"ablating a {Qab.shape[1]}-dim subspace after the embedding and every layer", flush=True)

    def proj(y: torch.Tensor) -> torch.Tensor:
        if Qab is None:
            return y
        q_ = Qab.to(y.device)
        yf = y.float()
        return (yf - a.ablate_scale * ((yf @ q_) @ q_.t())).to(y.dtype)

    if first == 0:
        emb = ck.get(P + "embed_tokens.weight", dtype=DT)
        for s in range(n_seq):
            mm[0][s] = proj(emb[torch.from_numpy(seqs[s])]).view(torch.uint16).numpy()
        mm[0].flush()
        del emb
    pos = torch.arange(T, device=dev).unsqueeze(0)
    dummy = torch.empty(1, 1, H, device=dev, dtype=DT)
    rope = {0: MiMoV2RotaryEmbedding(cfg, is_swa=False).to(dev)(dummy, pos),
            1: MiMoV2RotaryEmbedding(cfg, is_swa=True).to(dev)(dummy, pos)}
    i_ = torch.arange(T, device=dev)
    d = i_[:, None] - i_[None, :]
    neg = torch.finfo(DT).min
    masks = {0: torch.where(d >= 0, 0.0, neg).to(DT)[None, None],
             1: torch.where((d >= 0) & (d < int(cfg.sliding_window)), 0.0, neg).to(DT)[None, None]}

    for il in range(first, last + 1):
        t0 = time.time()
        src, dst = mm[il % 2], mm[(il + 1) % 2]
        layer = load_layer(ck, cfg, il, dev)
        moe = None
        if cfg.moe_layer_freq[il]:
            moe = StatMoE(ck, cfg, il, dev)
            layer.mlp = moe
        ffn = None
        if moe is not None and n_ffn:
            ffn = np.memmap(out / f"ffn_in_{il:02d}.bin", dtype=np.uint16, mode="w+", shape=(n_ffn, T, H))
        collect = [False]
        lin_stats, hooks = hook_linear_stats(layer, collect)
        t_load = time.time() - t0
        kind = pat[il]
        for b0 in range(0, n_seq, a.batch):
            ids = list(range(b0, min(n_seq, b0 + a.batch)))
            collect[0] = ids[0] < n_cal
            if moe is not None:
                moe.collect = collect[0]
                moe.ffn = None if ffn is None else (ffn, [s if s < n_ffn else -1 for s in ids])
            x = torch.stack([torch.from_numpy(np.asarray(src[s])) for s in ids]).view(DT).to(dev)
            with torch.no_grad():
                y = layer(x, attention_mask=masks[kind].expand(len(ids), 1, T, T), position_ids=pos,
                          past_key_values=None, use_cache=False, position_embeddings=rope[kind])
            y = proj(y[0] if isinstance(y, tuple) else y)
            a_np = y.contiguous().cpu().view(torch.uint16).numpy()
            for i, s in enumerate(ids):
                dst[s] = a_np[i]
        for hk in hooks:
            hk.remove()
        dst.flush()
        if ffn is not None:
            ffn.flush()
            del ffn
        st = {"layer": il, "linear": {k: {"ss": v.ss.float().cpu(), "n": v.n} for k, v in lin_stats.items()}}
        if moe is not None:
            st["moe"] = moe.stats()
        torch.save(st, out / f"stats_{il:02d}.pt")
        r = st.get("moe")
        extra = ""
        if r is not None:
            extra = f", tokens {int(r['n'].sum())}, experts used {int((r['n'] > 0).sum())}/{moe.E}"
        hn = torch.from_numpy(np.asarray(dst[0])).view(DT).float()
        print(f"layer {il:2d} ({'full' if kind == 0 else 'swa '}): {time.time() - t0:6.1f} s (load {t_load:.1f} s),"
              f" {len(lin_stats)} linears{extra}, |h| {hn.norm(dim=-1).mean():.1f}", flush=True)
        del layer, moe, lin_stats
        torch.cuda.empty_cache()
    print(f"gpu peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB", flush=True)
    if last == n_layers - 1:
        reference(ck, cfg, mm[(last + 1) % 2], seqs, n_cal, n_seq, T, a.topk, out, dev)
        print("CALIB DONE", flush=True)


def reference(ck, cfg, streams, seqs, n_cal, n_seq, T, topk, out, dev):
    """Eval sequences' next-token distributions (final norm + lm_head) and their NLL on the real next tokens."""
    if n_seq == n_cal:
        print("reference: no eval sequences, skipped", flush=True)
        return
    norm_w = ck.get(P + "norm.weight", dtype=torch.float32).to(dev)
    lm = ck.get("lm_head.weight", dev=dev)
    vals, idxs, lses, nll = [], [], [], []
    with torch.no_grad():
        for s in range(n_cal, n_seq):
            h = torch.from_numpy(np.asarray(streams[s])).view(DT).to(dev).float()
            h = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + float(cfg.layernorm_epsilon)) * norm_w
            lg = (h.to(DT) @ lm.t()).float()
            lse = torch.logsumexp(lg, -1)
            v, i = torch.topk(lg, topk, -1)
            vals.append((v - lse[:, None]).cpu().numpy())
            idxs.append(i.int().cpu().numpy())
            lses.append(lse.cpu().numpy())
            tgt = torch.from_numpy(seqs[s][1:]).to(dev)
            nll.append(float((lse[:-1] - lg[:-1].gather(1, tgt[:, None])[:, 0]).mean()))
    np.savez(out / "ref_topk.npz", logprob=np.stack(vals), ids=np.stack(idxs), lse=np.stack(lses))
    print(f"reference: {n_seq - n_cal} sequences x {T}, top-{topk}; NLL mean {np.mean(nll):.4f}"
          f" (ppl {np.exp(np.mean(nll)):.2f}), per seq {[round(x, 2) for x in nll]}", flush=True)


if __name__ == "__main__":
    main()
