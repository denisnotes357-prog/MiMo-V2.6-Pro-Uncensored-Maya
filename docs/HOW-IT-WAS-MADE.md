# How MiMo-V2.6-Pro Uncensored on Maya was made

The goal: Xiaomi's MiMo-V2.6-Pro — about a trillion parameters — running uncensored on one workstation, fast enough to
chat with and to drive coding agents, in Russian as well as in English, with its own vision and a long context.

The machine: 5× RTX 4090 48 GB + 2× A100 40 GB (320 GB VRAM, PCIe, no NVLink, no peer-to-peer between the 4090s),
503 GB RAM. Every number below was measured there.

## 1. The starting point

[`pqhaz/MiMo-V2.6-Pro-MOPD-heretic`](https://huggingface.co/pqhaz/MiMo-V2.6-Pro-MOPD-heretic) is Xiaomi's
MiMo-V2.6-Pro-MOPD with a two-stage refusal ablation in the 70 attention output projections; everything else is the
original checkpoint (routed experts MXFP4, attention and dense weights FP8, embeddings and `o_proj` BF16, ~1 TB).

The architecture, `mimo2`, is neither GLM nor Qwen:

| | |
| --- | --- |
| Layers | 70 + 3 multi-token-prediction (MTP) layers |
| Experts | 384 routed, top-8, expert FFN 2048; hidden 6144 |
| Attention | GQA 128 query / 8 KV heads, q·k 192, v 128, a value scale |
| Hybrid | 60 sliding-window layers (window 128, a learned sink logit per head) + 10 full-attention layers (0, 7, 15, 23, 31, 39, 47, 55, 62, 69) |
| Vision | its own ViT (28 blocks, 739M parameters, windowed + full attention), merged 2×2 into 6144-wide tokens |

Maya (the engine) ran GLM models only. llama.cpp knew `mimo2` and was the math reference, together with `transformers`.

## 2. The quant

To fit 320 GB the experts had to go to ~2 bits. Recipe `tools/mimo/recipes/m23.json`: gate/up IQ2_XXS, down IQ2_S
(IQ2_XS / IQ3_XXS in the first and last layers), attention and the MTP layers Q4_K, router and norms F32. Per-expert
importance and GPTQ rounding use activation statistics collected from the full-precision model on one GPU
(`tools/mimo/calib.py`, `quantize.py`). Result: 300 GB in seven GGUF shards.

Two problems on the way:

- **The MTP layers came out broken.** llama.cpp's converter dequantized the fused q/k/v of the three MTP layers as plain
  block-FP8, but the checkpoint stores them tensor-parallel-aware; the queries, keys and values were mixed up (relative
  error 1.41). Draft acceptance was 2–6 %. `tools/mimo/check_mtp_qkv.py` found it, `fix_mtp_qkv.py` rewrites those three
  tensors (relative error 0.072, the same as an ordinary layer); acceptance jumped to 68–84 % at depth 1.
- **Russian broke.** The first quant (calibrated on English, with a refusal vector baked into the weights at strength 1.0)
  had a Russian perplexity of 44.8 against ~3.8 for the full model. Two causes, measured separately: the vector alone
  raised it to 9.3, the English-only calibration did the rest. The final quant M23B is calibrated with ~14 % Russian
  text and has no baked vector: **Russian perplexity 4.03**.

## 3. Refusals (the vector is in development)

Through our quant the heretic base still refused 24 of 40 JailbreakBench requests (judged by the same quant without any
vector). Six refusal directions per layer were measured from the residual stream on harmful vs harmless instructions
(`tools/mimo/capture.py`, `derive.py`); the engine projects them out after every layer while generating
(`STRATA_GLM_ABLATE`), so the strength is a config value:

| Strength | Russian perplexity (KL to the full model) | English perplexity (KL) | Refusals / 40 |
| --- | --- | --- | --- |
| 0 | 4.03 (0.17) | 4.25 (0.20) | 24 |
| 0.25 | 4.44 (0.52) | 3.95 (0.52) | 5 |
| **0.35 (shipped)** | **5.12 (0.71)** | **4.06 (0.60)** | **1** |
| 0.5 | 7.31 (1.08) | 4.25 (0.70) | 1 |

Strength 1.0, the usual default, was 3–4× too strong for this model. Harmless prompts were untouched at every strength.

**Long answers.** These numbers come from short answers. A book-writing test showed what they miss: with the vector on
every token, 1000+-token Russian answers at temperature 1 started to loop ("Яма. Яма. Яма.") and coin words in 2–3 of
4 stories — at strength 0.35, and still at 0.2 or with the vector on layers 30–60 only; plain decoding without drafts did
the same, without the vector nothing broke. A refusal is decided at the start of an answer, so the vector now steers the
prompt and the first 64 answer tokens only (`STRATA_GLM_ABLATE_TOKENS`): 0 of 4 stories degraded, 5 / 40 refusals
(256 tokens: 1 of 4 degraded, 4 / 40).

**Invented facts.** A hallucination check (30 factual questions, 10 about people, events and films that do not exist)
gave 30 / 30 facts either way, but with the vector the model admitted not knowing in **1 of 10** trap questions against
**6 of 10** without it — it invented confident details instead. Refusing and saying "I don't know" share a direction.
So the release ships **without** the vector (24 / 40 refusals from the abliterated base); a vector that leaves honesty
intact, checked on long answers, invented facts and agent tasks, is the next step.

## 4. The engine port

`mimo2` lives inside Maya's `Glm5Model` behind an architecture flag (`STRATA_MIMO_DEV=1`): a geometry loader, a slow
reference path, the fast path with its own attention (sliding-window ring of 256 rows per layer, sink logits, value
scale), dense MTP layers with their own rings, and the vision protocol. Correctness was checked step by step: a tiny
synthetic `mimo2` against `transformers` (top-1 96/96, max |Δlogit| 2·10⁻⁶), fast path against slow path, the MTP layer
against a reference (93.7 % of drafts identical), then the real quant on seven GPUs: **31 tok/s** without MTP.

One crash on the way: a conversation snapshot copied GLM's indexer pool, which `mimo2` does not have — prompts of more
than ~1000 tokens segfaulted. The snapshot of `mimo2` is its sliding-window rings.

## 5. Making it fast

| Step | Result |
| --- | --- |
| Pipelined MTP window (drafts verified as one window through all 7 GPUs) | 45 / 59 tok/s (Russian / code) |
| Batched prompt reading for `mimo2` (sliding-window + sink attention on tensor cores) | prompt reading 23–26 → 1250–1800 tok/s |
| Split-K, GQA-sharing decode attention for the 10 full layers | 70K context: 7.4 → 32 tok/s |
| **All three MTP layers, as trained.** A `transformers` reference over the trunk's hidden states showed that chaining layer 0 three times gives 16 % / 3 % at depths 2 / 3, while layer *k* at depth *k* gives 56 % / 54 %. | 52 / 66 tok/s |
| The MTP layers' rings filled while the prompt is read (no cold start) | depth-2 acceptance 42 → 56 % |
| Speculative sampling of drafts at temperature > 0 (a buffer bug made it produce garbage before) | t = 1: +5–12 % — **off in the release**: with prompt lookup it still garbles long answers (see section 6) |
| A short chat turn through the pipelined window instead of token by token | 0.5 → 0.15 s |
| No experts on the CPU (they made greedy output depend on timing, and cost 0.8 ms a token) | 57 / 73 tok/s |
| Prompt-lookup drafts (repeat the context 7 tokens at a time) | code edits 81 → 106, proofreading 79 → 116 tok/s |
| Vision on the GPU (the engine lends 5.7 GB per picture) + a pre-warmed process that forks the encoder | 12 → 1.6 s an image |
| **Tensor-core decode attention.** The old kernel read the KV cache at only 0.3 TB/s — it was compute-bound. The new one puts the 16 query heads of a KV head into one `mma` tile and reaches 0.94 TB/s. | 200K: 20 → 33 tok/s |
| INT8 KV cache for the full layers (an FP32 scale per position and head) | 200K: 38–40 tok/s; 512K fits (~200–330 experts in RAM) |
| Conversation checkpoints at the start of the last turn | another question to 200K: 3 min → ~1 s |

## 6. Dead ends (measured, not kept)

- **Layer 0 at every depth without chaining**: no gain — its ring holds depth-1 entries.
- **Longer or shorter drafts** (3–7): no difference; the per-depth acceptance is what matters.
- **Moving layers from the A100s to the 4090s.** An A100 layer costs ~0.86 ms against ~0.43 ms on a 4090 (the IQ2
  unpacking is compute-bound there), but the 4090s are full of experts: moving three layers cost more in RAM fetches
  than it saved; moving one layer with a 256K context gained 2–3 %.
- **INT8 KV on the old attention kernel**: slower than FP16 — the kernel was not limited by memory.
- **Periodic checkpoints every 16K tokens**: each one ends a prompt-reading call, and a seven-GPU pipeline drains between
  calls — a 200K prompt took 286 s instead of 191. Only turn boundaries are checkpointed now.
- **DFlash** (MiMo ships a 5-layer block-diffusion drafter): on a PCIe pipeline every extra verified row costs ~7 ms,
  so 8-token blocks would add ~10 % on code and almost nothing on prose. Not ported yet.

## 7. How correctness was checked

- Kernels against FP64 CPU references (`tests/mimo_decode_attn_test.cu`, `tests/mimo_prefill_attention.cu`).
- Greedy output token-for-token equal between paths (prompt window vs token path, lookup drafts vs plain decoding).
- Speculative sampling vs plain sampling: the same digit frequencies (χ² 3.9, threshold 16.9) and early-stop rates. That
  check used short answers; 9000-character scenes at temperature 1 later showed that speculative sampling together with
  prompt lookup breaks long text (lines repeated up to 15 times, words falling apart). It is off in the release.
- Greedy output is still not bit-for-bit repeatable between requests on the seven-GPU pipeline (the same question at
  temperature 0 gave 403–470 tokens over six runs, every answer correct), whatever the drafting, KV or expert settings:
  numeric order, not a leaked state - the first request after a start is always the same.
- Long answers: `tools/mimo/bench/rep_test.py` (repeated lines anywhere in the text, not only adjacent ones, and
  Latin letters inside Russian), compared with plain sampling of the same seeds.
- A fact hidden in the middle of 100K tokens of text is found (`tools/mimo/bench/needle.py`).
- Long-context speed measured on a second question to the same context, so that the model cannot just copy the text.
