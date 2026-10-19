# The `mimo2` port of the Maya engine

How MiMo-V2.6-Pro runs inside Maya's `Glm5Model` (`src/core/glm_*.cu`), for people reading or extending the code.
The GLM paths are unchanged; everything below is selected by `Glm5Geometry::Glm5Arch::Mimo2` and needs
`STRATA_MIMO_DEV=1`.

## Geometry and weights

- GGUF architecture `mimo2` (llama.cpp's layout). `n_embd` 6144, 70 trunk layers + `nextn_predict_layers` = 3 MTP layers
  (`blk.70`–`blk.72`), 384 experts top-8 (a router with a selection bias), expert FFN 2048, no shared expert.
- Attention: 128 query / 8 KV heads, `head_k` 192, `head_v` 128, a `v_scale` on the attention output, RoPE on the first
  `rope_first` dims (separate bases for sliding-window and full layers), a learned sink logit per head on the
  sliding-window layers. `hybrid_layer_pattern` marks the 10 full layers.
- `tools/iq_pack.py` builds the pack (dense weights + an index of the expert offsets inside the shards);
  `--compat-bf16` stores the small projections the kernels read as BF16.

## Attention and the KV cache

- **Sliding-window layers** keep a ring of `min(max_ctx, swa_ring())` = 256 rows (window 128 + the verify window's
  margin), FP16, K `[ring][8][192]`, V `[ring][8][128]` at `dsa_lat_[il]` / `dsa_ik_[il]`.
- **Full layers** keep every position. FP16 by default; with `STRATA_MIMO_KV8=1` int8 K and V plus an FP32 scale per
  (position, KV head) (`include/strata/core/mimo_kv8.hpp`): half the bytes. Written once after RoPE
  (`mimo_kv8_store` / the prefill's `mimo_prefill_rope_kv8`).
- **Decode attention** (`src/kernels/cuda/mimo_decode_attn.cu`): split-K over the keys, GQA-shared (one block reads a key
  tile once for all 16 query heads of a KV head). For one row (the token path and every row of the MTP window)
  `dec_partial_mma<KV8>` runs Q·Kᵀ and P·V on `mma.m16n8k16` with the 16 heads as the m16 tile, P kept in registers, 64-key
  tiles through shared memory with `cp.async` double buffering (int8 tiles are dequantized once into the FP16 tile);
  `dec_combine` merges the slices, the sink and `v_scale`. `STRATA_MIMO_DEC_MMA=0` selects the scalar kernel.
- **Prompt attention** (`src/kernels/cuda/mimo_prefill.cu`): chunks of up to 4096 rows, tensor-core tiles, the window
  and the sink handled in the kernel, the full layers reading their past from the cache (FP16 or int8).

## Multi-token prediction

Each MTP layer is a dense block with its own sliding-window ring: input `eh_proj(cat(enorm(emb(t)), hnorm(h)))`.
MiMo was trained with **layer *k* predicting depth *k* from the trunk's hidden state** (`STRATA_MIMO_MTP3=1`):

- In a round at position *p*, layer 0 drafts from the trunk state `h(p−1)` and the last token; layer *k* from the same
  `h(p−1)` and draft *k−1*. Layer 2 first needs a provisional ring entry at *p−2* with draft 1.
- After the verify, the rings of layers 1 and 2 are rewritten with the accepted tokens (entry *q* of layer *k* holds
  `h(q)` and token *q+1+k*); while a prompt is read, the same entries are filled from the prompt (`glm_prefill.cu`).
- `decode_mtp_pipe` (`src/core/glm_mtp.cu`) pushes the drafts into the pipelined window (`win_pipe_*`, a thread per GPU
  part) as soon as each is out; the first wrong draft cancels the rows behind it.
- `STRATA_GLM_SPEC_SAMPLE=1`: at temperature > 0 the drafts are drawn from the MTP layers' own distributions and accepted
  with probability min(1, p/q) (residual draw on rejection). **Known bug:** combined with `STRATA_GLM_LOOKUP=1` it
  garbles long answers at temperature 1, so the shipped config keeps it off.
- `STRATA_GLM_LOOKUP=1`: a trigram index over the prompt and the written tokens; when the last three tokens occurred
  before, the up to 7 tokens that followed become the drafts of that round (block 0 still steps once so the MTP state
  stays consistent), verified like a plain draw. Chosen over MTP while its recent rounds kept more tokens.

## Prompts, turns and checkpoints

- Pieces shorter than `STRATA_GLM_PREFILL_MIN` (32 tokens — a chat turn) go through the pipelined window, 8 tokens a
  window, instead of one token at a time through all seven parts (`prompt_window`).
- A conversation snapshot (the next turn continues from it) and checkpoints (RAM, `ckpt_*`) hold the sliding-window rings
  of the trunk and the MTP layers (~80 MB); the full layers' caches are positional and stay in place. Checkpoints are taken
  at the start of the last user/tool turn (`<|im_start|>` + role), so a different follow-up reads only its own tokens.
  Periodic checkpoints are off for `mimo2` (`STRATA_GLM_CKPT_EVERY`): they split prompt reading into short calls and the
  seven-part pipeline drains between calls.

## Vision

`tools/mimo/mimo_vision.py` speaks Maya's `strata-vision` protocol with the checkpoint's own `MiMoVisionTransformer`
(code in `tools/mimo/hf/`, weights in the model's `vision/` folder, made by `tools/mimo/extract_vision.py`). With
`vision.gpu: true` the server measures the encoder's peak memory at start (`--measure`), the engine lends that much of the
first GPU's expert pool while a picture is encoded and takes it back afterwards. A pre-warmed process (PyTorch and the
tower loaded, no CUDA) forks one encoder per picture: ~1.6 s from request to features. Image rows replace the
`<|image_pad|>` tokens on plain 1-D positions.

## The refusal vector (experimental, not shipped)

`STRATA_GLM_ABLATE=<file>`, `STRATA_GLM_ABLATE_LAYERS=70`, `STRATA_GLM_ABLATE_SCALE=s`: after every layer the residual
stream loses `s` times its projection onto that layer's directions (`src/core/glm_ablate.cuh`; for MiMo 6 directions a
layer, `70 × 6 × 6144` float32). `STRATA_GLM_ABLATE_TOKENS=N` keeps it on for the prompt and the answer's first N tokens
only (`glm_ablate::live`, switched from `generate.cpp`).

## Switches for A/B tests

`STRATA_GLM_NO_SPEC=1` (plain decoding), `STRATA_MIMO_DEC_MMA=0`, `STRATA_GLM_PROMPT_WIN=0`, `STRATA_GLM_LOOKUP=0`,
`STRATA_GLM_PROF=16` (per-GPU, per-phase timing; printed when the engine shuts down), `STRATA_GLM_MTP_TRACE=1`.
