# MiMo-V2.6-Pro Uncensored on Maya

**Xiaomi's ~1-trillion-parameter MiMo-V2.6-Pro (abliterated "heretic" base), with its own vision, running at 58–85 tokens/s on seven GPUs with a 512K context.**

This repository is a fork of [Project Maya](https://github.com/mw00/project-maya) (MIT), the engine that runs large MoE models on your own GPUs. Maya did not know MiMo. Here the engine is ported to **MiMo-V2.6-Pro** (architecture `mimo2`: 70 layers, 384 experts, hybrid sliding-window + full attention with sinks, three multi-token-prediction layers, a native vision tower), a calibrated ~2-bit quant is built for it (300 GB, Russian kept intact), and decoding is made fast enough for daily use — from 31 tok/s at the first working build to 58–85 tok/s now, and from 20 to 38–40 tok/s at 200K context.

[Русская версия](README.ru.md) · [How it was made](docs/HOW-IT-WAS-MADE.md) · [Engine port](docs/MIMO2-ENGINE-PORT.md) · [Model on Hugging Face](#download-the-model) · [GLM-5.3 Uncensored on Maya](https://github.com/denisnotes357-prog/GLM-5.3-Uncensored-Maya) · [Original Project Maya README](README.project-maya.md)

---

## Results

Measured on 5× RTX 4090 48 GB + 2× A100 40 GB (320 GB VRAM), 503 GB RAM; 400-token answers unless noted.

| | **MiMo-V2.6-Pro Uncensored (this repo)** |
| --- | ---: |
| Refusals on 40 JailbreakBench requests | 24 / 40 (the abliterated base; a runtime refusal vector is [in development](#refusals-and-the-refusal-vector)) |
| Factual questions / questions about things that do not exist | 30 / 30 correct / 6 of 10 answered "unknown" honestly |
| Long answers (3 × ~9000-character Russian scenes, temperature 1, `rep_test.py`) | no repeated lines or broken words (identical to plain sampling); an earlier check that only looked for adjacent repeats missed a bug (see *Speculative sampling* under What is in here) |
| Decode, Russian / code, temperature 0 | **63–69 / 85 tok/s** |
| Decode, Russian / code, temperature 1 | **58–63 / 81 tok/s** |
| Decode when the answer repeats the context (code edits, proofreading) | **106–116 tok/s** |
| Decode at 100K / 200K context | **38 / 38–40 tok/s** |
| Prompt reading | **1500–1900 tok/s**; a 200K prompt in 155 s; a short chat turn in 0.15 s |
| A new question to an already-read 200K conversation | **~1 s** (only its own tokens are read) |
| An image | **~1.6 s** to encode (the model's own vision tower, on the GPU) |
| Context | **512K** (with the INT8 KV cache a few hundred experts live in RAM and are fetched when routed) |
| Russian / English perplexity (full-precision model ≈ 3.8 in Russian) | 4.03 / 4.25 |
| Size | 300 GB GGUF + 0.3 GB pack + 1.5 GB vision tower |

Speed history on that machine: 31 tok/s (plain decoding) → 45 / 59 (one MTP layer, pipelined window) → 52 / 66 (all three MTP layers, as the model was trained) → **63–69 / 85** at t = 0 (no CPU lane, prompt lookup). Long context: 7.4 tok/s at 70K → 32 (split-K attention) → **38–40 at 200K** (tensor-core attention + INT8 KV cache).

## What is in here

- **Engine port to `mimo2`** (`src/core/glm_*.cu`, `src/kernels/cuda/mimo_*.cu`, `include/strata/kernels/mimo_*.hpp`): MiMo-V2.6-Pro inside Maya's `Glm5Model`, behind an architecture flag, so the GLM models keep working and MiMo inherits Maya's expert tiers, layer split across GPUs, OpenAI/Anthropic-compatible server and browser chat. Sliding-window layers keep a 256-row ring per layer; full-attention layers keep the whole context.
- **All three native MTP layers** (`STRATA_MIMO_MTP3=1`): layer *k* drafts depth *k* from the trunk's hidden state, which is how MiMo was trained (chaining one layer three times leaves depth 2–3 nearly useless: 16 % / 3 % against 56 % / 54 % in a `transformers` reference). The drafts are verified as one window flowing through all seven GPUs.
- **Speculative sampling** of the drafts at temperature > 0 (`STRATA_GLM_SPEC_SAMPLE=1`) is **off in the shipped config**: together with prompt-lookup drafts it garbles long answers at temperature 1 (lines repeat, words break up; found on 2026-10-19 with long Russian scenes). Each alone is clean; the fix of the combination is in progress. With it off, sampled answers are exactly plain sampling (same text for the same seed).
- **Prompt-lookup drafts** (`STRATA_GLM_LOOKUP=1`): when the text repeats something from the conversation, up to 7 tokens of it are drafted at once.
- **Batched prompt reading** for `mimo2` (sliding-window + sink attention on tensor cores, chunks of 4096) and a pipelined window for short pieces (a new chat turn).
- **Tensor-core decode attention** for the full-attention layers (the 16 query heads of a KV head form one `mma` tile), memory-bound at 200K context; **INT8 KV cache** for those layers (`STRATA_MIMO_KV8=1`) — half the memory and traffic.
- **Conversation checkpoints**: the sliding-window rings at the start of the last turn are kept in RAM, so a different follow-up to a long conversation reads only its own tokens.
- **Native vision** (`tools/mimo/mimo_vision.py`): the checkpoint's own vision transformer, run on the first GPU in VRAM the engine lends it while a picture is encoded; a pre-warmed process forks one encoder per picture.
- **Runtime refusal-vector hook** (`STRATA_GLM_ABLATE*`, off in the shipped config): projects refusal directions out of the residual stream; the MiMo vector is still in development (see below).
- **Quantization toolchain** (`tools/mimo/`): calibration statistics with Russian text, imatrix/GPTQ quantization, the MTP-layer fix, refusal-direction capture and derivation, reference scripts, benchmarks.
- **Tests**: decode and prefill attention against FP64 references (`tests/mimo_decode_attn_test.cu`, `tests/mimo_prefill_attention.cu`), the `mimo2` geometry loader (`tests/mimo_geometry_test.cpp`).

## Fixes of 2026-10-20 (update if you cloned earlier)

- **Agent sessions could turn into garbage.** When one client's new conversation was dropped while its prompt was being read, the engine kept the previous conversation's snapshot record although it had already reset the caches; that conversation's next turn restored its sliding-window rings over zeroed full-attention caches and answered with nonsense from the first token (invented tool results, tools that do not exist). A reset now invalidates the snapshot. Reproduced with the exact request sequence (2 of 2 broken before, correct after).
- **`<think>` leaked into the reasoning.** MiMo opens its thinking itself; the server passed that tag on, agents sent it back in the history, and the model doubled it every turn. The server now strips it from answers and from returned history.
- Speculative sampling is off in the shipped config (see below).

## Quick start

### 1. Build the engine

The same installer as upstream Maya (Linux x86-64, NVIDIA driver, CUDA 12.x or 13, Python 3.10+; see [Project Maya's requirements](README.project-maya.md)):

```bash
git clone https://github.com/denisnotes357-prog/MiMo-V2.6-Pro-Uncensored-Maya
cd MiMo-V2.6-Pro-Uncensored-Maya
./setup.sh
```

It creates the Python environment `.venv`, fetches ggml at the pinned llama.cpp commit and builds `build/strata`. When it offers to download a model, answer **no** — it would fetch a different model. Rebuilding later: `cmake --build build -j --target strata`. The port was built and tested for `sm_80` (A100) and `sm_89` (RTX 4090).

For images, the vision tower runs in PyTorch inside the same environment:

```bash
.venv/bin/pip install torch transformers safetensors pillow
```

### 2. Download the model

```bash
pip install -U huggingface_hub
hf download nofacedfl/MiMo-V2.6-Pro-Uncensored-Maya --local-dir models/MiMo-V2.6-Pro-Uncensored-Maya
```

About 302 GB: seven GGUF shards, `pack/` (the engine's index of those shards, the dense weights and the tokenizer; it finds the shards next to itself) and `vision/` (the vision tower).

### 3. Run

```bash
tools/mimo/run-mimo-pro.sh
```

- Chat in the browser: http://127.0.0.1:8098/
- OpenAI API: `http://127.0.0.1:8098/v1/chat/completions`, model `mimo-pro` (images as `image_url`)
- Anthropic API: `http://127.0.0.1:8098/v1/messages` (Claude Code and other agents: `ANTHROPIC_BASE_URL=http://127.0.0.1:8098`)
- Thinking: `reasoning_effort` = `none` / `low` / `high` (default) / `max`, or `chat_template_kwargs: {"enable_thinking": false}`
- Live speed monitor: http://127.0.0.1:8098/#monitor

Loading takes 2–3 minutes. Stop any other GPU job first; Ctrl+C stops the model.

### Fitting your GPUs

The shipped config is for 7 GPUs with ~320 GB of VRAM (GPU 1 of that machine left out). The keys that matter:

| Key | Meaning |
| --- | --- |
| `gpu`, `CUDA_VISIBLE_DEVICES` | the cards along the layer pipeline |
| `STRATA_GLM_SPLIT=11,22,33,44,53,61` | the first layer of each next GPU (the two A100s get fewer layers) |
| `--max-context 512000` | with `STRATA_MIMO_KV8=1`; on this machine ~200–330 experts then live in RAM (the `glm stat` line shows `vram`/`ram`) |
| `STRATA_MIMO_MTP3=1`, `STRATA_GLM_MTP_PIPE=1`, `STRATA_GLM_MTP_DRAFT=7` | native three-layer MTP drafting, verified as a pipelined window |
| `STRATA_GLM_SPEC_SAMPLE=0`, `STRATA_GLM_LOOKUP=1` | speculative sampling off (with lookup it garbles long sampled answers); prompt-lookup drafts on |
| `STRATA_GLM_CPU_LANE=0`, `STRATA_GLM_PREFILL_CPU=0` | no experts on the CPU — faster here (greedy output still varies slightly between runs on the seven-GPU pipeline) |
| `STRATA_MIMO_KV8=1` | INT8 KV cache for the 10 full-attention layers |
| `vision.gpu: true` | the vision tower on the first GPU (5.7 GB lent by the engine per picture); `"device": N` runs it on a card of its own instead, with nothing lent |
| `STRATA_MIMO_DEV=1` | the `mimo2` path is marked experimental; the engine refuses it without this |

With less VRAM, Maya's tiers still work: experts that do not fit are served from RAM (slower).

## Refusals and the refusal vector

The weights are [`pqhaz/MiMo-V2.6-Pro-MOPD-heretic`](https://huggingface.co/pqhaz/MiMo-V2.6-Pro-MOPD-heretic) (MIT): Xiaomi's MiMo-V2.6-Pro-MOPD with a two-stage directional ablation of its refusal behaviour in the attention output projections. Through our quant it answers most requests the original refuses, but still refuses **24 of 40** JailbreakBench requests.

**A runtime refusal vector is in development and is not shipped.** We measured six refusal directions per layer (harmful vs harmless instructions, `tools/mimo/capture.py`, `derive.py`) and projected them out while generating. It brought refusals to 1–5 / 40, but the tests beyond refusal counts showed two side effects:

- with the vector on every token, 1000+-token Russian answers started to loop and coin words (limiting it to the first 64 answer tokens fixed that, `STRATA_GLM_ABLATE_TOKENS`);
- even then, the model stopped saying "I don't know": asked about people, events and films that do not exist, it answered honestly in **1 of 10** cases with the vector against **6 of 10** without it, and invented confident details instead. Refusing and admitting ignorance share a direction.

The next version will remove refusals without that side effect, checked on long answers, invented facts and agent tasks, not only on refusal counts. The engine hook stays available (`STRATA_GLM_ABLATE`, `_SCALE`, `_LAYERS`, `_TOKENS`) for anyone experimenting.

The quant itself is calibrated with ~14 % Russian text: an earlier quant calibrated on English only had a Russian perplexity of 44.8. Details, numbers and dead ends: [docs/HOW-IT-WAS-MADE.md](docs/HOW-IT-WAS-MADE.md).

## Download the model

**Hugging Face:** [nofacedfl/MiMo-V2.6-Pro-Uncensored-Maya](https://huggingface.co/nofacedfl/MiMo-V2.6-Pro-Uncensored-Maya)

| File | Size |
| --- | ---: |
| `MiMo-V2.6-Pro-heretic-M23B-0000{1..7}-of-00007.gguf` | 300 GB |
| `pack/` (engine index + dense weights + tokenizer + chat template) | 0.3 GB |
| `vision/` (the vision tower, 364 `visual.*` tensors in BF16, + config) | 1.5 GB |

Quant recipe (`tools/mimo/recipes/m23.json`): routed experts IQ2_XXS gate/up and IQ2_S down (IQ2_XS / IQ3_XXS in the first and last layers), imatrix/GPTQ against full-precision calibration statistics; attention and the MTP layers Q4_K; router and norms F32. The tensor layout follows llama.cpp's `mimo2`; the tested runtime is this engine.

## Responsible use

This model answers many requests that the original refuses. It is published for research, red-teaming, private use and creative work by adults. You are responsible for how you use it and for complying with the laws that apply to you. The output can be wrong or harmful; do not rely on it for medical, legal, safety-critical or other high-stakes decisions.

## Credits and licenses

- **Project Maya** by Niko1221 and the Strata contributors — the engine, server and quant tools this repo is built on. MIT, see [LICENSE](LICENSE).
- **MiMo-V2.6-Pro** by [Xiaomi MiMo](https://huggingface.co/XiaomiMiMo) — the model (MIT); its modeling code in `tools/mimo/hf/` (Apache-2.0).
- **pqhaz** — [MiMo-V2.6-Pro-MOPD-heretic](https://huggingface.co/pqhaz/MiMo-V2.6-Pro-MOPD-heretic), the attention-ablated base (MIT), using [Heretic](https://github.com/p-e-w/heretic).
- **alesha-pro** — the runtime ablation hook for Maya, [project-maya `glm-ablate`](https://github.com/alesha-pro/project-maya/blob/glm-ablate/docs/GLM-ABLATE.md).
- **llama.cpp** and **transformers** — the reference implementations the port was checked against.

See [NOTICE.md](NOTICE.md) for the full list. If this project is useful to you, a ⭐ helps others find it.
