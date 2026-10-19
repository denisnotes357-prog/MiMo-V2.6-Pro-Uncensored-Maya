---
license: mit
base_model:
- pqhaz/MiMo-V2.6-Pro-MOPD-heretic
- XiaomiMiMo/MiMo-V2.6-Pro-MOPD
tags:
- mimo
- mimo2
- uncensored
- abliterated
- gguf
- moe
- vision
- maya
pipeline_tag: image-text-to-text
language:
- en
- ru
- zh
---

# MiMo-V2.6-Pro Uncensored on Maya

Xiaomi's ~1-trillion-parameter **MiMo-V2.6-Pro** (the abliterated "heretic" base), quantized to ~2 bits (300 GB) for the
[Maya](https://github.com/denisnotes357-prog/MiMo-V2.6-Pro-Uncensored-Maya) engine, with its own vision tower. On 5× RTX 4090 48 GB + 2× A100 40 GB: **55–58 tok/s in Russian, 71–75 tok/s on code,
106–116 tok/s when the answer repeats the context, 38–40 tok/s at 200K context**, 512K context, an image in ~1.6 s.

**Engine, code and full documentation:** https://github.com/denisnotes357-prog/MiMo-V2.6-Pro-Uncensored-Maya

## Files

| File | Size | What |
| --- | ---: | --- |
| `MiMo-V2.6-Pro-heretic-M23B-0000{1..7}-of-00007.gguf` | 300 GB | the quantized model (llama.cpp `mimo2` layout) |
| `pack/` | 0.3 GB | Maya's index of the shards, the dense weights in the layout the kernels read, the tokenizer and chat template |
| `vision/` | 1.5 GB | the model's own vision transformer (364 `visual.*` tensors, BF16, byte-identical to the checkpoint) + config |

## Use

```bash
git clone https://github.com/denisnotes357-prog/MiMo-V2.6-Pro-Uncensored-Maya
cd MiMo-V2.6-Pro-Uncensored-Maya && ./setup.sh            # answer "no" to its model download
.venv/bin/pip install torch transformers safetensors pillow   # for images
hf download nofacedfl/MiMo-V2.6-Pro-Uncensored-Maya --local-dir models/MiMo-V2.6-Pro-Uncensored-Maya
tools/mimo/run-mimo-pro.sh                                # http://127.0.0.1:8098 — chat, OpenAI and Anthropic APIs
```

The shipped config (`configs/mimo-pro-uncensored.json`) is for 7 GPUs with ~320 GB VRAM; adjust the GPU list and the
layer split to yours.

The GGUF shards follow llama.cpp's `mimo2` layout, but this release is built and tested only with the Maya engine (its
MTP drafting, attention kernels, INT8 KV cache and vision path are what the numbers above measure).

## How it was made

- **Base:** [pqhaz/MiMo-V2.6-Pro-MOPD-heretic](https://huggingface.co/pqhaz/MiMo-V2.6-Pro-MOPD-heretic) — Xiaomi's
  MiMo-V2.6-Pro-MOPD with a two-stage refusal ablation in the attention output projections.
- **Quant M23B:** routed experts IQ2_XXS gate/up and IQ2_S down (IQ2_XS / IQ3_XXS in the first and last layers),
  imatrix/GPTQ against statistics of the full-precision model, calibration with ~14 % Russian text; attention and the
  three MTP layers Q4_K (their fused q/k/v re-dequantized tensor-parallel-aware — the stock conversion scrambles them);
  router and norms F32.

**Refusals:** the abliterated base refuses 24 of 40 JailbreakBench requests through this quant. Facts 30 / 30 correct;
about things that do not exist it answers "unknown" honestly in 6 of 10 cases; long Russian answers stay clean.
A runtime refusal vector is in development: it cut refusals to 1–5 / 40 but made the model invent confident answers
about non-existent things (honest 1 / 10), so it is not shipped.

Details: [How it was made](https://github.com/denisnotes357-prog/MiMo-V2.6-Pro-Uncensored-Maya/blob/main/docs/HOW-IT-WAS-MADE.md).

## Limits

- ~2-bit experts: the answers are good but not identical to the full model; at temperature 1 Russian sometimes picks
  odd words. Lower the temperature for precise work.
- The numbers are from one machine (7 GPUs, PCIe). With less VRAM Maya serves the experts that do not fit from RAM,
  more slowly.
- No capability benchmarks were run beyond perplexity, a 100K-token retrieval test and manual checks.

## Responsible use

This model answers requests that the original refuses. It is published for research, red-teaming, private use and
creative work by adults. You are responsible for how you use it and for complying with the laws that apply to you.
Its output can be wrong or harmful; do not rely on it for medical, legal, safety-critical or other high-stakes decisions.

## Credits and license

MIT, like the base model ([license text](https://github.com/denisnotes357-prog/MiMo-V2.6-Pro-Uncensored-Maya/blob/main/LICENSES/MiMo-V2.6-Pro-LICENSE.md)).
[Xiaomi MiMo](https://huggingface.co/XiaomiMiMo) for the model, [pqhaz](https://huggingface.co/pqhaz) for the ablated base
([Heretic](https://github.com/p-e-w/heretic)), [Project Maya](https://github.com/mw00/project-maya) for the engine,
[alesha-pro](https://github.com/alesha-pro/project-maya/blob/glm-ablate/docs/GLM-ABLATE.md) for the runtime ablation hook.
