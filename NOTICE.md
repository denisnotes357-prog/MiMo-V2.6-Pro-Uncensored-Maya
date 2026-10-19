# Notice

MiMo-V2.6-Pro Uncensored on Maya combines work from several projects. Their licenses apply to their parts.

| Part | Author | License | Where |
| --- | --- | --- | --- |
| The engine, server, dashboard and quant tools (everything not listed below) | Niko1221 and the Strata contributors — [Project Maya](https://github.com/mw00/project-maya) | MIT | [LICENSE](LICENSE). Forked at upstream commit `d187195` (2026-10-15). |
| The MiMo-V2.6-Pro port (`mimo2` in `src/`, `include/`, `tests/mimo_*`), `tools/mimo/` (except `hf/`), `configs/mimo-pro-*`, the docs of this repo | denisnotes357-prog | MIT | this repository |
| The GLM-5.3 port (`glm-dsa`, `tools/glm53/`, `configs/glm53-*`, `docs/GLM53-*`, `docs/glm53-*`) — see [GLM-5.3 Uncensored on Maya](https://github.com/denisnotes357-prog/GLM-5.3-Uncensored-Maya) | denisnotes357-prog | MIT | this repository |
| The runtime ablation hook (`src/core/glm_ablate.cuh`) and the multi-direction refusal method | [alesha-pro](https://github.com/alesha-pro/project-maya/blob/glm-ablate/docs/GLM-ABLATE.md) | MIT (fork of Project Maya) | used for both models |
| MiMo-V2.6-Pro model weights (the quantized model on Hugging Face) | Xiaomi — [XiaomiMiMo/MiMo-V2.6-Pro-MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD) | MIT | [LICENSES/MiMo-V2.6-Pro-LICENSE.md](LICENSES/MiMo-V2.6-Pro-LICENSE.md) |
| The attention-ablated base weights | [pqhaz/MiMo-V2.6-Pro-MOPD-heretic](https://huggingface.co/pqhaz/MiMo-V2.6-Pro-MOPD-heretic) (with [Heretic](https://github.com/p-e-w/heretic)) | MIT | quantized here |
| MiMo-V2 modeling code (`tools/mimo/hf/`, modified for standalone loading) | Xiaomi Corporation and The HuggingFace Inc. team | Apache-2.0 | [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt), `tools/mimo/hf/NOTICE` |
| GLM-5.3 model weights (for the GLM-5.3 configs) | Z.AI — [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) | GLM-5.3 License | [LICENSES/GLM-5.3-LICENSE.txt](LICENSES/GLM-5.3-LICENSE.txt) |
| ggml (fetched at build time at the commit Maya pins) and the `mimo2` / `glm-dsa` references | [llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT | not vendored |

The refusal vectors (`refusal-vector-L70.f32` on Hugging Face for MiMo, `tools/glm53/ablate/dir3.f32` for GLM-5.3) were
measured from the models by this project.
