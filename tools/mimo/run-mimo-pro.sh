#!/usr/bin/env bash
# tools/mimo/run-mimo-pro.sh - MiMo-V2.6-Pro Uncensored (arch mimo2) on the 7 GPUs (5x 4090 + 2x A100) through Maya's
# server: the browser chat and the OpenAI / Anthropic APIs on http://127.0.0.1:8098 (/v1/chat/completions, /v1/messages).
# Any other GPU job (e.g. a vLLM server) must be stopped first.  Ctrl+C here stops the model.
#   tools/mimo/run-mimo-pro.sh [config]   (default configs/mimo-pro-uncensored.json; without the refusal vector:
#   configs/mimo-pro-uncensored-novector.json).  The config's "gpu" list, CUDA_VISIBLE_DEVICES and STRATA_GLM_SPLIT
#   describe this box (GPU 1 left out) - adjust them to yours.
set -euo pipefail
cd "$(dirname "$0")/../.."
cfg=${1:-configs/mimo-pro-uncensored.json}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
py=python3; [ -x .venv/bin/python ] && { py=.venv/bin/python; export PATH="$PWD/.venv/bin:$PATH"; }   # (the vision encoder
# is started as `python3 tools/mimo/mimo_vision.py` from PATH: torch, transformers, safetensors, pillow must be there)
exec "$py" serve/server.py --engine strata --config "$cfg" --port 8098
