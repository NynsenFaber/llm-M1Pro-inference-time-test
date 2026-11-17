#!/usr/bin/env bash
# Service D: SGLang (scheduler + MLX on Metal), port 8004. Install first: servers/sglang/install.sh
# Extra args pass through.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
[ -x "$PY" ] || { echo "SGLang not installed: run servers/sglang/install.sh" >&2; exit 1; }
# SGLANG_USE_MLX=1: MLX runtime (without it SGLang falls back to torch.mps, less supported).
# --disable-radix-cache: repeated benchmark prompts must not reuse cached KV (would fake TTFT).
# --context-length 4096: same as vllm-metal's --max-model-len; the prompts are short.
SGLANG_USE_MLX=1 exec "$PY" -m sglang.launch_server --model-path "${MODEL:-google/gemma-3-1b-it}" \
  --host 127.0.0.1 --port 8004 --disable-cuda-graph --disable-radix-cache --context-length 4096 "$@"
