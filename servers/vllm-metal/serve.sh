#!/usr/bin/env bash
# Service C: vllm-metal (vLLM engine + MLX on Metal), port 8003.
# Uses the official installer's venv (~/.venv-vllm-metal), else a `vllm` on PATH (Homebrew).
# Extra args pass through.
set -euo pipefail
VLLM="$HOME/.venv-vllm-metal/bin/vllm"
[ -x "$VLLM" ] || VLLM="$(command -v vllm)" || { echo "vllm not found: install vllm-metal (see README)" >&2; exit 1; }
# --generation-config vllm: ignore the model's generation_config.json, use the client's params.
# --no-enable-prefix-caching: repeated benchmark prompts must not reuse cached KV (would fake TTFT).
# --max-model-len 4096: the benchmark prompts are short; avoids sizing KV for the full 32K context.
exec "$VLLM" serve "${MODEL:-google/gemma-3-1b-it}" \
  --host 127.0.0.1 --port 8003 --dtype bfloat16 \
  --generation-config vllm --no-enable-prefix-caching --max-model-len 4096 "$@"
