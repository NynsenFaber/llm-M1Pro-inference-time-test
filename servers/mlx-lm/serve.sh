#!/usr/bin/env bash
# Service B: mlx-lm (MLX on Metal), port 8002.
# Extra args pass through and win over the defaults below (argparse keeps the last value).
set -euo pipefail
cd "$(dirname "$0")"
# --prompt-cache-size 0: repeated benchmark prompts must not reuse cached KV (would fake TTFT).
exec uv run mlx_lm.server --model "${MODEL:-google/gemma-3-1b-it}" \
  --host 127.0.0.1 --port 8002 --prompt-cache-size 0 "$@"
