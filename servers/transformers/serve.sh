#!/usr/bin/env bash
# Service A: Hugging Face transformers on PyTorch MPS, port 8001.
# Extra args pass through, e.g. `./serve.sh --continuous-batching` for the paged/batched variant.
set -euo pipefail
cd "$(dirname "$0")"
unset VIRTUAL_ENV  # the caller's venv (e.g. the client's) is not this project's
# No silent CPU fallback: an op missing on MPS should fail, not skew the timings.
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-0}"
exec uv run python serve.py serve "${MODEL:-google/gemma-3-1b-it}" \
  --device mps --dtype bfloat16 --host 127.0.0.1 --port 8001 "$@"
