#!/usr/bin/env bash
# Install SGLang (MLX backend) as a uv project, like servers/transformers and servers/mlx-lm.
# SGLang has no macOS wheel, and its default pyproject is the CUDA build, so first fetch
# its source (pinned commit) and apply the step from its Apple Metal guide, then `uv sync`
# installs it with the versions locked in uv.lock. Safe to run again.
set -euo pipefail
cd "$(dirname "$0")"
unset VIRTUAL_ENV
COMMIT=703e763d5f3af75ea3587275d79155c28d3c5d4f  # sglang main, 2026-11-17 (uv.lock matches it)

if [ ! -d src/.git ]; then
  git init -q src
  git -C src remote add origin https://github.com/sgl-project/sglang.git
fi
git -C src fetch -q --depth 1 origin "$COMMIT"
git -C src checkout -q --force "$COMMIT"  # --force also restores the file moved below
# The guide's step: replace the CUDA pyproject with the one for other platforms (srt_mps extra).
mv src/python/pyproject_other.toml src/python/pyproject.toml
# mlx-lm's gemma3 has a sliding_window_pattern but no layer_types, so the MLX runner
# refused to load it ("declares sliding-window attention but ... no per-layer window map").
git -C src apply "$PWD/gemma3-window.patch"

uv sync
echo "SGLang installed. Start it with servers/sglang/serve.sh"
