#!/usr/bin/env bash
# Install SGLang (MLX backend) from source into servers/sglang/.venv, as its Apple Metal
# guide describes (no macOS wheel). Pinned to one commit so every install is the same.
# Idempotent: run again to reinstall. Takes a few minutes.
set -euo pipefail
cd "$(dirname "$0")"
unset VIRTUAL_ENV
COMMIT=703e763d5f3af75ea3587275d79155c28d3c5d4f  # sglang main, 2026-11-17

if [ ! -d src/.git ]; then
  git init -q src
  git -C src remote add origin https://github.com/sgl-project/sglang.git
fi
git -C src fetch -q --depth 1 origin "$COMMIT"
git -C src checkout -q --force "$COMMIT"  # --force also restores the pyproject swapped below

# The guide's step: use the pyproject for non-CUDA platforms.
mv src/python/pyproject_other.toml src/python/pyproject.toml

uv venv -q --allow-existing -p 3.12 .venv
# SGLANG_BUILD_RUST_EXTS=none: skip the Rust "multimodal" extension (needs cargo; only
# some image/video models use it). srt_mps = the serving runtime: MLX, mlx-lm, torch.
SGLANG_BUILD_RUST_EXTS=none VIRTUAL_ENV="$PWD/.venv" uv pip install -e "src/python[srt_mps]"
echo "SGLang installed. Start it with servers/sglang/serve.sh"
