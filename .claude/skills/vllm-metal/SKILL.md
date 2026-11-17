---
name: vllm-metal
description: Install and serve models with vllm-metal (the vLLM platform plugin for Apple Silicon, MLX compute backend, optional PyTorch MPS backend). Use when installing vllm-metal on a Mac, running `vllm serve` on Metal, choosing vllm-metal env vars/flags, checking which architectures (e.g. Gemma 3) it supports, or benchmarking a vllm-metal server fairly.
---

# vllm-metal on Apple Silicon

Project-local skill, written from the vllm-metal repository README and `docs/`
(github.com/vllm-project/vllm-metal, main as of 2026-11-16, version 0.33.0).
Re-check `docs/configuration.md` and `docs/supported_models.md` there if behaviour
differs: the project moves fast.

## Architecture (why B vs C is a meaningful comparison)

Upstream vLLM supplies the API server, scheduler and paged block manager;
**mlx_lm supplies the model layers**; vllm-metal owns the attention path
(paged varlen Metal kernel, M5 NAX prefill, speculative decoding). So against
plain `mlx_lm.server`, the difference is the vLLM engine: continuous batching,
paged KV cache and the custom attention kernel.

## Requirements

- macOS 15 (Sequoia) or later, Apple Silicon.
- Python 3.12 or 3.13 (`requires-python = ">=3.12,<3.14"`).

## Install (pick one)

```bash
# Latest dev build: prebuilt vllm + vllm-metal wheels into ~/.venv-vllm-metal
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
source ~/.venv-vllm-metal/bin/activate
```

```bash
# Stable release via Homebrew (no venv activation needed)
brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
brew install vllm-project/vllm-metal/vllm-metal
```

Do not put vllm-metal in the same venv as the transformers or mlx-lm servers:
it pins its own vllm/torch/mlx versions.

## Gemma 3 support

`docs/supported_models.md` lists **Gemma 3 as supported on MLX** with
"GQA + per-layer sliding window (paged)", example checkpoint
`mlx-community/gemma-3-1b-it-qat-4bit`. For a precision-matched benchmark use
bf16 instead: try `google/gemma-3-1b-it` first, fall back to
`mlx-community/gemma-3-1b-it-bf16` if loading fails. Never use the 4-bit
example checkpoint for this comparison.

## Serve

```bash
vllm serve google/gemma-3-1b-it \
  --host 127.0.0.1 --port 8003 \
  --served-model-name gemma-3-1b-it \
  --dtype bfloat16 \
  --generation-config vllm
```

- `--generation-config vllm` stops vLLM from applying the model's
  `generation_config.json` defaults, so `temperature=0` / `max_tokens` from the
  client are what actually run.
- For N=1 latency runs you can add `--max-num-seqs 1`; for the concurrency sweep
  (N = 4, 16) leave `--max-num-seqs` at least 16.
- Consider `--no-enable-prefix-caching` so repeated identical benchmark prompts
  are not served from the prefix cache (otherwise TTFT for C is not comparable to
  A/B). Record whichever choice you make.

### Paged attention flag in the plan is obsolete

`VLLM_METAL_USE_PAGED_ATTENTION=1` is no longer needed: since v0.2.0 the unified
paged varlen Metal kernel is the **default** attention backend. Likewise
`VLLM_METAL_MEMORY_FRACTION` was removed; use `--gpu-memory-utilization` or
`--kv-cache-memory-bytes`. vllm-metal validates its env vars at startup and fails
on unknown/invalid values.

### Useful env vars (docs/configuration.md)

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_MLX_DEVICE` | `gpu` | `gpu` or `cpu` |
| `VLLM_METAL_BACKEND` | MLX | set `mps` for the experimental PyTorch MPS backend |
| `VLLM_METAL_DECODE_PIPELINE` | `1` | one-step-ahead decode pipeline; greedy output unchanged |
| `VLLM_METAL_COMPILED_MLP` | `0` | opt-in `mx.compile` MLP dispatch |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | — | changes process topology and throughput; keep fixed across runs and record it |

Optional extra arm: `VLLM_METAL_BACKEND=mps PYTORCH_ENABLE_MPS_FALLBACK=0 vllm serve google/gemma-3-1b-it --dtype bfloat16`
gives "vLLM engine + PyTorch MPS", which isolates engine vs backend against
service A. Check the model table in `docs/pytorch-mps.md` for Gemma 3 first.

## Smoke test

```bash
curl -s http://127.0.0.1:8003/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma-3-1b-it","messages":[{"role":"user","content":"write me a joke"}],"max_tokens":32,"temperature":0}'
```

## Benchmarking guidance (docs/benchmarking-macos.md)

- Measure a real `vllm serve` process, not offline `LLM(...)` generation.
- Warm every arm with the real workload; exclude load/compile and warm-up requests.
- Alternate or randomize A/B run order; keep power source and power mode fixed
  (plugged in, same Low Power Mode setting); close browsers and other GPU users.
- Read-only machine-state checks worth logging per run:

```bash
pmset -g batt
pmset -g therm
ioreg -r -c AGXAccelerator -d 1 | grep -oE '"Device Utilization %"=[0-9]+'
```

- `vllm bench serve` (see the `vllm-bench-serve` skill) can target any
  OpenAI-compatible endpoint, so it can cross-check `bench.py` numbers on all
  three ports with `--backend openai-chat --endpoint /v1/chat/completions`.
