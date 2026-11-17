# CLAUDE.md

Benchmark of **Gemma 3 1B (`google/gemma-3-1b-it`, bf16)** on a MacBook Pro M1 Pro
under three frameworks, run **natively** (no Docker: it has no Metal GPU access on macOS).

| Service | Framework | Backend | Port | Environment |
|---|---|---|---|---|
| A | Hugging Face transformers (`transformers serve`) | PyTorch MPS | 8001 | `servers/transformers/` (uv project) |
| B | mlx-lm (`mlx_lm.server`) | MLX / Metal | 8002 | `servers/mlx-lm/` (uv project) |
| C | vllm-metal (`vllm serve`) | vLLM engine + MLX / Metal | 8003 | `~/.venv-vllm-metal` (official installer) or Homebrew |

What each comparison isolates: **A vs B** = PyTorch MPS vs MLX; **B vs C** = what the
vLLM engine (scheduler, continuous batching, paged KV cache) adds on top of MLX.

## Where code runs

The servers and the benchmark only run on the Mac (Apple Silicon, macOS 15+ for
vllm-metal). A Linux cloud container can edit code, lock dependencies and lint, but
cannot run MPS/MLX/Metal. Don't claim benchmark numbers that weren't measured on the Mac.

## Layout

- `pyproject.toml` (root) — benchmark client env: httpx, pandas, matplotlib, `hf` CLI. `uv sync` here.
- `servers/transformers/pyproject.toml` — service A deps (torch, `transformers[serving]`, accelerate).
- `servers/mlx-lm/pyproject.toml` — service B deps (mlx-lm).
- vllm-metal is **not** a uv project here: it pins its own vllm/torch/mlx and is installed by its own script.
- Each server project has its own lockfile and `.venv`, because the torch/MLX pins conflict.
  Server lockfiles are restricted to `darwin/arm64` via `[tool.uv] environments`.
- `.claude/skills/` — vendored project skills; see `.claude/skills/SOURCES.md`.

## Commands (on the Mac)

```bash
# one-time
uv sync                                   # client env
uv run hf auth login                      # Gemma is gated: accept the license on the model page first
uv run hf download google/gemma-3-1b-it   # shared cache ~/.cache/huggingface/hub

# A — transformers on MPS
uv run --project servers/transformers transformers serve --port 8001
# B — mlx-lm
uv run --project servers/mlx-lm mlx_lm.server --model google/gemma-3-1b-it --port 8002
# C — vllm-metal
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
source ~/.venv-vllm-metal/bin/activate
vllm serve google/gemma-3-1b-it --port 8003 --dtype bfloat16 --generation-config vllm

# smoke test any port
curl -s localhost:8002/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"google/gemma-3-1b-it","messages":[{"role":"user","content":"write me a joke"}],"max_tokens":32,"temperature":0}'

# benchmark (client env)
uv run python bench.py ...
```

Always check exact flags with each tool's `--help`; the commands above are a starting point.

## Benchmark rules (keep these invariant)

- **One server running at a time** during measurements.
- Same prompts for every framework (a short "write me a joke" and a longer one),
  `temperature=0`, fixed `max_tokens` (256), streaming on.
- 3 warm-up requests discarded, then 10 timed runs per prompt; concurrency N = 1, 4, 16.
- Output token counts come from the response `usage` field (request
  `stream_options: {"include_usage": true}`), never from splitting text.
- Metrics: TTFT = first streamed token − send time; decode tok/s =
  (output_tokens − 1) / (end − first_token); end-to-end (only comparable at equal
  output length); throughput = total output tokens / wall-clock at N concurrent.
- Report median and spread (IQR or min/max), never a single run.
- Results CSV in long format: `framework, prompt, N, metric, value` (+ run index); plot afterwards.
- Same precision everywhere: bf16. If a framework can't load the original repo, use
  `mlx-community/gemma-3-1b-it-bf16`, never a 4-bit/QAT checkpoint.
- Record machine state with each run: plugged in, Low Power Mode off, browsers closed.

## Known facts (verified 2026-11-17)

- vllm-metal lists **Gemma 3 as supported** on MLX (GQA + per-layer sliding window, paged).
- `VLLM_METAL_USE_PAGED_ATTENTION` is obsolete: paged attention is the default since
  vllm-metal v0.2.0. `VLLM_METAL_MEMORY_FRACTION` was removed (use `--gpu-memory-utilization`).
- vllm-metal by default applies the model's `generation_config.json`; pass
  `--generation-config vllm` so the client's sampling params are what run.
- Consider `--no-enable-prefix-caching` on vllm-metal so repeated prompts don't
  shortcut TTFT; mlx-lm's server also has a prompt cache. Note which is on in results.

## Open questions (update this list as they are answered)

- Does `transformers serve` handle Gemma 3 on MPS, or is a FastAPI + `TextIteratorStreamer` wrapper needed?
- Do mlx-lm and vllm-metal load `google/gemma-3-1b-it` directly, or is the mlx-community bf16 copy needed?
- Does each server return `usage` in the final streamed chunk?
- Expectation to test: at N = 1 the gap between B and C is small; C should pull ahead at N = 16.

## Conventions

- Python 3.12 everywhere (`.python-version`). Manage deps with `uv add` / `uv add --project servers/<name>`; commit lockfiles.
- Ruff for lint/format: `uv run ruff check . && uv run ruff format .`
