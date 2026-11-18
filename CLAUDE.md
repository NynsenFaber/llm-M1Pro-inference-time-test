# CLAUDE.md

Benchmark of **Gemma 3 1B (`google/gemma-3-1b-it`, bf16)** on a MacBook Pro M1 Pro
under four frameworks, run **natively** (no Docker: it has no Metal GPU access on macOS).

| Service | Framework | Backend | Port | Environment |
|---|---|---|---|---|
| A | Hugging Face transformers (`transformers serve`) | PyTorch MPS | 8001 | `servers/transformers/` (uv project) |
| B | mlx-lm (`mlx_lm.server`) | MLX / Metal | 8002 | `servers/mlx-lm/` (uv project) |
| C | vllm-metal (`vllm serve`) | vLLM engine + MLX / Metal | 8003 | `~/.venv-vllm-metal` (official installer) or Homebrew |
| D | SGLang (`sglang.launch_server`, `SGLANG_USE_MLX=1`) | SGLang engine + MLX / Metal | 8004 | `servers/sglang/` (uv project; `install.sh` fetches the source, then `uv sync`) |

What each comparison isolates: **A vs B** = PyTorch MPS vs MLX; **B vs C** = what the
vLLM engine (scheduler, continuous batching, paged KV cache) adds on top of MLX; **B vs D**
the same for SGLang's engine; **C vs D** compares the two engines on the same backend.

## Where code runs

The servers and the benchmark only run on the Mac (Apple Silicon, macOS 15+ for
vllm-metal). A Linux cloud container can edit code, lock dependencies and lint, but
cannot run MPS/MLX/Metal. Don't claim benchmark numbers that weren't measured on the Mac.

## Layout

- `pyproject.toml` (root) — benchmark client env: httpx, pandas, matplotlib, `hf` CLI. `uv sync` here.
- `run_experiment.py` — the experiment: starts each server's `serve.sh` in turn, 3 warm-up +
  10 timed streamed requests, stops it; writes `results/results.csv` (per request),
  `results/token_times.csv` (elapsed time at each generated token), `results/time_per_token.png`
  (bar: mean ms per generated token, std error bars), `results/latency_vs_tokens.png`
  (line: time per token vs tokens generated so far, mean over runs, moving average), `results/logs/`.
- `servers/<name>/serve.sh` — one launcher per server (transformers, mlx-lm, vllm-metal, sglang).
- `servers/sglang/pyproject.toml` — service D: uv project depending on `sglang[srt_mps]` from the
  local checkout `src/python` (editable). Its default pyproject is CUDA-only, so a plain git
  dependency can't work. `[tool.uv.extra-build-variables]` sets `SGLANG_BUILD_RUST_EXTS=none`
  (skips the cargo-built multimodal extension). `uv.lock` resolved: torch 2.13.0, mlx 0.32.4,
  mlx-lm 0.32.2, no CUDA packages.
- `servers/sglang/install.sh` — fetches sglang at commit 703e763 into `servers/sglang/src`
  (gitignored), swaps in `pyproject_other.toml` (the guide's step), applies
  `gemma3-window.patch`, then `uv sync`. If you bump
  the commit, rerun `uv lock` in `servers/sglang`.
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

# A — transformers on MPS (bf16, pinned model, no CPU fallback)
servers/transformers/serve.sh                          # sequential model.generate()
servers/transformers/serve.sh --continuous-batching    # optional variant: paged KV + batching
# B — mlx-lm (prompt cache off; MODEL=mlx-community/gemma-3-1b-it-bf16 if the original repo fails)
servers/mlx-lm/serve.sh
# C — vllm-metal (install once; serve.sh finds ~/.venv-vllm-metal or a Homebrew `vllm`)
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
servers/vllm-metal/serve.sh
# D — SGLang (install once from source, then serve)
servers/sglang/install.sh
servers/sglang/serve.sh

# smoke test any port
curl -s localhost:8002/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"google/gemma-3-1b-it","messages":[{"role":"user","content":"write me a joke"}],"max_tokens":32,"temperature":0}'

# the experiment: A, then B, C, D (one server at a time)
uv run python run_experiment.py            # --only mlx-lm vllm, --runs, --max-tokens, --prompt
```

Always check exact flags with each tool's `--help`; the commands above are a starting point.

## Benchmark rules (keep these invariant)

- **One server running at a time** during measurements.
- Same prompt for every framework (a long-story request, so outputs reach `max_tokens`),
  `temperature=0`, `max_tokens` 1024, streaming on.
- 3 warm-up requests discarded, then 10 timed runs, one request at a time (N = 1).
  Concurrency N = 4, 16 is a possible follow-up, not implemented.
- Output token counts come from the response `usage` field (request
  `stream_options: {"include_usage": true}`), never from splitting text.
- Metrics: TTFT = first streamed token − send time; decode tok/s =
  (output_tokens − 1) / (end − first_token); end-to-end (only comparable at equal
  output length); throughput = total output tokens / wall-clock at N concurrent.
- Report median and spread (IQR or min/max), never a single run.
- Results CSV: one row per timed request, `framework, run, total_s, ttft_s, tokens, ms_per_token`.
- Per-token times: chunk arrival times spread evenly over `usage.completion_tokens`
  (a chunk can hold several tokens). Token 1 (prefill/TTFT) is left out of the line plot.
- Same precision everywhere: bf16. If a framework can't load the original repo, use
  `mlx-community/gemma-3-1b-it-bf16`, never a 4-bit/QAT checkpoint.
- Record machine state with each run: plugged in, Low Power Mode off, browsers closed.

## Known facts (verified 2026-11-17)

- vllm-metal lists **Gemma 3 as supported** on MLX (GQA + per-layer sliding window, paged).
- `VLLM_METAL_USE_PAGED_ATTENTION` is obsolete: paged attention is the default since
  vllm-metal v0.2.0. `VLLM_METAL_MEMORY_FRACTION` was removed (use `--gpu-memory-utilization`).
- vllm-metal by default applies the model's `generation_config.json`; pass
  `--generation-config vllm` so the client's sampling params are what run.
- Prompt/prefix caching is **off** on every server so repeated prompts don't shortcut
  TTFT: mlx-lm `--prompt-cache-size 0` (set in `serve.sh`), vllm-metal
  `--no-enable-prefix-caching`, SGLang `--disable-radix-cache`; transformers serve has no
  cross-request prompt cache.
- `mlx_lm.server` (0.32.2, read from source): batches concurrent requests itself
  (`--decode-concurrency` 32, `--prompt-concurrency` 8 by default), so B vs C at N = 4/16
  is batched vs batched. It sends `usage` only when the request has
  `stream_options: {"include_usage": true}`. Requests may send
  `"model": "google/gemma-3-1b-it"` (same as `--model`, no reload) or omit it.
  Gemma 3 text (`gemma3_text`) is a supported architecture. A request carrying `seed`
  is not batchable (`_is_batchable` requires `seed is None`) and runs on the sequential
  path, so `run_experiment.py` sends `seed` only when `--temperature > 0`.

- SGLang (commit 703e763, run on the Mac 2026-11-18): Apple Silicon serving runs on MLX
  with `SGLANG_USE_MLX=1` (else torch.mps fallback, less supported); needs macOS 14+, torch
  2.13.0 + MLX >= 0.32.4 via the `srt_mps` extra. The MLX runner loads weights with
  `mlx_lm.load` (so Gemma 3 text architecture is known) and its KV cache handles per-layer
  sliding windows (gemma3-style). `--disable-radix-cache` turns prefix caching off.
  Streamed `usage` honours `stream_options.include_usage`. The Rust `multimodal` extension
  is only loaded lazily by some image/video processors, so skipping it is fine for text.
  Optional `sgl-kernel` Metal kernels need full Xcode and are skipped.

- `transformers serve` (5.23, read from source): `--device mps --dtype bfloat16` are
  passed straight to `from_pretrained`; the positional model arg pins the server, so
  requests must send `"model": "google/gemma-3-1b-it"` (or omit it). Streaming
  responses always end with a chunk carrying `usage`. Without `--continuous-batching`
  requests run one at a time on a single inference thread, so N = 4/16 queue up;
  with it, it uses paged attention (Metal flash kernel on MPS only if the `kernels`
  package is installed, otherwise its default attention). `temperature: 0` sets `do_sample=False`.

## Open questions (update this list as they are answered)

- Does `transformers serve` run Gemma 3 on MPS end to end? First Mac run: it failed to load
  `AutoProcessor` (Gemma3Processor wants preprocessor_config.json, absent in the 1B text repo);
  `servers/transformers/serve.py` now falls back to `AutoTokenizer`. Not yet re-run.
- Do mlx-lm, vllm-metal and SGLang load `google/gemma-3-1b-it` directly, or is the mlx-community bf16 copy needed?
- SGLang serves Gemma 3 on MLX only with `servers/sglang/gemma3-window.patch` (applied by
  `install.sh`): mlx-lm's gemma3 has `sliding_window_pattern` but no `layer_types`, which the
  MLX runner needs to size sliding-window KV. Verified: 1024-token output stays coherent.
- Does each server return `usage` in the final streamed chunk?
- Expectation to test: at N = 1 the gap between B and C is small; C should pull ahead at N = 16.

## Conventions

- When writing or reviewing code, use the `ponytail` skill (smallest complete change,
  stdlib/existing tools first, no wrappers nobody asked for); `ponytail-review` for reviews.

- Python 3.12 everywhere (`.python-version`). Manage deps with `uv add` / `uv add --project servers/<name>`; commit lockfiles.
- Ruff for lint/format: `uv run ruff check . && uv run ruff format .`
