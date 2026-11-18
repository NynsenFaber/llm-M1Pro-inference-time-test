# llm-M1Pro-inference-time-test

How fast does the same model, **Gemma 3 1B (`google/gemma-3-1b-it`, bf16)**, run on a
MacBook Pro M1 Pro under four frameworks? Everything runs natively (Docker has no
Metal GPU access on macOS).

| | Framework | Compute backend | Port |
|---|---|---|---|
| A | Hugging Face transformers (`transformers serve`) | PyTorch MPS | 8001 |
| B | mlx-lm (`mlx_lm.server`) | MLX (Metal) | 8002 |
| C | vllm-metal (`vllm serve`) | vLLM engine + MLX (Metal) | 8003 |
| D | SGLang (`sglang.launch_server`) | SGLang engine + MLX (Metal) | 8004 |

A vs B compares PyTorch MPS with MLX. C and D each put a serving engine (scheduler,
paged KV cache) on top of MLX, so B vs C vs D shows what each engine adds, and C vs D
compares the two engines.

`run_experiment.py` starts A, sends it 3 warm-up and 10 timed requests, stops it, then
does the same for B, C and D. It produces two plots:

- `results/time_per_token.png`: bar chart of the mean time per generated token
  (request time / generated tokens) for each framework.
- `results/latency_vs_tokens.png`: time per token against the number of tokens generated so
  far, one line per framework. Each step attends to a longer context, so the lines are
  expected to rise slowly; the gap between them shows which framework is faster.

## Results

Measured on a MacBook Pro M1 Pro on 2026-11-18 (data in `results/`). Settings: one request
at a time, greedy decoding (`temperature=0`), a long-story prompt, `max_tokens=1024`,
3 warm-up requests, then 10 timed runs per framework. Every one of the 40 timed requests
generated exactly 1,024 tokens, so all four frameworks are compared on identical output
lengths. SGLang runs with a small local fix for Gemma 3 (see [SGLang note](#sglang-note)).

![Mean time per generated token](results/time_per_token.png)

| Framework | Time per token (median) | Range over 10 runs | Decode speed | Time to first token | Whole request |
|---|---|---|---|---|---|
| **SGLang** (MLX) | **13.9 ms** | 13.86–13.88 ms | **72.4 tokens/s** | **61 ms** | 14.2 s |
| mlx-lm (MLX) | 15.6 ms | 15.53–15.59 ms | 64.5 tokens/s | 75 ms | 15.9 s |
| vLLM (vllm-metal, MLX) | 18.0 ms | 17.99–18.05 ms | 55.7 tokens/s | 67 ms | 18.4 s |
| Hugging Face (PyTorch MPS) | 18.6 ms | 18.58–19.15 ms | 54.0 tokens/s | 97 ms | 19.1 s |

Decode speed = (tokens − 1) / (request time − time to first token), so it excludes the
prompt processing.

![Time per token vs tokens generated so far](results/latency_vs_tokens.png)

| Framework | Tokens 2–64 | Tokens 768–1024 | Change |
|---|---|---|---|
| SGLang | 13.5 ms | 14.0 ms | +3% |
| mlx-lm | 15.1 ms | 15.6 ms | +3% |
| Hugging Face | 17.9 ms | 18.9 ms | +5% |
| vLLM | 15.7 ms | 18.9 ms | +20% (in steps) |

### Discussion

**SGLang is the fastest, then mlx-lm, then vLLM and Hugging Face.** SGLang generates
tokens 11% faster than mlx-lm and about 23–25% faster than vLLM and Hugging Face, and it
also has the quickest first token. Each framework's 10 runs stay within a few hundredths
of a millisecond of each other, so every gap in the table is a real difference, not noise.

**PyTorch MPS vs MLX (A vs B).** Same weights and precision, different compute library:
MLX is 16% faster per token and reaches the first token about 20 ms sooner.

**What a serving engine adds on top of MLX (B vs C vs D).** vLLM and SGLang both run
mlx-lm's own model code (they load it with `mlx_lm.load`), so comparing them with plain
mlx-lm isolates what each engine adds. The two engines go in opposite directions:

- **SGLang makes MLX faster**: 11% per token, and 14 ms sooner to the first token. The
  likely reasons are its own pre-allocated KV cache and its "overlap scheduling", which
  prepares the next step on the CPU while the GPU is still computing the current one
  (`mx.async_eval`). These are hypotheses; running SGLang with `--disable-overlap-schedule`
  would show how much the overlap contributes.
- **vLLM makes MLX slower** after the first ~100 tokens. It starts exactly at mlx-lm's
  speed (15.7 ms per token) and has a fast first token (67 ms), but then slows down in
  steps, to about 19 ms per token. The steps fall at the same token positions in all 10
  runs, and at the same positions as in an earlier, separate run, so they are caused by
  the growing context, not by heat or background activity. The most likely cause is its
  paged KV cache and attention path behaving differently as the context grows; the dip
  around tokens 600–880 may be Gemma 3's sliding-window layers releasing old cache
  blocks. Both are hypotheses, not verified.

**vLLM vs SGLang (C vs D).** On the same backend and model code, SGLang is 23% faster per
token and its speed barely changes as the output grows. Both engines are built to serve
many requests at once, though; with one request at a time their scheduling is mostly
overhead, so this is the least favourable case for both. Testing 4 and 16 simultaneous
requests would show how they compare where they are meant to shine.

**Longer context costs little.** Each new token attends to everything generated before
it, so time per token should rise as the output grows. It does, but only slightly: Gemma 3
uses a 512-token sliding window on 5 of every 6 layers, which keeps attention cheap.
SGLang and mlx-lm rise 3% over 1,000 tokens, Hugging Face 5%, and only vLLM rises
noticeably (20%), for the engine-related reasons above.

**How close to the hardware limit.** Generating a token means reading all ~2 GB of bf16
weights once, and the M1 Pro's memory bandwidth is about 200 GB/s, which caps a 1B model
at roughly 100 tokens/s. SGLang reaches about 72% of that ceiling (72.4 tokens/s × 2 GB
≈ 145 GB/s), mlx-lm 65%, vLLM and Hugging Face about 55%. So even this small model is
largely limited by memory bandwidth, and the frameworks differ in how much overhead
they add on top.

### SGLang note

SGLang's MLX backend refused to load Gemma 3 at the pinned commit: it needs a per-layer
list of which layers use the sliding window, and mlx-lm's Gemma 3 model only provides the
repeating pattern (`sliding_window_pattern`). `servers/sglang/gemma3-window.patch`,
applied by `install.sh`, builds that list from the pattern using the same rule as mlx-lm
(every 6th layer is global, the rest slide). With it, the 1,024-token outputs stay
coherent.

### Limits of this run

- One model size, one prompt and one request at a time. Bigger models (more
  bandwidth-bound) or many simultaneous users (where vLLM's and SGLang's batching helps)
  may rank the frameworks differently.
- The frameworks ran in a fixed order (Hugging Face, mlx-lm, vLLM, SGLang) rather than
  alternating. The very small spread and the repeat of the earlier run's numbers make heat
  an unlikely factor, but alternating the order would rule it out completely.
- The bandwidth figures assume ~2 GB of weights read per token; they are estimates.

## Requirements

- Apple Silicon Mac, macOS 15 (Sequoia) or later (needed by vllm-metal)
- About 15 GB free disk (model weights + four Python environments + the SGLang source)
- [uv](https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- A Hugging Face account

## Setup (one time)

**1. Clone and install the benchmark client**

```bash
git clone https://github.com/NynsenFaber/llm-M1Pro-inference-time-test.git
cd llm-M1Pro-inference-time-test
uv sync
```

**2. Get access to Gemma**

Gemma is a gated model. Open https://huggingface.co/google/gemma-3-1b-it, accept the
license, then create a read token at https://huggingface.co/settings/tokens and log in:

```bash
uv run hf auth login
uv run hf download google/gemma-3-1b-it
```

The weights go to `~/.cache/huggingface/hub`, which all four frameworks share.

**3. Install the transformers and mlx-lm servers**

Each has its own environment because their dependencies conflict.

```bash
uv sync --project servers/transformers
uv sync --project servers/mlx-lm
```

**4. Install vllm-metal**

The official installer creates its own environment in `~/.venv-vllm-metal`:

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

(Homebrew also works: `brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal`
then `brew install vllm-project/vllm-metal/vllm-metal`. The start script finds either.)

**5. Install SGLang**

SGLang is a uv project like the transformers and mlx-lm servers (`servers/sglang/pyproject.toml`
and `uv.lock`), but it has no macOS package, so its [Apple Silicon guide](https://docs.sglang.io/docs/hardware-platforms/apple_metal)
installs it from source. This script downloads that source at a fixed commit into
`servers/sglang/src`, applies the guide's one setup step and a small Gemma 3 fix
(`gemma3-window.patch`, see [SGLang note](#sglang-note)), then runs `uv sync`. It takes a
few minutes:

```bash
servers/sglang/install.sh
```

It needs macOS 14 or newer and git. It skips two optional native parts: SGLang's own Metal
kernels (they need the full Xcode app) and a Rust extension used only by some
image/video models. The server runs on SGLang's MLX backend (`SGLANG_USE_MLX=1`).

## Run the experiment

Plug in the charger, turn Low Power Mode off and close browsers and other GPU-heavy apps.
Make sure nothing else is using ports 8001-8004. Then:

```bash
uv run python run_experiment.py
```

It prints each request's time, a summary table, and saves in `results/`:

- `time_per_token.png` and `latency_vs_tokens.png`: the two plots above
- `results.csv`: one row per timed request (`framework, run, total_s, ttft_s, tokens, ms_per_token`)
- `token_times.csv`: elapsed time at every generated token of every request
  (`framework, run, token, elapsed_s`), the data behind the line plot
- `logs/<framework>.log`: server output, for when something fails

If a framework fails, the script says so, keeps going with the others and plots the ones
that worked. Useful options:

```bash
uv run python run_experiment.py --only mlx-lm vllm      # rerun a subset
uv run python run_experiment.py --runs 20 --max-tokens 2048   # longer outputs
uv run python run_experiment.py --smooth 32                    # smoother lines
uv run python run_experiment.py --temperature 0.7 --seed 42    # sampling, reproducible
```

All requests use `temperature=0`, the same prompt (a request for a long story, so each
answer is long enough to show the trend) and `max_tokens` 1024 by default. Times are per
generated token, so frameworks stay comparable even if they stop at slightly different
lengths.

**Reproducibility.** By default decoding is greedy (`temperature=0`): the model always
picks the most likely token, so there is nothing random and every run of the same prompt
gives the same text. With `--temperature > 0` the script also sends `--seed` (default 0)
with every request, so sampled outputs repeat too. The seed is not sent with greedy
decoding because mlx-lm turns off its batched generation path for seeded requests, which
would change what is measured. Timings still vary a little between runs (that is what
the error bars show); no seed can remove that.

Servers may stream several tokens in one chunk; the script then spreads that
chunk's time evenly over its tokens, using the token count from the response's `usage`.

## Run a server by hand

```bash
servers/transformers/serve.sh   # A, port 8001
servers/mlx-lm/serve.sh         # B, port 8002
servers/vllm-metal/serve.sh     # C, port 8003
servers/sglang/serve.sh         # D, port 8004
```

```bash
curl -s localhost:8002/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"google/gemma-3-1b-it","messages":[{"role":"user","content":"write me a joke"}],"max_tokens":32,"temperature":0}'
```

## Troubleshooting

- **A server never becomes ready / 401 or "gated repo" in the log:** the license isn't
  accepted or `hf auth login` wasn't run. mlx-lm keeps its port open and answers 404
  when the model failed to load, so look in `results/logs/mlx-lm.log`.
- **mlx-lm or vllm-metal can't load the original repo:** use the MLX bf16 copy (same
  precision) for that run, e.g.
  `MODEL=mlx-community/gemma-3-1b-it-bf16 uv run python run_experiment.py --only mlx-lm`.
  Avoid 4-bit or QAT checkpoints: they would not be a fair comparison.
- **transformers fails with an op "not implemented for MPS":** CPU fallback is disabled
  on purpose so timings stay GPU-only. To see whether it runs at all, start it with
  `PYTORCH_ENABLE_MPS_FALLBACK=1`, and note it in the results.
- **"vllm not found":** install vllm-metal (step 4).
- **"SGLang not installed":** run `servers/sglang/install.sh` (step 5). SGLang's Apple
  Silicon support is new: if it fails on Gemma 3, the other three still run; skip it with
  `--only transformers mlx-lm vllm` and check `results/logs/sglang.log`.
