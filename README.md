# llm-M1Pro-inference-time-test

How fast does the same model, **Gemma 3 1B (`google/gemma-3-1b-it`, bf16)**, run on a
MacBook Pro M1 Pro under three frameworks? Everything runs natively (Docker has no
Metal GPU access on macOS).

| | Framework | Compute backend | Port |
|---|---|---|---|
| A | Hugging Face transformers (`transformers serve`) | PyTorch MPS | 8001 |
| B | mlx-lm (`mlx_lm.server`) | MLX (Metal) | 8002 |
| C | vllm-metal (`vllm serve`) | vLLM engine + MLX (Metal) | 8003 |

A vs B compares PyTorch MPS with MLX; B vs C shows what the vLLM engine adds on top of MLX.

`run_experiment.py` starts A, sends it 3 warm-up and 10 timed requests, stops it, then
does the same for B and C. It produces two plots:

- `results/time_per_token.png`: bar chart of the mean time per generated token
  (request time / generated tokens) for each framework.
- `results/latency_vs_tokens.png`: time per token against the number of tokens generated so
  far, one line per framework. Each step attends to a longer context, so the lines are
  expected to rise slowly; the gap between them shows which framework is faster.

## Requirements

- Apple Silicon Mac, macOS 15 (Sequoia) or later (needed by vllm-metal)
- About 10 GB free disk (model weights + three Python environments)
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

The weights go to `~/.cache/huggingface/hub`, which all three frameworks share.

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

## Run the experiment

Plug in the charger, turn Low Power Mode off and close browsers and other GPU-heavy apps.
Make sure nothing else is using ports 8001-8003. Then:

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
