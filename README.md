# llm-M1Pro-inference-time-test
Testing inference time of a Gemma model locally on my M1Pro, using Hugging Face, mxl-lm and vLLM

## Quick start (macOS, Apple Silicon)

```bash
uv sync                                   # benchmark client + hf CLI
uv sync --project servers/transformers    # service A (PyTorch MPS)
uv sync --project servers/mlx-lm          # service B (MLX)
# service C: vllm-metal installs into ~/.venv-vllm-metal with its own script, see CLAUDE.md
```

See `CLAUDE.md` for the full plan, ports, commands and benchmark rules.
