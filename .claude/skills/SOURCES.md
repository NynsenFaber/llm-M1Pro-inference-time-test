# Vendored skills

Skills in this folder are loaded automatically by Claude Code as project skills.
They are copies (no local edits unless noted) taken on 2026-11-17, because
`/plugin marketplace add` / `/plugin install` cannot run in a cloud session.
To get updates, re-copy from upstream or install the plugins locally instead:

```
/plugin marketplace add huggingface/skills
/plugin marketplace add vllm-project/vllm-skills
```

| Skill | Upstream | Commit | License |
|---|---|---|---|
| `hf-cli` | github.com/huggingface/skills `skills/hf-cli` | 25910b3 | Apache-2.0 |
| `hf-mem` | github.com/huggingface/skills `skills/hf-mem` | 25910b3 | Apache-2.0 |
| `huggingface-local-models` | github.com/huggingface/skills `skills/huggingface-local-models` | 25910b3 | Apache-2.0 |
| `vllm-bench-serve` | github.com/vllm-project/vllm-skills `plugins/vllm-skills/skills/vllm-bench-serve` | c996234 | Apache-2.0 |
| `mlx-lm` | github.com/itsmostafa/llm-engineering-skills `skills/mlx` | c364a15 | MIT (c) 2026 Mostafa. Edit: frontmatter `name` changed `mlx` → `mlx-lm` |
| `vllm-metal` | written for this project from github.com/vllm-project/vllm-metal docs (main, 2026-11-16) | — | — |

Why these and not others:

- `hf-cli`: gated-model login and `hf download` into the shared cache (step 1).
- `hf-mem`: quick memory estimate for the bf16 weights before loading.
- `huggingface-local-models`: llama.cpp/GGUF serving, for the llama.cpp reference point.
- `vllm-bench-serve`: `vllm bench serve` works against any OpenAI-compatible
  endpoint, so it can cross-check `bench.py` on all three ports.
- `mlx-lm`: `mlx_lm.server`, generation and conversion on Apple Silicon.
- `vllm-metal`: the official `vllm-deploy-simple` skill only covers CUDA/ROCm/TPU/CPU,
  so a Metal-specific skill was written instead.

The remaining huggingface/skills (SageMaker, Spaces, training, Gradio, ...) are
not relevant to a local inference benchmark and were left out.
