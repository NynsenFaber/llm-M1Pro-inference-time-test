"""Start each server in turn (A: transformers, B: mlx-lm, C: vllm-metal), time 10 requests,
stop it, then save CSVs and two plots: mean time per generated token (bar) and
time per token vs position in the output (line).

uv run python run_experiment.py                       # all three
uv run python run_experiment.py --only mlx-lm vllm    # a subset
"""

import argparse
import csv
import json
import os
import signal
import statistics
import subprocess
import time
from pathlib import Path

import httpx
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
OUT = ROOT / "results"
MODEL = os.environ.get("MODEL", "google/gemma-3-1b-it")
SERVERS = {  # key: (plot label, start script, port)
    "transformers": ("Hugging Face\n(PyTorch MPS)", "servers/transformers/serve.sh", 8001),
    "mlx-lm": ("mlx-lm\n(MLX)", "servers/mlx-lm/serve.sh", 8002),
    "vllm": ("vLLM\n(vllm-metal)", "servers/vllm-metal/serve.sh", 8003),
}

p = argparse.ArgumentParser()
p.add_argument("--only", nargs="+", choices=SERVERS, default=list(SERVERS))
p.add_argument("--runs", type=int, default=10)
p.add_argument("--warmup", type=int, default=3)
p.add_argument("--max-tokens", type=int, default=1024)
p.add_argument(
    "--prompt",  # long output on purpose: the line plot needs many tokens to show slowdown
    default="Write a long, detailed story of at least 1500 words about a lighthouse keeper.",
)
p.add_argument("--smooth", type=int, default=16, help="moving-average window for the line plot")
p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy, already deterministic")
p.add_argument("--seed", type=int, default=0, help="sampling seed, sent only if temperature > 0")
p.add_argument("--startup-timeout", type=int, default=1800, help="s; first start downloads")
args = p.parse_args()


def chat(port, max_tokens):
    """One streamed request. Returns (total_s, completion_tokens, chunk arrival times in s)."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": max_tokens,
        "temperature": args.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.temperature > 0:
        # Only when sampling: mlx-lm switches off its batched path for seeded requests,
        # so sending it with greedy decoding would change what is measured for nothing.
        body["seed"] = args.seed
    times, usage = [], None
    t0 = time.perf_counter()
    with httpx.stream(
        "POST", f"http://127.0.0.1:{port}/v1/chat/completions", json=body, timeout=600
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or usage
            if any((c.get("delta") or {}).get("content") for c in chunk.get("choices", [])):
                times.append(time.perf_counter() - t0)
    total = time.perf_counter() - t0
    if not times or not usage:
        raise RuntimeError(f"no streamed text or no usage from port {port}")
    return total, usage["completion_tokens"], times


def per_token(times, tokens):
    """Elapsed time at tokens 1..tokens. A chunk may hold several tokens, so spread the
    chunk arrival times evenly over the token count reported in usage."""
    pos = np.arange(1, len(times) + 1) * tokens / len(times)
    return np.interp(np.arange(1, tokens + 1), pos, times)


def wait_ready(proc, port, log):
    deadline = time.time() + args.startup_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}, see {log}")
        try:
            chat(port, max_tokens=1)
            return
        except (httpx.HTTPError, RuntimeError):
            time.sleep(3)  # still loading (mlx-lm answers 404 until the model is in)
    raise RuntimeError(f"server not ready after {args.startup_timeout}s, see {log}")


def stop(proc):
    os.killpg(proc.pid, signal.SIGTERM)  # whole group: uv run spawns a child process
    try:
        proc.wait(30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


(OUT / "logs").mkdir(parents=True, exist_ok=True)
rows, token_rows = [], []
for key in args.only:
    label, script, port = SERVERS[key]
    log = OUT / "logs" / f"{key}.log"
    print(f"\n== {key}: starting {script} (log: {log})", flush=True)
    with open(log, "w") as f:
        proc = subprocess.Popen(
            [ROOT / script], stdout=f, stderr=subprocess.STDOUT, start_new_session=True
        )
    try:
        wait_ready(proc, port, log)
        for _ in range(args.warmup):
            chat(port, args.max_tokens)
        for run in range(args.runs):
            total, tokens, times = chat(port, args.max_tokens)
            ms_tok = 1000 * total / tokens
            print(
                f"  run {run + 1:2d}: {total:6.3f}s  {tokens} tokens  {ms_tok:.2f} ms/token  "
                f"ttft {times[0]:.3f}s",
                flush=True,
            )
            rows.append(
                {"framework": key, "run": run, "total_s": total, "ttft_s": times[0]}
                | {"tokens": tokens, "ms_per_token": ms_tok}
            )
            elapsed = per_token(times, tokens)
            token_rows += [
                {"framework": key, "run": run, "token": i + 1, "elapsed_s": t}
                for i, t in enumerate(elapsed)
            ]
    except (
        httpx.HTTPError,
        RuntimeError,
        KeyError,
        OSError,
    ) as e:  # keep going: one broken framework shouldn't lose the others
        print(f"  FAILED: {e}", flush=True)
    finally:
        stop(proc)

if not rows:
    raise SystemExit("no successful runs, check results/logs/")

for name, data in (("results.csv", rows), ("token_times.csv", token_rows)):
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=data[0])
        w.writeheader()
        w.writerows(data)

done = [k for k in args.only if any(r["framework"] == k for r in rows)]
labels = [SERVERS[k][0] for k in done]
colors = dict(zip(SERVERS, ["#4C72B0", "#55A868", "#C44E52"]))
title = f"{MODEL}, bf16, M1 Pro, {args.runs} runs, max_tokens={args.max_tokens}, " + (
    f"temperature={args.temperature}, seed={args.seed}" if args.temperature > 0 else "greedy"
)

# Bar plot: mean time per generated token (whole request / tokens), error bar = std.
ms = {k: [r["ms_per_token"] for r in rows if r["framework"] == k] for k in done}
toks = {k: statistics.median(r["tokens"] for r in rows if r["framework"] == k) for k in done}
means = [statistics.mean(ms[k]) for k in done]
stds = [statistics.stdev(ms[k]) if len(ms[k]) > 1 else 0 for k in done]
print("\nframework     ms/token (mean)  std  median tokens")
for k, m, sd in zip(done, means, stds):
    print(f"{k:13s} {m:14.2f} {sd:5.2f} {toks[k]:8.0f}")

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(labels, means, yerr=stds, capsize=6, color=[colors[k] for k in done])
ax.bar_label(bars, labels=[f"{m:.1f} ms" for m in means], padding=4)
ax.set_ylabel("Mean time per generated token (ms)")
ax.set_title(title + "\n(request time / generated tokens, error bar = std)", fontsize=9)
ax.set_ylim(0, max(m + sd for m, sd in zip(means, stds)) * 1.25)
fig.tight_layout()
fig.savefig(OUT / "time_per_token.png", dpi=150)

# Line plot: time per token at each position in the output, mean over runs.
# Token 1 is skipped: its time is the prompt prefill (TTFT), not a decode step.
fig, ax = plt.subplots(figsize=(8, 4.5))
for k, label in zip(done, labels):
    runs = [
        np.diff([t["elapsed_s"] for t in token_rows if t["framework"] == k and t["run"] == r])
        for r in sorted({r["run"] for r in rows if r["framework"] == k})
    ]
    n = max(map(len, runs))
    padded = np.full((len(runs), n), np.nan)
    for j, d in enumerate(runs):
        padded[j, : len(d)] = d
    mean_ms = 1000 * np.nanmean(padded, axis=0)  # runs can stop at different lengths
    w = min(args.smooth, len(mean_ms))
    smooth = np.convolve(mean_ms, np.ones(w) / w, mode="valid")
    ax.plot(
        np.arange(len(smooth)) + 2 + (w - 1) / 2,
        smooth,
        label=label.replace("\n", " "),
        color=colors[k],
    )
ax.set_xlabel("Generated tokens so far")
ax.set_ylabel("Time per token (ms)")
ax.set_title(title + f"\n(mean over runs, moving average of {args.smooth} tokens)", fontsize=9)
ax.set_ylim(bottom=0)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "latency_vs_tokens.png", dpi=150)
print(f"\nsaved results.csv, token_times.csv, time_per_token.png, latency_vs_tokens.png in {OUT}")
