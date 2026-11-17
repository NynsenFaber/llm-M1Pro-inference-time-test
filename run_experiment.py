"""Start each server in turn (A: transformers, B: mlx-lm, C: vllm-metal), time 10 requests,
stop it, then save results/results.csv and a bar plot of mean inference time.

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
p.add_argument("--max-tokens", type=int, default=256)
p.add_argument("--prompt", default="Write me a joke about computers.")
p.add_argument("--startup-timeout", type=int, default=1800, help="s; first start downloads")
args = p.parse_args()


def chat(port, max_tokens):
    """One streamed request. Returns (total_s, ttft_s, completion_tokens)."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    ttft, usage = None, None
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
            if ttft is None and any(
                (c.get("delta") or {}).get("content") for c in chunk.get("choices", [])
            ):
                ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    if ttft is None or not usage:
        raise RuntimeError(f"no streamed text or no usage from port {port}")
    return total, ttft, usage["completion_tokens"]


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
rows = []
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
            total, ttft, tokens = chat(port, args.max_tokens)
            print(
                f"  run {run + 1:2d}: {total:6.3f}s  ttft {ttft:.3f}s  {tokens} tokens", flush=True
            )
            rows.append(
                {"framework": key, "run": run, "total_s": total, "ttft_s": ttft, "tokens": tokens}
            )
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

with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0])
    w.writeheader()
    w.writerows(rows)

done = [k for k in args.only if any(r["framework"] == k for r in rows)]
times = {k: [r["total_s"] for r in rows if r["framework"] == k] for k in done}
toks = {k: statistics.median(r["tokens"] for r in rows if r["framework"] == k) for k in done}
means = [statistics.mean(times[k]) for k in done]
stds = [statistics.stdev(times[k]) if len(times[k]) > 1 else 0 for k in done]

print("\nframework      mean (s)  std (s)  median tokens")
for k, m, s in zip(done, means, stds):
    print(f"{k:13s} {m:8.3f} {s:8.3f} {toks[k]:8.0f}")

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(
    [SERVERS[k][0] for k in done],
    means,
    yerr=stds,
    capsize=6,
    color=["#4C72B0", "#55A868", "#C44E52"][: len(done)],
)
ax.bar_label(
    bars, labels=[f"{m:.2f}s\n({toks[k]:.0f} tok)" for k, m in zip(done, means)], padding=4
)
ax.set_ylabel("Mean inference time per request (s)")
ax.set_title(
    f"{MODEL}, bf16, M1 Pro: {args.runs} runs, max_tokens={args.max_tokens} (error bar = std)"
)
ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.3)
fig.tight_layout()
fig.savefig(OUT / "inference_time.png", dpi=150)
print(f"\nsaved {OUT / 'results.csv'} and {OUT / 'inference_time.png'}")
