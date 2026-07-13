#!/usr/bin/env python3
"""LR sensitivity sweep driver (ICML workshop camera-ready).

Grid: lr {1e-4, 1e-3} x models {transformer(attn), mamba, stu, alt_attn_mamba}
      x 11 headline tasks = 88 runs, seed 1, 1m scale.
Existing 5-seed data at lr 3e-4 is the middle point.

Runs 2 jobs concurrently. NOTE: della-fongpu currently has a single H100 NVL
(95GB) instead of the earlier 2x A100, so both workers share physical GPU 0
(CUDA_VISIBLE_DEVICES=0). The prior seed sweep ran 4 workers across GPUs at
this scale, so co-locating two ~1M-param runs is safe.

Output: /scratch/gpfs/EHAZAN/tharuntk/lr_sweep/{lr1e-4,lr1e-3}/<task>-<arch>-1m/history.json
Log:    /scratch/gpfs/EHAZAN/tharuntk/lr_sweep/sweep.log
"""
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

REPO = Path("/home/tt6444/mechanistic-bench")
PY = str(REPO / ".venv/bin/python")
ROOT = Path("/scratch/gpfs/EHAZAN/tharuntk/lr_sweep")
LOG = ROOT / "sweep.log"
RUNLOGS = ROOT / "run_logs"

LRS = [("lr1e-4", "1e-4"), ("lr1e-3", "1e-3")]
# (model_yaml_basename, arch_name_used_in_run_dir)
MODELS = [("transformer", "attn"), ("mamba", "mamba"),
          ("stu", "stu"), ("alt_attn_mamba", "alt_attn_mamba")]
TASKS = ["copy", "induction", "associative", "selective_copy", "needle",
         "counting", "parity", "state_tracking", "copy_count",
         "state_retrieve", "selective_parity"]

# Both workers pinned to GPU 0 (single-GPU node; see module docstring).
WORKER_GPUS = ["0", "0"]

_log_lock = threading.Lock()


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with _log_lock:
        with open(LOG, "a") as f:
            f.write(line + "\n")
    print(line, flush=True)


def final_tok_acc(hist_path: Path):
    try:
        h = json.loads(hist_path.read_text())
        return h["history"][-1]["tok_acc"]
    except Exception:
        return None


def already_done(out_dir: Path) -> bool:
    hp = out_dir / "history.json"
    return hp.exists() and final_tok_acc(hp) is not None


def worker(wid: int, jobs: "queue.Queue", counters: dict) -> None:
    gpu = WORKER_GPUS[wid]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    while True:
        try:
            lr_dir, lr, model, arch, task = jobs.get_nowait()
        except queue.Empty:
            return
        run_name = f"{task}-{arch}-1m"
        out_root = ROOT / lr_dir
        out_dir = out_root / run_name
        if already_done(out_dir):
            with _log_lock:
                counters["done"] += 1
            log(f"[{counters['done']}/{counters['total']}] SKIP  gpu{gpu} {lr_dir}/{run_name} (already complete)")
            continue
        cmd = [PY, "-u", "scripts/train.py", "--scale", "1m",
               "--model", model, "--task", task,
               "--lr", lr, "--seed", "1", "--out_dir", str(out_root)]
        run_log = RUNLOGS / f"{lr_dir}-{run_name}.log"
        t0 = time.time()
        with open(run_log, "w") as lf:
            rc = subprocess.call(cmd, cwd=str(REPO), env=env,
                                 stdout=lf, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        acc = final_tok_acc(out_dir / "history.json")
        with _log_lock:
            counters["done"] += 1
            if rc != 0 or acc is None:
                counters["fail"] += 1
        status = "OK  " if (rc == 0 and acc is not None) else "FAIL"
        rem = counters["total"] - counters["done"]
        log(f"[{counters['done']}/{counters['total']}] {status} gpu{gpu} "
            f"{lr_dir}/{run_name} rc={rc} {dt:.0f}s tok_acc={acc} "
            f"fails={counters['fail']} remaining={rem}")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    RUNLOGS.mkdir(parents=True, exist_ok=True)
    jobs = queue.Queue()
    n = 0
    for lr_dir, lr in LRS:
        for task in TASKS:
            for model, arch in MODELS:
                jobs.put((lr_dir, lr, model, arch, task))
                n += 1
    counters = {"done": 0, "fail": 0, "total": n}
    log(f"START lr sweep: {n} jobs, lrs={[l for l, _ in LRS]}, "
        f"models={[m for m, _ in MODELS]}, tasks={len(TASKS)}, "
        f"2 workers on gpus={WORKER_GPUS} (single H100 NVL node)")
    threads = [threading.Thread(target=worker, args=(i, jobs, counters))
               for i in range(2)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log(f"DONE {counters['done']}/{n} in {(time.time()-t0)/60:.1f} min, "
        f"fails={counters['fail']}")


if __name__ == "__main__":
    main()
