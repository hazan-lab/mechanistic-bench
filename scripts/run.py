"""Dispatcher: run one model across many tasks at a given scale.

Usage:
    # all tasks in configs/scale_1m/tasks.yaml
    uv run python scripts/run.py --scale 1m --model transformer

    # a subset of tasks
    uv run python scripts/run.py --scale 10m --model mamba --tasks k_hop two_hop

    # print what would launch
    uv run python scripts/run.py --scale 10m --model attn --dry_run

Each task runs in its own subprocess (one crash doesn't kill the sweep) and
writes its resolved config.yaml to ``<out_dir>/<task>-<arch>-<scale>/``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

from mechbench.configs import list_scales


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_tasks_yaml(scale: str) -> list[str]:
    p = _repo_root() / "configs" / f"scale_{scale}" / "tasks.yaml"
    if not p.exists():
        raise FileNotFoundError(p)
    data = yaml.safe_load(p.read_text()) or {}
    return list((data.get("tasks") or {}).keys())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scale", required=True, choices=list_scales())
    p.add_argument("--model", required=True,
                   help="Model YAML basename under configs/scale_<scale>/models/")
    p.add_argument("--tasks", nargs="*", default=None,
                   help="Subset of tasks to run (default: every task in tasks.yaml).")
    p.add_argument("--out_dir", default=None,
                   help="Override TrainConfig.out_dir for every dispatched run.")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip runs whose out_dir already has a model.pt.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print the commands that would run, then exit.")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="All remaining args (must come last on the command line) "
                        "forwarded to each train.py invocation "
                        "(e.g. --extra --max_steps 50 --wandb). "
                        "Do not pass --out_dir here; use the top-level --out_dir "
                        "so --skip_existing checks the right directory.")
    return p.parse_args()


def _arch_for(scale: str, model: str) -> str:
    p = _repo_root() / "configs" / f"scale_{scale}" / "models" / f"{model}.yaml"
    return yaml.safe_load(p.read_text())["arch"]


def main() -> None:
    args = parse_args()
    all_tasks = _load_tasks_yaml(args.scale)
    tasks = args.tasks or all_tasks
    unknown = set(tasks) - set(all_tasks)
    if unknown:
        raise SystemExit(
            f"Tasks not listed in configs/scale_{args.scale}/tasks.yaml: {sorted(unknown)}"
        )

    if "--out_dir" in (args.extra or []):
        raise SystemExit(
            "--out_dir must be supplied at the run.py level, not inside --extra "
            "(otherwise --skip_existing checks the wrong directory)."
        )

    arch = _arch_for(args.scale, args.model)

    # Resolve default out_dir the same way TrainConfig does, so skip_existing
    # can check model.pt before dispatching.
    from mechbench.training import TrainConfig
    default_out_dir = args.out_dir or TrainConfig.out_dir
    out_root = Path(default_out_dir)

    cmds: list[list[str]] = []
    skipped = 0
    for task in tasks:
        run_name = f"{task}-{arch}-{args.scale}"
        if args.skip_existing and (out_root / run_name / "model.pt").exists():
            skipped += 1
            continue
        cmd = [
            sys.executable, "-u", "scripts/train.py",
            "--scale", args.scale,
            "--model", args.model,
            "--task", task,
        ]
        if args.out_dir:
            cmd += ["--out_dir", args.out_dir]
        cmd += list(args.extra or [])
        cmds.append(cmd)

    msg = f"dispatching {len(cmds)} runs (scale={args.scale}, model={args.model}, arch={arch})"
    if args.skip_existing:
        msg += f" [skipping {skipped} already-complete]"
    print(msg, flush=True)

    if args.dry_run:
        for c in cmds:
            print(" ".join(c))
        return

    t_start = time.time()
    failures: list[str] = []
    for i, cmd in enumerate(cmds, 1):
        task = cmd[cmd.index("--task") + 1]
        print(f"[{i}/{len(cmds)}] {task}: {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd, cwd=_repo_root()).returncode
        if rc != 0:
            failures.append(f"{task} (rc={rc})")

    elapsed = time.time() - t_start
    print(f"done in {elapsed/60:.1f} min; {len(cmds) - len(failures)}/{len(cmds)} succeeded")
    if failures:
        print("failed runs:", ", ".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
