"""Dispatch the full mechanistic-suite sweep (task × architecture) at a scale."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from mechbench.configs import list_archs, list_scales
from mechbench.tasks import list_tasks


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", default="1m", choices=list_scales())
    p.add_argument("--tasks", nargs="*", default=None, help="subset of tasks (default: all)")
    p.add_argument("--archs", nargs="*", default=None, help="subset of architectures (default: all)")
    p.add_argument("--max_steps", type=int, default=4000)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--out_dir", default="/scratch/gpfs/EHAZAN/tharuntk/mechbench/runs")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    tasks = args.tasks or list_tasks()
    archs = args.archs or list_archs()
    cmds = []
    for task in tasks:
        for arch in archs:
            cmd = [
                sys.executable, "scripts/train.py",
                "--task", task, "--arch", arch, "--scale", args.scale,
                "--max_steps", str(args.max_steps),
                "--seq_len", str(args.seq_len),
                "--batch_size", str(args.batch_size),
                "--out_dir", args.out_dir,
            ]
            cmds.append(cmd)

    print(f"dispatching {len(cmds)} runs (scale={args.scale})")
    if args.dry_run:
        for c in cmds:
            print(" ".join(c))
        return

    for i, cmd in enumerate(cmds, 1):
        print(f"[{i}/{len(cmds)}] {' '.join(cmd)}")
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
