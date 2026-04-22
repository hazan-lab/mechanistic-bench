"""Standalone evaluator: load a checkpoint and run the configured evaluators."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from mechbench.configs.lm_config import LMTrainConfig  # noqa: E402
from mechbench.eval import build_evaluators  # noqa: E402
from mechbench.models.model import build_model  # noqa: E402
from mechbench.training.lm_trainer import _precision_dtype, run_evaluators  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("checkpoint", help="Path to a .pt file from lm_trainer")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--no-validate-paths", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = LMTrainConfig.load(
        args.config, overrides=args.overrides, validate_paths=not args.no_validate_paths
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.model.to_mech_config()).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    evaluators = build_evaluators(cfg, device)
    precision_dtype = _precision_dtype(cfg.precision)
    metrics = run_evaluators(model, evaluators, device, precision_dtype)
    for k, v in sorted(metrics.items()):
        print(f"{k}\t{v:.6f}")


if __name__ == "__main__":
    main()
