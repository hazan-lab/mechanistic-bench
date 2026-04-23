"""CLI entry point for language-modeling pretraining.

Usage::

    uv run python scripts/train_lm.py configs/lm/scale_10m/mechbench.yaml \
        [key.subkey=value] [other.key=value] ...

Any trailing ``key=value`` arguments are applied as OmegaConf dotlist
overrides on top of the YAML config.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project src/ is importable when running from the repo root
# without installing the package in editable mode.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mechbench.configs.lm_config import LMTrainConfig  # noqa: E402
from mechbench.training.lm_trainer import train  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. 'optimizer.learning_rate=1e-4'",
    )
    parser.add_argument("--no-validate-paths", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = LMTrainConfig.load(
        args.config,
        overrides=args.overrides,
        validate_paths=not args.no_validate_paths,
    )
    train(cfg)


if __name__ == "__main__":
    main()
