"""Run a list of downstream ICL / perplexity evaluators on a saved checkpoint.

Builds the model from a training YAML, loads the checkpoint, then builds
a *custom* list of evaluators (by task label) and runs them, writing a
JSON dict of metrics to stdout.

Intended for extending the 4-evaluator training sweep with additional
ICL tasks (winogrande, openbookqa, sciq, arc_easy, commonsense_qa,
social_iqa, mmlu_*) without retraining.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from mechbench.configs.lm_config import EvaluatorConfig, EvaluatorType, LMTrainConfig  # noqa: E402
from mechbench.eval import build_evaluator  # noqa: E402
from mechbench.models.model import build_model  # noqa: E402
from mechbench.tokenizer import Tokenizer  # noqa: E402
from mechbench.training.lm_trainer import _precision_dtype, run_evaluators  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Training YAML (for model/tokenizer)")
    parser.add_argument("checkpoint", help="Path to model.pt saved by lm_trainer")
    parser.add_argument("--tasks", required=True, help="Comma-separated downstream task labels")
    parser.add_argument("--out", required=True, help="JSON output path")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-validate-paths", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("eval_lm_extended")

    cfg = LMTrainConfig.load(args.config, validate_paths=not args.no_validate_paths)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("building model")
    model = build_model(cfg.model.to_mech_config()).to(device)
    log.info("loading checkpoint %s", args.checkpoint)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    tokenizer = Tokenizer.from_identifier(
        cfg.tokenizer.identifier,
        eos_token_id=cfg.model.eos_token_id,
        pad_token_id=cfg.model.pad_token_id,
        vocab_size=cfg.model.vocab_size,
        truncate_direction=cfg.tokenizer.truncate_direction,
    )

    precision_dtype = _precision_dtype(cfg.precision)

    task_labels = [t.strip() for t in args.tasks.split(",") if t.strip()]
    results: dict[str, dict] = {}
    for label in task_labels:
        log.info("=== running task: %s ===", label)
        t0 = time.time()
        eval_cfg = EvaluatorConfig(label=label, type=EvaluatorType.downstream)
        eval_cfg.device_eval_batch_size = args.batch_size
        try:
            evaluator = build_evaluator(cfg, eval_cfg, tokenizer, device)
            metrics = run_evaluators(model, [evaluator], device, precision_dtype)
            dur = time.time() - t0
            log.info("task %s metrics: %s (%.1fs)", label, metrics, dur)
            results[label] = {"metrics": {k: float(v) for k, v in metrics.items()}, "seconds": dur, "ok": True}
        except Exception as e:
            dur = time.time() - t0
            log.exception("task %s failed: %s", label, e)
            results[label] = {"error": repr(e), "seconds": dur, "ok": False}

        # Write incrementally after each task in case we OOM later.
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(
                {
                    "config": str(args.config),
                    "checkpoint": str(args.checkpoint),
                    "results": results,
                },
                f,
                indent=2,
            )

    log.info("done -> %s", args.out)


if __name__ == "__main__":
    main()
