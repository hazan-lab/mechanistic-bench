"""Load each 150m YAML, build the model, and print its param count."""

from __future__ import annotations

from pathlib import Path

import torch

from mechbench.configs.lm_config import LMTrainConfig
from mechbench.models.model import build_model


CONFIGS = [
    "configs/lm/150m/attn.yaml",
    "configs/lm/150m/mamba.yaml",
    "configs/lm/150m/alt_attn_mamba.yaml",
    "configs/lm/150m/headwise_alt_attn_mamba.yaml",
]

TARGET = 150_000_000


def main():
    print(f"{'config':>50s}  {'d_model':>8s}  {'layers':>6s}  {'params':>12s}  {'delta':>8s}")
    print("-" * 100)
    for path in CONFIGS:
        cfg = LMTrainConfig.load(Path(path))
        # build_model takes a MechConfig; compose from cfg.model fields.
        from mechbench.models.model import MechConfig
        mech_cfg = MechConfig(**{k: v for k, v in vars(cfg.model).items()
                                 if k in MechConfig.__dataclass_fields__})
        with torch.device("meta"):
            model = build_model(mech_cfg)
        n = sum(p.numel() for p in model.parameters())
        pct = 100.0 * (n - TARGET) / TARGET
        print(f"{path:>50s}  {mech_cfg.d_model:>8d}  {len(mech_cfg.block_types):>6d}  "
              f"{n/1e6:>10.2f}M  {pct:>+6.2f}%")


if __name__ == "__main__":
    main()
