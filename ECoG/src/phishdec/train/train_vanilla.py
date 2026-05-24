from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from phishdec.train.train_ccia import run_ccia_from_config


def run_vanilla_from_config(cfg: Dict[str, Any], run_dir: str) -> str:
    cfg_local = deepcopy(cfg)
    cfg_local.setdefault("train", {})
    cfg_local["train"]["variant"] = "vanilla"
    return run_ccia_from_config(cfg=cfg_local, run_dir=run_dir)
