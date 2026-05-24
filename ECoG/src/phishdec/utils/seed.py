from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class SeedConfig:
    seed: int = 10
    deterministic: bool = False
    benchmark: bool = False


def set_seed(seed: int = 10, deterministic: bool = True, benchmark: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            # Required by CUDA deterministic kernels on some setups.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = bool(deterministic)
            torch.backends.cudnn.benchmark = bool(benchmark if not deterministic else False)

        if deterministic:
            try:
                torch.use_deterministic_algorithms(False)
            except Exception:
                pass
    except Exception:
        pass


def seed_everything(cfg: Optional[SeedConfig] = None) -> None:
    cfg = cfg or SeedConfig()
    set_seed(cfg.seed, deterministic=cfg.deterministic, benchmark=cfg.benchmark)
