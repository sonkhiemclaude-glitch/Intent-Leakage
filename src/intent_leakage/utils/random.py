"""Seed control for deterministic synthetic and analysis runs."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> None:
    """Seed supported local RNGs without requiring optional dependencies."""

    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a declared dependency
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
