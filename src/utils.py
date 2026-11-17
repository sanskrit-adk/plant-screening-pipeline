"""Shared helpers: project paths, config loading, seeding, device selection."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

# Project root = parent of src/. All relative paths in config.yaml resolve against it.
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"   # CSV / JSON outputs
FIGURES = ROOT / "figures"   # PNG outputs


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load config.yaml (default: <ROOT>/config.yaml) into a plain dict."""
    path = Path(path) if path else ROOT / "config.yaml"
    if not path.is_absolute():
        path = ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Seed every RNG we use and make cuDNN deterministic."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(name: str = "auto") -> torch.device:
    """'auto' picks cuda, then Apple mps, then cpu; anything else is passed through."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def ensure_dirs() -> None:
    """Create the output folders (and the data cache folder) if missing."""
    for d in (RESULTS, FIGURES, ROOT / "data" / "cache"):
        d.mkdir(parents=True, exist_ok=True)


def resolve(path: str | os.PathLike) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
