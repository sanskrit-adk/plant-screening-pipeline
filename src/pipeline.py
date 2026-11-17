"""Pipeline composition and safety gates.

Order (what happens to one field capture):

    degraded -> [GATE 1: exposure] -> restore -> enhance -> compress (transmit)
             -> decode -> segment -> [GATE 2: mask area] -> CNN

Gate 1 checks that the raw capture is neither almost black nor saturated.
Gate 2 checks that the segmented leaf covers a plausible fraction of the frame.
A failed gate means "this image cannot be trusted" and the decision layer
re-images (and never CLEARs after a second failure).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict, replace

import cv2
import numpy as np
from joblib import Parallel, delayed

from .degrade import degrade, rng_for
from .stages import apply_mask, compress, enhance, restore, segment_mask

STAGES = ("restore", "enhance", "compress", "segment")


@dataclass(frozen=True)
class PipelineConfig:
    """Which stages run. Every stage can be switched off independently."""
    restore: str = "nlm"           # "none" disables restoration
    enhance: str = "clahe"         # "none" disables enhancement
    jpeg_q: int | None = 50        # None = lossless transmission (no JPEG artefacts)
    segment: bool = True           # False skips segmentation (and gate 2)

    @classmethod
    def from_cfg(cls, cfg: dict, **overrides) -> "PipelineConfig":
        """Build from the config's `pipeline` section, then apply keyword overrides."""
        pc = cfg.get("pipeline", {})
        base = cls(restore=str(pc.get("restore", "nlm") or "none"),
                   enhance=str(pc.get("enhance", "clahe") or "none"),
                   jpeg_q=pc.get("jpeg_q", 50),
                   segment=bool(pc.get("segment", True)))
        return replace(base, **overrides)

    def label(self) -> str:
        q = "lossless" if self.jpeg_q is None else f"q{self.jpeg_q}"
        return f"restore={self.restore}|enhance={self.enhance}|jpeg={q}|seg={'on' if self.segment else 'off'}"

    def as_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------
# Safety gates
# ----------------------------------------------------------------------------
def input_gate_ok(img: np.ndarray, gates: dict) -> bool:
    """SAFETY GATE 1: mean grey level of the raw capture within [min_mean, max_mean]."""
    m = float(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).mean())
    return gates["min_mean"] <= m <= gates["max_mean"]


def mask_gate_ok(area: float, gates: dict) -> bool:
    """SAFETY GATE 2: leaf mask area fraction within [mask_area_min, mask_area_max]."""
    return gates["mask_area_min"] <= area <= gates["mask_area_max"]


# ----------------------------------------------------------------------------
# One image
# ----------------------------------------------------------------------------
def run_one(degraded: np.ndarray, p: PipelineConfig, img_size: int, gates: dict) -> dict:
    """Run the enabled stages on one degraded capture, timing each stage (ms)."""
    times = {}
    g1 = input_gate_ok(degraded, gates)

    t = time.perf_counter()
    x = restore(degraded, p.restore)
    times["restore"] = (time.perf_counter() - t) * 1e3

    t = time.perf_counter()
    x = enhance(x, p.enhance)
    times["enhance"] = (time.perf_counter() - t) * 1e3

    # Transmission: the receiver only ever sees the decoded image
    t = time.perf_counter()
    x, nbytes = compress(x, p.jpeg_q)
    times["compress"] = (time.perf_counter() - t) * 1e3

    t = time.perf_counter()
    if p.segment:
        mask = segment_mask(x)
        area = float(mask.mean())
        g2 = mask_gate_ok(area, gates)
        x = apply_mask(x, mask, img_size)
    else:
        mask, area, g2 = None, float("nan"), True  # gate 2 always passes without segmentation
    times["segment"] = (time.perf_counter() - t) * 1e3

    return dict(image=x, nbytes=nbytes, area=area, gate_ok=bool(g1 and g2),
                mask=mask, times=times)


def _work(clean, gidx, p, img_size, gates, dcfg, seed, attempt):
    """joblib worker: degrade deterministically, then run the pipeline."""
    cv2.setNumThreads(1)  # one OpenCV thread per process avoids oversubscription
    deg = degrade(clean, dcfg, rng_for(seed, gidx, attempt))
    return run_one(deg, p, img_size, gates)


def process(X: np.ndarray, idxs, p: PipelineConfig, cfg: dict, seed: int,
            attempt: int = 0, dcfg: dict | None = None, keep_mask: bool = False):
    """Degrade + process the images X[idxs] in parallel.

    The degradation RNG is keyed on the GLOBAL index, so a given leaf always gets
    the same noise/lighting for a given attempt regardless of config.

    Returns (X_proc [n,H,W,3] uint8, nbytes [n], area [n], gate_ok [n] bool, masks or None).
    """
    idxs = np.asarray(idxs)
    dcfg = dcfg or cfg["degrade"]
    size = int(cfg["data"]["img_size"])
    out = Parallel(n_jobs=cfg["train"].get("n_jobs", -1), batch_size=32)(
        delayed(_work)(X[i], int(i), p, size, cfg["gates"], dcfg, seed, attempt) for i in idxs)
    Xp = np.stack([o["image"] for o in out]).astype(np.uint8)
    nbytes = np.array([o["nbytes"] for o in out], dtype=np.int64)
    area = np.array([o["area"] for o in out], dtype=np.float64)
    gate = np.array([o["gate_ok"] for o in out], dtype=bool)
    masks = None
    if keep_mask and p.segment:
        masks = np.stack([o["mask"] for o in out]).astype(np.uint8)
    return Xp, nbytes, area, gate, masks


def benchmark_latency(X: np.ndarray, idxs, p: PipelineConfig, cfg: dict, seed: int,
                      n: int | None = None) -> dict:
    """Sequential, single-threaded per-stage latency (mean ms over n images)."""
    n = int(n or cfg["experiments"]["latency_n"])
    idxs = np.asarray(idxs)[:n]
    size = int(cfg["data"]["img_size"])
    prev = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        acc = {s: [] for s in STAGES}
        for i in idxs:
            deg = degrade(X[i], cfg["degrade"], rng_for(seed, int(i), 0))
            r = run_one(deg, p, size, cfg["gates"])
            for s in STAGES:
                acc[s].append(r["times"][s])
    finally:
        cv2.setNumThreads(prev)
    return {f"ms_{s}": float(np.mean(v)) if v else 0.0 for s, v in acc.items()}
