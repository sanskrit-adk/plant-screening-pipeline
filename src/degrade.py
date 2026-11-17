"""Controlled, reproducible field degradation of clean PlantVillage images.

Model of a field capture (all per-pixel, per-channel):

    reference = clip( G(x,y) * blur(clean) )                  noise-free but unevenly lit
    degraded  = clip( reference + N(0, sigma^2) ) [+ salt & pepper]

G is a smooth illumination field in [illum_min, illum_max]. Because we keep the
noise-free `reference`, restoration (denoising) can be scored objectively
(PSNR/SSIM against reference), while enhancement (fixing the lighting) is a
separate, subjective step.
"""
from __future__ import annotations

import cv2
import numpy as np


def rng_for(seed: int, idx: int, attempt: int) -> np.random.Generator:
    """Deterministic RNG for (seed, image index, attempt).

    attempt 0 = first photo, attempt 1 = the re-image. The same image/attempt pair
    always receives exactly the same degradation, whatever the pipeline config.
    """
    return np.random.default_rng([int(seed), int(idx), int(attempt)])


def illumination_field(h: int, w: int, rng: np.random.Generator,
                       gmin: float, gmax: float) -> np.ndarray:
    """Smooth gain field: random-direction linear gradient blended with an off-centre vignette.

    Returns float32 [h, w] with values spanning exactly [gmin, gmax].
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx /= max(w - 1, 1)
    yy /= max(h - 1, 1)

    # 1) linear gradient along a random direction
    theta = rng.uniform(0, 2 * np.pi)
    lin = np.cos(theta) * xx + np.sin(theta) * yy
    lin = (lin - lin.min()) / (np.ptp(lin) + 1e-8)

    # 2) radial vignette around an off-centre point (bright centre, dark rim)
    cx, cy = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rad = 1.0 - (r - r.min()) / (np.ptp(r) + 1e-8)

    # 3) blend with a random weight, then normalise to [gmin, gmax]
    a = rng.uniform(0.3, 0.7)
    f = a * lin + (1 - a) * rad
    f = (f - f.min()) / (np.ptp(f) + 1e-8)
    return (gmin + f * (gmax - gmin)).astype(np.float32)


def degrade(clean: np.ndarray, dcfg: dict, rng: np.random.Generator,
            return_reference: bool = False):
    """Apply blur -> illumination -> Gaussian noise -> clip -> salt & pepper.

    Args:
        clean: uint8 RGB [H,W,3].
        dcfg:  the `degrade` section of the config.
        rng:   generator from rng_for(...).
        return_reference: also return the noise-free, unevenly lit reference.
    """
    img = clean.astype(np.float32)
    h, w = img.shape[:2]

    # 1. optional blur (focus / motion softness)
    k = int(dcfg.get("blur_ksize", 0) or 0)
    if k > 1:
        k += (k + 1) % 2  # kernel must be odd
        img = cv2.GaussianBlur(img, (k, k), 0)

    # 2. uneven illumination -> REFERENCE (noise-free)
    g = illumination_field(h, w, rng, float(dcfg["illum_min"]), float(dcfg["illum_max"]))
    lit = img * g[..., None]
    reference = np.clip(lit, 0, 255).astype(np.uint8)

    # 3. additive Gaussian sensor noise, 4. clip
    sigma = float(dcfg.get("noise_sigma", 0) or 0)
    noisy = lit + rng.normal(0.0, sigma, size=lit.shape).astype(np.float32) if sigma > 0 else lit
    out = np.clip(noisy, 0, 255).astype(np.uint8)

    # 5. optional salt & pepper (dead / hot pixels)
    p = float(dcfg.get("sp_prob", 0) or 0)
    if p > 0:
        u = rng.random((h, w))
        out[u < p / 2] = 0
        out[(u >= p / 2) & (u < p)] = 255

    return (out, reference) if return_reference else out
