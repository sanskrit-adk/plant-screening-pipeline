"""Pipeline stages. Every function takes/returns uint8 RGB and has an "off" option.

5.1 restore      RESTORATION  (objective: undo a known degradation, scored vs reference)
5.2 enhance      ENHANCEMENT  (subjective: make the image easier to look at / analyse)
5.3 compress     JPEG transmission over a limited-bandwidth link (None = lossless)
5.4 segment_mask leaf segmentation with morphology, apply_mask, iou
"""
from __future__ import annotations

import cv2
import numpy as np

RESTORE_METHODS = ("none", "gaussian", "median", "bilateral", "nlm", "wavelet")
ENHANCE_METHODS = ("none", "clahe", "histeq", "gamma")


# ----------------------------------------------------------------------------
# 5.1 Restoration (denoising)
# ----------------------------------------------------------------------------
def restore(img: np.ndarray, method: str | None) -> np.ndarray:
    """Denoise an RGB uint8 image with the chosen method ("none" returns it unchanged)."""
    method = (method or "none").lower()
    if method == "none":
        return img
    if method == "gaussian":
        return cv2.GaussianBlur(img, (5, 5), 0)
    if method == "median":
        return cv2.medianBlur(img, 5)
    if method == "bilateral":
        return cv2.bilateralFilter(img, 7, 50, 50)
    if method == "nlm":
        # OpenCV's colour NLM works in CIELAB internally; channel order does not matter here.
        return cv2.fastNlMeansDenoisingColored(img, None, 12, 12, 7, 21)
    if method == "wavelet":
        from skimage.restoration import denoise_wavelet
        out = denoise_wavelet(img.astype(np.float64) / 255.0, channel_axis=-1,
                              convert2ycbcr=True, method="BayesShrink", mode="soft",
                              rescale_sigma=True)
        return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    raise ValueError(f"unknown restore method {method!r}; choose from {RESTORE_METHODS}")


# ----------------------------------------------------------------------------
# 5.2 Enhancement (contrast / illumination)
# ----------------------------------------------------------------------------
_GAMMA_LUT = np.array([((i / 255.0) ** 0.7) * 255 for i in range(256)]).clip(0, 255).astype(np.uint8)


def enhance(img: np.ndarray, method: str | None) -> np.ndarray:
    """Contrast enhancement on luminance only, so leaf colour (a disease cue) is preserved."""
    method = (method or "none").lower()
    if method == "none":
        return img
    if method == "clahe":
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if method == "histeq":
        ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        ycc[..., 0] = cv2.equalizeHist(ycc[..., 0])
        return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)
    if method == "gamma":
        return cv2.LUT(img, _GAMMA_LUT)  # gamma 0.7 brightens dark regions
    raise ValueError(f"unknown enhance method {method!r}; choose from {ENHANCE_METHODS}")


# ----------------------------------------------------------------------------
# 5.3 Compression (transmission)
# ----------------------------------------------------------------------------
def raw_bytes(img: np.ndarray) -> int:
    """Uncompressed size of an 8-bit RGB image: H * W * 3."""
    return int(img.shape[0] * img.shape[1] * 3)


def compress(img: np.ndarray, quality: int | None) -> tuple[np.ndarray, int]:
    """Encode + decode as the receiver would see it. Returns (decoded_rgb, n_bytes).

    quality None = lossless: PNG (level 9) is encoded only to measure the size and
    the image is returned unchanged.
    """
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if quality is None:
        ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        if not ok:
            raise RuntimeError("PNG encoding failed")
        return img, int(buf.size)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB), int(buf.size)


# ----------------------------------------------------------------------------
# 5.4 Segmentation + morphology
# ----------------------------------------------------------------------------
_OPEN_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_CLOSE_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


def saturation_channel(img: np.ndarray) -> np.ndarray:
    """HSV saturation with very dark pixels (V < 25) zeroed: they have unreliable hue/sat."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    s = hsv[..., 1].copy()
    s[hsv[..., 2] < 25] = 0
    return s


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes of a 0/1 mask (e.g. lesions that failed the threshold).

    The mask is padded by one background pixel so the flood fill from the corner
    always starts outside the leaf, even if the leaf touches the image border.
    """
    m = np.pad(mask.astype(np.uint8), 1)
    flood = m.copy() * 255
    h, w = flood.shape
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)       # fill the outside background
    holes = cv2.bitwise_not(flood)                   # what was not reached = holes
    out = cv2.bitwise_or(m * 255, holes)
    return (out[1:-1, 1:-1] > 0).astype(np.uint8)


def segment_mask(img: np.ndarray) -> np.ndarray:
    """Leaf mask (uint8 0/1): saturation -> blur -> Otsu -> open -> close -> largest CC -> fill."""
    s = saturation_channel(img)
    s = cv2.GaussianBlur(s, (5, 5), 0)
    _, th = cv2.threshold(s, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, _OPEN_K)                 # remove speckle
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, _CLOSE_K, iterations=2)  # bridge gaps
    n, lab, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    if n <= 1:
        return np.zeros_like(th, dtype=np.uint8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))       # label 0 = background
    return fill_holes((lab == largest).astype(np.uint8))


def apply_mask(img: np.ndarray, mask: np.ndarray, out_size: int, pad: int = 4) -> np.ndarray:
    """Zero the background, crop to the mask bounding box (+pad), resize to out_size."""
    if mask is None or mask.sum() == 0:
        return img
    out = img * mask[..., None].astype(img.dtype)
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    crop = out[y0:y1, x0:x1]
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two binary masks (1.0 if both are empty)."""
    a = a.astype(bool)
    b = b.astype(bool)
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)
