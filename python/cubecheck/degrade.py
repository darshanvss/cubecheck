"""Make clean renders look like they came from a cheap webcam or a phone.

Only photometric changes live here: colour, exposure, blur, noise, sharpening,
vignetting and JPEG. None of them moves a pixel, so keypoint labels stay valid.
Geometric augmentation (flip, rotate, scale, crop) moves keypoints too, so it is
left to the training library, which transforms the labels along with the image.

Every step runs with some probability, so the dataset mixes clean and terrible
images. All randomness comes from the generator passed in, so a build is
reproducible.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import numpy as np


def _odd(n: float) -> int:
    k = max(3, int(round(n)))
    return k if k % 2 else k + 1


def white_balance(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Auto white balance gets fooled by warm bulbs and blue daylight: tint the whole frame."""
    warm = rng.uniform(-0.25, 0.25)  # >0 warmer (more red, less blue)
    gains = np.array([1 - warm, 1 + rng.uniform(-0.06, 0.06), 1 + warm])  # BGR order
    # One 256-entry table per channel: the same result as multiplying every pixel, much faster.
    lut = np.clip(np.arange(256, dtype=np.float32)[:, None] * gains.astype(np.float32), 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut[None]), {"gains_bgr": gains.round(3).tolist()}


def exposure(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    gain = float(np.exp(rng.normal(0, 0.3)))
    gamma = float(rng.uniform(0.75, 1.35))
    lut = ((np.clip(np.arange(256, dtype=np.float32) / 255 * gain, 0, 1) ** gamma) * 255).astype(np.uint8)
    return cv2.LUT(img, lut), {"gain": round(gain, 3), "gamma": round(gamma, 3)}


def low_resolution(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Small sensors and video scaling: shrink, then blow back up to the same size."""
    h, w = img.shape[:2]
    factor = float(rng.uniform(1.5, 4.0))
    small = cv2.resize(img, (max(16, int(w / factor)), max(16, int(h / factor))), interpolation=cv2.INTER_AREA)
    interp = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST][int(rng.integers(0, 3))]
    return cv2.resize(small, (w, h), interpolation=interp), {"factor": round(factor, 2)}


def blur(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Missed focus (round blur) or a shaky hand (streak in one direction)."""
    kind = ["gaussian", "defocus", "motion"][int(rng.integers(0, 3))]
    if kind == "gaussian":
        sigma = float(rng.uniform(0.6, 2.5))
        return cv2.GaussianBlur(img, (0, 0), sigma), {"kind": kind, "sigma": round(sigma, 2)}
    size = _odd(rng.uniform(3, 13))
    kernel = np.zeros((size, size), np.float32)
    if kind == "defocus":
        cv2.circle(kernel, (size // 2, size // 2), size // 2, 1.0, -1)
    else:
        angle = float(rng.uniform(0, 180))
        dx, dy = np.cos(np.radians(angle)) * size / 2, np.sin(np.radians(angle)) * size / 2
        c = size // 2
        cv2.line(kernel, (int(c - dx), int(c - dy)), (int(c + dx), int(c + dy)), 1.0, 1)
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel), {"kind": kind, "size": size}


def sensor_noise(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Shot noise grows with brightness, read noise is constant; dark scenes look grainier."""
    x = img.astype(np.float32) / 255
    shot = float(rng.uniform(0.002, 0.03))
    read = float(rng.uniform(0.002, 0.02))
    sigma = np.sqrt(x * shot + read**2)
    noise = rng.standard_normal(x.shape, dtype=np.float32) * sigma
    if rng.random() < 0.5:  # colour speckle, not just brightness grain
        noise += rng.standard_normal(x.shape, dtype=np.float32) * np.float32(read)
    return (np.clip(x + noise, 0, 1) * 255).astype(np.uint8), {"shot": round(shot, 4), "read": round(read, 4)}


def oversharpen(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Webcams sharpen aggressively, leaving bright halos along edges."""
    amount = float(rng.uniform(0.4, 1.5))
    blurred = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(1, 3)))
    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0), {"amount": round(amount, 2)}


@lru_cache(maxsize=8)
def _falloff(h: int, w: int) -> np.ndarray:
    """Squared distance from the centre (1 at the middle of each edge), halved and capped at 1."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    return np.clip(r2 / 2, 0, 1)


def vignette(img: np.ndarray, rng: np.random.Generator, frame: tuple[int, int, int, int] | None = None) -> tuple[np.ndarray, dict]:
    """Darker towards the frame's edges. `frame` = (x0, y0, width, height) when `img` is a cut-out
    of a bigger frame, so the darkening follows the whole frame, not the cut-out."""
    h, w = img.shape[:2]
    strength = float(rng.uniform(0.15, 0.55))
    if frame is None:
        falloff = _falloff(h, w)
    else:
        x0, y0, fw, fh = frame
        xx = ((np.arange(w, dtype=np.float32) + x0 - fw / 2) / (fw / 2)) ** 2
        yy = ((np.arange(h, dtype=np.float32) + y0 - fh / 2) / (fh / 2)) ** 2
        falloff = np.clip((yy[:, None] + xx[None, :]) / 2, 0, 1)
    mask = 1 - np.float32(strength) * falloff
    return (img.astype(np.float32) * mask[..., None]).astype(np.uint8), {"strength": round(strength, 2)}


def jpeg(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    quality = int(rng.integers(25, 92))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), {"quality": quality}


# (name, function, probability), applied in camera-pipeline order.
PIPELINE = [
    ("white_balance", white_balance, 0.5),
    ("exposure", exposure, 0.5),
    ("low_resolution", low_resolution, 0.35),
    ("blur", blur, 0.35),
    ("sensor_noise", sensor_noise, 0.55),
    ("oversharpen", oversharpen, 0.2),
    ("vignette", vignette, 0.25),
    ("jpeg", jpeg, 0.7),
]


def degrade(img: np.ndarray, rng: np.random.Generator, clean_chance: float = 0.1,
            frame: tuple[int, int, int, int] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a degraded copy of a BGR uint8 image and a record of what was applied.

    Every step is local except the vignette, so a cut-out degrades the same as
    that part of the whole frame would; pass `frame` = (x0, y0, width, height)
    for a cut-out (training uses this to skip the parts of the frame it never sees).
    """
    applied: dict[str, Any] = {}
    if rng.random() < clean_chance:
        return img.copy(), applied
    out = img
    for name, fn, p in PIPELINE:
        if rng.random() < p:
            out, params = vignette(out, rng, frame) if fn is vignette else fn(out, rng)
            applied[name] = params
    return out, applied
