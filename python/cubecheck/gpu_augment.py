"""Camera effects and view cutting on the GPU, for training the corner model.

Preparing a training picture on the CPU (decode, camera effects, cut-out) cost
more than the GPU's work, so on RunPod the GPU sat mostly idle (docs/training.md).
Here the work is split:

- CPU, in the data workers: work out which part of the frame the view needs
  (`canvas_box`) and draw the random settings of every camera effect
  (`draw_effects`). On an NVIDIA GPU that's all: the JPEG bytes go to the GPU,
  which decodes them (nvJPEG) and cuts the part out (`cut_canvas_gpu`). Elsewhere
  the workers decode and cut it out themselves (`cut_canvas`).
- GPU, a whole batch at once (`render`): apply the camera effects, the same ones
  in the same order and with the same random ranges as degrade.py, then cut the
  final turned, zoomed and maybe mirrored 256 x 256 view.

Effects happen at camera resolution, as in degrade.py. When a cut-out is bigger
than the canvas it's shrunk by k first, and sizes measured in camera pixels
(blur, the low-resolution factor, sharpening) are divided by k; noise is too,
since shrinking by k averages k x k pixels. JPEG blocks stay 8 canvas pixels.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .corners import IN_PX, apply, invert
from .degrade import _odd

CANVAS = 384  # cut-out size on the GPU; a view is at most ~1.4x this after turning
FILL = 114 / 255  # outside the frame, as corners.warp
BLUR_K = 17  # largest blur kernel: gaussian sigma 2.5 -> 17 taps, like cv2.GaussianBlur
SHARP_K = 19  # sharpening blur, sigma up to 3


# --- CPU side: cut-out and random settings ---

def canvas_box(m: np.ndarray, w: int, h: int, margin: int = 16) -> dict[str, np.ndarray]:
    """Which part of a w x h frame view `m` shows (plus `margin` px for blur) and how it goes on the canvas:
    at camera resolution, top-left, unless it's bigger than CANVAS (then shrunk by k).

    Returns box (x0, y0, x1, y1, filled width, filled height), theta (affine_grid's map from the
    output view to the canvas), the filled size, and where the canvas sits in the frame (vignetting).
    """
    seen = apply(invert(m), np.array([[0, 0], [IN_PX, 0], [IN_PX, IN_PX], [0, IN_PX]], float))
    x0, y0 = np.clip(np.floor(seen.min(0)) - margin, 0, [w - 1, h - 1]).astype(int)
    x1, y1 = np.clip(np.ceil(seen.max(0)) + margin, [x0 + 1, y0 + 1], [w, h]).astype(int)
    k = max(1.0, max(x1 - x0, y1 - y0) / CANVAS)
    cw, ch = (max(1, round((x1 - x0) / k)), max(1, round((y1 - y0) / k))) if k > 1 else (x1 - x0, y1 - y0)
    kx, ky = (x1 - x0) / cw, (y1 - y0) / ch
    # output pixel -> frame pixel -> canvas pixel (pixel centres, as cv2.resize with INTER_AREA)
    to_canvas = np.array([[1 / kx, 0, (0.5 - x0) / kx - 0.5], [0, 1 / ky, (0.5 - y0) / ky - 0.5], [0, 0, 1]])
    a = to_canvas @ np.vstack([invert(m), [0, 0, 1]])
    # affine_grid works in -1..1 (align_corners=False): out_n -> out px -> canvas px -> canvas_n
    from_out = np.array([[IN_PX / 2, 0, (IN_PX - 1) / 2], [0, IN_PX / 2, (IN_PX - 1) / 2], [0, 0, 1]])
    to_n = np.array([[2 / CANVAS, 0, 1 / CANVAS - 1], [0, 2 / CANVAS, 1 / CANVAS - 1], [0, 0, 1]])
    return {
        "box": np.array([x0, y0, x1, y1, cw, ch], np.int64),
        "theta": (to_n @ a @ from_out)[:2].astype(np.float32),
        "filled": np.array([cw, ch], np.float32),
        "frame": np.array([x0, y0, kx, ky, w, h], np.float32),
        "k": np.float32(max(kx, ky)),
    }


def cut_canvas(img_bgr: np.ndarray, m: np.ndarray, margin: int = 16) -> tuple[np.ndarray, dict]:
    """CPU cut-out: the canvas (RGB uint8, CANVAS x CANVAS, grey where empty) and canvas_box's info."""
    info = canvas_box(m, img_bgr.shape[1], img_bgr.shape[0], margin)
    x0, y0, x1, y1, cw, ch = info["box"]
    crop = img_bgr[y0:y1, x0:x1, ::-1]
    if (cw, ch) != (x1 - x0, y1 - y0):
        crop = cv2.resize(np.ascontiguousarray(crop), (int(cw), int(ch)), interpolation=cv2.INTER_AREA)
    canvas = np.full((CANVAS, CANVAS, 3), 114, np.uint8)
    canvas[:ch, :cw] = crop
    return canvas, info


def cut_canvas_gpu(images: list[torch.Tensor], box: torch.Tensor) -> torch.Tensor:
    """GPU cut-out from decoded RGB frames (3, H, W) uint8: (B, 3, CANVAS, CANVAS) float 0..1."""
    out = torch.full((len(images), 3, CANVAS, CANVAS), FILL, device=images[0].device)
    for i, (img, (x0, y0, x1, y1, cw, ch)) in enumerate(zip(images, box.tolist())):
        crop = img[None, :, y0:y1, x0:x1].float() / 255
        if (cw, ch) != (x1 - x0, y1 - y0):  # antialiased shrink, like INTER_AREA
            crop = F.interpolate(crop, size=(ch, cw), mode="bilinear", antialias=True, align_corners=False)
        out[i, :, :ch, :cw] = crop[0]
    return out


def decode_gpu(jpegs: list[torch.Tensor], device: str) -> list[torch.Tensor]:
    """JPEG files (uint8 byte tensors) -> RGB frames on the GPU, decoded there (nvJPEG)."""
    from torchvision.io import decode_jpeg

    return decode_jpeg(jpegs, device=device)


def _delta(size: int) -> np.ndarray:
    kernel = np.zeros((size, size), np.float32)
    kernel[size // 2, size // 2] = 1
    return kernel


def _gauss1d(sigma: float, size: int) -> np.ndarray:
    taps = min(_odd(sigma * 6 + 1), size)  # cv2.GaussianBlur's own kernel size for 8-bit images
    g = np.zeros(size, np.float32)
    start = (size - taps) // 2
    g[start:start + taps] = cv2.getGaussianKernel(taps, sigma).ravel()
    return g / g.sum()


def draw_effects(rng: np.random.Generator, k: float = 1.0, clean_chance: float = 0.1) -> dict[str, np.ndarray]:
    """Random settings for degrade.py's camera effects, scaled to a canvas shrunk by k.
    Every effect has a fixed-size entry (a do-nothing value when it's off), so a batch stacks."""
    fx = {
        "gains": np.ones(3, np.float32), "exposure": np.ones(2, np.float32), "lowres": np.zeros(2, np.float32),
        "blur": _delta(BLUR_K), "noise": np.zeros(3, np.float32), "sharpen": np.zeros(1, np.float32),
        "sharpen_kernel": _gauss1d(1.0, SHARP_K), "vignette": np.zeros(1, np.float32), "jpeg": np.zeros(1, np.float32),
    }
    if rng.random() < clean_chance:
        return fx
    if rng.random() < 0.5:  # white balance; degrade.py's gains are BGR, the canvas is RGB
        warm = rng.uniform(-0.25, 0.25)
        fx["gains"] = np.array([1 + warm, 1 + rng.uniform(-0.06, 0.06), 1 - warm], np.float32)
    if rng.random() < 0.5:
        fx["exposure"] = np.array([np.exp(rng.normal(0, 0.3)), rng.uniform(0.75, 1.35)], np.float32)
    if rng.random() < 0.35:
        factor, interp = rng.uniform(1.5, 4.0) / k, int(rng.integers(0, 3))
        if factor > 1.1:
            fx["lowres"] = np.array([factor, interp], np.float32)
    if rng.random() < 0.35:
        kind = ["gaussian", "defocus", "motion"][int(rng.integers(0, 3))]
        if kind == "gaussian":
            sigma = rng.uniform(0.6, 2.5) / k
            if sigma >= 0.3:
                g = _gauss1d(sigma, BLUR_K)
                fx["blur"] = np.outer(g, g)
        else:
            size_px = rng.uniform(3, 13) / k
            angle = rng.uniform(0, 180)
            if size_px >= 1.5:
                size = min(_odd(size_px), BLUR_K)
                kernel = np.zeros((size, size), np.float32)
                c = size // 2
                if kind == "defocus":
                    cv2.circle(kernel, (c, c), c, 1.0, -1)
                else:
                    dx, dy = np.cos(np.radians(angle)) * size / 2, np.sin(np.radians(angle)) * size / 2
                    cv2.line(kernel, (int(c - dx), int(c - dy)), (int(c + dx), int(c + dy)), 1.0, 1)
                pad = (BLUR_K - size) // 2
                fx["blur"] = np.pad(kernel / kernel.sum(), pad)
    if rng.random() < 0.55:
        shot, read = rng.uniform(0.002, 0.03), rng.uniform(0.002, 0.02)
        fx["noise"] = np.array([shot / k**2, read / k, float(rng.random() < 0.5)], np.float32)
    if rng.random() < 0.2:
        amount, sigma = rng.uniform(0.4, 1.5), rng.uniform(1, 3) / k
        if sigma >= 0.3:
            fx["sharpen"] = np.array([amount], np.float32)
            fx["sharpen_kernel"] = _gauss1d(sigma, SHARP_K)
    if rng.random() < 0.25:
        fx["vignette"] = np.array([rng.uniform(0.15, 0.55)], np.float32)
    if rng.random() < 0.7:
        fx["jpeg"] = np.array([rng.integers(25, 92)], np.float32)
    return fx


# --- GPU side ---

def _depthwise(x: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
    """Each picture filtered with its own kernel (B, kh, kw), edges mirrored like cv2.filter2D."""
    b, c, h, w = x.shape
    kh, kw = kernels.shape[-2:]
    x = F.pad(x.reshape(1, b * c, h, w), (kw // 2, kw // 2, kh // 2, kh // 2), mode="reflect")
    return F.conv2d(x, kernels.repeat_interleave(c, 0)[:, None], groups=b * c).view(b, c, h, w)


def _low_resolution(x: torch.Tensor, lowres: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    for i in torch.nonzero(lowres[:, 0] > 0).flatten().tolist():
        factor, interp = float(lowres[i, 0]), int(lowres[i, 1])
        # antialiased shrink ~ cv2's INTER_AREA ("area" mode needs whole-number factors on Apple GPUs)
        small = F.interpolate(x[i:i + 1], size=(max(8, int(CANVAS / factor)),) * 2, mode="bilinear", antialias=True, align_corners=False)
        mode = ["bilinear", "bicubic", "nearest"][interp]
        out[i:i + 1] = F.interpolate(small, size=(CANVAS, CANVAS), mode=mode, **({} if mode == "nearest" else {"align_corners": False}))
    return out


def _dct_matrix(device) -> torch.Tensor:
    n = torch.arange(8, dtype=torch.float32, device=device)
    d = torch.cos((2 * n[None, :] + 1) * n[:, None] * math.pi / 16) * math.sqrt(2 / 8)
    d[0] /= math.sqrt(2)
    return d


# libjpeg's standard tables (the JPEG spec, Annex K), scaled by quality like libjpeg does.
_LUMA = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
         18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]
_CHROMA = [17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99] + [99] * 32


def _quant(table: list[int], quality: torch.Tensor) -> torch.Tensor:
    q = quality.clamp(1, 100)
    scale = torch.where(q < 50, 5000 / q, 200 - 2 * q)
    t = torch.tensor(table, dtype=torch.float32, device=quality.device).view(1, 8, 8)
    return ((t * scale.view(-1, 1, 1) + 50) / 100).floor().clamp(1, 255)


def _blocks(fn, plane: torch.Tensor) -> torch.Tensor:
    b, h, w = plane.shape
    blk = plane.view(b, h // 8, 8, w // 8, 8).permute(0, 1, 3, 2, 4)
    return fn(blk).permute(0, 1, 3, 2, 4).reshape(b, h, w)


def jpeg(x: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
    """JPEG's lossy steps (YCbCr, 4:2:0 chroma, 8x8 DCT, quantising) for RGB 0..1, per-picture quality."""
    d = _dct_matrix(x.device)
    v = x * 255
    r, g, bl = v[:, 0], v[:, 1], v[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * bl
    cb = -0.168736 * r - 0.331264 * g + 0.5 * bl + 128
    cr = 0.5 * r - 0.418688 * g - 0.081312 * bl + 128
    cb, cr = (F.avg_pool2d(c[:, None], 2)[:, 0] for c in (cb, cr))

    def lossy(plane: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        qv = q.view(-1, 1, 1, 8, 8)
        return _blocks(lambda blk: d.T @ (torch.round((d @ (blk - 128) @ d.T) / qv) * qv) @ d + 128, plane)

    y = lossy(y, _quant(_LUMA, quality))
    qc = _quant(_CHROMA, quality)
    cb, cr = (F.interpolate(lossy(c, qc)[:, None], scale_factor=2, mode="bilinear", align_corners=False)[:, 0] for c in (cb, cr))
    r = y + 1.402 * (cr - 128)
    g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
    bl = y + 1.772 * (cb - 128)
    return (torch.stack([r, g, bl], 1).round() / 255).clamp(0, 1)


def apply_effects(x: torch.Tensor, fx: dict[str, torch.Tensor], frame: torch.Tensor) -> torch.Tensor:
    """degrade.py's camera pipeline on a batch of RGB 0..1 canvases (B, 3, CANVAS, CANVAS)."""
    x = (x * fx["gains"][:, :, None, None]).clamp(0, 1)
    gain, gamma = fx["exposure"][:, 0, None, None, None], fx["exposure"][:, 1, None, None, None]
    x = (x * gain).clamp(0, 1) ** gamma
    if bool((fx["lowres"][:, 0] > 0).any()):
        x = _low_resolution(x, fx["lowres"])
    x = _depthwise(x, fx["blur"]).clamp(0, 1)
    shot, read, colour = (fx["noise"][:, i, None, None, None] for i in range(3))
    noise = torch.randn_like(x) * torch.sqrt(x * shot + read**2) + torch.randn_like(x) * read * colour
    x = (x + noise).clamp(0, 1)
    amount = fx["sharpen"][:, :, None, None]
    g = fx["sharpen_kernel"]
    blurred = _depthwise(_depthwise(x, g[:, None, :]), g[:, :, None])
    x = (x * (1 + amount) - blurred * amount).clamp(0, 1)
    # vignette: darker towards the edges of the whole frame, wherever this canvas sits in it
    idx = torch.arange(CANVAS, device=x.device, dtype=torch.float32)
    x0, y0, kx, ky, w, h = (frame[:, i, None] for i in range(6))
    xx = (((x0 + (idx + 0.5) * kx - 0.5) - w / 2) / (w / 2)) ** 2
    yy = (((y0 + (idx + 0.5) * ky - 0.5) - h / 2) / (h / 2)) ** 2
    falloff = ((yy[:, :, None] + xx[:, None, :]) / 2).clamp(0, 1)
    x = x * (1 - fx["vignette"][:, :, None, None] * falloff[:, None])
    q = fx["jpeg"][:, 0]
    if bool((q > 0).any()):
        x = torch.where((q > 0)[:, None, None, None], jpeg(x, q.clamp_min(1)), x)
    return (x * 255).round() / 255


def cut_view(x: torch.Tensor, theta: torch.Tensor, filled: torch.Tensor) -> torch.Tensor:
    """The final IN_PX x IN_PX views; outside the filled part of each canvas is grey."""
    idx = torch.arange(CANVAS, device=x.device)
    inside = (idx[None, None, :] < filled[:, 0, None, None]) & (idx[None, :, None] < filled[:, 1, None, None])
    x = torch.where(inside[:, None], x, torch.full_like(x, FILL)) - FILL
    grid = F.affine_grid(theta, (len(x), 3, IN_PX, IN_PX), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False) + FILL


def render(x: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Canvases (B, 3, CANVAS, CANVAS) float 0..1 + settings -> network input (B, 3, IN_PX, IN_PX)."""
    fx = {k[3:]: v for k, v in batch.items() if k.startswith("fx_")}
    x = apply_effects(x, fx, batch["frame"])
    return cut_view(x, batch["theta"], batch["filled"])
