"""Draw labels on top of images so a person can check them at a glance."""

from __future__ import annotations

import cv2
import numpy as np

from .keypoints import KEYPOINT_NAMES, OCCLUDED, OUTSIDE, SKELETON

# BGR display colours for the sticker colour letters.
STICKER_BGR = {
    "U": (255, 255, 255), "D": (0, 213, 255), "F": (80, 192, 0),
    "B": (255, 96, 16), "R": (48, 16, 224), "L": (0, 122, 255),
}


def draw_labels(img: np.ndarray, labels: dict) -> np.ndarray:
    out = img.copy()
    scale = max(1.0, out.shape[1] / 960)
    kp = labels["keypoints"]
    for a, b in SKELETON:
        if kp[a][2] != OUTSIDE and kp[b][2] != OUTSIDE:
            cv2.line(out, (int(kp[a][0]), int(kp[a][1])), (int(kp[b][0]), int(kp[b][1])), (255, 255, 0), max(1, int(scale)), cv2.LINE_AA)
    for st in labels["stickers"]:
        if st["facing"] <= 0 or not st["inFrame"]:
            continue
        c = (int(st["x"]), int(st["y"]))
        r = int(6 * scale)
        cv2.circle(out, c, r, STICKER_BGR[st["color"]], -1, cv2.LINE_AA)
        if st["visible"]:
            cv2.circle(out, c, r, (0, 0, 0), max(1, int(scale)), cv2.LINE_AA)
        else:  # hidden behind a finger: magenta cross
            d = int(r * 1.3)
            cv2.line(out, (c[0] - d, c[1] - d), (c[0] + d, c[1] + d), (255, 0, 255), max(2, int(scale * 2)), cv2.LINE_AA)
            cv2.line(out, (c[0] - d, c[1] + d), (c[0] + d, c[1] - d), (255, 0, 255), max(2, int(scale * 2)), cv2.LINE_AA)
    for name, (x, y, v) in zip(KEYPOINT_NAMES, kp):
        if v == OUTSIDE:
            continue
        color = (255, 0, 255) if v == OCCLUDED else (255, 255, 0)
        cv2.circle(out, (int(x), int(y)), int(5 * scale), color, -1, cv2.LINE_AA)
        cv2.putText(out, name, (int(x + 7 * scale), int(y - 7 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, color, max(1, int(scale)), cv2.LINE_AA)
    return out


def mosaic(images: list[np.ndarray], cols: int = 4, tile: tuple[int, int] = (480, 320)) -> np.ndarray:
    tw, th = tile
    rows = (len(images) + cols - 1) // cols
    canvas = np.full((rows * th, cols * tw, 3), 24, np.uint8)
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        s = min(tw / w, th / h)
        small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        y, x = (i // cols) * th, (i % cols) * tw
        canvas[y : y + small.shape[0], x : x + small.shape[1]] = small
    return canvas
