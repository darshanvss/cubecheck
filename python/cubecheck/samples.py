"""Read a dataset folder (images/ and labels/) and check that every label is self-consistent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .cube import facelets_after
from .keypoints import FACE_QUADS, OUTSIDE

Sample = dict[str, Any]


def load_samples(raw_dir: Path) -> list[Sample]:
    return [json.loads(p.read_text()) for p in sorted((raw_dir / "labels").glob("*.json"))]


def _signed_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)) / 2


def check_sample(s: Sample) -> list[str]:
    """Return a list of problems (empty when the sample is consistent)."""
    problems = []
    lab = s["labels"]

    # 1. The rendered cube state must match the scramble, computed independently.
    expected = facelets_after(s["scramble"])
    if lab["facelets"] != expected:
        problems.append("facelets differ from the simulated scramble")

    # 2. A corner view shows exactly three faces, i.e. 27 stickers facing the camera.
    facing = [st for st in lab["stickers"] if st["facing"] > 0]
    if len(facing) != 27:
        problems.append(f"{len(facing)} stickers face the camera, expected 27")
    faces_seen = {"URFDLB"[st["facelet"] // 9] for st in facing}
    if faces_seen != set(lab["visibleFaces"]):
        problems.append(f"facing stickers are on {sorted(faces_seen)}, labels say {lab['visibleFaces']}")
    if lab["visibleFaces"][0] not in ("U", "D"):
        problems.append("first visible face is not the white/yellow one")

    # 3. Keypoint geometry: clockwise convex silhouette with C inside it.
    kp = np.array(lab["keypoints"], dtype=np.float64)
    if (kp[:, 2] != OUTSIDE).all():
        hexagon = kp[:6, :2]
        if _signed_area(hexagon) <= 0:  # image y points down, so positive area = clockwise
            problems.append("silhouette P1..P6 is not clockwise")
        if cv2.pointPolygonTest(hexagon.astype(np.float32), (float(kp[6, 0]), float(kp[6, 1])), False) <= 0:
            problems.append("C is not inside the silhouette")

        # 4. Every visible sticker must sit inside the quad of the face it belongs to.
        for face, quad in zip(lab["visibleFaces"], FACE_QUADS):
            poly = kp[list(quad), :2].astype(np.float32)
            for st in facing:
                if "URFDLB"[st["facelet"] // 9] == face and st["inFrame"]:
                    if cv2.pointPolygonTest(poly, (st["x"], st["y"]), False) <= 0:
                        problems.append(f"sticker {st['facelet']} lies outside face {face}'s quad")
                        break
    return problems


def clipped_bbox(s: Sample) -> tuple[float, float, float, float] | None:
    x0, y0, x1, y1 = s["labels"]["bbox"]
    w, h = s["width"], s["height"]
    x0, x1 = max(0.0, x0), min(float(w), x1)
    y0, y1 = max(0.0, y0), min(float(h), y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return x0, y0, x1, y1
