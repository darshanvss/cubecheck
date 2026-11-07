"""Read generator output and check that every label is self-consistent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .keypoints import FACE_QUADS, OUTSIDE, face_windings
from .nxn import cube

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
    n = s.get("cube", {}).get("size", 3)  # labels from before 1 Nov 2026 are all 3x3 and don't say
    c = cube(n)
    if lab["facelets"] != c.to_string(c.after(s["scramble"])):
        problems.append("facelets differ from the simulated scramble")

    # 2. A corner view shows exactly three faces, i.e. 3N² stickers facing the camera. An
    #    exactly-two-face view (generator --edge-on) shows 2N²: the third face is seen from behind.
    facing = [st for st in lab["stickers"] if st["facing"] > 0]
    faces_seen = {"URFDLB"[st["facelet"] // (n * n)] for st in facing}
    if not faces_seen <= set(lab["visibleFaces"]) or len(faces_seen) < 2:
        problems.append(f"facing stickers are on {sorted(faces_seen)}, labels say {lab['visibleFaces']}")
    if len(facing) != len(faces_seen) * n * n:
        problems.append(f"{len(facing)} stickers face the camera, expected {len(faces_seen) * n * n}")
    if lab["visibleFaces"][0] not in ("U", "D"):
        problems.append("first visible face is not the white/yellow one")

    # 3. Keypoint geometry: clockwise convex silhouette with C inside it. In a two-face view,
    #    the hidden face's quad must wind the other way instead (or be edge-on: about no area).
    kp = np.array(lab["keypoints"], dtype=np.float64)
    if (kp[:, 2] != OUTSIDE).all():
        hexagon = kp[:6, :2]
        hidden = [f not in faces_seen for f in lab["visibleFaces"]]
        if not any(hidden):
            if _signed_area(hexagon) <= 0:  # image y points down, so positive area = clockwise
                problems.append("silhouette P1..P6 is not clockwise")
            if cv2.pointPolygonTest(hexagon.astype(np.float32), (float(kp[6, 0]), float(kp[6, 1])), False) <= 0:
                problems.append("C is not inside the silhouette")
        else:
            wind = face_windings(kp[:, :2].tolist())
            front = max((w for w, h in zip(wind, hidden) if not h), key=abs)
            for w, h, f in zip(wind, hidden, lab["visibleFaces"]):
                if (h and w * front > 0 and abs(w) > 0.03 * abs(front)) or (not h and w * front <= 0):
                    problems.append(f"face {f}'s corners wind the wrong way for a two-face view")

        # 4. Every visible sticker must sit inside the quad of the face it belongs to.
        for face, quad in zip(lab["visibleFaces"], FACE_QUADS):
            poly = kp[list(quad), :2].astype(np.float32)
            for st in facing:
                if "URFDLB"[st["facelet"] // (n * n)] == face and st["inFrame"]:
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


def yolo_pose_line(s: Sample) -> str | None:
    """One line of an Ultralytics pose label: class, box, then x y visibility per keypoint (all 0..1)."""
    box = clipped_bbox(s)
    if box is None:
        return None
    w, h = s["width"], s["height"]
    x0, y0, x1, y1 = box
    parts = [0, (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h]
    for x, y, v in s["labels"]["keypoints"]:
        parts += [x / w, y / h, int(v)] if v != OUTSIDE else [0, 0, 0]
    return " ".join(str(p) if isinstance(p, int) else f"{p:.6f}" for p in parts)
