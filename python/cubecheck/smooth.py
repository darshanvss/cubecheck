"""Steadier corners in live video.

Each frame's corners wobble by a pixel or two even when the cube is held still,
which makes the outline shimmer and the sticker reading flick between answers.
A One Euro filter (Casiez et al., 2012) smooths that away: when the corners
barely move it averages over many frames, and as they move faster it trusts
the newest frame more, so a moving cube doesn't trail behind.

Speeds are measured in cube edges per second, so the same settings work for a
cube close to the camera or far away.
"""

from __future__ import annotations

import math

import numpy as np

# The 3 ways to start the corner order: shift P1..P6 by two places (one face), C stays.
ROTATIONS = [[(i + 2 * r) % 6 for i in range(6)] + [6] for r in range(3)]


SPOKES = (1, 3, 5)  # the corners joined to C by an edge


def edge(kp: np.ndarray, spokes: tuple[int, ...] = SPOKES) -> float:
    """Rough cube edge length in pixels: mean distance from the centre corner C (the last one) to P2, P4, P6."""
    return float(np.linalg.norm(kp[list(spokes)] - kp[-1], axis=1).mean())


def align(kp: np.ndarray, ref: np.ndarray, rotations: list[list[int]] = ROTATIONS) -> np.ndarray:
    """kp with its corner order turned to match ref (a model may start at any of the 3 faces)."""
    return min((kp[r] for r in rotations), key=lambda k: float(np.linalg.norm(k - ref, axis=1).sum()))


def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
    tau = 1 / (2 * math.pi * cutoff)
    return 1 / (1 + tau / dt)


class CornerSmoother:
    """One Euro filter over the 7 corners of one live video.

    rotations and spokes are a cube's by default; shapes.py's puzzles pass their own
    (Shape.relabellings, Shape.spokes), with any number of corners.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 2.0, d_cutoff: float = 1.0, max_gap: float = 0.5,
                 rotations: list[list[int]] = ROTATIONS, spokes: tuple[int, ...] = SPOKES) -> None:
        self.min_cutoff, self.beta, self.d_cutoff, self.max_gap = min_cutoff, beta, d_cutoff, max_gap
        self.rotations, self.spokes = rotations, spokes
        self.kp: np.ndarray | None = None  # smoothed corners
        self.speed = np.zeros(len(rotations[0]))  # smoothed speed of each corner, edges per second
        self.t = 0.0

    def reset(self) -> None:
        self.kp = None

    def fresh(self, t: float) -> bool:
        """There's a recent position to start from."""
        return self.kp is not None and t - self.t <= self.max_gap

    def update(self, kp: np.ndarray, t: float) -> np.ndarray:
        kp = np.asarray(kp, float)
        if not self.fresh(t):
            self.kp, self.speed, self.t = kp, np.zeros(len(kp)), t
            return kp
        kp = align(kp, self.kp, self.rotations)
        dt = max(t - self.t, 1e-3)
        size = max(edge(self.kp, self.spokes), 1.0)
        if np.linalg.norm(kp - self.kp, axis=1).max() > 0.5 * size:  # jumped: a different find, start over
            self.kp, self.speed, self.t = kp, np.zeros(len(kp)), t
            return kp
        speed = np.linalg.norm(kp - self.kp, axis=1) / size / dt
        self.speed += _alpha(self.d_cutoff, dt) * (speed - self.speed)
        a = _alpha(self.min_cutoff + self.beta * self.speed, dt)[:, None]
        self.kp = self.kp + a * (kp - self.kp)
        self.t = t
        return self.kp
