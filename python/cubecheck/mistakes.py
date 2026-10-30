"""Realistic scrambling mistakes, for testing the checker.

Testing a mis-scramble needs no new picture: keep the image (the cube as
it really is) and claim a slightly different scramble. The checker should then
see a mismatch, exactly as if the cube had been scrambled wrongly.
"""

from __future__ import annotations

import numpy as np

from .cube import FACES, SOLVED, apply

MODS = ["", "'", "2"]


def _invert(move: str) -> str:
    return move[0] + {"": "'", "'": "", "2": "2"}[move[1:]]


def corrupt(moves: list[str], rng: np.random.Generator) -> tuple[list[str], str]:
    """One random human mistake, guaranteed to change the cube state."""
    while True:
        kind = rng.choice(["skip", "direction", "half", "face", "extra", "swap", "stopped short"])
        if not moves and kind != "extra":  # a solved cube: the only slip is a move that shouldn't be there
            continue
        i = int(rng.integers(len(moves))) if moves else 0
        m = list(moves)
        if kind == "skip":
            del m[i]
        elif kind == "direction":
            m[i] = m[i][0] + str(rng.choice(["", "'"])) if m[i].endswith("2") else _invert(m[i])
        elif kind == "half":
            m[i] = m[i][0] + ("" if m[i].endswith("2") else "2")
        elif kind == "face":
            m[i] = str(rng.choice([f for f in FACES if f != m[i][0]])) + m[i][1:]
        elif kind == "extra":
            m.insert(i, str(rng.choice(list(FACES))) + str(rng.choice(MODS)))
        elif kind == "swap" and i + 1 < len(m):
            m[i], m[i + 1] = m[i + 1], m[i]
        elif kind == "stopped short":
            m = m[:-1]
        if not np.array_equal(apply(SOLVED, m), apply(SOLVED, moves)):
            return m, str(kind)
