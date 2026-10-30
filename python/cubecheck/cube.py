"""54-facelet 3x3 model: apply a scramble, get the expected sticker colours.

Facelets use Kociemba order U R F D L B, nine per face in reading order, each
letter naming the face whose colour the sticker has (U = white, D = yellow,
F = green, B = blue, R = red, L = orange in the WCA scheme). The dataset labels
use the same format, and tests check the two agree.

Built geometrically: a facelet is (cubie position, outward normal), and a face
turn rotates every facelet in that layer by -90 degrees about the face normal.
"""

from __future__ import annotations

import numpy as np

FACES = "URFDLB"
NORMALS = {
    "U": (0, 1, 0), "R": (1, 0, 0), "F": (0, 0, 1),
    "D": (0, -1, 0), "L": (-1, 0, 0), "B": (0, 0, -1),
}


def _face_positions(face: str) -> list[tuple[int, int, int]]:
    """Cubie positions (x right, y up, z front) of a face's 9 facelets in reading order."""
    pts = []
    for r in range(3):
        for c in range(3):
            a, b = r - 1, c - 1
            pts.append({
                "U": (b, 1, a),     # rows back->front, columns left->right
                "R": (1, -a, -b),   # rows top->bottom, columns front->back
                "F": (b, -a, 1),    # rows top->bottom, columns left->right
                "D": (b, -1, -a),   # rows front->back, columns left->right
                "L": (-1, -a, b),   # rows top->bottom, columns back->front
                "B": (-b, -a, -1),  # rows top->bottom, columns right->left
            }[face])
    return pts


FACELETS = [(p, NORMALS[f]) for f in FACES for p in _face_positions(f)]
INDEX = {fl: i for i, fl in enumerate(FACELETS)}


def _rotation(axis: tuple[int, int, int], quarter_turns: int) -> np.ndarray:
    ax = np.array(axis, dtype=float)
    th = -np.pi / 2 * quarter_turns
    k = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.rint(np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * k @ k).astype(int)


def _move_permutation(face: str, quarter_turns: int) -> np.ndarray:
    n = np.array(NORMALS[face])
    rot = _rotation(NORMALS[face], quarter_turns)
    perm = np.arange(54)  # new[dst] = old[src]
    for src, (p, nv) in enumerate(FACELETS):
        if np.dot(p, n) == 1:
            p2 = tuple(int(v) for v in rot @ np.array(p))
            n2 = tuple(int(v) for v in rot @ np.array(nv))
            perm[INDEX[(p2, n2)]] = src
    return perm


MOVES: dict[str, np.ndarray] = {}
for _f in FACES:
    MOVES[_f] = _move_permutation(_f, 1)
    MOVES[_f + "2"] = _move_permutation(_f, 2)
    MOVES[_f + "'"] = _move_permutation(_f, 3)
    MOVES[_f + "2'"] = MOVES[_f + "2"]

SOLVED = np.array([i // 9 for i in range(54)])


def apply(state: np.ndarray, alg: str | list[str]) -> np.ndarray:
    s = state.copy()
    for m in alg.split() if isinstance(alg, str) else alg:
        s = s[MOVES[m]]
    return s


def to_string(state: np.ndarray) -> str:
    return "".join(FACES[c] for c in state)


def facelets_after(scramble: str) -> str:
    """Expected facelet string for a scramble applied to a solved cube."""
    return to_string(apply(SOLVED, scramble))
