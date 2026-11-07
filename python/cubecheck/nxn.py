"""Any N×N×N cube (2 to 7): apply a scramble, get the expected sticker colours.

The same model as cube.py, for every size. Facelets are in Kociemba face order
U R F D L B, N×N per face in reading order (as seen looking at that face, with
the same row and column directions as cube.py), each value naming the face
whose colour the sticker has. For N = 3 it gives exactly cube.py's numbers
(tests check this), and the training renders use the same order.

Built geometrically, like cube.py. Positions use doubled coordinates so even
cubes stay whole numbers: a 4×4's layers sit at -3, -1, 1, 3. A move turns
every facelet whose position along the face normal is within its layers.

Notation (WCA, as TNoodle writes it): R, R', R2 turn one layer; Rw turns two;
3Rw turns three (any depth below N); x, y, z turn the whole cube.
"""

from __future__ import annotations

import re
from functools import cached_property, lru_cache

import numpy as np

FACES = "URFDLB"
NORMALS = {
    "U": (0, 1, 0), "R": (1, 0, 0), "F": (0, 0, 1),
    "D": (0, -1, 0), "L": (-1, 0, 0), "B": (0, 0, -1),
}
ROTATIONS = {"x": "R", "y": "U", "z": "F"}  # whole-cube turns go the way of these faces
MOVE = re.compile(r"^(?:(\d+)?([URFDLB])(w)?|([xyz]))(2'?|')?$")

# What WCA scrambles (TNoodle) use for each size, and their usual length. Wide turns only
# come from the U, F and R sides. A 2x2 scramble never moves its D-B-L corner; bigger even
# cubes have no piece that stays put, so their orientation has to be found from the photo.
EVENT_FAMILIES = {
    2: ["R", "U", "F"],
    3: list(FACES),
    4: list(FACES) + ["Rw", "Uw", "Fw"],
    5: list(FACES) + [f + "w" for f in FACES],
    6: list(FACES) + [f + "w" for f in FACES] + ["3Rw", "3Uw", "3Fw"],
    7: list(FACES) + [f + "w" for f in FACES] + ["3" + f + "w" for f in FACES],
}
EVENT_LENGTH = {2: 11, 3: 25, 4: 45, 5: 60, 6: 80, 7: 100}


def _rotation(axis: tuple[int, int, int], quarter_turns: int) -> np.ndarray:
    ax = np.array(axis, dtype=float)
    th = -np.pi / 2 * quarter_turns
    k = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.rint(np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * k @ k).astype(int)


class Cube:
    """The N×N×N cube's facelets and moves. Use cube(n) to share one per size."""

    def __init__(self, n: int) -> None:
        if not 2 <= n <= 7:
            raise ValueError(f"cube size {n}: 2 to 7 supported")
        self.n = n
        self.size = 6 * n * n
        h = n - 1  # a face sits at +-h in doubled coordinates
        self.facelets: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
        for f in FACES:
            for r in range(n):
                for c in range(n):
                    a, b = 2 * r - h, 2 * c - h
                    pos = {
                        "U": (b, h, a),     # rows back->front, columns left->right
                        "R": (h, -a, -b),   # rows top->bottom, columns front->back
                        "F": (b, -a, h),    # rows top->bottom, columns left->right
                        "D": (b, -h, -a),   # rows front->back, columns left->right
                        "L": (-h, -a, b),   # rows top->bottom, columns back->front
                        "B": (-b, -a, -h),  # rows top->bottom, columns right->left
                    }[f]
                    self.facelets.append((pos, NORMALS[f]))
        self.index = {fl: i for i, fl in enumerate(self.facelets)}
        self._pos = np.array([p for p, _ in self.facelets])
        self._nrm = np.array([nv for _, nv in self.facelets])
        self.solved = np.repeat(np.arange(6), n * n)
        self._moves: dict[str, np.ndarray] = {}

    def _permutation(self, face: str, depth: int, quarter_turns: int) -> np.ndarray:
        """new[dst] = old[src] for turning the outer `depth` layers of `face` (depth n = whole cube)."""
        normal = np.array(NORMALS[face])
        rot = _rotation(NORMALS[face], quarter_turns)
        moving = self._pos @ normal >= (self.n - 1) - 2 * (depth - 1)
        perm = np.arange(self.size)
        for src in np.flatnonzero(moving):
            p2 = tuple(int(v) for v in rot @ self._pos[src])
            n2 = tuple(int(v) for v in rot @ self._nrm[src])
            perm[self.index[(p2, n2)]] = src
        return perm

    def move(self, name: str) -> np.ndarray:
        """The permutation for one move in WCA notation; ValueError if it isn't one for this size."""
        perm = self._moves.get(name)
        if perm is not None:
            return perm
        m = MOVE.match(name)
        if not m:
            raise ValueError(f"not a move: {name}")
        digits, face, wide, rotation, amount = m.groups()
        if rotation:
            face, depth = ROTATIONS[rotation], self.n
        else:
            if digits and not wide:
                raise ValueError(f"not a WCA move: {name} (slice moves aren't used in scrambles)")
            depth = int(digits) if digits else 2 if wide else 1
            if depth < 1 or depth > max(1, self.n - 1) or (wide and self.n < 3):
                raise ValueError(f"not a {self.n}x{self.n} move: {name}")
        turns = {None: 1, "'": 3, "2": 2, "2'": 2}[amount]
        perm = self._permutation(face, depth, turns)
        self._moves[name] = perm
        return perm

    def parse(self, text: str) -> list[str]:
        """Split a scramble into moves, checking each one; ValueError names the bad ones."""
        moves = text.replace("’", "'").split()
        bad = []
        for m in moves:
            try:
                self.move(m)
            except ValueError:
                bad.append(m)
        if bad:
            raise ValueError(f"not {self.n}x{self.n} moves: {' '.join(bad)}")
        return moves

    def apply(self, state: np.ndarray, alg: str | list[str]) -> np.ndarray:
        s = state.copy()
        for m in self.parse(alg) if isinstance(alg, str) else alg:
            s = s[self.move(m)]
        return s

    def after(self, scramble: str) -> np.ndarray:
        """Expected facelets after a scramble from solved."""
        return self.apply(self.solved, scramble)

    def to_string(self, state: np.ndarray) -> str:
        return "".join(FACES[c] for c in state)

    @cached_property
    def event_turns(self) -> list[str]:
        """Every move WCA scrambles of this size can contain (R, R', R2, Rw, ...)."""
        return [f + a for f in EVENT_FAMILIES[self.n] for a in ("", "'", "2")]

    def face_slice(self, face: str) -> slice:
        k = FACES.index(face) * self.n * self.n
        return slice(k, k + self.n * self.n)


@lru_cache(maxsize=None)
def cube(n: int) -> Cube:
    return Cube(n)


def random_scramble(n: int, rng: np.random.Generator, length: int | None = None) -> list[str]:
    """A scramble shaped like TNoodle's: the size's WCA moves,
    no face turned twice in a row, no three turns on one axis in a row. 4x4 opens with 15-21
    outer turns only, like a 3x3 scramble, then mixes in wide turns (about 40%); the other
    sizes are random moves from the start. One 4x4 scramble in four also uses wide turns from
    the start, as other scramblers might. A shorter length is the start of a full scramble."""
    axis = {"U": 0, "D": 0, "R": 1, "L": 1, "F": 2, "B": 2}
    families = EVENT_FAMILIES[n]
    outer_first = int(rng.integers(15, 22)) if n == 4 and rng.random() < 0.75 else 0

    def pick() -> str:
        if len(out) < outer_first:
            return FACES[int(rng.integers(6))]
        if n == 4 and outer_first:
            wide = len(out) == outer_first or rng.random() < 0.4
            return ("Rw", "Uw", "Fw")[int(rng.integers(3))] if wide else FACES[int(rng.integers(6))]
        return families[int(rng.integers(len(families)))]

    out: list[str] = []
    want = EVENT_LENGTH[n] if length is None else length
    while len(out) < want:
        fam = pick()
        face = fam.rstrip("w").lstrip("0123456789")
        if out:
            last = out[-1].rstrip("2'")
            if last == fam:
                continue
            prev = [m.rstrip("2'").rstrip("w").lstrip("0123456789") for m in out[-2:]]
            if len(prev) == 2 and axis[prev[0]] == axis[prev[1]] == axis[face]:
                continue
        out.append(fam + ("", "'", "2")[int(rng.integers(3))])
    return out
