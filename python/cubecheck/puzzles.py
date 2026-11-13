"""Skewb, pyraminx and FTO as lists of stickers (experimental).

A fixed-shape puzzle is a list of sticker places, each with a centre and the face it's on,
and a move is a reshuffle of that list, as in nxn.py. puzzle_data/<event>.json holds each
puzzle's shape: its faces (corners), its stickers (face, piece, centre, outline), its
colours and, for every turn, the axis, the slab of the puzzle it turns and the angle. The
reshuffles are worked out here: a piece turns when its centre is inside the slab, and its
stickers land on other stickers' places.

The sticker places are numbered as in puzzle_data, and everything else (the readers'
outputs, the check, the flat pictures, the website) uses those numbers.
"""

from __future__ import annotations

import json
from functools import cache, cached_property
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "puzzle_data"
EVENTS = ("skewb", "pyram", "fto")
NAMES = {"skewb": "Skewb", "pyram": "Pyraminx", "fto": "FTO"}


def rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Right-handed rotation matrix about an axis."""
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.cos(angle) * np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * np.outer(k, k)


def _closure(gens: list[np.ndarray]) -> list[np.ndarray]:
    """Every reshuffle the generators make together (for whole-puzzle turns: a small group)."""
    ident = np.arange(len(gens[0]))
    seen, todo = {ident.tobytes(): ident}, [ident]
    while todo:
        p = todo.pop()
        for g in gens:
            q = p[g]
            if q.tobytes() not in seen:
                seen[q.tobytes()] = q
                todo.append(q)
    return list(seen.values())


class Fixed:
    """A puzzle whose shape never changes. States are arrays of colours, one per sticker place."""

    def __init__(self, event: str, data: dict):
        self.event, self.name = event, NAMES[event]
        faces = {f["key"]: np.array(f["vertices"], float) for f in data["faces"]}
        self.face_keys = list(faces)
        self.normal_of = {}
        for key, v in faces.items():
            n = np.cross(v[1] - v[0], v[2] - v[0])
            self.normal_of[key] = n / np.linalg.norm(n) * np.sign(n @ v.mean(axis=0))
        self.face_vertices = faces
        st = data["stickers"]
        self.centroids = np.array([s["centroid"] for s in st])
        self.sticker_vertices = [np.array(s["vertices"], float) for s in st]
        self.colours: dict[str, int] = data.get("colors", {})
        self.normals = np.array([self.normal_of[s["face"]] for s in st])
        self.piece = np.array([s["piece"] for s in st])
        self.solved = np.array([self.face_keys.index(s["face"]) for s in st], dtype=np.int8)
        self.size = len(st)
        self.piece_centres = np.array([self.centroids[self.piece == p].mean(axis=0) for p in range(self.piece.max() + 1)])
        self.defs = data["moves"]
        self.turns: list[str] = data["turns"]
        self._perm: dict[str, np.ndarray] = {}

    def _place(self, points: np.ndarray) -> np.ndarray:
        """Index of the sticker place at each point (they must all be places)."""
        d = np.linalg.norm(points[:, None] - self.centroids[None], axis=2)
        idx = d.argmin(axis=1)
        assert d[np.arange(len(points)), idx].max() < 1e-4, f"{self.name}: a turn moved a sticker off the grid"
        return idx

    def _reshuffle(self, matrices_by_sticker: np.ndarray) -> np.ndarray:
        """Places from per-sticker rotations: new_state = state[p] (p[j] = sticker that lands on j)."""
        moved = np.einsum("sij,sj->si", matrices_by_sticker, self.centroids)
        dest = self._place(moved)
        p = np.empty(self.size, dtype=np.int64)
        p[dest] = np.arange(self.size)
        assert len(set(dest)) == self.size
        return p

    def move(self, token: str) -> np.ndarray:
        if token not in self._perm:
            mats = np.repeat(np.eye(3)[None], self.size, axis=0)
            taken = np.zeros(len(self.piece_centres), bool)
            for part in self.defs[token]:
                proj = self.piece_centres @ np.array(part["axis"])
                lo = -np.inf if part["min"] is None else part["min"]
                hi = np.inf if part["max"] is None else part["max"]
                inside = ~taken & (proj > lo + 1e-6) & (proj < hi - 1e-6)
                taken |= inside
                mats[inside[self.piece]] = rotation(part["axis"], part["angle"])
            self._perm[token] = self._reshuffle(mats)
        return self._perm[token]

    def apply(self, state: np.ndarray, moves: list[str]) -> np.ndarray:
        for m in moves:
            state = state[self.move(m)]
        return state

    @cached_property
    def vertices(self) -> np.ndarray:
        """The shape's corners, one each."""
        pts = np.concatenate(list(self.face_vertices.values()))
        return np.unique(np.round(pts, 6), axis=0)

    @cached_property
    def orientations(self) -> list[np.ndarray]:
        """Every way the whole puzzle can be turned onto itself, as reshuffles (24 for a skewb or an FTO, 12 for a pyraminx)."""
        gens = []
        for key, v in self.face_vertices.items():
            gens.append(self._reshuffle(np.repeat(rotation(self.normal_of[key], 2 * np.pi / len(v))[None], self.size, 0)))
        for c in self.vertices:
            meeting = sum(np.isclose(np.linalg.norm(v - c, axis=1), 0, atol=1e-5).any() for v in self.face_vertices.values())
            gens.append(self._reshuffle(np.repeat(rotation(c, 2 * np.pi / meeting)[None], self.size, 0)))
        return _closure(gens)


@cache
def puzzle(event: str) -> Fixed:
    return Fixed(event, json.loads((DATA / f"{event}.json").read_text()))
