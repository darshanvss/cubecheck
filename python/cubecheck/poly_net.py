"""Flat layouts (nets) of the skewb, pyraminx and FTO, for showing a whole state at once.

Each sticker place of shapes.py's puzzles gets an outline on a flat picture, so a state
(the colour on every place) can be drawn the way the website draws a cube's net: what the
scramble should give, next to what the camera read.

- skewb: a cube's cross, as the cube nets: U on top, then L F R B in a row, D below.
- pyraminx: F in the middle with the other three faces folded out from its edges, which
  makes one big triangle (the usual pyraminx picture).
- FTO: two squares, each cut by its diagonals into four faces. The left one is the puzzle
  seen straight at the corner where U, L, R and F meet (the view the check wants); the
  right one is the opposite corner, as if the puzzle were turned round on the table.
  This is a view, not a paper net: the faces look like right-angled triangles.

The first two are true unfoldings: a face is swung flat about the edge it shares with a
face already laid down. Everything comes from the puzzle's own geometry (puzzles.py), so
the place numbers are the same as everywhere else. See docs/other-puzzles.md.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .shapes import shape

# Which face each face is folded out from; the root lies flat first. For the skewb the
# tree is chosen to give the cube nets' cross.
UNFOLD = {
    "skewb": ("F", {"U": "F", "D": "F", "L": "F", "R": "F", "B": "R"}),
    "pyram": ("F", {"D": "F", "BL": "F", "BR": "F"}),
}
UP = np.array([0.0, 1.0, 0.0])


def _frame(normal: np.ndarray, up: np.ndarray = UP) -> tuple[np.ndarray, np.ndarray]:
    """Right and up directions of a face seen from outside (right x up = normal)."""
    if abs(normal @ up) > 0.99:  # the face is the top or the bottom: any "up" along it will do
        up = np.array([0.0, 0.0, -1.0])
    u = up - (up @ normal) * normal
    u /= np.linalg.norm(u)
    return np.cross(u, normal), u


def _unfolded(event: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """For each face, a 2 x 3 matrix and an offset taking a point on it to the flat picture (y up)."""
    sh = shape(event)
    keys = sh.puzzle.face_keys
    root, parents = UNFOLD[event]
    r, u = _frame(sh.normals[keys.index(root)])
    maps = {keys.index(root): (np.stack([r, u]), np.zeros(2))}
    todo = list(parents)
    while todo:
        key = next(k for k in todo if keys.index(parents[k]) in maps)
        todo.remove(key)
        f, parent = keys.index(key), keys.index(parents[key])
        a, b = (sh.vertices[v] for v in sh.face_corners[f] if v in sh.face_corners[parent])  # the hinge
        m = np.stack(_frame(sh.normals[f]))
        pm, pt = maps[parent]
        # Turn and shift the face's own flat picture so the hinge lands where the parent has it.
        turn = np.arctan2(*(pm @ (b - a))[::-1]) - np.arctan2(*(m @ (b - a))[::-1])
        rot = np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]])
        maps[f] = (rot @ m, pm @ a + pt - rot @ m @ a)
    return maps


def _two_views(event: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """An FTO seen straight at the corner where U, L, R and F meet, and at the opposite corner."""
    sh = shape(event)
    keys = sh.puzzle.face_keys
    front = next(c for c, fs in enumerate(sh.faces_at) if {keys[f] for f in fs} == {"U", "L", "R", "F"})
    axis = sh.vertices[front] / np.linalg.norm(sh.vertices[front])
    width = 2.15 * np.linalg.norm(sh.vertices[front])  # the square's diagonal, plus a gap
    maps = {}
    for side, a in ((0, axis), (1, -axis)):
        m = np.stack(_frame(a))
        for f, n in enumerate(sh.normals):
            if n @ a > 0:
                maps[f] = (m, np.array([side * width, 0.0]))
    return maps


@lru_cache(maxsize=None)
def net(event: str) -> dict:
    """The flat layout: {"polys": outline of every sticker place, "faces": outline of every face,
    "labels": [face key, x, y], "width", "height"}. x runs right and y down, from 0."""
    sh = shape(event)
    p = sh.puzzle
    maps = _two_views(event) if event == "fto" else _unfolded(event)
    flat = lambda f, pts: pts @ maps[f][0].T + maps[f][1]  # noqa: E731
    polys = [flat(int(p.solved[i]), p.sticker_vertices[i]) for i in range(p.size)]
    faces = [flat(f, sh.vertices[sh.face_corners[f]]) for f in range(len(p.face_keys))]
    lo = np.min([q.min(axis=0) for q in faces], axis=0)
    hi = np.max([q.max(axis=0) for q in faces], axis=0)
    to_page = lambda q: np.column_stack([q[:, 0] - lo[0], hi[1] - q[:, 1]])  # noqa: E731  (y down)
    return {
        "polys": [np.round(to_page(q), 3).tolist() for q in polys],
        "faces": [np.round(to_page(q), 3).tolist() for q in faces],
        "labels": [[k, *np.round(to_page(faces[f].mean(axis=0)[None])[0], 3).tolist()] for f, k in enumerate(p.face_keys)],
        "width": round(float(hi[0] - lo[0]), 3),
        "height": round(float(hi[1] - lo[1]), 3),
    }
