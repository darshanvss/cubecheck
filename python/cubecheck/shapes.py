"""Skewb, pyraminx and FTO for the camera check: corners, faces and sticker places (experimental).

The cube pipeline works from 7 corners: C, the corner nearest the camera, and the six
around the outline; each visible face is the quad of 4 of them, straightened to read its
N×N stickers (keypoints.py, reader.py, nxn_check.py). This is the same idea for any shape
whose visible faces all meet at the corner nearest the camera:

- skewb: a cube, so the same 7 corners and 3 quads; each face holds 5 stickers.
- pyraminx: 4 corners (its tips): C and the three around it; 3 triangles of 9 stickers.
- FTO: 5 corners: C and the four around it (the sixth is behind C); 4 triangles of 9.

Corners are ordered [ring..., C], the ring clockwise in the picture, as for the cube. A
shape looks the same from several sides, so a picture has several correct labellings:
the ring can start at any face (3 for skewb and pyraminx, 4 for FTO), and when a pyraminx
shows only two faces, C can be either end of the edge they share. `views` lists every
labelling as indices into the shape's own corners; training and scoring take the best
fitting one, and the check tries them all, as it tries all 24 for a cube.

Within a face, stickers are placed by (u, v): the face's corner C is (0, 0), the next
corner anticlockwise in the picture (1, 0), the previous one (0, 1). The `slots` of a face
are its stickers in a fixed order (nearest C first), each a polygon in (u, v).

The shapes themselves come from puzzles.py. See docs/other-puzzles.md.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from .puzzles import EVENTS, NAMES, Fixed, puzzle

VISIBLE, OCCLUDED, OUTSIDE = 2, 1, 0
# A face counts as in view from this cosine up (about 85° from facing the camera).
OPEN = 0.08


class Shape:
    def __init__(self, puzzle: Fixed) -> None:
        self.puzzle, self.event = puzzle, puzzle.event
        # Corners in the order the training renders label them: first appearance going through the faces.
        verts: list[np.ndarray] = []
        self.face_corners: list[list[int]] = []
        for v in puzzle.face_vertices.values():
            ids = []
            for p in v:
                hit = [i for i, q in enumerate(verts) if np.allclose(p, q, atol=1e-6)]
                if not hit:
                    verts.append(p)
                ids.append(hit[0] if hit else len(verts) - 1)
            self.face_corners.append(ids)
        self.vertices = np.array(verts)
        self.normals = np.array(list(puzzle.normal_of.values()))
        self.quads = len(self.face_corners[0]) == 4
        self.faces_at = [[f for f, ids in enumerate(self.face_corners) if c in ids] for c in range(len(verts))]
        self.per_face = puzzle.size // len(self.face_corners)  # stickers on a face
        self.n_faces = len(self.faces_at[0])  # faces in view around C
        self.colours = len(self.face_corners)

        # Every labelling: for each corner as C and each face to start at, the ring and the faces.
        self.views: list[list[int]] = []  # [ring..., C] as indices into self.vertices
        self.view_faces: list[list[int]] = []  # the model face behind each face of the picture
        frames = []  # per view and face: (C, next anticlockwise, previous) as corner indices
        for c in range(len(verts)):
            around = self._clockwise(c)
            for start in range(len(around)):
                faces = around[start:] + around[:start]
                ring, frame = [], []
                for k, f in enumerate(faces):
                    nxt = faces[(k + 1) % len(faces)]
                    shared = [v for v in self.face_corners[f] if v in self.face_corners[nxt] and v != c]
                    assert len(shared) == 1
                    if self.quads:  # the face's own far corner, then the one it shares with the next face
                        ring += [next(v for v in self.face_corners[f] if v != c and all(v not in self.face_corners[g] for g in faces if g != f)), shared[0]]
                    else:
                        ring.append(shared[0])
                if not self.quads:  # ring[k] is shared by faces k and k+1: face k spans ring[k-1], ring[k]
                    ring = ring[-1:] + ring[:-1]
                self.views.append(ring + [c])
                self.view_faces.append(faces)
        self.keypoints = len(self.views[0])
        m = self.keypoints - 1
        # Faces of the picture as corner indices, C first, then anticlockwise (keypoints.FACE_QUADS for a cube).
        if self.quads:
            self.polys = [(m, (2 * k + 1) % m, 2 * k, (2 * k - 1) % m) for k in range(m // 2)]
        else:
            self.polys = [(m, (k + 1) % m, k) for k in range(m)]
        # A mirrored picture: the ring runs the other way (P1 and C stay), as keypoints.FLIP_IDX.
        self.flip = [0] + list(range(m - 1, 0, -1)) + [m]
        # The corners joined to C by an edge, and every other order the same corners can come in
        # (smooth.CornerSmoother lines one frame's answer up with the last frame's).
        self.spokes = tuple(poly[1] for poly in self.polys)
        first = self.views[0]
        self.relabellings = [[first.index(c) for c in v] for v in self.views if set(v) == set(first)]

        # Sticker slots of a face, and the sticker place behind each slot in each view.
        self.slots = self._slots()
        self.cells = np.array([self._cells(v) for v in range(len(self.views))])
        self.slot_centres = np.array([s.mean(axis=0) for s in self.slots])

    def _clockwise(self, c: int) -> list[int]:
        """The faces around corner c, clockwise as seen from outside."""
        axis = self.vertices[c] / np.linalg.norm(self.vertices[c])
        e1 = np.cross(axis, [0.3, 0.5, 0.81])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(axis, e1)  # (e1, e2, axis) is right-handed: angles grow anticlockwise from outside
        centre = lambda f: self.vertices[self.face_corners[f]].mean(axis=0)  # noqa: E731
        return sorted(self.faces_at[c], key=lambda f: -np.arctan2(centre(f) @ e2, centre(f) @ e1))

    def _uv(self, points: np.ndarray, c: int, nxt: int, prev: int) -> np.ndarray:
        """(u, v) of points on a face whose corner C is c, with `nxt` at (1, 0) and `prev` at (0, 1)."""
        o, a, b = self.vertices[c], self.vertices[nxt] - self.vertices[c], self.vertices[prev] - self.vertices[c]
        return np.linalg.lstsq(np.stack([a, b], axis=1), (points - o).T, rcond=None)[0].T

    def _frame(self, view: int, k: int) -> tuple[int, int, int]:
        kp, poly = self.views[view], self.polys[k]
        return kp[poly[0]], kp[poly[1]], kp[poly[-1]]

    def _slots(self) -> list[np.ndarray]:
        """The stickers of one face as (u, v) polygons, nearest C first, then from the (1, 0) side to the (0, 1) side."""
        f = self.view_faces[0][0]
        on = np.flatnonzero(self.puzzle.solved == f)
        polys = [self._uv(self.puzzle.sticker_vertices[i], *self._frame(0, 0)) for i in on]
        centres = np.array([p.mean(axis=0) for p in polys])
        order = sorted(range(len(polys)), key=lambda i: (round(centres[i].sum(), 3), round(centres[i][1] - centres[i][0], 3)))
        return [polys[i] for i in order]

    def _cells(self, view: int) -> list[int]:
        """The sticker place behind each (face, slot) of this view."""
        centres = np.array([s.mean(axis=0) for s in self.slots])
        out = []
        for k, f in enumerate(self.view_faces[view]):
            on = np.flatnonzero(self.puzzle.solved == f)
            uv = self._uv(self.puzzle.centroids[on], *self._frame(view, k))
            d = np.linalg.norm(uv[None] - centres[:, None], axis=2)  # (slots, stickers)
            assert d.min(axis=1).max() < 1e-6, f"{self.event}: a face's stickers don't fit the slots"
            out += on[d.argmin(axis=1)].tolist()
        return out

    # --- from a label ---

    def labellings(self, labels: dict) -> list[int]:
        """Every view (index into self.views) that correctly labels this render.

        C is a corner with the most faces in view around it: the one where all the visible
        faces meet, or for a pyraminx showing two faces, either end of the edge they share.
        """
        facing = np.array(labels["facing"])
        count = [int((facing[fs] > OPEN).sum()) for fs in self.faces_at]
        best = max(count)
        cs = [c for c, n in enumerate(count) if n == best]
        return [v for v, kp in enumerate(self.views) if kp[-1] in cs]

    def keypoints_of(self, labels: dict, view: int) -> np.ndarray:
        """(K, 3) x, y, visibility of this labelling's corners, as the cube labels' keypoints."""
        out = []
        for i in self.views[view]:
            v = labels["vertices"][i]
            out.append([v["x"], v["y"], OCCLUDED if v["occluded"] else VISIBLE] if v["inFrame"] else [0, 0, OUTSIDE])
        return np.array(out, np.float32)

    # --- in the picture ---

    def pose(self, keypoints: np.ndarray, img_shape: tuple[int, ...]) -> np.ndarray:
        return self.fit(keypoints, img_shape)[0]

    def fit(self, keypoints: np.ndarray, img_shape: tuple[int, ...]) -> tuple[np.ndarray, float]:
        """3 x 4 camera matrix that puts the shape's own corners on these keypoints, and how
        badly they fit: the mean distance left between them, as a share of the puzzle's size.
        Corners a network placed well fit to about half a percent; one put on the wrong spot
        can't be matched by any rigid pose, and the error says so (poly_check.MAX_FIT).

        A triangle's 3 corners don't say how it's foreshortened (they fix an affine map, and
        stickers placed with one are off by up to a fifth of the edge on close-ups), so the
        whole shape is fitted instead: a pinhole camera looking through the middle of the
        picture, its zoom unknown. Four zooms are tried (about 33° to 64° of view top to
        bottom) and the one that fits the corners best is kept.
        """
        h, w = img_shape[:2]
        obj = self.vertices[self.views[0]].astype(np.float64)
        img = keypoints[:, :2].astype(np.float64)
        best, best_err = None, np.inf
        for zoom in (0.8, 1.0, 1.3, 1.7):
            cam = np.array([[zoom * h, 0, w / 2], [0, zoom * h, h / 2], [0, 0, 1]], np.float64)
            ok, rvec, tvec = cv2.solvePnP(obj, img, cam, None, flags=cv2.SOLVEPNP_SQPNP)
            if not ok:
                continue
            rvec, tvec = cv2.solvePnPRefineLM(obj, img, cam, None, rvec, tvec)
            err = float(np.linalg.norm(cv2.projectPoints(obj, rvec, tvec, cam, None)[0][:, 0] - img, axis=1).mean())
            if tvec[2, 0] > 0 and err < best_err:
                best, best_err = cam @ np.hstack([cv2.Rodrigues(rvec)[0], tvec]), err
        if best is None:  # no fit (corners in a line, say): fall back to no perspective at all
            a = cv2.estimateAffine2D(obj[:, :2].astype(np.float32), img.astype(np.float32))[0]
            best = np.array([[a[0, 0], a[0, 1], 0, a[0, 2]], [a[1, 0], a[1, 1], 0, a[1, 2]], [0, 0, 0, 1]])
        return best, best_err / max(self.size(keypoints), 1e-6)

    def face_matrix(self, keypoints: np.ndarray, k: int, pose: np.ndarray | None = None) -> np.ndarray:
        """3 x 3 map from a face's (u, v) to the picture: from the face's own 4 corners for a
        quad, from the whole shape's pose (self.pose) for a triangle."""
        if self.quads:
            pts = keypoints[list(self.polys[k]), :2].astype(np.float32)
            return cv2.getPerspectiveTransform(np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), pts)
        c, nxt, prev = (self.vertices[i] for i in self._frame(0, k))
        return pose @ np.column_stack([np.append(nxt - c, 0), np.append(prev - c, 0), np.append(c, 1)])

    def face_matrices(self, keypoints: np.ndarray, img_shape: tuple[int, ...]) -> list[np.ndarray]:
        pose = None if self.quads else self.pose(keypoints, img_shape)
        return [self.face_matrix(keypoints, k, pose) for k in range(self.n_faces)]

    def cell_centres(self, keypoints: np.ndarray, img_shape: tuple[int, ...]) -> np.ndarray:
        """Where every (face, slot) sticker's middle falls in the picture, (faces x slots, 2)."""
        centres = self.slot_centres[None].astype(np.float64)
        return np.concatenate([cv2.perspectiveTransform(centres, m)[0] for m in self.face_matrices(keypoints, img_shape)])

    def windings(self, keypoints: np.ndarray) -> np.ndarray:
        """Signed area of each face in the picture: negative for a face seen from the front (its corners
        run C, then anticlockwise), positive for one seen from behind. Unlike a cube's hidden face, a
        pyraminx's can be the largest of the three, so the sign itself decides, not the largest face."""
        out = []
        for poly in self.polys:
            p = keypoints[list(poly), :2]
            out.append(0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])))
        return np.array(out)

    def size(self, keypoints: np.ndarray) -> float:
        """The puzzle's size in the picture, in pixels: the mean length of the edges out of C."""
        m = self.keypoints - 1
        spokes = [poly[1] for poly in self.polys]
        return float(np.mean([np.linalg.norm(keypoints[i, :2] - keypoints[m, :2]) for i in spokes]))

    def well_formed(self, keypoints: np.ndarray, min_faces: int) -> bool:
        """At least `min_faces` faces seen from the front, each a proper convex shape (as corners.cube_shaped)."""
        front = self.windings(keypoints) < 0
        ok = 0
        for poly, f in zip(self.polys, front):
            p = keypoints[list(poly), :2]
            e = np.roll(p, -1, axis=0) - p
            turn = e[:, 0] * np.roll(e[:, 1], -1) - e[:, 1] * np.roll(e[:, 0], -1)
            ok += bool(f and (np.all(turn > 0) or np.all(turn < 0)))
        return ok >= min_faces


# A full scramble: the faces turned and how many turns (WCA random-move scrambles; FTO as the training renders').
SCRAMBLE = {"skewb": ("R U L B", 11), "pyram": ("U L R B", 11), "fto": ("U F L R D B BL BR", 30)}


def random_scramble(event: str, rng: np.random.Generator, length: int | None = None) -> list[str]:
    """A random-move scramble in WCA notation: never the same face twice in a row, each turn a
    third of a turn either way. A full pyraminx scramble ends with its tips (u l r b), each turned or not."""
    faces, full = SCRAMBLE[event][0].split(), SCRAMBLE[event][1]
    moves: list[str] = []
    while len(moves) < (length or full):
        face = str(rng.choice(faces))
        if moves and moves[-1].rstrip("'") == face:
            continue
        moves.append(face + str(rng.choice(["", "'"])))
    if event == "pyram" and not length:
        for tip in "ulrb":
            turn = int(rng.integers(3))
            if turn:
                moves.append(tip + ("'" if turn == 2 else ""))
    return moves


def parse_scramble(event: str, text: str) -> list[str]:
    """The scramble's moves; ValueError when one isn't a turn of this puzzle."""
    moves = text.replace("’", "'").replace("′", "'").split()
    turns = set(shape(event).puzzle.turns)
    for m in moves:
        if m not in turns:
            raise ValueError(f"{m} isn't a {NAMES[event]} move ({' '.join(t for t in sorted(turns) if not t.endswith(chr(39)))}, each also with ')")
    return moves


# How many faces a view must show to be judged: a skewb and an FTO show all the faces
# around C; a pyraminx is usually held edge-on, showing two. (poly_check.THRESHOLDS'
# min_faces; fewer gets "re-check" with HOLD_HINT.)
MIN_FACES = {"skewb": 3, "pyram": 2, "fto": 4}
# How few faces the corner model's answer may show and still count as the puzzle found.
# From c2 on the pyraminx and FTO models know every hold (face-on shows one face), so a
# poor hold is found and answered with a hint instead of nothing at all.
FOUND_FACES = {"skewb": 3, "pyram": 1, "fto": 1}
# The current models of each puzzle: runs/corners/<first>, runs/reader/<second>/best.pt.
# c2/r2 (12 Nov 2026) were trained on the puzzle held any way round; FOUND_FACES assumes them.
MODELS = {"skewb": ("skewb-c1", "skewb-r1"), "pyram": ("pyram-c2", "pyram-r2"), "fto": ("fto-c2", "fto-r2")}
HOLD_HINT = {
    "skewb": "turn a corner toward the camera, so three faces show",
    "pyram": "turn an edge toward the camera, so two faces show",
    "fto": "turn a corner toward the camera, so four faces show",
}


@lru_cache(maxsize=None)
def shape(event: str) -> Shape:
    return Shape(puzzle(event))
