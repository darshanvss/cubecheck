"""The check for any N×N cube: 7 corners, N×N stickers per face, scramble -> verdict.

The same steps as verify.verify_probs() (see its docstring),
for every size:

1. Geometry. Each visible face is a perspective-warped square; a homography
   from its 4 corners gives the N×N sticker centres (3N² in all).
2. Orientation. All 24 ways the cube can face the camera are tried. Odd cubes
   have a fixed middle sticker on each face, which narrows it down first, as on
   a 3x3. Even cubes (2x2, 4x4, 6x6) have none, so the photo decides alone. A
   cube that's scrambled wrongly looks wrong in every orientation, so the
   verdict doesn't depend on picking it.
3. Comparison, from the sticker reader's probabilities: a single slip that fits
   clearly better than the scramble as written, or enough stickers the reader
   is sure are wrong, means mis-scramble.

The cube rules that pick colours piece by piece (verify.decode_pieces) are 3x3
only, so here each sticker's colour is the reader's own best guess. For N = 3
this uses the same cells as verify.CELLS (tests check it).
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from .keypoints import FACE_QUADS, face_windings
from .nxn import cube
from .verify import HYPOTHESES, U_FLIP, V_FLIP, Alternatives, Verdict

FACES = "URFDLB"
UNIT_SQUARE = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])


@lru_cache(maxsize=None)
def cells(n: int) -> tuple[np.ndarray, ...]:
    """For each of the 24 orientations (verify.HYPOTHESES), the facelet behind each of the 3N² cells
    (face k of the image, row j, column i)."""
    c, h = cube(n), n - 1
    out = []
    for s, axes in HYPOTHESES:
        idx = []
        for k in range(3):
            a, au, av = axes[k], axes[U_FLIP[k]], axes[V_FLIP[k]]
            normal = [0, 0, 0]
            normal[a] = s[a]
            for j in range(n):
                for i in range(n):
                    p = [0, 0, 0]
                    p[a] = s[a] * h
                    p[au] = s[au] * (h - 2 * i)
                    p[av] = s[av] * (h - 2 * j)
                    idx.append(c.index[(tuple(p), tuple(normal))])
        out.append(np.array(idx))
    return tuple(out)


def middle_cells(n: int) -> list[int]:
    """The fixed middle sticker of each visible face (odd cubes only)."""
    return [k * n * n + (n // 2) * n + n // 2 for k in range(3)] if n % 2 else []


# Layer widths measured on real cubes with a caliper, in mm: (outer layer, inner layer).
# Big cubes are built with thicker outer layers, so their sticker grid isn't even; the
# training renders have equal layers. Measured 12 Nov 2026 on one 6x6 and one 7x7; other
# makes differ, which is why the website lets this be changed (see docs/how-it-works.md).
LAYER_MM = {6: (13.5, 9.5), 7: (13.0, 8.2)}


def layer_edges(n: int, mm: tuple[float, float] | None = None) -> np.ndarray:
    """Where the N layers of a face start and end, as fractions 0..1 of its width (N+1 numbers).

    mm is (outer layer, inner layer) in any unit; None means equal layers.
    """
    outer, inner = mm or (1.0, 1.0)
    widths = [outer] + [inner] * (n - 2) + [outer] if n > 1 else [1.0]
    return np.concatenate([[0.0], np.cumsum(widths) / np.sum(widths)])


def cell_centres(keypoints: np.ndarray, n: int, edges: np.ndarray | None = None) -> np.ndarray:
    """Image positions of the 3N² sticker centres, (3N², 2). edges: layer_edges() of a cube with uneven layers."""
    mid = (np.arange(n) + 0.5) / n if edges is None else (edges[:-1] + edges[1:]) / 2
    uv = np.float32([[mid[i], mid[j]] for j in range(n) for i in range(n)])
    out = []
    for quad in FACE_QUADS:
        H = cv2.getPerspectiveTransform(UNIT_SQUARE, keypoints[list(quad)].astype(np.float32))
        out.append(cv2.perspectiveTransform(uv[None], H)[0])
    return np.concatenate(out)


# How small a face may look (its area over the largest face's) and still be read, per size.
# Chosen on 4 Nov 2026 (true corners, reader r4): 0.3, the
# 3x3 value from verify.sample_colours, skipped readable faces in 31% of ordinary views and
# sent 40% of two-face-and-sliver views to "re-check". Smaller stickers need a bigger face
# before they read reliably; a 2x2 needs every sticker it can get.
MIN_FACE_SHARE = {2: 0.03, 3: 0.06, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1}

# Sizes where two faces aren't enough to judge: a 2x2 shows 8 of its 24 stickers, and
# 40% of single slips change none of them (4 Nov 2026, true corners).
NEEDS_THREE_FACES = {2}

# The check's thresholds per size, chosen on 4 Nov 2026 (c4 corners,
# reader r4, 400 mixed-scramble renders a size). A 2x2 shows only 12 stickers, so a slip
# there changes few of them; a 7x7 shows 147, so "6 clearly wrong" happened by chance.
# PX_PER_STICKER: smaller than this per sticker is "too small, move closer"; renders read
# well down to 10-12 px, 14 leaves room for a blurrier real webcam.
THRESHOLDS = {
    2: dict(min_evidence=6.0, min_slip_seen=2, min_mismatches=3),
    3: dict(min_evidence=8.0, min_slip_seen=3, min_mismatches=6),
    4: dict(min_evidence=8.0, min_slip_seen=3, min_mismatches=8),
    5: dict(min_evidence=8.0, min_slip_seen=3, min_mismatches=8),
    6: dict(min_evidence=8.0, min_slip_seen=3, min_mismatches=12),
    7: dict(min_evidence=8.0, min_slip_seen=3, min_mismatches=16),
}
PX_PER_STICKER = {n: 14 for n in THRESHOLDS}


def readable(keypoints: np.ndarray, n: int, img_shape: tuple[int, ...], min_face_share: float | None = None) -> np.ndarray:
    """Which of the 3N² stickers can be read: inside the picture, on a face that isn't seen
    nearly edge-on (at least `min_face_share` of the largest face's area; MIN_FACE_SHARE by default).

    A cube held toward the person scrambling often shows the camera two faces and a sliver
    of the third; the sliver's stickers are a few pixels thin, so they count as unknown and
    the check works from the other two faces (in simulation two faces are nearly as good).
    When exactly two faces are in view, the third is seen from behind (its corners wind the
    other way in the picture, and its outline covers the visible faces): it is skipped too.
    """
    c = cell_centres(keypoints, n)
    h, w = img_shape[:2]
    inside = (c[:, 0] >= 0) & (c[:, 0] < w) & (c[:, 1] >= 0) & (c[:, 1] < h)
    if min_face_share is None:
        min_face_share = MIN_FACE_SHARE.get(n, 0.1)
    signed = np.array(face_windings(keypoints))
    areas = np.abs(signed)
    front = signed * signed[areas.argmax()] > 0
    return inside & np.repeat(front & (areas >= min_face_share * areas.max()), n * n)


def single_slips(scramble: str, n: int) -> Alternatives:
    """States one human slip away from the scramble (as verify.single_slips), with this size's WCA moves.

    A 7x7 scramble has about 11,000 of them, so instead of replaying the whole scramble
    for each, the states before each move (prefix) and the combined effect of the moves
    after it (suffix) are worked out once; a slip at move k is then prefix[k], the slip,
    and suffix[k] (single_slips_slow does it the long way, and tests check they agree).
    """
    c = cube(n)
    moves = c.parse(scramble)
    perms = [c.move(m) for m in moves]
    prefix = [c.solved]
    for p in perms:
        prefix.append(prefix[-1][p])
    suffix = [np.arange(c.size)] * (len(moves) + 1)  # state after all moves = state after k moves [suffix[k]]
    for k in range(len(moves) - 1, -1, -1):
        suffix[k] = perms[k][suffix[k + 1]]
    base = prefix[-1].tobytes()
    seen, states, slips, where = {base}, [], [], []

    def add(st: np.ndarray, text: str, move: int, kind: str, done: str | None = None) -> None:
        key = st.tobytes()
        if key not in seen:
            seen.add(key)
            states.append(st)
            slips.append(text)
            where.append({"move": move, "kind": kind, "done": done})

    turns = [(t, c.move(t)) for t in c.event_turns]
    for i, m in enumerate(moves):
        before = prefix[i]
        add(before[suffix[i + 1]], f"move {i + 1} ({m}) skipped", i, "skipped")
        for t, pt in turns:
            if t != m:
                add(before[pt][suffix[i + 1]], f"move {i + 1} done as {t} instead of {m}", i, "wrong", t)
        if i + 1 < len(moves):
            add(before[perms[i + 1]][perms[i]][suffix[i + 2]], f"moves {i + 1} and {i + 2} swapped", i, "swapped")
    for i in range(len(moves) + 1):
        for t, pt in turns:
            add(prefix[i][pt][suffix[i]], f"extra {t} before move {i + 1}" if i < len(moves) else f"extra {t} at the end",
                i, "extra", t)
    return Alternatives(np.array(states), slips, where)


def single_slips_slow(scramble: str, n: int) -> Alternatives:
    """single_slips() the straightforward way, replaying each variant from solved (for tests)."""
    c = cube(n)
    moves = c.parse(scramble)
    base = c.apply(c.solved, moves).tobytes()
    seen, states, slips, where = {base}, [], [], []

    def add(variant: list[str], text: str, move: int, kind: str, done: str | None = None) -> None:
        st = c.apply(c.solved, variant)
        key = st.tobytes()
        if key not in seen:
            seen.add(key)
            states.append(st)
            slips.append(text)
            where.append({"move": move, "kind": kind, "done": done})

    turns = c.event_turns
    for i, m in enumerate(moves):
        add(moves[:i] + moves[i + 1:], f"move {i + 1} ({m}) skipped", i, "skipped")
        for t in turns:
            if t != m:
                add(moves[:i] + [t] + moves[i + 1:], f"move {i + 1} done as {t} instead of {m}", i, "wrong", t)
        if i + 1 < len(moves):
            add(moves[:i] + [moves[i + 1], m] + moves[i + 2:], f"moves {i + 1} and {i + 2} swapped", i, "swapped")
    for i in range(len(moves) + 1):
        for t in turns:
            add(moves[:i] + [t] + moves[i:], f"extra {t} before move {i + 1}" if i < len(moves) else f"extra {t} at the end",
                i, "extra", t)
    return Alternatives(np.array(states), slips, where)


def check(
    probs: np.ndarray,
    inside: np.ndarray,
    n: int,
    state: np.ndarray,
    alternatives: Alternatives | None = None,
    covered: float = 0.5,
    cap: float = 5.0,
    min_evidence: float | None = None,
    sure: float = 0.05,
    min_mismatches: int | None = None,
    max_unknown: float = 0.48,
    unsure: float = 0.6,
    max_unsure: float = 0.33,
    min_slip_seen: int | None = None,
    max_slip_left: int = 1,
) -> Verdict:
    """verify_probs() for any N. probs: (3N², 7) from the sticker reader (6 colours URFDLB, then covered).

    max_unknown and max_unsure are shares of the 3N² stickers (13 and 9 of 27 on a 3x3).
    min_evidence, min_slip_seen and min_mismatches default to THRESHOLDS for this size.
    """
    t = THRESHOLDS.get(n, THRESHOLDS[3])
    min_evidence = t["min_evidence"] if min_evidence is None else min_evidence
    min_slip_seen = t["min_slip_seen"] if min_slip_seen is None else min_slip_seen
    min_mismatches = t["min_mismatches"] if min_mismatches is None else min_mismatches
    m = 3 * n * n
    if n in NEEDS_THREE_FACES and inside.reshape(3, -1).any(axis=1).sum() < 3:
        return Verdict("re-check", "turn a corner toward the camera: this cube needs three faces in view",
                       unknown=np.flatnonzero(~inside).tolist())
    unknown = ~inside | (probs[:, 6] > covered)
    if unknown.sum() > max_unknown * m:
        return Verdict("re-check", "too much of the cube is hidden or out of view", unknown=np.flatnonzero(unknown).tolist())
    colour = probs[:, :6] / np.maximum(probs[:, :6].sum(axis=1, keepdims=True), 1e-9)
    cost = np.minimum(-np.log(np.maximum(colour, 1e-9)), cap)
    if (~unknown & (colour.max(axis=1) < unsure)).sum() > max_unsure * m:
        return Verdict("re-check", "colours aren't clear enough (glare, too dark or washed out)", unknown=np.flatnonzero(unknown).tolist())
    rows = np.flatnonzero(~unknown)
    all_cells = cells(n)
    fits = np.array([cost[rows, state[c][rows]].sum() for c in all_cells])
    middles = [k for k in middle_cells(n) if not unknown[k]]
    if middles:  # odd cubes: the middle stickers never move, so they say which way round the cube is
        centre = np.array([cost[middles, state[c][middles]].sum() for c in all_cells])
        fits = np.where(centre <= centre.min() + 1.0, fits, np.inf)
    best = int(np.argmin(fits))
    cell = all_cells[best]
    expected = state[cell]
    nearest = cost.argmin(axis=1)
    mismatch = ~unknown & (nearest != expected) & (colour[np.arange(m), expected] < sure)
    base = dict(
        mismatches=np.flatnonzero(mismatch).tolist(),
        unknown=np.flatnonzero(unknown).tolist(),
        facelets=cell.tolist(),
        expected="".join(FACES[e] for e in expected),
        observed="".join("?" if unknown[i] else FACES[nearest[i]] for i in range(m)),
        hypothesis=best,
    )
    if alternatives is not None and len(alternatives.states):
        # Each slip is judged in every orientation still possible (on an even cube, all 24),
        # one orientation at a time so a 7x7's ~11,000 slips stay small in memory.
        cost_rows = cost[rows].astype(np.float32)
        pick = np.arange(len(rows))
        k, h, slip_fit = 0, best, np.inf
        for o in np.flatnonzero(np.isfinite(fits)):
            f = cost_rows[pick, alternatives.states[:, all_cells[o][rows]]].sum(axis=1)  # (slips,)
            j = int(np.argmin(f))
            if f[j] < slip_fit:
                k, h, slip_fit = j, int(o), float(f[j])
        evidence = float(fits[best] - slip_fit)
        slipped = alternatives.states[k][all_cells[h]]
        changed = (slipped != expected) & ~unknown
        seen = int((changed & (colour[np.arange(m), expected] < 0.5)).sum())
        left = int((~unknown & (colour[np.arange(m), slipped] < sure)).sum())
        base.update(evidence=round(evidence, 2), slip_seen=seen, slip_left=left)
        if evidence > min_evidence and seen >= min_slip_seen:
            if left > max_slip_left:
                return Verdict("mis-scramble", "doesn't match the scramble", **base)
            base.update(mismatches=np.flatnonzero(changed).tolist())
            return Verdict("mis-scramble", "doesn't match the scramble", slip=alternatives.slips[k],
                           slip_where=alternatives.where[k], **base)
    if mismatch.sum() >= min_mismatches:
        return Verdict("mis-scramble", "doesn't match the scramble", **base)
    return Verdict("correct", "matches the scramble", **base)
