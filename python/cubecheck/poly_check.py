"""The check for skewb, pyraminx and FTO (experimental): nxn_check.py for shapes.py's puzzles.

The same three steps as for a cube:

1. Geometry. The corners give every visible sticker's place in the picture (shapes.py).
2. Orientation. Every way the puzzle can face the camera is tried: 24 for a skewb or an
   FTO, 12 for a pyraminx (shapes.Shape.views). None of these puzzles has stickers that
   never move the way a 3x3's centres do, so the photo decides alone, as on an even cube.
3. Comparison, from the sticker reader's probabilities: a single slip that fits clearly
   better than the scramble as written, or enough stickers the reader is sure are wrong,
   means mis-scramble.

See docs/other-puzzles.md.
"""

from __future__ import annotations

import numpy as np

from .shapes import HOLD_HINT, MIN_FACES, Shape
from .verify import Alternatives, Verdict

# As nxn_check.THRESHOLDS; min_known is how many stickers must be readable to judge at all.
# Skewb and FTO: by analogy with the cubes (a skewb shows 15 stickers, about a 2x2's 12).
# Pyraminx: a wrong tip changes only that tip's 3 stickers, and a two-face view shows 1 or 2
# of them, so one sticker the reader is sure about has to be enough. With 3 stickers needed,
# 37% of single slips passed as correct; with 1 (and evidence 4.5: one sure sticker costs 5),
# 1.6% did, and no right scramble in 1,000 was called wrong (11 Nov 2026, labelled corners,
# reader pyram-r1).
THRESHOLDS = {
    "skewb": dict(min_evidence=6.0, min_slip_seen=2, min_mismatches=3, min_known=9),
    "pyram": dict(min_evidence=4.5, min_slip_seen=1, min_mismatches=5, min_known=12),
    "fto": dict(min_evidence=8.0, min_slip_seen=3, min_mismatches=8, min_known=20),
}
# Smaller than this (pixels along an edge out of C) is "too small, move closer": about 14 px per sticker.
MIN_SIZE = {"skewb": 30, "pyram": 42, "fto": 42}


# How badly the corners may fit the puzzle's rigid shape (shapes.Shape.fit) and still be read.
# 12 Nov 2026, fto-c2 + fto-r2 on 1,000 renders held any way: all 23 false alarms had a fit
# error of 3.4% or more (median 13.8%: one corner on the wrong spot), right verdicts a median
# of 0.5% and under 1.5% nine times in ten. A limit of 3% left no false alarm; 5% left 2 in
# 1,000. But the 9 real FTO frames saved that day, with the outline on the puzzle by eye, fit
# to 0.5-4.9% (a real puzzle's corners are rounder and a close webcam bends the picture), so
# the limit is 5% until real frames with a known scramble say more.
# A skewb's faces are quads, read without a pose.
MAX_FIT = {"pyram": 0.05, "fto": 0.05}


def readable(keypoints: np.ndarray, sh: Shape, img_shape: tuple[int, ...], min_face_share: float = 0.1,
             max_fit: float | None = None) -> np.ndarray:
    """Which (face, slot) stickers can be read: inside the picture, on a face seen from the front
    that isn't nearly edge-on (at least `min_face_share` of the largest face's area). None at all
    when the corners don't fit the puzzle's shape (MAX_FIT): the grid can't be trusted then."""
    limit = MAX_FIT.get(sh.event) if max_fit is None else max_fit
    if limit and not sh.quads and sh.fit(keypoints, img_shape)[1] > limit:
        return np.zeros(sh.n_faces * sh.per_face, bool)
    c = sh.cell_centres(keypoints, img_shape)
    h, w = img_shape[:2]
    inside = (c[:, 0] >= 0) & (c[:, 0] < w) & (c[:, 1] >= 0) & (c[:, 1] < h)
    signed = sh.windings(keypoints)
    areas = np.abs(signed)
    front = signed < 0
    return inside & np.repeat(front & (areas >= min_face_share * areas[front].max(initial=0.0)), sh.per_face)


def single_slips(sh: Shape, scramble: str) -> Alternatives:
    """States one human slip away from the scramble (as nxn_check.single_slips), with this puzzle's turns."""
    p = sh.puzzle
    moves = scramble.split()
    perms = [p.move(m) for m in moves]
    prefix = [p.solved]
    for q in perms:
        prefix.append(prefix[-1][q])
    suffix = [np.arange(p.size)] * (len(moves) + 1)
    for k in range(len(moves) - 1, -1, -1):
        suffix[k] = perms[k][suffix[k + 1]]
    seen, states, slips, where = {prefix[-1].tobytes()}, [], [], []

    def add(st: np.ndarray, text: str, move: int, kind: str, done: str | None = None) -> None:
        if st.tobytes() not in seen:
            seen.add(st.tobytes())
            states.append(st)
            slips.append(text)
            where.append({"move": move, "kind": kind, "done": done})

    turns = [(t, p.move(t)) for t in p.turns]
    for i, m in enumerate(moves):
        add(prefix[i][suffix[i + 1]], f"move {i + 1} ({m}) skipped", i, "skipped")
        for t, pt in turns:
            if t != m:
                add(prefix[i][pt][suffix[i + 1]], f"move {i + 1} done as {t} instead of {m}", i, "wrong", t)
        if i + 1 < len(moves):
            add(prefix[i][perms[i + 1]][perms[i]][suffix[i + 2]], f"moves {i + 1} and {i + 2} swapped", i, "swapped")
    for i in range(len(moves) + 1):
        for t, pt in turns:
            add(prefix[i][pt][suffix[i]], f"extra {t} before move {i + 1}" if i < len(moves) else f"extra {t} at the end", i, "extra", t)
    return Alternatives(np.array(states).reshape(len(states), p.size), slips, where)


def check(
    probs: np.ndarray,
    inside: np.ndarray,
    sh: Shape,
    state: np.ndarray,
    alternatives: Alternatives | None = None,
    covered: float = 0.5,
    cap: float = 5.0,
    sure: float = 0.05,
    unsure: float = 0.6,
    max_unsure: float = 0.33,
    max_slip_left: int = 1,
    **thresholds: float,
) -> Verdict:
    """nxn_check.check() for a shapes.py puzzle. probs: (faces x slots, colours + 1) from the
    reader (the puzzle's colours in its model's face order, then covered)."""
    t = {"min_faces": MIN_FACES[sh.event], **THRESHOLDS[sh.event], **thresholds}
    m, colours = len(probs), sh.colours
    keys = sh.puzzle.face_keys
    faces_read = int(inside.reshape(sh.n_faces, -1).any(axis=1).sum())
    # Too few faces to be sure it's right (shapes.MIN_FACES): a slip that shows is still a slip,
    # but "correct" needs the better view. An FTO's single slips, labelled corners and colours,
    # renders held any way (12 Nov 2026): caught 100% from 4 faces, 96% from 3, 85% from 2.
    poor_hold = faces_read < t["min_faces"]
    unknown = ~inside | (probs[:, colours] > covered)
    if faces_read == 0 or (~unknown).sum() < t["min_known"]:
        reason = HOLD_HINT[sh.event] if poor_hold else "too much of the puzzle is hidden or out of view"
        return Verdict("re-check", reason, unknown=np.flatnonzero(unknown).tolist())
    colour = probs[:, :colours] / np.maximum(probs[:, :colours].sum(axis=1, keepdims=True), 1e-9)
    cost = np.minimum(-np.log(np.maximum(colour, 1e-9)), cap)
    if (~unknown & (colour.max(axis=1) < unsure)).sum() > max_unsure * (~unknown).sum():
        return Verdict("re-check", "colours aren't clear enough (glare, too dark or washed out)", unknown=np.flatnonzero(unknown).tolist())
    rows = np.flatnonzero(~unknown)
    fits = np.array([cost[rows, state[c][rows]].sum() for c in sh.cells])
    best = int(np.argmin(fits))
    cell = sh.cells[best]
    expected = state[cell]
    nearest = cost.argmin(axis=1)
    mismatch = ~unknown & (nearest != expected) & (colour[np.arange(m), expected] < sure)
    base = dict(
        mismatches=np.flatnonzero(mismatch).tolist(),
        unknown=np.flatnonzero(unknown).tolist(),
        facelets=cell.tolist(),
        expected=",".join(keys[e] for e in expected),
        observed=",".join("?" if unknown[i] else keys[nearest[i]] for i in range(m)),
        hypothesis=best,
    )
    if alternatives is not None and len(alternatives.states):
        cost_rows = cost[rows].astype(np.float32)
        pick = np.arange(len(rows))
        k, h, slip_fit = 0, best, np.inf
        for o in range(len(sh.cells)):
            f = cost_rows[pick, alternatives.states[:, sh.cells[o][rows]]].sum(axis=1)  # (slips,)
            j = int(np.argmin(f))
            if f[j] < slip_fit:
                k, h, slip_fit = j, o, float(f[j])
        evidence = float(fits[best] - slip_fit)
        slipped = alternatives.states[k][sh.cells[h]]
        changed = (slipped != expected) & ~unknown
        seen = int((changed & (colour[np.arange(m), expected] < 0.5)).sum())
        left = int((~unknown & (colour[np.arange(m), slipped] < sure)).sum())
        base.update(evidence=round(evidence, 2), slip_seen=seen, slip_left=left)
        if evidence > t["min_evidence"] and seen >= t["min_slip_seen"]:
            if left > max_slip_left:
                return Verdict("mis-scramble", "doesn't match the scramble", **base)
            base.update(mismatches=np.flatnonzero(changed).tolist())
            return Verdict("mis-scramble", "doesn't match the scramble", slip=alternatives.slips[k], slip_where=alternatives.where[k], **base)
    if mismatch.sum() >= t["min_mismatches"]:
        return Verdict("mis-scramble", "doesn't match the scramble", **base)
    if poor_hold:
        return Verdict("re-check", f"fits so far: {HOLD_HINT[sh.event]}", **base)
    return Verdict("correct", "matches the scramble", **base)
