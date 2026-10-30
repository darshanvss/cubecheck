"""From 7 corners and a scramble to a verdict: correct, mis-scramble or re-check.

1. Geometry. The corners split the image into three faces; each face is a
   perspective-warped square, so a homography gives the 3x3 sticker centres.
2. Reading. A small patch around each centre gives the sticker's colour
   (median, in CIELAB, so glare and noise matter less).
3. Orientation. Which physical corner faces the camera is unknown, so all 24
   ways of reading the three faces are tried. Under the right one, stickers
   that should share a colour look alike and the rest look different.
4. Comparison. Colours are judged relative to each other in the same photo,
   never against fixed reference colours, so warm light or a colour cast
   shifts everything together and doesn't matter. A sticker counts as a
   mismatch only when it clearly looks like another colour group. One odd
   sticker is treated as a misread; a real mistake changes several.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import cv2
import numpy as np

from .cube import FACELETS, FACES, INDEX, MOVES, SOLVED, apply
from .keypoints import FACE_QUADS

# Image face k's u direction (C -> second quad corner) leaves face U_FLIP[k];
# its v direction (C -> fourth quad corner) leaves face V_FLIP[k].
U_FLIP = (2, 0, 1)
V_FLIP = (1, 2, 0)


def _hypotheses() -> list[tuple[tuple[int, int, int], list[int]]]:
    """24 readings: which cube corner faces the camera, and which visible face is image face 0.

    Seen from outside, the faces around corner (sx, sy, sz) run clockwise
    y, x, z when sx*sy*sz > 0 and y, z, x otherwise. Rotating the start covers
    the model mistaking which face is white/yellow.
    """
    out = []
    for s in itertools.product((1, -1), repeat=3):
        order = (1, 0, 2) if s[0] * s[1] * s[2] > 0 else (1, 2, 0)
        for r in range(3):
            out.append((s, [order[(k + r) % 3] for k in range(3)]))
    return out


def cell_facelets(s: tuple[int, int, int], axes: list[int]) -> np.ndarray:
    """Facelet index (0..53) of each of the 27 cells, ordered face k, row j, column i."""
    idx = []
    for k in range(3):
        a, au, av = axes[k], axes[U_FLIP[k]], axes[V_FLIP[k]]
        normal = [0, 0, 0]
        normal[a] = s[a]
        for j in range(3):
            for i in range(3):
                p = [0, 0, 0]
                p[a] = s[a]
                p[au] = s[au] * (1 - i)
                p[av] = s[av] * (1 - j)
                idx.append(INDEX[(tuple(p), tuple(normal))])
    return np.array(idx)


HYPOTHESES = _hypotheses()
CELLS = [cell_facelets(s, axes) for s, axes in HYPOTHESES]


def _piece_options() -> dict[tuple[int, ...], np.ndarray]:
    """For every corner and edge slot (its facelet indices, sorted), each piece's possible colourings there.

    Straight from the colour scheme. The solved cube has 8 corners, each with
    its own 3 colours, and 12 edges with 2. Any corner can sit in any corner
    slot, twisted 3 ways, but its colours always keep the same clockwise order
    (a mirror-image corner can't exist). Any edge can sit in any edge slot,
    either way round. Result per slot: an array (piece, orientation, sticker)
    of colours, 8 x 3 x 3 for corners and 12 x 2 x 2 for edges.
    """
    by_pos: dict[tuple[int, int, int], list[int]] = {}
    for i, (pos, _normal) in enumerate(FACELETS):
        if sum(abs(v) for v in pos) >= 2:  # corners and edges; centres never move
            by_pos.setdefault(pos, []).append(i)

    def clockwise(slot: list[int]) -> list[int]:
        # Order a corner's 3 facelets so their normals form a right-handed set:
        # that's one fixed rotational direction around the corner, the same everywhere.
        a, b, c = slot
        normals = [np.array(FACELETS[i][1]) for i in (a, b, c)]
        return [a, b, c] if np.linalg.det(np.array(normals)) > 0 else [a, c, b]

    corners = [clockwise(f) for f in by_pos.values() if len(f) == 3]
    edges = [f for f in by_pos.values() if len(f) == 2]
    corner_colours = [[int(SOLVED[i]) for i in c] for c in corners]  # each solved corner, clockwise
    edge_colours = [[int(SOLVED[i]) for i in e] for e in edges]

    options: dict[tuple[int, ...], np.ndarray] = {}
    for slot in corners:
        order = sorted(slot)  # stored in the slot's sorted facelet order, as used for lookups
        twists = [[cols[(slot.index(f) + t) % 3] for f in order] for cols in corner_colours for t in range(3)]
        options[tuple(order)] = np.array(twists).reshape(8, 3, 3)
    for slot in edges:
        order = sorted(slot)
        first = slot.index(order[0])
        flips = [[cols[first], cols[1 - first]] for cols in edge_colours]
        options[tuple(order)] = np.array([[f, f[::-1]] for f in flips])
    return options


OPTIONS = _piece_options()
LEGAL = {slot: {tuple(c) for c in opts.reshape(-1, len(slot)).tolist()} for slot, opts in OPTIONS.items()}
SLOT_OF = {f: slot for slot in OPTIONS for f in slot}


def _distinct(cost: np.ndarray) -> list[int]:
    """Give each row (slot) a different column (piece) with the lowest total cost.

    Dynamic programming over which pieces are already used: after each slot,
    keep only the cheapest way to have used every set of pieces. 12 edge
    pieces make 4,096 sets, so this is instant.
    """
    slots, pieces = cost.shape
    masks = np.arange(1 << pieces)
    best = np.full(1 << pieces, np.inf)
    best[0] = 0.0
    choices = []
    for s in range(slots):
        new = np.full_like(best, np.inf)
        choice = np.full(1 << pieces, -1)
        for p in range(pieces):
            has = (masks >> p) & 1 == 1
            cand = np.full_like(best, np.inf)
            cand[has] = best[masks[has] ^ (1 << p)] + cost[s, p]
            better = cand < new
            new[better], choice[better] = cand[better], p
        best = new
        choices.append(choice)
    mask, out = int(best.argmin()), []
    for choice in reversed(choices):
        out.append(int(choice[mask]))
        mask ^= 1 << out[-1]
    return out[::-1]


def decode_pieces(cells: np.ndarray, dist6: np.ndarray, known: np.ndarray, distinct: bool = True) -> np.ndarray:
    """Colour of each cell, reading every corner and edge as a whole, real piece.

    Each visible piece gets the colouring of an actual cube piece that best
    fits all of its readable stickers, so one misread sticker gets pulled back
    to a colour that can be there (red next to blue on a corner means the third
    sticker is white or yellow, and the turning direction says which). With
    `distinct`, no piece is used twice in view, which also means no colour
    shows more than 9 times. Centres keep their nearest colour.
    """
    out = dist6.argmin(axis=1)
    groups: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for n, f in enumerate(cells.tolist()):
        if known[n] and f in SLOT_OF:
            groups.setdefault(SLOT_OF[f], []).append((n, SLOT_OF[f].index(f)))
    for size in (3, 2):
        slots = [s for s in groups if len(s) == size]
        if not slots:
            continue
        costs, orients = [], []
        for s in slots:
            opts = OPTIONS[s]  # (piece, orientation, sticker)
            c = sum(dist6[n, opts[:, :, k]] for n, k in groups[s])  # (piece, orientation)
            costs.append(c.min(axis=1))
            orients.append(c.argmin(axis=1))
        costs = np.array(costs)
        chosen = _distinct(costs) if distinct else costs.argmin(axis=1).tolist()
        for s, piece, orient in zip(slots, chosen, orients):
            for n, k in groups[s]:
                out[n] = OPTIONS[s][piece, orient[piece], k]
    return out


UNIT_SQUARE = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])


def parse_scramble(text: str) -> list[str]:
    """WCA face-turn notation (R U' F2 ...); raises ValueError on anything else."""
    moves = text.replace("’", "'").split()
    bad = [m for m in moves if m not in MOVES]
    if bad:
        raise ValueError(f"not a 3x3 face turn: {' '.join(bad)}")
    return moves


def expected_state(scramble: str) -> np.ndarray:
    return apply(SOLVED, parse_scramble(scramble))


@dataclass
class Alternatives:
    """Every cube state one slip away from the scramble, with a description of each slip."""

    states: np.ndarray  # (M, 54)
    slips: list[str]


def single_slips(scramble: str) -> Alternatives:
    """States from one human slip: a move skipped, reversed/changed, added, or two moves swapped.

    Different slips often give the same state; each state is kept once, with the first slip found.
    """
    moves = parse_scramble(scramble)
    base = apply(SOLVED, moves).tobytes()
    seen, states, slips = {base}, [], []

    def add(variant: list[str], text: str) -> None:
        st = apply(SOLVED, variant)
        key = st.tobytes()
        if key not in seen:
            seen.add(key)
            states.append(st)
            slips.append(text)

    turns = [f + m for f in FACES for m in ("", "'", "2")]
    for i, m in enumerate(moves):
        add(moves[:i] + moves[i + 1 :], f"move {i + 1} ({m}) skipped")
        for t in turns:
            if t != m:
                add(moves[:i] + [t] + moves[i + 1 :], f"move {i + 1} done as {t} instead of {m}")
        if i + 1 < len(moves):
            add(moves[:i] + [moves[i + 1], m] + moves[i + 2 :], f"moves {i + 1} and {i + 2} swapped")
    for i in range(len(moves) + 1):
        for t in turns:
            add(moves[:i] + [t] + moves[i:], f"extra {t} before move {i + 1}" if i < len(moves) else f"extra {t} at the end")
    return Alternatives(np.array(states), slips)


def cell_centres(keypoints: np.ndarray) -> np.ndarray:
    """Image positions of the 27 sticker centres, (27, 2)."""
    uv = np.float32([[(i + 0.5) / 3, (j + 0.5) / 3] for j in range(3) for i in range(3)])
    out = []
    for quad in FACE_QUADS:
        H = cv2.getPerspectiveTransform(UNIT_SQUARE, keypoints[list(quad)].astype(np.float32))
        out.append(cv2.perspectiveTransform(uv[None], H)[0])
    return np.concatenate(out)


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = x / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_lab(rgb: np.ndarray) -> np.ndarray:
    srgb = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.clip(rgb, 0, None) ** (1 / 2.4) - 0.055)
    img = np.clip(srgb, 0, 1).astype(np.float32).reshape(-1, 1, 3)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return lab


def cube_size(keypoints: np.ndarray) -> float:
    """Mean length of the 9 visible cube edges, in pixels."""
    from .keypoints import SKELETON

    return float(np.mean([np.linalg.norm(keypoints[a] - keypoints[b]) for a, b in SKELETON]))


def sample_colours(
    img_bgr: np.ndarray,
    keypoints: np.ndarray,
    radius: float = 0.07,
    min_face_share: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Colour of a patch around each sticker centre.

    Returns the median colour in linear RGB (light adds up linearly there, so a
    shaded face is just a scaled copy), how uneven each patch is (a patch
    half-covered by a finger is uneven), and which patches can be used: inside
    the image, on a face that isn't seen nearly edge-on (smaller than
    `min_face_share` of the largest face's area; stickers there are thin
    slivers full of glare).
    `radius` is in face units (a sticker is 1/3), so 0.07 covers the middle 40% of a sticker.
    """
    offsets = np.linspace(-radius, radius, 5)
    grid = np.array([[du, dv] for dv in offsets for du in offsets], np.float32)
    pts = []
    for quad in FACE_QUADS:
        H = cv2.getPerspectiveTransform(UNIT_SQUARE, keypoints[list(quad)].astype(np.float32))
        for j in range(3):
            for i in range(3):
                centre = np.float32([(i + 0.5) / 3, (j + 0.5) / 3])
                pts.append(cv2.perspectiveTransform((grid + centre)[None], H)[0])
    pts = np.array(pts)  # (27, 25, 2)
    h, w = img_bgr.shape[:2]
    inside = (pts[..., 0] >= 0) & (pts[..., 0] < w - 1) & (pts[..., 1] >= 0) & (pts[..., 1] < h - 1)
    rgb = img_bgr[..., ::-1].astype(np.float32)
    flat = pts.reshape(-1, 1, 2).astype(np.float32)
    samples = cv2.remap(rgb, flat[..., 0], flat[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    samples = _srgb_to_linear(samples.reshape(27, -1, 3))

    def patch_colour(s: np.ndarray, m: np.ndarray) -> np.ndarray:
        if not m.any():
            return np.full(3, 1e-3)
        s = s[m]
        # Drop the brightest third: on glossy stickers that's where glare sits.
        luma = s @ np.float32([0.2126, 0.7152, 0.0722])
        return np.median(s[luma <= np.percentile(luma, 67)], axis=0)

    colours = np.array([patch_colour(s, m) for s, m in zip(samples, inside)])
    lab = _linear_to_lab(samples.reshape(-1, 3)).reshape(27, -1, 3)
    spread = np.array([np.median(np.linalg.norm(l - np.median(l, axis=0), axis=1)) for l in lab])
    areas = np.array([cv2.contourArea(keypoints[list(q)].astype(np.float32)) for q in FACE_QUADS])
    face_ok = np.repeat(areas >= min_face_share * areas.max(), 9)
    return colours, spread, (inside.mean(axis=1) > 0.6) & face_ok


def _face_balance(rgb: np.ndarray, expected: np.ndarray, valid: np.ndarray, rounds: int = 3) -> np.ndarray:
    """Per-face colour gains so the three faces agree, given which stickers should match.

    Each face gets its own light (a lamp on one side, shade on another). Light
    scales linear RGB, so one multiplier per colour channel per face undoes it.
    Gains come from the median ratio over a face's stickers, so a few stickers
    that really differ (a mis-scramble, a finger) don't drag them.
    """
    out = np.maximum(rgb, 1e-4)
    for _ in range(rounds):
        protos = {c: np.median(out[valid & (expected == c)], axis=0) for c in range(6) if (valid & (expected == c)).any()}
        gains = np.ones((3, 3))
        for k in range(3):
            cells = [n for n in range(9 * k, 9 * k + 9) if valid[n]]
            if cells:
                gains[k] = np.median([protos[expected[n]] / out[n] for n in cells], axis=0)
        gains /= np.exp(np.log(gains).mean(axis=0))  # keep the overall level
        out = out * np.repeat(gains, 9, axis=0)
    return out


def _features(rgb: np.ndarray) -> np.ndarray:
    # CIELAB distances follow how different colours look; lightness gets half weight
    # because leftover shading mostly changes lightness.
    return _linear_to_lab(rgb) * np.float32([0.5, 1, 1])


def _separation(feat: np.ndarray, expected: np.ndarray, valid: np.ndarray) -> float:
    """Mean distance between stickers expected to match, over mean distance between those expected to differ."""
    d = np.linalg.norm(feat[:, None] - feat[None], axis=2)
    both = valid[:, None] & valid[None] & ~np.eye(len(feat), dtype=bool)
    same = both & (expected[:, None] == expected[None])
    diff = both & (expected[:, None] != expected[None])
    if not same.any() or not diff.any():
        return np.inf
    return float(d[same].mean() / d[diff].mean())


@dataclass
class Verdict:
    status: str  # "correct", "mis-scramble" or "re-check"
    reason: str
    mismatches: list[int] = field(default_factory=list)  # cell numbers 0..26
    unknown: list[int] = field(default_factory=list)
    facelets: list[int] = field(default_factory=list)  # facelet index of each cell
    expected: str = ""  # colour letter per cell
    observed: str = ""  # colour letter per cell ("?" = unknown)
    separation: float = 0.0
    hypothesis: int = -1
    slip: str = ""  # most likely slip, when one explains the photo
    evidence: float = 0.0  # how much better the best slip fits than the scramble as written


def verify(
    colours: np.ndarray,
    spread: np.ndarray,
    valid: np.ndarray,
    state: np.ndarray,
    margin: float = 8.0,
    ratio: float = 2.0,
    max_spread: float = 25.0,
    max_distance: float = 30.0,
    min_mismatches: int = 10,
    max_unknown: int = 13,
    alternatives: Alternatives | None = None,
    cap: float = 25.0,
    min_evidence: float = 30.0,
    max_separation: float = 0.35,
    rules: str = "unique",
) -> Verdict:
    """Compare 27 observed sticker colours (linear RGB) with the state a scramble should give.

    With `alternatives` (see single_slips), the main test is whether some slip
    explains the photo clearly better than the scramble as written: a misread
    sticker or a finger can't be explained by a slip, because a slip moves whole
    pieces. Without a matching slip, it takes `min_mismatches` clearly wrong
    stickers to call a mis-scramble (bigger mistakes change many stickers).

    `rules` is how colours are read: "none" (each sticker on its own),
    "pieces" (each corner and edge as a real piece) or "unique" (also no piece twice).
    """
    usable = valid & (spread < max_spread)  # very uneven patches: finger edges, glare
    if usable.sum() < 27 - max_unknown:
        return Verdict("re-check", "can't see enough stickers clearly", unknown=np.flatnonzero(~usable).tolist())
    scores, feats = [], []
    for cells in CELLS:
        feat = _features(_face_balance(colours, state[cells], usable))
        feats.append(feat)
        scores.append(_separation(feat, state[cells], usable))
    best = int(np.argmin(scores))
    cells = CELLS[best]
    expected = state[cells]
    feat = feats[best]

    # Reference colour of each group = median of the stickers expected to have it,
    # then refined from what each sticker actually looks closest to.
    def prototypes(labels: np.ndarray) -> dict[int, np.ndarray]:
        return {c: np.median(feat[usable & (labels == c)], axis=0) for c in range(6) if (usable & (labels == c)).any()}

    protos = prototypes(expected)
    for _ in range(2):
        keys = list(protos)
        dist = np.stack([np.linalg.norm(feat - protos[c], axis=1) for c in keys], axis=1)
        protos = {**protos, **prototypes(np.array(keys)[dist.argmin(axis=1)])}
    keys = list(protos)
    dist = np.stack([np.linalg.norm(feat - protos[c], axis=1) for c in keys], axis=1)
    closest = dist.min(axis=1)
    dist6 = np.full((27, 6), 2.0 * max_distance)  # colours with no reference in view count as far
    for j, c in enumerate(keys):
        dist6[:, c] = dist[:, j]
    own = dist6[np.arange(27), expected]

    # Unknown: off the image, uneven, or unlike every sticker colour in view (a finger).
    unknown = ~usable | (closest > max_distance)
    nearest = dist6.argmin(axis=1) if rules == "none" else decode_pieces(cells, dist6, ~unknown, rules == "unique")
    read = dist6[np.arange(27), nearest]
    # Mismatch: clearly another colour than expected, both in absolute terms and relative to it.
    mismatch = ~unknown & (nearest != expected) & (own - read > margin) & (own > ratio * read)
    observed = "".join("?" if unknown[n] else FACES[nearest[n]] for n in range(27))
    base = dict(
        mismatches=np.flatnonzero(mismatch).tolist(),
        unknown=np.flatnonzero(unknown).tolist(),
        facelets=cells.tolist(),
        expected="".join(FACES[e] for e in expected),
        observed=observed,
        separation=round(scores[best], 3),
        hypothesis=best,
    )
    n_mis = int(mismatch.sum())
    best_labels = expected

    # 1. Does one slip explain the photo clearly better than the scramble as written?
    #    Fit = distance from each readable sticker to the colour group a state puts
    #    there, capped so one wild reading (glare, a finger) can't dominate.
    if alternatives is not None and len(alternatives.states):
        dists = np.full((27, 6), cap)
        for j, c in enumerate(keys):
            dists[:, c] = np.minimum(dist[:, j], cap)
        rows = np.flatnonzero(~unknown)
        fit = lambda labels: dists[rows, labels[..., rows]].sum(axis=-1)  # noqa: E731
        slips = fit(alternatives.states[:, cells])
        k = int(np.argmin(slips))
        evidence = float(fit(expected)) - float(slips[k])
        base.update(evidence=round(evidence, 1))
        if evidence > min_evidence:
            # Highlight exactly the stickers that slip changes.
            changed = (alternatives.states[k][cells] != expected) & ~unknown
            base.update(mismatches=np.flatnonzero(changed).tolist())
            return Verdict("mis-scramble", "doesn't match the scramble", slip=alternatives.slips[k], **base)
        if evidence > 0:
            best_labels = alternatives.states[k][cells]

    # 2. Are the colours clear enough to trust a verdict? Judged on the best explanation,
    #    so a real mistake doesn't make a good photo look unclear.
    clarity = _separation(feat, best_labels, ~unknown)
    base.update(separation=round(clarity, 3))
    if unknown.sum() > max_unknown:
        return Verdict("re-check", "too much of the cube is hidden or out of view", **base)
    if clarity > max_separation:
        return Verdict("re-check", "colours aren't clear enough (glare, too dark or washed out)", **base)

    # 3. Bigger mistakes (several slips, wrong orientation) change many stickers.
    if n_mis >= min_mismatches:
        return Verdict("mis-scramble", "doesn't match the scramble", **base)
    return Verdict("correct", "matches the scramble" if n_mis == 0 else "matches (a sticker or two looked odd)", **base)


def verify_probs(
    probs: np.ndarray,
    inside: np.ndarray,
    state: np.ndarray,
    alternatives: Alternatives | None = None,
    rules: str = "unique",
    covered: float = 0.5,
    cap: float = 5.0,
    min_evidence: float = 8.0,
    sure: float = 0.05,
    min_mismatches: int = 6,
    max_unknown: int = 13,
    unsure: float = 0.6,
    max_unsure: int = 9,
) -> Verdict:
    """Like verify(), but from the sticker reader's probabilities (27, 7) instead of raw colours.

    A colour's cost at a sticker is -log(probability): 0 when the reader is
    certain, 2.3 at 10%, 4.6 at 1%. Costs are capped at `cap`, so one confident
    misread can't outweigh everything else. Then the same steps as verify():
    pick the orientation (from the centres, then the photo), ask whether a single slip fits
    clearly better (`min_evidence`), and otherwise count stickers the reader
    is sure are wrong (probability of the expected colour under `sure`).
    More than `max_unsure` stickers with no colour above `unsure` means the
    photo isn't clear enough (glare, washed out): re-check. `rules` sets how the shown colours are chosen ("none", "pieces", "unique").
    """
    unknown = ~inside | (probs[:, 6] > covered)
    if unknown.sum() > max_unknown:
        return Verdict("re-check", "too much of the cube is hidden or out of view", unknown=np.flatnonzero(unknown).tolist())
    colour = probs[:, :6] / np.maximum(probs[:, :6].sum(axis=1, keepdims=True), 1e-9)  # given it's a sticker, which colour
    cost = np.minimum(-np.log(np.maximum(colour, 1e-9)), cap)
    if (~unknown & (colour.max(axis=1) < unsure)).sum() > max_unsure:
        return Verdict("re-check", "colours aren't clear enough (glare, too dark or washed out)", unknown=np.flatnonzero(unknown).tolist())
    rows = np.flatnonzero(~unknown)
    fits = [cost[rows, state[cells][rows]].sum() for cells in CELLS]
    # Face turns never move the centres, so they say which way round the cube is: of the orientations
    # whose visible centres fit, the one the whole photo fits best. (Without this, a badly wrong cube
    # can fit a turned-around orientation better and get compared sticker for sticker the wrong way.)
    centres = [n for n in (4, 13, 22) if not unknown[n]]
    centre = np.array([cost[centres, state[cells][centres]].sum() for cells in CELLS])
    best = int(np.argmin(np.where(centre <= centre.min() + 1.0, fits, np.inf)))
    cells = CELLS[best]
    expected = state[cells]

    nearest = cost.argmin(axis=1) if rules == "none" else decode_pieces(cells, cost, ~unknown, rules == "unique")
    mismatch = ~unknown & (nearest != expected) & (colour[np.arange(27), expected] < sure)
    base = dict(
        mismatches=np.flatnonzero(mismatch).tolist(),
        unknown=np.flatnonzero(unknown).tolist(),
        facelets=cells.tolist(),
        expected="".join(FACES[e] for e in expected),
        observed="".join("?" if unknown[n] else FACES[nearest[n]] for n in range(27)),
        hypothesis=best,
    )
    if alternatives is not None and len(alternatives.states):
        slips = cost[rows[None, :], alternatives.states[:, cells][:, rows]].sum(axis=1)
        k = int(np.argmin(slips))
        evidence = float(fits[best] - slips[k])
        base.update(evidence=round(evidence, 2))
        if evidence > min_evidence:
            changed = (alternatives.states[k][cells] != expected) & ~unknown
            base.update(mismatches=np.flatnonzero(changed).tolist())
            return Verdict("mis-scramble", "doesn't match the scramble", slip=alternatives.slips[k], **base)
    if mismatch.sum() >= min_mismatches:
        return Verdict("mis-scramble", "doesn't match the scramble", **base)
    return Verdict("correct", "matches the scramble", **base)


def inside_image(img_shape: tuple[int, ...], keypoints: np.ndarray) -> np.ndarray:
    """Which of the 27 sticker centres fall inside the image."""
    c = cell_centres(keypoints)
    h, w = img_shape[:2]
    return (c[:, 0] >= 0) & (c[:, 0] < w) & (c[:, 1] >= 0) & (c[:, 1] < h)


def check_image(
    img_bgr: np.ndarray,
    keypoints: np.ndarray,
    state: np.ndarray,
    alternatives: Alternatives,
    min_cube_px: float = 60.0,
    **kwargs,
) -> tuple[Verdict, np.ndarray]:
    """Full pipeline for one image and its 7 corners: the verdict and the 27 sticker centres.

    `state` and `alternatives` come from expected_state() and single_slips(); compute
    them once per scramble, not per frame.
    """
    centres = cell_centres(keypoints)
    if cube_size(keypoints) < min_cube_px:
        return Verdict("re-check", "cube too small: move it closer"), centres
    colours, spread, valid = sample_colours(img_bgr, keypoints)
    return verify(colours, spread, valid, state, alternatives=alternatives, **kwargs), centres
