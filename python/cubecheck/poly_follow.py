"""Follow-along for skewb, pyraminx and FTO (experimental): follow.py's tracker on shapes.py's puzzles.

The tracker is the cubes' (follow.py: a score for every point of the scramble, updated look
by look, and one slip near where the puzzle is believed to be). What differs here:

- Every turn is a third of a turn, so there are no half turns and no "halfway".
- No sticker stays put the way a 3x3's centres do, so every way round is tried at each look
  (24 for a skewb or an FTO, 12 for a pyraminx), as for an even cube.
- Some turns don't disturb each other: a pyraminx's tips and anything else, an FTO's opposite
  faces (U and D). Which ones is worked out from the moves themselves (`apart`), and used in
  three places:
  1. A later move done early is not a mistake. A pyraminx with tip b turned before tip u is
     the scramble's puzzle once u is done too, so it counts as the point before u (`ahead`
     in the report). These are the states follow.py keeps for half turns (`halves`).
  2. The way back keeps the moves it needn't undo. A tip turned the wrong way three moves
     ago is put right with that tip alone.
  3. The way back is shortened across them: U D U is D U' (three thirds of U make nothing).
- How much a look must show and how much better a slip must fit are per puzzle (SETTINGS).

See docs/other-puzzles.md.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .follow import Follower, Mistake, inverse
from .shapes import parse_scramble, shape

# What differs from the cubes' settings, measured on test renders with the real models
# and on simulated looks, 12 Nov 2026:
# - skewb, max_cost: a look shows 15 stickers at most, so 0.6 a sticker allows a single wrong one
#   (5 each); 0.75 allows two. Position right from one look 98.1% -> 98.9%, no false mistake in 360.
# - pyram, min_evidence: a tip is 3 stickers and an edge-on view shows 1 or 2 of them, so one
#   sticker the reader is sure about has to be enough (as poly_check.THRESHOLDS). Slips caught
#   with two looks: 72% at 8, 95% at 4.5; false mistakes 0 in 356 either way.
# - pyram, min_faces: a look at one face changes nothing. From one face the position was right
#   71% of the time (two faces: 97-99%) and a slip was told with the wrong way back in 21% of looks.
DEFAULTS = dict(max_cost=0.6, min_evidence=8.0, min_faces=1)  # the cubes' (follow.Follower)
SETTINGS = {
    "skewb": dict(max_cost=0.75),
    "pyram": dict(min_evidence=4.5, min_faces=2),
    "fto": dict(),
}
MAX_EARLY = 4  # later moves that may be done early at one point (a pyraminx has 4 tips)


def layer(move: str) -> str:
    """What a move turns: R' -> R, BL -> BL, u' -> u."""
    return move.rstrip("'")


class PolyFollower(Follower):
    # A move done early can be out of sight (a tip on the far side): the look then fits the point
    # with and without it equally, and that is not "a later move is already done".
    half_margin = 1.0

    def __init__(self, scramble: str, event: str, after: int = 6, before: int = 2, jump: float = 0.02, temper: float = 3.0,
                 min_seen: float = 0.3, max_cost: float | None = None, min_evidence: float | None = None, confirm: int = 2,
                 min_faces: int | None = None) -> None:
        sh = shape(event)
        self.event, self.sh, self.c, self.colours = event, sh, sh.puzzle, sh.colours
        self.after, self.before, self.jump, self.temper = after, before, jump, temper
        self.min_seen, self.confirm = min_seen, confirm
        given = dict(max_cost=max_cost, min_evidence=min_evidence, min_faces=min_faces)
        for key, value in {**DEFAULTS, **SETTINGS[event]}.items():
            setattr(self, key, value if given[key] is None else given[key])
        p = sh.puzzle
        layers = sorted({layer(t) for t in p.turns})
        # Pairs of layers whose turns don't disturb each other (in either order, the same puzzle).
        self.apart = {(a, b) for a in layers for b in layers
                      if a == b or np.array_equal(p.move(a)[p.move(b)], p.move(b)[p.move(a)])}
        self.prepare(parse_scramble(event, scramble), p.turns, sh.cells, [])
        # Later moves done early: for point k, the moves after it that nothing in between
        # disturbs, in every combination. They count as point k (follow.py's `halves`).
        self.halves = {}
        for k in range(len(self.moves)):
            early = [j for j in range(k + 1, len(self.moves))
                     if all((layer(self.moves[j]), layer(self.moves[i])) in self.apart for i in range(k, j))][:MAX_EARLY]
            states = [self.c.apply(self.points[k], [self.moves[j] for j in js])
                      for r in range(1, len(early) + 1) for js in combinations(early, r)]
            on = {q.tobytes() for q in self.points}
            states = [s for s in states if s.tobytes() not in on]
            if states:
                self.halves[k] = np.array(states)

    def look(self, probs: np.ndarray, inside: np.ndarray, covered: float = 0.5) -> dict:
        """One look: the reader's (faces x slots, colours + 1) probabilities and which stickers can be read."""
        if int(inside.reshape(self.sh.n_faces, -1).any(axis=1).sum()) < self.min_faces:
            return self.report(clear=False)
        return super().look(probs, inside, covered)

    def commutes(self, a: str, b: str) -> bool:
        return (layer(a), layer(b)) in self.apart

    def shorten(self, moves: list[str]) -> list[str]:
        """Add up the turns of one layer that meet, through turns that don't disturb them:
        R R -> R', R R' -> nothing, U D U -> D U', u L u' -> L."""
        out: list[list] = []  # [layer, thirds of a turn]
        for m in moves:
            k, q = layer(m), 2 if m.endswith("'") else 1
            j = len(out) - 1
            while j >= 0 and out[j][0] != k and (out[j][0], k) in self.apart:
                j -= 1
            if j >= 0 and out[j][0] == k:
                out[j][1] = (out[j][1] + q) % 3
                if not out[j][1]:
                    del out[j]
            else:
                out.append([k, q])
        new = [k + ("'" if q == 2 else "") for k, q in out]
        return new if new == moves else self.shorten(new)

    def fix(self, m: Mistake) -> dict:
        """The way back, as Mistake.fix, but the moves done since the slip stay when putting the
        slip right doesn't disturb them (`resume` is then past them)."""
        k, moves = m.move, self.moves
        if m.kind == "swapped":  # moves k+1 then k: undo all of it, carry on from move k
            since = [moves[k + 1], moves[k]] + moves[k + 2:k + 2 + m.after]
            return self.way([inverse(x) for x in reversed(since)], [], k)
        start = k if m.kind == "extra" else k + 1  # the first right move done after the slip
        since = moves[start:start + m.after]
        undo = [inverse(m.done)] if m.done else []
        then = [] if m.kind == "extra" else [moves[k]]
        keep = 0
        while keep < len(since) and all(self.commutes(since[keep], t) for t in undo + then):
            keep += 1
        undo = [inverse(x) for x in reversed(since[keep:])] + undo
        if m.kind == "skipped" and not keep:  # everything since undone: carry on from the skipped move
            return self.way(undo, [], k)
        return self.way(undo, then, start + keep if m.kind == "extra" else k + 1 + keep)

    def way(self, undo: list[str], then: list[str], resume: int) -> dict:
        whole = self.shorten(undo + then)
        if undo and self.shorten(undo) + then == whole:
            return {"undo": self.shorten(undo), "then": then, "resume": resume, "merged": False}
        return {"undo": whole, "then": [], "resume": resume, "merged": True}

    def rank(self, m: Mistake, at: int) -> tuple:
        f = self.fix(m)
        return m.move < at, len(f["undo"]) + len(f["then"]), ("skipped", "swapped", "wrong", "extra").index(m.kind), -m.move, m.after

    def meets(self, i: int, t: str) -> bool:
        """An extra turn t before move i reaches a move of its own layer, on either side, through
        moves that don't disturb it: with that move it's a wrong move, a skip or a move done early."""
        k = layer(t)
        for step, j in ((-1, i - 1), (1, i)):
            while 0 <= j < len(self.moves):
                other = layer(self.moves[j])
                if other == k:
                    return True
                if (other, k) not in self.apart:
                    break
                j += step
        return False

    def report(self, clear: bool) -> dict:
        out = super().report(clear)
        out["ahead"] = out.pop("half")  # the best fit has a later move already done
        out["half"] = False
        return out
