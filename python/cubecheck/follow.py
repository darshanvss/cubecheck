"""Follow-along: where in the scramble the cube is, look by look, and the first wrong move.

The scrambler turns the cube while the camera watches. Each still look gives the sticker
reader's probabilities for the visible stickers; this keeps a score for every point of the
scramble (0 = solved, up to every move done) and updates it with each look, the way a
hidden Markov model does:

1. Before looking: probably the same point or a few moves further on, but a jump
   anywhere is possible (a fast scrambler, another cube, a solved cube for the next one).
2. After looking: each point's score is multiplied by how well its stickers match the
   look, in whichever of the 24 orientations fits it best (as nxn_check.check does).

When no point fits, the likely mistakes near the best one are tried: a move done wrong,
an extra turn, a skipped move or two moves swapped, followed by up to `after` correct moves
(6: docs/follow-along.md says why). A mistake must
fit clearly better (by `min_evidence`, as the check) in two looks in a row before it is
reported, with the way back: the moves to undo, newest first, then the right one. It is
cleared only after `confirm` looks in a row are back on track, so it doesn't flash on and off,
except that a cube along the way back shown is taken at once (Follower.show).
An extra turn after the last move counts too.

The way back is shortened: turns of one axis (U and D, say) don't disturb each other, so
next to each other they add up (D2 then D is D') or cancel (D' U D is U).

Different slips can leave the very same cube (skipping move k, then doing move k+1, is
the cube an extra turn of move k+1 would give). Of those, the story shown is the one that
doesn't put the slip before where the cube was last seen on track, then the shortest way
back, then a skip over a wrong move over an extra one. An extra turn of the layers of a
move next to it (or next to it but for moves of its axis, which it passes through) isn't
tried: with that move it's a wrong move or a skip.

A half turn (R2) is done as two quarter turns, so the cube after one of them (R or R') is
halfway through the move, not a mistake: it counts as the move before, with `half` set.

Looks that show too little (fingers, mid-turn blur) change nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nxn import cube
from .nxn_check import cells, middle_cells

# How far on the scrambler probably is at the next still look: 0, 1, ... 6 moves.
STEP = np.array([0.40, 0.22, 0.12, 0.09, 0.07, 0.05, 0.05])


def quarter(move: str) -> str | None:
    """The quarter turn a half turn is made of: R2 -> R, Rw2' -> Rw; None for a quarter turn."""
    m = move.rstrip("'")
    return m[:-1] if m.endswith("2") else None


def inverse(move: str) -> str:
    """The turn that undoes a move: R -> R', Rw' -> Rw, R2 -> R2."""
    if move.endswith("2"):
        return move
    return move[:-1] if move.endswith("'") else move + "'"


AXIS = {f: a for a, fs in enumerate(("UDy", "RLx", "FBz")) for f in fs}


def turns(move: str) -> tuple[str, int]:
    """A move as (layers, quarter turns clockwise): R2 -> ("R", 2), 3Rw' -> ("3Rw", 3)."""
    m = move.rstrip("'")
    q = 2 if m.endswith("2") else 3 if move.endswith("'") else 1
    return m.rstrip("2"), q


def shorten(moves: list[str]) -> list[str]:
    """Add up the turns of one axis that come together: D2 D -> D', D' U D -> U, R R' -> nothing."""
    out = list(moves)
    while True:
        new, run = [], {}  # run: layers -> quarter turns, for the current axis
        axis = None

        def flush() -> None:
            new.extend(k + {1: "", 2: "2", 3: "'"}[q % 4] for k, q in run.items() if q % 4)
            run.clear()

        for m in out:
            k, q = turns(m)
            a = AXIS[k.lstrip("0123456789")[0]]
            if a != axis:
                flush()
                axis = a
            run[k] = run.get(k, 0) + q
        flush()
        if new == out:  # again, as a run that cancels out brings two runs of one axis together
            return out
        out = new


KINDS = ("skipped", "swapped", "wrong", "extra")  # the order a tie between stories goes


@dataclass
class Mistake:
    kind: str  # "wrong", "extra", "skipped" or "swapped" (moves `move` and `move`+1)
    move: int  # 0-based index of the scramble move where it went wrong
    done: str | None  # what was done instead (wrong), or the extra turn
    after: int  # correct moves done since
    state: np.ndarray

    def fix(self, moves: list[str]) -> dict:
        """The way back: undo these (newest first), then do these, then carry on from `resume`.

        Shortened (see shorten): when that merges the undoing with the right move, it's all
        under "undo" and `merged` is set (D2 done for D: D', not D2 then D).
        """
        k = self.move
        if self.kind == "wrong":
            since = moves[k + 1:k + 1 + self.after]
            undo, then, resume = [inverse(m) for m in reversed(since)] + [inverse(self.done)], [moves[k]], k + 1
        elif self.kind == "swapped":  # moves k+1 then k, then moves k+2..: undo all of it, carry on from move k
            since = [moves[k + 1], moves[k]] + moves[k + 2:k + 2 + self.after]
            undo, then, resume = [inverse(m) for m in reversed(since)], [], k
        elif self.kind == "extra":
            since = moves[k:k + self.after]
            undo, then, resume = [inverse(m) for m in reversed(since)] + [inverse(self.done)], [], k
        else:  # skipped: moves k+1.. were done without move k; undo them and carry on from move k
            since = moves[k + 1:k + 1 + self.after]
            undo, then, resume = [inverse(m) for m in reversed(since)], [], k
        way = shorten(undo + then)
        if shorten(undo) + then == way:
            return {"undo": shorten(undo), "then": then, "resume": resume, "merged": False}
        return {"undo": way, "then": [], "resume": resume, "merged": True}

    def rank(self, moves: list[str], at: int) -> tuple:
        """Smaller is the likelier story for the same cube (`at`: where it was last seen on track)."""
        f = self.fix(moves)
        return self.move < at, len(f["undo"]) + len(f["then"]), KINDS.index(self.kind), -self.move, self.after


class Follower:
    colours = 6  # the reader's answers per sticker: this many colours, then "covered"
    half_margin = 0.0  # how much better "halfway" must fit than the point itself to be reported

    def __init__(self, scramble: str, n: int, after: int = 6, before: int = 2, jump: float = 0.02, temper: float = 3.0,
                 min_seen: float = 0.3, max_cost: float = 0.6, min_evidence: float = 8.0, confirm: int = 2) -> None:
        self.n, self.c = n, cube(n)
        self.after, self.before, self.jump, self.temper = after, before, jump, temper
        self.min_seen, self.max_cost, self.min_evidence, self.confirm = min_seen, max_cost, min_evidence, confirm
        self.prepare(self.c.parse(scramble), self.c.event_turns, cells(n), middle_cells(n))

    def prepare(self, moves: list[str], turn_list: list[str], cells: np.ndarray, middles: list[int]) -> None:
        """The scramble's points and a fresh start. The puzzle itself is self.c (its solved state
        and move()), `turn_list` every turn a slip can be, `cells` the sticker place behind each
        sticker read, for every way round the puzzle can be seen (poly_follow.py: other puzzles)."""
        self.moves, self.turn_list = moves, turn_list
        states = [self.c.solved]
        for m in self.moves:
            states.append(states[-1][self.c.move(m)])
        self.points = np.array(states)  # (L+1, 6N²): the cube after 0..L moves
        # Halfway through each half turn, either way round: move i -> (2, 6N²).
        self.halves = {i: np.array([states[i][self.c.move(q)], states[i][self.c.move(q + "'")]])
                       for i, m in enumerate(self.moves) if (q := quarter(m))}
        L = len(self.moves)
        self.belief = np.full(L + 1, 0.1 / (L + 1))
        self.belief[0] += 0.9  # a scramble starts from a solved cube
        self.cells, self.middles = cells, middles
        self.streak, self.mistake, self.looks, self.back, self.half = 0, None, 0, 0, False
        self.path: set[bytes] = set()  # the cubes along the way back shown (see show)

    # --- the two HMM steps -------------------------------------------------

    def predict(self) -> np.ndarray:
        """Spread the scores the way scrambling goes: a few moves on, sometimes a jump anywhere."""
        L1 = len(self.belief)
        out = np.zeros(L1)
        for d, w in enumerate(STEP):
            out[d:] += w * self.belief[:L1 - d]
            out[-1] += w * self.belief[L1 - d:].sum() if d else 0  # can't go past the end
        out = (1 - self.jump) * out / out.sum() + self.jump / L1
        return out

    def fits(self, probs: np.ndarray, unknown: np.ndarray, states: np.ndarray, cap: float = 5.0) -> np.ndarray:
        """Total cost of the seen stickers for each state, in its best orientation: (K,)."""
        rows = np.flatnonzero(~unknown)
        colour = probs[:, :self.colours] / np.maximum(probs[:, :self.colours].sum(axis=1, keepdims=True), 1e-9)
        cost = np.minimum(-np.log(np.maximum(colour, 1e-9)), cap)[rows].astype(np.float32)
        pick = np.arange(len(rows))
        orients = range(len(self.cells))
        middles = [k for k in self.middles if not unknown[k]]
        if middles:  # odd cubes: the middle stickers never move, so they say which way round the cube is
            full = np.minimum(-np.log(np.maximum(colour, 1e-9)), cap)
            centre = np.array([full[middles, self.c.solved[cl][middles]].sum() for cl in self.cells])
            orients = np.flatnonzero(centre <= centre.min() + 1.0)
        best = np.full(len(states), np.inf, np.float32)
        for o in orients:
            best = np.minimum(best, cost[pick, states[:, self.cells[o][rows]]].sum(axis=1))
        return best

    # --- one look -----------------------------------------------------------

    def look(self, probs: np.ndarray, inside: np.ndarray, covered: float = 0.5) -> dict:
        """Update with one look (the reader's (3N², 7) probabilities and which stickers are in view)."""
        unknown = ~inside | (probs[:, self.colours] > covered)
        seen = int((~unknown).sum())
        if seen < self.min_seen * len(probs):
            return self.report(clear=False)
        f = self.fits(probs, unknown, self.points)
        half = np.full(len(f), np.inf, np.float32)  # halfway through move k's half turn, as point k
        if self.halves:  # each point's own few (two for a half turn), the best of them
            idx = list(self.halves)
            g = self.fits(probs, unknown, np.concatenate([self.halves[i] for i in idx]))
            half[idx] = np.minimum.reduceat(g, np.cumsum([0] + [len(self.halves[i]) for i in idx[:-1]]))
        plain, f = f, np.minimum(f, half)
        # A mistake first, judged from where the cube was before this look: a look that a
        # mistake explains clearly better than any point of the scramble doesn't move the position.
        # Near where the cube was believed to be, and near the point this look fits best
        # (a jump, or a tracker that started late).
        at = int(self.belief.argmax())
        ks = {at, int(f.argmin())} | ({self.mistake.move} if self.mistake is not None else set())
        cands = [m for k in sorted(ks) for m in self.mistakes_near(k, at)]
        if cands:
            g = self.fits(probs, unknown, np.array([m.state for m in cands]))
            # Of the mistakes that fit about as well as the best, the likeliest story.
            j = min(np.flatnonzero(g <= g.min() + 1.0), key=lambda i: self.rank(cands[i], at))
            if f.min() - g[j] > self.min_evidence and g[j] / seen <= self.max_cost:
                self.streak += 1
                self.back = 0
                if self.streak >= self.confirm or cands[j].state.tobytes() in self.path:
                    self.show(cands[j])
                return self.report(clear=True)
        self.streak = 0
        if f.min() / seen > self.max_cost:  # nothing fits: mid-turn, glare, a mistake further back, another puzzle
            return self.report(clear=False)
        if self.mistake is not None:  # back on track (undone, or never wrong), once it's held
            self.back += 1
            p = int(f.argmin())
            if self.points[p].tobytes() in self.path:  # where the way back shown leads, or passes: done right
                self.belief = np.full(len(self.belief), 0.1 / len(self.belief))
                self.belief[p] += 0.9
                self.show(None)
            elif self.back >= self.confirm:
                self.show(None)
        self.looks += 1
        prior = self.predict()
        like = np.exp(-(f - f.min()) / self.temper)
        self.belief = prior * like / (prior * like).sum()
        k = int(self.belief.argmax())
        self.half = bool(k in self.halves and half[k] <= plain[k] - self.half_margin)
        return self.report(clear=True)

    def show(self, m: Mistake | None) -> None:
        """Show this mistake (None: back on track), and note the cubes along its way back.

        A cube along the way back is what the scrambler is expected to show next, so it's
        taken at the first look (the next part of the way back, or back on track), where
        any other change needs `confirm` looks in a row. Without that, a noisy look while
        undoing kept the old way back on screen, with a move in it that was already undone.
        """
        self.mistake, self.back, self.path = m, 0, set()
        if m is None:
            return
        st = m.state
        f = self.fix(m)
        for move in f["undo"] + f["then"]:
            if (q := quarter(move)):  # halfway through a half turn of the way back
                self.path |= {st[self.c.move(q)].tobytes(), st[self.c.move(q + "'")].tobytes()}
            st = st[self.c.move(move)]
            self.path.add(st.tobytes())

    def mistakes_near(self, k: int, at: int | None = None) -> list[Mistake]:
        """One slip at moves k-before .. k (0-based), then up to `after` correct moves; an extra
        turn after the last move too. A cube two slips explain is kept as the likelier story
        (Mistake.rank, `at` being where the cube was last seen on track: k if not given)."""
        L, out = len(self.moves), {}
        at = k if at is None else at
        scramble = {p.tobytes() for p in self.points} | {h.tobytes() for hs in self.halves.values() for h in hs}
        for i in range(max(0, k - self.before), min(L, k) + 1):
            base = self.points[i]
            tails = {"wrong": self.moves[i + 1:], "extra": self.moves[i:], "skipped": self.moves[i + 1:]}
            if i + 1 < L:  # moves i and i+1 done the other way round (nothing to see when they commute)
                tails["swapped"] = self.moves[i + 2:]
            if i == L:  # after the last move: only an extra turn
                tails = {"extra": []}
            for kind, tail in tails.items():
                if kind in ("skipped", "swapped"):
                    firsts = [None]
                elif kind == "wrong":
                    firsts = [t for t in self.turn_list if t != self.moves[i]]
                else:  # not the layers of a neighbouring move: R R2 is a wrong R, R' R a skip
                    firsts = [t for t in self.turn_list if not self.meets(i, t)]
                for t in firsts:
                    if kind == "swapped":
                        st = base[self.c.move(self.moves[i + 1])][self.c.move(self.moves[i])]
                    else:
                        st = base if t is None else base[self.c.move(t)]
                    for a in range(0, self.after + 1):
                        if a:
                            if a > len(tail):
                                break
                            st = st[self.c.move(tail[a - 1])]
                        if kind == "skipped" and a == 0:
                            continue
                        if kind == "extra" and a == 0 and i < L:  # until move i is done too, it's move i done wrong
                            continue
                        key = st.tobytes()
                        if key in scramble:
                            continue
                        m = Mistake(kind, i, t, a, st)
                        if key not in out or self.rank(m, at) < self.rank(out[key], at):
                            out[key] = m
        return list(out.values())

    def fix(self, m: Mistake) -> dict:
        """The way back from a mistake (Mistake.fix)."""
        return m.fix(self.moves)

    def rank(self, m: Mistake, at: int) -> tuple:
        """Smaller is the likelier story for the same cube (Mistake.rank)."""
        return m.rank(self.moves, at)

    def meets(self, i: int, t: str) -> bool:
        """An extra turn t before move i that is really a wrong move or a skip (see neighbours)."""
        return turns(t)[0] in self.neighbours(i, t)

    def neighbours(self, i: int, t: str) -> set[str]:
        """The layers of the moves an extra turn t before move i meets: the moves on either side,
        and on past those of t's axis (B F2 B' is F2: the B' meets the B)."""
        axis = AXIS[turns(t)[0].lstrip("0123456789")[0]]
        out = set()
        for step, j in ((-1, i - 1), (1, i)):
            while 0 <= j < len(self.moves):
                k = turns(self.moves[j])[0]
                if AXIS[k.lstrip("0123456789")[0]] != axis:
                    break
                out.add(k)
                j += step
        return out

    def report(self, clear: bool) -> dict:
        k = int(self.belief.argmax())
        out = {"position": k, "total": len(self.moves), "sure": round(float(self.belief[k]), 3), "clear": clear,
               "half": self.half and self.mistake is None}
        if self.mistake is not None:
            m = self.mistake
            out.update(status="mistake", mistake={"kind": m.kind, "move": m.move, "done": m.done, "after": m.after}, **self.fix(m))
        elif not clear and self.looks:
            out["status"] = "unclear"
        else:
            out["status"] = "done" if k == len(self.moves) and self.belief[k] > 0.8 else "following"
        return out
