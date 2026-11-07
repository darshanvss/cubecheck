"""Python's follow-along answers on simulated looks, for checking the browser version against.

    uv run python scripts/follow_reference.py --out ../web/tests/fixtures/follow.json

Each run follows a random scramble to a random point (a look every 1-2 moves), makes one
slip there (a move done wrong, an extra turn, a skipped move or two moves swapped), does
0-3 more moves, takes looks until the slip is shown, undoes it the way shown one turn at a
time, then carries on to the end. The looks are the reader's probabilities made from the
true cube (tests/test_follow.py's look_at: 15% of stickers covered, 3% misread, a random
orientation each time). reference.json gets every look and Follower's report after it;
web/tests/follow.test.ts feeds the same looks to the TypeScript and compares.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_follow import look_at  # noqa: E402

from cubecheck.follow import Follower, turns  # noqa: E402
from cubecheck.nxn import cube, random_scramble  # noqa: E402

KEEP = ("status", "position", "half", "mistake", "undo", "then", "resume", "merged")


def slip(c, moves: list[str], k: int, rng: np.random.Generator) -> tuple[str, list[str]]:
    """One slip at move k: its kind, and the moves done up to and including it."""
    kind = str(rng.choice(["wrong", "extra", "skipped", "swapped"]))
    if kind == "swapped":
        return kind, moves[:k] + [moves[k + 1], moves[k]]
    if kind == "wrong":
        return kind, moves[:k] + [str(rng.choice([t for t in c.event_turns if t != moves[k]]))]
    if kind == "extra":
        near = {turns(m)[0] for m in moves[max(0, k - 1):k + 1]}
        return kind, moves[:k] + [str(rng.choice([t for t in c.event_turns if turns(t)[0] not in near])), moves[k]]
    return kind, moves[:k]


def run(n: int, rng: np.random.Generator) -> dict:
    c = cube(n)
    moves = random_scramble(n, rng)
    f = Follower(" ".join(moves), n)
    looks = []

    def look(state: np.ndarray) -> dict:
        probs, inside = look_at(state, n, rng)
        probs = np.round(probs, 4)
        r = f.look(probs, inside)
        looks.append({"probs": probs.flatten().tolist(), "report": {k: r[k] for k in KEEP if k in r}})
        return r

    k, j = int(rng.integers(2, len(moves) - 8)), 0
    while j < k:
        j = min(k, j + int(rng.integers(1, 3)))
        look(c.apply(c.solved, moves[:j]))
    kind, seq = slip(c, moves, k, rng)
    nxt = {"wrong": k + 1, "extra": k + 1, "swapped": k + 2, "skipped": k + 1}[kind]
    seq = seq + moves[nxt:nxt + int(rng.integers(kind == "skipped", 4))]
    state = c.apply(c.solved, seq)
    r = {}
    for _ in range(4):
        r = look(state)
        if r["status"] == "mistake":
            break
    resume = len(seq) if r["status"] != "mistake" else r["resume"]
    if r["status"] == "mistake":
        for m in r["undo"] + r["then"]:
            state = c.apply(state, [m])
            look(state)
    for m in moves[resume:]:
        state = c.apply(state, [m])
        if rng.random() < 0.6:
            look(state)
    return {"size": n, "scramble": " ".join(moves), "slip": kind, "at": k, "looks": looks}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runs", type=int, default=3, help="runs per size")
    ap.add_argument("--sizes", type=int, nargs="+", default=[2, 3, 4, 5])
    args = ap.parse_args()
    runs = [run(n, np.random.default_rng([n, i])) for n in args.sizes for i in range(args.runs)]
    args.out.write_text(json.dumps(runs, separators=(",", ":")))
    shown = sum(any(lk["report"]["status"] == "mistake" for lk in r["looks"]) for r in runs)
    print(f"{len(runs)} runs, {sum(len(r['looks']) for r in runs)} looks, a mistake shown in {shown} -> {args.out}")


if __name__ == "__main__":
    main()
