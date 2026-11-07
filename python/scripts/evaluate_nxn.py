"""The whole check on test renders of every cube size: corners, sticker reading, verdict.

    uv run python scripts/evaluate_nxn.py ../runs/corners/c7 ../runs/reader/r4/best.pt \
        ../data/raw/test-mixed-{222,333,444,555,666,777} --limit 300

For each render (with the same fixed camera degradation as the other evaluations),
the corner model finds the cube, the size-aware reader reads its 3N² stickers, and
the check judges three claimed scrambles (on a 3x3 verify_probs, with the cube rules, as
the website does; other sizes nxn_check): the true one (should be "correct"), one
with a single realistic slip and one with two (both should be "mis-scramble").
--gt-corners uses the labelled corners instead, to separate reading from finding.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from cubecheck.corners import CornerFinder
from cubecheck.degrade import degrade
from cubecheck.nxn import EVENT_LENGTH, cube
from cubecheck.nxn_check import PX_PER_STICKER, check, readable, single_slips
from cubecheck.reader import load_reader, read_grid
from cubecheck.samples import load_samples
from cubecheck.verify import cube_size, verify_probs
from cubecheck.verify import single_slips as single_slips_3x3


def corrupt(moves: list[str], n: int, rng: np.random.Generator) -> list[str]:
    """One realistic slip (skipped, wrong turn, extra turn, swapped, stopped short) that changes the cube."""
    c = cube(n)
    right = c.apply(c.solved, moves)
    while True:
        m = list(moves)
        kind = rng.choice(["skip", "wrong", "extra", "swap", "short"])
        i = int(rng.integers(len(m))) if m else 0
        if not m and kind != "extra":
            continue
        if kind == "skip":
            del m[i]
        elif kind == "wrong":
            m[i] = str(rng.choice([t for t in c.event_turns if t != m[i]]))
        elif kind == "extra":
            m.insert(i, str(rng.choice(c.event_turns)))
        elif kind == "swap" and i + 1 < len(m):
            m[i], m[i + 1] = m[i + 1], m[i]
        elif kind == "short":
            m = m[:-1]
        if not np.array_equal(c.apply(c.solved, m), right):
            return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corners", type=Path, help="folder with finder.pt and refiner.pt")
    ap.add_argument("reader", type=Path, help="size-aware reader (GridReader) weights")
    ap.add_argument("raw", type=Path, nargs="+")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--gt-corners", action="store_true")
    ap.add_argument("--px-per-sticker", type=float, help="smaller cubes get re-check (default: nxn_check.PX_PER_STICKER for the size)")
    ap.add_argument("--min-find", type=float, default=0.5, help="the finder's cube-present bar on the whole frame (the website's)")
    ap.add_argument("--zoom", type=float, default=2.0, help="the finder's second look, this much closer, when it's unsure (the website's)")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    finder = CornerFinder.load(str(args.corners / "finder.pt"), str(args.corners / "refiner.pt"), args.device)
    reader = load_reader(str(args.reader), args.device)
    rng = np.random.default_rng(11)

    print(f"{'set':<20}{'renders':>8}  {'true: correct':>13}{'false alarm':>12}{'re-check':>9}  {'1 slip caught':>13}{'missed':>7}  {'2 slips caught':>14}{'missed':>7}"
          f"   correct by scramble: solved / 1-4 / between / full   read from 2 faces")
    for raw in args.raw:
        out = {k: Counter() for k in ("true", "one slip", "two slips")}
        by_band = defaultdict(Counter)
        faces_read = Counter()  # renders judged from 3 faces, or from 2 (the third a sliver or hidden)
        samples = load_samples(raw)[: args.limit]
        for s in samples:
            n = s.get("cube", {}).get("size", 3)
            moves = s["scramble"].split()
            k = len(moves)
            band = "solved" if k == 0 else "1-4" if k <= 4 else "full" if k >= EVENT_LENGTH[n] else "between"
            img, _ = degrade(cv2.imread(str(raw / s["image"])), np.random.default_rng([999, int(s["seed"])]))
            if args.gt_corners:
                kp = np.array(s["labels"]["keypoints"], np.float32)[:, :2] if all(p[2] > 0 for p in s["labels"]["keypoints"]) else None
            else:
                found = finder.find(img, min_find=args.min_find, zoom=args.zoom)
                kp = None if found is None else found[0].astype(np.float32)
            while True:  # two slips that don't cancel out
                two = corrupt(corrupt(moves, n, rng), n, rng)
                if not np.array_equal(cube(n).apply(cube(n).solved, two), cube(n).apply(cube(n).solved, moves)):
                    break
            claims = {"true": moves, "one slip": corrupt(moves, n, rng), "two slips": two}
            smallest = args.px_per_sticker * n if args.px_per_sticker else 60 if n == 3 else PX_PER_STICKER[n] * n  # as the website
            if kp is None or cube_size(kp) < smallest:
                for claim in claims:
                    out[claim]["re-check"] += 1
                by_band[band]["re-check"] += 1
                continue
            probs = read_grid(reader, img, kp, n, args.device)
            inside = readable(kp, n, img.shape)
            faces_read[int(inside.reshape(3, -1).any(axis=1).sum())] += 1
            for claim, ms in claims.items():
                text = " ".join(ms)
                state = cube(n).apply(cube(n).solved, ms)
                if n == 3:
                    v = verify_probs(probs, inside, state, single_slips_3x3(text))
                else:
                    v = check(probs, inside, n, state, single_slips(text, n))
                out[claim][v.status] += 1
                if claim == "true":
                    by_band[band][v.status] += 1
        pct = lambda c, k: 100 * c[k] / max(sum(c.values()), 1)
        bands = " / ".join(f"{pct(by_band[b], 'correct'):.0f}%" for b in ("solved", "1-4", "between", "full"))
        t, o, w2 = out["true"], out["one slip"], out["two slips"]
        print(f"{raw.parent.name + '/' + raw.name:<20}{len(samples):>8}  {pct(t, 'correct'):>12.1f}%{pct(t, 'mis-scramble'):>11.1f}%{pct(t, 're-check'):>8.1f}%"
              f"  {pct(o, 'mis-scramble'):>12.1f}%{pct(o, 'correct'):>6.1f}%  {pct(w2, 'mis-scramble'):>13.1f}%{pct(w2, 'correct'):>6.1f}%   {bands}   {pct(faces_read, 2):>5.1f}%", flush=True)


if __name__ == "__main__":
    main()
