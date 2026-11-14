"""The whole check on test renders of a skewb, a pyraminx or an FTO: corners, sticker reading, verdict.

    uv run python scripts/evaluate_poly.py skewb <test renders: images/ and labels/> --corners gt --reader perfect
    uv run python scripts/evaluate_poly.py skewb <test renders: images/ and labels/> --corners ../runs/corners/skewb-c1 --reader ../runs/reader/skewb-r1/best.pt

evaluate_nxn.py for shapes.py's puzzles. For each render (the same fixed camera degradation
as the other evaluations), poly_check judges three claimed scrambles: the true one (should
be "correct"), one with a single realistic slip and one with two (both should be
"mis-scramble").

--corners gt uses the labelled corners instead of a corner model, and --reader perfect the
labelled colours instead of a reader: with both, only the geometry and the check's own
logic are tested, and what's missed is what one look
can't show.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from cubecheck.degrade import degrade
from cubecheck.poly_check import MIN_SIZE, check, readable, single_slips
from cubecheck.poly_reader import cell_labels, load_poly_reader, read_stickers
from cubecheck.shapes import shape

FULL = {"skewb": 11, "pyram": 11, "fto": 30}


def corrupt(moves: list[str], sh, rng: np.random.Generator) -> list[str]:
    """One realistic slip (skipped, wrong turn, extra turn, swapped, stopped short) that changes the puzzle."""
    p = sh.puzzle
    right = p.apply(p.solved, moves)
    while True:
        m = list(moves)
        kind = rng.choice(["skip", "wrong", "extra", "swap", "short"])
        i = int(rng.integers(len(m))) if m else 0
        if not m and kind != "extra":
            continue
        if kind == "skip":
            del m[i]
        elif kind == "wrong":
            m[i] = str(rng.choice([t for t in p.turns if t != m[i]]))
        elif kind == "extra":
            m.insert(i, str(rng.choice(p.turns)))
        elif kind == "swap" and i + 1 < len(m):
            m[i], m[i + 1] = m[i + 1], m[i]
        elif kind == "short":
            m = m[:-1]
        if not np.array_equal(p.apply(p.solved, m), right):
            return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("puzzle", choices=list(FULL))
    ap.add_argument("raw", type=Path, nargs="+")
    ap.add_argument("--corners", default="gt", help="'gt' (the labelled corners) or a folder with finder.pt and refiner.pt")
    ap.add_argument("--reader", default="perfect", help="'perfect' (the labelled colours) or a PolyReader's weights")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--min-find", type=float, default=0.5)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override the check's thresholds, e.g. min_slip_seen=1")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    sh = shape(args.puzzle)
    p = sh.puzzle
    finder = None
    if args.corners != "gt":
        from cubecheck.poly_corners import PolyCornerFinder
        finder = PolyCornerFinder.load(args.puzzle, Path(args.corners), args.device)
    reader = None if args.reader == "perfect" else load_poly_reader(args.reader, args.device)
    rng = np.random.default_rng(11)
    overrides = {k: float(v) for k, v in (kv.split("=") for kv in args.set)}

    print(f"corners: {args.corners}   reader: {args.reader}   {overrides or ''}")
    print(f"{'set':<22}{'renders':>8}  {'true: correct':>13}{'false alarm':>12}{'re-check':>9}  {'1 slip caught':>13}{'missed':>7}{'re-check':>9}  {'2 slips caught':>14}{'missed':>7}"
          f"   correct by scramble: solved / 1-4 / between / full   by faces read")
    for raw in args.raw:
        out = {k: Counter() for k in ("true", "one slip", "two slips")}
        by_band, by_faces = defaultdict(Counter), defaultdict(Counter)
        files = sorted((raw / "labels").glob("*.json"))[: args.limit]
        for f in files:
            s = json.loads(f.read_text())
            lab, moves = s["labels"], s["scramble"].split()
            k = len(moves)
            band = "solved" if k == 0 else "1-4" if k <= 4 else "full" if k >= FULL[args.puzzle] else "between"
            img, _ = degrade(cv2.imread(str(raw / s["image"])), np.random.default_rng([999, int(s["seed"])]))
            view = sh.labellings(lab)[0]
            if finder is None:
                kp = sh.keypoints_of(lab, view)
                kp = kp[:, :2] if (kp[:, 2] > 0).all() else None
            else:
                found = finder.find(img, min_find=args.min_find)
                kp = None if found is None else found[0].astype(np.float32)
            while True:  # two slips that don't cancel out
                two = corrupt(corrupt(moves, sh, rng), sh, rng)
                if not np.array_equal(p.apply(p.solved, two), p.apply(p.solved, moves)):
                    break
            claims = {"true": moves, "one slip": corrupt(moves, sh, rng), "two slips": two}
            if kp is None or sh.size(kp) < MIN_SIZE[args.puzzle]:
                for claim in claims:
                    out[claim]["re-check"] += 1
                by_band[band]["re-check"] += 1
                continue
            inside = readable(kp, sh, img.shape)
            if reader is None:  # the labelled colours, under the labelling the corners came with
                y = cell_labels(lab, view, sh)
                probs = np.zeros((len(y), sh.colours + 1))
                probs[np.arange(len(y)), np.where(y < 0, 0, y)] = 1.0
                inside = inside & (y >= 0)
            else:
                probs = read_stickers(reader, img, kp, args.device)
            faces = int(inside.reshape(sh.n_faces, -1).any(axis=1).sum())
            for claim, ms in claims.items():
                v = check(probs, inside, sh, p.apply(p.solved, ms), single_slips(sh, " ".join(ms)), **overrides)
                out[claim][v.status] += 1
                by_faces[f"{faces} {claim}"][v.status] += 1
                if claim == "true":
                    by_band[band][v.status] += 1
        pct = lambda c, k: 100 * c[k] / max(sum(c.values()), 1)  # noqa: E731
        bands = " / ".join(f"{pct(by_band[b], 'correct'):.0f}%" for b in ("solved", "1-4", "between", "full"))
        t, o, w2 = out["true"], out["one slip"], out["two slips"]
        seen = sorted({k.split()[0] for k in by_faces})
        faces_txt = "; ".join(f"{n} faces ({sum(by_faces[n + ' true'].values())}): correct {pct(by_faces[n + ' true'], 'correct'):.0f}%, false alarm {pct(by_faces[n + ' true'], 'mis-scramble'):.1f}%, 1 slip caught {pct(by_faces[n + ' one slip'], 'mis-scramble'):.0f}%" for n in seen)
        print(f"{raw.parent.name + '/' + raw.name:<22}{len(files):>8}  {pct(t, 'correct'):>12.1f}%{pct(t, 'mis-scramble'):>11.1f}%{pct(t, 're-check'):>8.1f}%"
              f"  {pct(o, 'mis-scramble'):>12.1f}%{pct(o, 'correct'):>6.1f}%{pct(o, 're-check'):>8.1f}%  {pct(w2, 'mis-scramble'):>13.1f}%{pct(w2, 'correct'):>6.1f}%   {bands}   {faces_txt}", flush=True)


if __name__ == "__main__":
    main()
