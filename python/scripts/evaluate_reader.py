"""Compare the sticker reader with the hand-made colour reading, end to end.

    uv run python scripts/evaluate_reader.py ../runs/corners/c2 ../runs/reader/r2/best.pt ../data/raw/test-v1

Every test render gets a fixed random camera degradation, and the corner
model finds the corners, exactly as on the live page. Then:

1. Reading: how many stickers each method gets right, with the reader's raw
   top answer and with the cube rules choosing among its probabilities.
2. Fingers: how often covered stickers are recognised as covered.
3. Honesty: when the reader says it's 90% sure, is it right about 90% of the time?
4. The gap between true and wrong claims, which is where the slip threshold
   goes (verify_probs' min_evidence).
5. Verdicts for the true scramble, one slip and two slips.

Model outputs are cached next to the reader's weights, so re-running only
redoes the cheap parts.
"""

from __future__ import annotations

import argparse
import pickle
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from cubecheck.corners import CornerFinder
from cubecheck.degrade import degrade
from cubecheck.mistakes import corrupt
from cubecheck.reader import COVERED, cell_labels, load_reader, read_stickers
from cubecheck.samples import load_samples
from cubecheck.verify import (
    cube_size, expected_state, inside_image, sample_colours, single_slips, verify, verify_probs,
)

CLAIMS = ("true", "one slip", "two slips")


def collect(args) -> list[dict]:
    cache = args.reader.parent / f"eval_{args.raw.name}_{args.limit}_{args.corners.name}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    corners = CornerFinder.load(str(args.corners / "finder.pt"), str(args.corners / "refiner.pt"), args.device)

    def find(img):
        found = corners.find(img)
        return None if found is None else found[0]
    reader = load_reader(str(args.reader), args.device)
    rng = np.random.default_rng(11)
    out = []
    for s in load_samples(args.raw)[: args.limit]:
        img, _ = degrade(cv2.imread(str(args.raw / s["image"])), np.random.default_rng([999, int(s["seed"])]))
        kp = find(img)
        moves = s["scramble"].split()
        one, _ = corrupt(moves, rng)
        while True:  # two slips that don't cancel out
            two, _ = corrupt(corrupt(moves, rng)[0], rng)
            if not np.array_equal(expected_state(" ".join(two)), expected_state(s["scramble"])):
                break
        row = {"id": s["id"], "claims": {"true": s["scramble"], "one slip": " ".join(one), "two slips": " ".join(two)}}
        if kp is not None:
            row.update(kp=kp, truth=cell_labels(s, kp), hand=sample_colours(img, kp),
                       probs=read_stickers(reader, img, kp, args.device), inside=inside_image(img.shape, kp))
        out.append(row)
    cache.write_bytes(pickle.dumps(out))
    return out


def verdicts(rows: list[dict], judge) -> dict[str, Counter]:
    out = {c: Counter() for c in CLAIMS}
    for r in rows:
        for claim in CLAIMS:
            if "kp" not in r or cube_size(r["kp"]) < 60:
                out[claim]["re-check"] += 1
                continue
            out[claim][judge(r, *r["states"][claim]).status] += 1
    return out


def pct(c: Counter, k: str) -> float:
    return 100 * c[k] / max(sum(c.values()), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corners", type=Path, help="corner model: folder with finder.pt and refiner.pt")
    ap.add_argument("reader", type=Path, help="sticker reader")
    ap.add_argument("raw", type=Path)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    rows = collect(args)
    for r in rows:  # expected cube and every one-slip version, once per claim
        r["states"] = {c: (expected_state(text), single_slips(text)) for c, text in r["claims"].items()}
    seen = [r for r in rows if "kp" in r]
    print(f"{len(rows)} test renders, cube found in {len(seen)}\n")

    # 1. Reading
    methods = {
        "hand-made": lambda r: verify(*r["hand"], expected_state(r["claims"]["true"]), rules="none").observed,
        "hand-made + rules": lambda r: verify(*r["hand"], expected_state(r["claims"]["true"]), rules="unique").observed,
        "reader, top answer": lambda r: "".join("?" if not r["inside"][n] or p.argmax() == COVERED else "URFDLB"[p.argmax()] for n, p in enumerate(r["probs"])),
        "reader + pieces": lambda r: verify_probs(r["probs"], r["inside"], expected_state(r["claims"]["true"]), rules="pieces", max_unsure=99).observed,
        "reader + all rules": lambda r: verify_probs(r["probs"], r["inside"], expected_state(r["claims"]["true"]), rules="unique", max_unsure=99).observed,
    }
    print(f"{'reading':<22}{'stickers right':>15}{'answered':>10}")
    for name, read in methods.items():
        right = answered = total = 0
        for r in seen:
            obs = read(r)
            for n, t in enumerate(r["truth"]):
                if 0 <= t < COVERED:
                    total += 1
                    if obs and obs[n] != "?":
                        answered += 1
                        right += obs[n] == "URFDLB"[t]
        print(f"{name:<22}{100 * right / max(answered, 1):14.1f}%{100 * answered / max(total, 1):9.1f}%")

    # 2. Fingers
    truth = np.concatenate([r["truth"] for r in seen])
    probs = np.concatenate([r["probs"] for r in seen])
    said_covered = probs[:, COVERED] > 0.5
    print(f"\ncovered stickers recognised: {100 * said_covered[truth == COVERED].mean():.1f}% of {(truth == COVERED).sum()}"
          f" | real stickers wrongly called covered: {100 * said_covered[(truth >= 0) & (truth < COVERED)].mean():.2f}%")

    # 3. Honesty (calibration)
    colour = (truth >= 0) & (truth < COVERED)
    top, conf = probs[colour].argmax(1), probs[colour].max(1)
    print("\nhow sure the reader says it is -> how often it's right:")
    for lo, hi in ((0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.99), (0.99, 1.01)):
        m = (conf >= lo) & (conf < hi)
        if m.any():
            print(f"  {lo:.0%}-{min(hi, 1):.0%}: right {100 * (top[m] == truth[colour][m]).mean():5.1f}%  ({m.sum()} stickers)")

    # 4. The gap: how strongly the best single slip beats the claim, for true and wrong claims.
    #    A threshold in the middle of the gap leaves room for real photos being messier.
    print("\nslip evidence (how much better the best slip fits than the claim):")
    for claim in CLAIMS:
        e = np.array([verify_probs(r["probs"], r["inside"], *r["states"][claim], min_evidence=np.inf, min_mismatches=99, max_unsure=99).evidence
                      for r in seen])
        print(f"  {claim:<10} lowest 1% {np.percentile(e, 1):6.1f}   median {np.median(e):6.1f}   highest {e.max():6.1f}")

    # 5. Verdicts, with verify_probs' default thresholds
    print(f"\nverdicts ({len(rows)} renders):")
    for name, judge in (
        ("hand-made", lambda r, st, alts: verify(*r["hand"], st, alternatives=alts)),
        ("reader", lambda r, st, alts: verify_probs(r["probs"], r["inside"], st, alts)),
    ):
        v = verdicts(rows, judge)
        print(f"  {name:<10} true: correct {pct(v['true'], 'correct'):4.1f} / false alarm {pct(v['true'], 'mis-scramble'):4.1f} / re-check {pct(v['true'], 're-check'):4.1f}"
              f" | one slip: caught {pct(v['one slip'], 'mis-scramble'):4.1f} / missed {pct(v['one slip'], 'correct'):4.1f} / re-check {pct(v['one slip'], 're-check'):4.1f}"
              f" | two slips: caught {pct(v['two slips'], 'mis-scramble'):4.1f} / missed {pct(v['two slips'], 'correct'):4.1f}")

if __name__ == "__main__":
    main()
