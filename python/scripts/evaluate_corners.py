"""How precisely the corner model finds the 7 corners, on degraded test renders.

    uv run python scripts/evaluate_corners.py ../runs/corners/c2 ../data/raw/test-v1 ../data/raw/test-easy-v1

Error is the distance from each corner to the label, as a share of the cube's
edge. The model may start the corner order at any of the three faces (the check
tries all three), so each render is scored under the best of the 3 rotations of
the labels. Also reports how long one frame takes (full search, no tracking).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from cubecheck.corners import CornerFinder
from cubecheck.degrade import degrade
from cubecheck.samples import load_samples
from cubecheck.verify import cube_size

ROTATIONS = [[(i + 2 * r) % 6 for i in range(6)] + [6] for r in range(3)]


def score(pred: np.ndarray, gt: np.ndarray, ok: np.ndarray) -> np.ndarray:
    errs = [np.linalg.norm(pred - gt[rot], axis=1)[ok[rot]] for rot in ROTATIONS]
    return min(errs, key=np.mean) / cube_size(gt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corners", type=Path, help="folder with finder.pt and refiner.pt")
    ap.add_argument("raw", type=Path, nargs="+")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    finder = CornerFinder.load(str(args.corners / "finder.pt"), str(args.corners / "refiner.pt"), args.device)

    for raw in args.raw:
        samples = [s for s in load_samples(raw) if all(k[2] > 0 for k in s["labels"]["keypoints"])]
        errs, secs = [], 0.0
        for s in samples:
            img, _ = degrade(cv2.imread(str(raw / s["image"])), np.random.default_rng([999, int(s["seed"])]))
            gt = np.array(s["labels"]["keypoints"], float)
            t0 = time.perf_counter()
            found = finder.find(img)
            secs += time.perf_counter() - t0
            if found is not None:
                errs.append(score(found[0], gt[:, :2], gt[:, 2] > 0))
        e = np.concatenate(errs) * 100
        worst = np.array([x.max() for x in errs]) * 100
        print(f"{raw.name}: found {len(errs)}/{len(samples)}  median {np.median(e):.2f}%  p95 {np.percentile(e, 95):.2f}%"
              f"  renders with a corner off >5%: {np.mean(worst > 5) * 100:.1f}%  |  {1000 * secs / len(samples):.0f} ms per frame")


if __name__ == "__main__":
    main()
