"""Check a folder of generated samples and summarise what is in it.

    uv run python scripts/check_samples.py ../data/raw/dev

Every label is tested for internal consistency (state matches the scramble,
27 stickers face the camera, stickers sit inside their face, and so on), then
the script prints how the dataset is distributed and writes preview mosaics with
the labels drawn on, so you can look at the data before training on it.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from cubecheck.draw import draw_labels, mosaic
from cubecheck.keypoints import KEYPOINT_NAMES, OCCLUDED, OUTSIDE
from cubecheck.samples import check_sample, load_samples


def pct(n: int, total: int) -> str:
    return f"{n} ({100 * n / max(total, 1):.0f}%)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path, help="dataset folder (with images/ and labels/)")
    ap.add_argument("--previews", type=int, default=24, help="how many samples to draw")
    args = ap.parse_args()

    samples = load_samples(args.raw)
    n = len(samples)
    print(f"{n} samples in {args.raw}\n")

    bad = {}
    for s in samples:
        problems = check_sample(s)
        if problems:
            bad[s["id"]] = problems
    print(f"label checks: {n - len(bad)} passed, {len(bad)} failed")
    for sid, problems in list(bad.items())[:10]:
        print(f"  {sid}: {'; '.join(problems)}")

    def dist(key, values):
        counts = Counter(values)
        print(f"{key:<22}" + ", ".join(f"{k} {pct(v, n)}" for k, v in counts.most_common()))

    print()
    dist("scene", [s["mode"] for s in samples])
    dist("cube type", [s["cube"]["type"] for s in samples])
    dist("finish", [s["cube"]["finish"] for s in samples])
    dist("palette", [s["cube"]["palette"] for s in samples])
    dist("logo on white", [s["cube"]["logo"] for s in samples])
    dist("background", [s["background"]["kind"].split(":")[0] for s in samples])
    def table_kind(s):
        t = s["background"]["table"]
        return "none" if t is None else t if t.startswith("procedural") else "photo"

    dist("table", [table_kind(s) for s in samples])
    dist("image size", [f"{s['width']}x{s['height']}" for s in samples])

    fill = np.array([s["camera"]["fill"] for s in samples])
    kelvin = np.array([s["lighting"]["keyKelvin"] for s in samples])
    print(f"{'cube size':<22}{fill.min():.2f}–{fill.max():.2f} of the short side (median {np.median(fill):.2f})")
    print(f"{'key light':<22}{kelvin.min()}–{kelvin.max()} K (warm < 3500 K: {pct(int((kelvin < 3500).sum()), n)})")

    hidden = [sum(1 for st in s["labels"]["stickers"] if st["facing"] > 0 and not st["visible"]) for s in samples]
    print(f"{'stickers hidden':<22}" + ", ".join(f"{k}: {v}" for k, v in sorted(Counter(min(h, 9) for h in hidden).items())) + "  (of 27; 9 means 9+)")
    kp = np.array([s["labels"]["keypoints"] for s in samples])
    occluded = (kp[:, :, 2] == OCCLUDED).mean(axis=0)
    outside = (kp[:, :, 2] == OUTSIDE).mean(axis=0)
    print(f"{'keypoints hidden':<22}" + ", ".join(f"{name} {100 * o:.0f}%" for name, o in zip(KEYPOINT_NAMES, occluded)))
    if outside.any():
        print(f"{'keypoints off-image':<22}" + ", ".join(f"{name} {100 * o:.0f}%" for name, o in zip(KEYPOINT_NAMES, outside)))

    previews = []
    for s in samples[: args.previews]:
        img = cv2.imread(str(args.raw / s["image"]))
        previews.append(draw_labels(img, s["labels"]))
    if previews:
        out = args.raw / "preview.jpg"
        cv2.imwrite(str(out), mosaic(previews), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"\npreview mosaic → {out}")

    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
