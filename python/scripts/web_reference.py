"""Python's answers on a few test renders, for checking the browser version against.

    uv run python scripts/web_reference.py ../runs/corners/c2 ../runs/reader/r2/best.pt ../data/raw/test-v1 \
        --out ../web/public/check

Copies the renders and writes reference.json: for each one, the corners the
finder and the refiner found, the sticker reader's 27 x 7 probabilities, and
the verdict for the true scramble and for a version with one move slipped.
The website's /test page runs the ONNX models in the browser on the same
renders and compares; web/tests/check.test.ts checks the TypeScript verdicts.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from cubecheck.corners import CornerFinder
from cubecheck.mistakes import corrupt
from cubecheck.reader import load_reader, read_stickers
from cubecheck.samples import load_samples
from cubecheck.verify import expected_state, inside_image, single_slips, verify_probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corners", type=Path)
    ap.add_argument("reader", type=Path)
    ap.add_argument("raw", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    finder = CornerFinder.load(str(args.corners / "finder.pt"), str(args.corners / "refiner.pt"), "cpu")
    reader = load_reader(str(args.reader), "cpu")

    rows, rng = [], np.random.default_rng(5)
    for s in load_samples(args.raw)[: args.n]:
        name = Path(s["image"]).name
        shutil.copy(args.raw / s["image"], args.out / name)
        img = cv2.imread(str(args.out / name))
        steps: list[dict] = []
        found = finder.find(img, steps=steps)
        row = {"image": name, "scramble": s["scramble"], "finder": np.round(steps[0]["corners"], 3).tolist(), "corners": None}
        if found is not None:
            row["corners"] = np.round(found[0], 3).tolist()
            probs = read_stickers(reader, img, found[0])
            row["probs"] = np.round(probs, 5).tolist()
            inside = inside_image(img.shape, found[0])
            row["inside"] = inside.tolist()
            row["verdicts"] = {}
            # Verdicts from the rounded probabilities, exactly what the TypeScript test gets.
            for kind, claim in (("true", s["scramble"]), ("slip", " ".join(corrupt(s["scramble"].split(), rng)[0]))):
                v = verify_probs(np.array(row["probs"]), inside, expected_state(claim), single_slips(claim))
                row["verdicts"][kind] = {"scramble": claim, "status": v.status, "observed": v.observed,
                                         "mismatches": v.mismatches, "hypothesis": v.hypothesis, "slip": v.slip}
        rows.append(row)
    (args.out / "reference.json").write_text(json.dumps(rows))
    print(f"{len(rows)} renders, cube found in {sum(r['corners'] is not None for r in rows)} -> {args.out}")


if __name__ == "__main__":
    main()
