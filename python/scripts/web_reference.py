"""Python's answers on a few test renders, for checking the browser version against.

    uv run python scripts/web_reference.py ../runs/corners/c7 ../runs/reader/r4/best.pt \
        ../data/raw/test-v1 ../data/raw/test-mixed-{222,444,555,666,777} --out ../web/public/check

Copies the renders and writes reference.json: for each one, the cube size, the
corners the finder and the refiner found, the sticker reader's 3N² x 7
probabilities, which stickers can be read, and the verdict for the true scramble
and for a version with one move slipped (with how many one-slip cubes it has).
3x3 is checked with verify_probs (the cube rules), other sizes with nxn_check.
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

from cubecheck import nxn_check
from cubecheck.corners import CornerFinder
from cubecheck.nxn import cube
from cubecheck.reader import GridReader, load_reader, read_grid
from cubecheck.samples import load_samples
from cubecheck.verify import single_slips, verify_probs


def slipped(moves: list[str], n: int, rng: np.random.Generator) -> str:
    """The scramble with one move done as another of the size's turns."""
    k = int(rng.integers(len(moves)))
    turns = [t for t in cube(n).event_turns if t != moves[k]]
    return " ".join(moves[:k] + [turns[int(rng.integers(len(turns)))]] + moves[k + 1:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corners", type=Path)
    ap.add_argument("reader", type=Path)
    ap.add_argument("raw", type=Path, nargs="+", help="test render folders (the first gets --n renders, the others --per-size)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--per-size", type=int, default=4)
    ap.add_argument("--min-find", type=float, default=0.5)
    ap.add_argument("--zoom", type=float, default=2.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    finder = CornerFinder.load(str(args.corners / "finder.pt"), str(args.corners / "refiner.pt"), "cpu")
    reader = load_reader(str(args.reader), "cpu")
    assert isinstance(reader, GridReader), "the any-size reader (r3 and later)"

    rows, rng = [], np.random.default_rng(5)
    for i, raw in enumerate(args.raw):
        for s in load_samples(raw)[: args.n if i == 0 else args.per_size]:
            n = s.get("cube", {}).get("size", 3)
            name = f"{n}x{n}-{Path(s['image']).name}"
            shutil.copy(raw / s["image"], args.out / name)
            img = cv2.imread(str(args.out / name))
            steps: list[dict] = []
            found = finder.find(img, steps=steps, min_find=args.min_find, zoom=args.zoom)
            row = {"image": name, "size": n, "width": img.shape[1], "height": img.shape[0], "scramble": s["scramble"], "finder": np.round(steps[0]["corners"], 3).tolist(), "corners": None}
            if found is not None:
                kp = found[0].astype(np.float32)
                row["corners"] = np.round(kp, 3).tolist()
                row["probs"] = np.round(read_grid(reader, img, kp, n), 5).tolist()
                inside = nxn_check.readable(np.array(row["corners"], np.float32), n, img.shape)
                row["inside"] = inside.tolist()
                row["verdicts"] = {}
                # Verdicts from the rounded probabilities and corners, exactly what the TypeScript test gets.
                for kind, claim in (("true", s["scramble"]), ("slip", slipped(s["scramble"].split(), n, rng))):
                    if n == 3:
                        alts = single_slips(claim)
                        v = verify_probs(np.array(row["probs"]), inside, cube(3).after(claim), alts)
                    else:
                        alts = nxn_check.single_slips(claim, n)
                        v = nxn_check.check(np.array(row["probs"]), inside, n, cube(n).after(claim), alts)
                    row["verdicts"][kind] = {"scramble": claim, "status": v.status, "observed": v.observed, "mismatches": v.mismatches,
                                             "hypothesis": v.hypothesis, "slip": v.slip, "evidence": v.evidence, "slips": len(alts.states)}
            rows.append(row)
    (args.out / "reference.json").write_text(json.dumps(rows))
    print(f"{len(rows)} renders, cube found in {sum(r['corners'] is not None for r in rows)} -> {args.out}")


if __name__ == "__main__":
    main()
