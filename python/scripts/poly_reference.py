"""Python's answers for skewb, pyraminx and FTO, for checking the browser version against (experimental).

    uv run python scripts/poly_reference.py skewb=../data/poly/skewb/test pyram=... fto=... \
        --out ../web/public/check --fixtures ../web/tests/fixtures

For a few test renders per puzzle (folders of images/ and labels/, as the training renders):

- poly/<event>-<image> and poly-reference.json in --out: the render, the corners the finder
  and the refiner found, the reader's probabilities, which stickers can be read, and the
  verdict for the true scramble and for one with a move slipped. The /test page runs the
  ONNX models in the browser on the same renders and compares.
- poly-reference.json in --fixtures: the same numbers without the pictures, plus the pose
  fit (shapes.Shape.fit) and sticker centres for those corners and for jittered copies of
  them, which web/tests/poly.test.ts checks the TypeScript against.
- poly-follow.json in --fixtures: whole scrambles followed on simulated looks (with a slip
  and the way back), every look and PolyFollower's report after it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from cubecheck import poly_check
from cubecheck.poly_corners import PolyCornerFinder
from cubecheck.poly_follow import PolyFollower
from cubecheck.poly_reader import load_poly_reader, read_stickers
from cubecheck.shapes import EVENTS, MODELS, random_scramble, shape

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_poly_follow import look_at  # noqa: E402

RUNS = Path(__file__).resolve().parents[2] / "runs"
KEEP = ("status", "position", "ahead", "mistake", "undo", "then", "resume", "merged")


def slipped(moves: list[str], turns: list[str], rng: np.random.Generator) -> str:
    """The scramble with one move done as another turn (an empty scramble: one extra turn)."""
    if not moves:
        return turns[int(rng.integers(len(turns)))]
    k = int(rng.integers(len(moves)))
    other = [t for t in turns if t != moves[k]]
    return " ".join(moves[:k] + [other[int(rng.integers(len(other)))]] + moves[k + 1:])


def geometry(sh, kp: np.ndarray, img_shape) -> dict:
    pose, err = sh.fit(kp, img_shape)
    return {"corners": np.round(kp, 3).tolist(), "fit": round(float(err), 6), "pose": np.round(pose, 6).tolist(),
            "centres": np.round(sh.cell_centres(kp, img_shape), 3).tolist(),
            "inside": poly_check.readable(kp, sh, img_shape).tolist(),
            "wellFormed": [bool(sh.well_formed(kp, f)) for f in range(sh.n_faces + 1)]}


def follow_run(event: str, rng: np.random.Generator) -> dict:
    """A scramble followed with one slip: as follow_reference.py, with this puzzle's moves."""
    sh = shape(event)
    p = sh.puzzle
    moves = random_scramble(event, rng)
    f = PolyFollower(" ".join(moves), event)
    looks = []

    def look(state: np.ndarray, faces=None) -> dict:
        probs, inside = look_at(state, sh, rng, faces=faces)
        probs = np.round(probs, 4)
        out = f.look(probs, inside)
        looks.append({"probs": probs.flatten().tolist(), "inside": inside.tolist(), "report": {k: out[k] for k in KEEP if k in out}})
        return out

    k, j = int(rng.integers(2, max(3, len(moves) - 4))), 0
    while j < k:
        j = min(k, j + int(rng.integers(1, 3)))
        look(p.apply(p.solved, moves[:j]))
    kind = str(rng.choice(["wrong", "extra", "skipped", "swapped"]))
    if kind == "swapped" and k + 1 < len(moves):
        seq, nxt = moves[:k] + [moves[k + 1], moves[k]], k + 2
    elif kind == "skipped":
        seq, nxt = moves[:k], k + 1
    else:
        kind = "wrong" if kind == "swapped" else kind
        t = str(rng.choice([t for t in p.turns if t.rstrip("'") != moves[k].rstrip("'")]))
        seq, nxt = (moves[:k] + [t] if kind == "wrong" else moves[:k] + [t, moves[k]]), k + 1
    seq = seq + moves[nxt:nxt + int(rng.integers(kind == "skipped", 3))]
    state = p.apply(p.solved, seq)
    out = {}
    for _ in range(4):
        out = look(state)
        if out["status"] == "mistake":
            break
    resume = len(seq) if out["status"] != "mistake" else out["resume"]
    if out["status"] == "mistake":
        for m in out["undo"] + out["then"]:
            state = p.apply(state, [m])
            look(state)
    look(state, faces=1)  # a look at one face only
    for m in moves[resume:]:
        state = p.apply(state, [m])
        if rng.random() < 0.6:
            look(state)
    return {"event": event, "scramble": " ".join(moves), "slip": kind, "at": k, "looks": looks}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+", help="event=folder of test renders (images/, labels/); several per event allowed")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fixtures", type=Path, required=True)
    ap.add_argument("--n", type=int, default=6, help="renders per folder")
    ap.add_argument("--runs", type=int, default=3, help="follow-along runs per puzzle")
    ap.add_argument("--min-find", type=float, default=0.5)
    ap.add_argument("--zoom", type=float, default=2.0)
    args = ap.parse_args()
    (args.out / "poly").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    rows, loaded = [], {}
    for item in args.sets:
        event, folder = item.split("=", 1)
        folder = Path(folder)
        sh = shape(event)
        if event not in loaded:
            corners, reader = MODELS[event]
            loaded[event] = (PolyCornerFinder.load(event, RUNS / "corners" / corners), load_poly_reader(str(RUNS / "reader" / reader / "best.pt")))
        finder, reader = loaded[event]
        for label in sorted((folder / "labels").glob("*.json"))[: args.n]:
            s = json.loads(label.read_text())
            name = f"{event}-{folder.name}-{Path(s['image']).name}"
            shutil.copy(folder / "images" / Path(s["image"]).name, args.out / "poly" / name)
            img = cv2.imread(str(args.out / "poly" / name))
            steps: list[dict] = []
            found = finder.find(img, steps=steps, min_find=args.min_find, zoom=args.zoom)
            row = {"event": event, "image": f"poly/{name}", "width": img.shape[1], "height": img.shape[0], "scramble": s["scramble"],
                   "finder": np.round(steps[0]["corners"], 3).tolist(), "found": None}
            if found is not None:
                kp = np.array(np.round(found[0], 3), np.float32)
                row["found"] = geometry(sh, kp, img.shape)
                probs = np.round(read_stickers(reader, img, kp), 5)
                row["probs"] = probs.tolist()
                inside = np.array(row["found"]["inside"])
                row["verdicts"] = {}
                for kind, claim in (("true", s["scramble"]), ("slip", slipped(s["scramble"].split(), sh.puzzle.turns, rng))):
                    alts = poly_check.single_slips(sh, claim)
                    v = poly_check.check(probs, inside, sh, sh.puzzle.apply(sh.puzzle.solved, claim.split()), alts)
                    row["verdicts"][kind] = {"scramble": claim, "status": v.status, "reason": v.reason, "observed": v.observed,
                                             "mismatches": v.mismatches, "hypothesis": v.hypothesis, "slip": v.slip,
                                             "evidence": v.evidence, "slips": len(alts.states)}
                # The pose fit on jittered corners too: corners a network puts a little off, or one on the wrong spot.
                row["jittered"] = []
                for scale in (2.0, 6.0, 25.0):
                    j = (kp + rng.normal(0, scale, kp.shape)).astype(np.float32)
                    row["jittered"].append(geometry(sh, np.array(np.round(j, 3), np.float32), img.shape))
            rows.append(row)
    (args.out / "poly-reference.json").write_text(json.dumps(rows))
    (args.fixtures / "poly-reference.json").write_text(json.dumps(rows, separators=(",", ":")))
    print(f"{len(rows)} renders, puzzle found in {sum(r['found'] is not None for r in rows)}")
    runs = [follow_run(e, np.random.default_rng([EVENTS.index(e), i])) for e in loaded for i in range(args.runs)]
    (args.fixtures / "poly-follow.json").write_text(json.dumps(runs, separators=(",", ":")))
    shown = sum(any(lk["report"]["status"] == "mistake" for lk in r["looks"]) for r in runs)
    print(f"{len(runs)} follow-along runs, {sum(len(r['looks']) for r in runs)} looks, a mistake shown in {shown}")


if __name__ == "__main__":
    main()
