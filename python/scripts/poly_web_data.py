"""Skewb, pyraminx and FTO for the website, as numbers: web/src/lib/poly-data.json (experimental).

    uv run python scripts/poly_web_data.py --out ../web/src/lib/poly-data.json

Everything the browser needs that Python works out from the puzzles' shapes, so the
TypeScript doesn't redo the geometry: each turn's reshuffle of the sticker places, the
puzzle's corners and every way to label them in a picture (shapes.Shape.views), the
sticker places behind each face of the picture, the sticker outlines on a face, the flat
picture (poly_net.py), and the check's and follow-along's settings. web/tests/poly.test.ts
checks the TypeScript that uses it against Python's answers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cubecheck import poly_check, poly_follow, poly_net, shapes
from cubecheck.poly_reader import face_px, map_matrix
from cubecheck.shapes import EVENTS, shape


def r(a, d: int = 9):
    return np.round(np.asarray(a, float), d).tolist()


def data(event: str) -> dict:
    sh = shape(event)
    p = sh.puzzle
    follow = {**poly_follow.DEFAULTS, **poly_follow.SETTINGS[event]}
    return {
        "event": event,
        "name": shapes.NAMES[event],
        "faceKeys": p.face_keys,
        "css": [f"#{p.colours[k]:06x}" for k in p.face_keys],
        "places": p.size,
        "solved": p.solved.tolist(),
        "turns": p.turns,
        "perms": {t: p.move(t).tolist() for t in p.turns},
        "vertices": r(sh.vertices),
        "views": sh.views,
        "viewFaces": sh.view_faces,
        "polys": [list(q) for q in sh.polys],
        "quads": sh.quads,
        "nFaces": sh.n_faces,
        "perFace": sh.per_face,
        "keypoints": sh.keypoints,
        "colours": sh.colours,
        "spokes": list(sh.spokes),
        "relabellings": sh.relabellings,
        "frames": [list(sh._frame(0, k)) for k in range(sh.n_faces)],  # each face's C, (1, 0) and (0, 1) corners
        "slots": [r(s) for s in sh.slots],
        "slotCentres": r(sh.slot_centres),
        "cells": sh.cells.tolist(),
        "facePx": face_px(sh),
        "mapMatrix": r(map_matrix(sh)),
        "net": poly_net.net(event),
        "minFaces": shapes.MIN_FACES[event],
        "foundFaces": shapes.FOUND_FACES[event],
        "holdHint": shapes.HOLD_HINT[event],
        "thresholds": poly_check.THRESHOLDS[event],
        "minSize": poly_check.MIN_SIZE[event],
        "maxFit": poly_check.MAX_FIT.get(event, 0),
        "scramble": {"faces": shapes.SCRAMBLE[event][0].split(), "length": shapes.SCRAMBLE[event][1], "tips": event == "pyram"},
        "follow": follow,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.write_text(json.dumps({e: data(e) for e in EVENTS}, separators=(",", ":")))
    print(f"{args.out}: {args.out.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
