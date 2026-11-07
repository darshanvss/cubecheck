"""The any-size check, end to end on renders, with a perfect sticker reader.

Each fixture is the labels of one render (tests/fixtures/nxn/<puzzle>.json,
no image). The "reader" here reads every sticker from the labels, so these tests
check the geometry (corners -> N×N cells), the 24 orientations and the verdicts,
not the colour reading.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from cubecheck.nxn import cube
from cubecheck.nxn_check import cell_centres, cells, check, readable, single_slips, single_slips_slow
from cubecheck.verify import CELLS

FIXTURES = Path(__file__).parent / "fixtures" / "nxn"
COLOURS = "URFDLB"


def perfect_reader(sample: dict, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(3N², 7) probabilities read off the labels: the sticker nearest each cell centre."""
    kp = np.array([k[:2] for k in sample["labels"]["keypoints"]], np.float32)
    centres = cell_centres(kp, n)
    stickers = [s for s in sample["labels"]["stickers"] if s["facing"] > 0]
    xy = np.array([[s["x"], s["y"]] for s in stickers])
    probs = np.full((len(centres), 7), 0.004)
    for i, c in enumerate(centres):
        s = stickers[int(np.linalg.norm(xy - c, axis=1).argmin())]
        probs[i, 6 if s["visibleFraction"] < 0.4 else COLOURS.index(s["color"])] = 0.976
    w, h = sample["width"], sample["height"]
    inside = (centres[:, 0] >= 0) & (centres[:, 0] < w) & (centres[:, 1] >= 0) & (centres[:, 1] < h)
    return probs, inside


def load(puzzle: str) -> tuple[dict, int]:
    return json.loads((FIXTURES / f"{puzzle}.json").read_text()), int(puzzle[0])


def test_three_by_three_cells_are_verify_cells():
    for a, b in zip(cells(3), CELLS):
        assert (a == b).all()


@pytest.mark.parametrize("puzzle", ["222", "444", "555", "666", "777"])
def test_render_labels_match_the_model(puzzle):
    sample, n = load(puzzle)
    c = cube(n)
    assert c.to_string(c.after(sample["scramble"])) == sample["labels"]["facelets"]


@pytest.mark.parametrize("puzzle", ["222", "444", "555", "666", "777"])
def test_true_scramble_is_correct(puzzle):
    sample, n = load(puzzle)
    probs, inside = perfect_reader(sample, n)
    v = check(probs, inside, n, cube(n).after(sample["scramble"]), single_slips(sample["scramble"], n))
    assert v.status == "correct", (v.reason, len(v.mismatches))


@pytest.mark.parametrize("puzzle", ["222", "444", "555", "666", "777"])
def test_one_slip_is_caught_and_named(puzzle):
    sample, n = load(puzzle)
    probs, inside = perfect_reader(sample, n)
    moves = sample["scramble"].split()
    # Claim a scramble with move 3 done the other way: the photo shows the real cube, so it's one slip off.
    m = moves[2]
    moves[2] = m[:-1] if m.endswith("'") else m.rstrip("2") + "'" if not m.endswith("2") else m[:-1]
    claim = " ".join(moves)
    v = check(probs, inside, n, cube(n).after(claim), single_slips(claim, n))
    assert v.status == "mis-scramble"
    assert v.slip_where.get("move") == 2, v.slip


@pytest.mark.parametrize("puzzle", ["222", "444", "777"])
def test_a_different_cube_is_caught_but_not_explained(puzzle):
    sample, n = load(puzzle)
    probs, inside = perfect_reader(sample, n)
    other = " ".join(reversed(sample["scramble"].split()))
    v = check(probs, inside, n, cube(n).after(other), single_slips(other, n))
    assert v.status == "mis-scramble"
    assert not v.slip


@pytest.mark.parametrize("n", [2, 3, 4, 7])
def test_fast_slips_match_the_slow_way(n):
    from cubecheck.nxn import random_scramble
    text = " ".join(random_scramble(n, np.random.default_rng(n), length=12))
    fast, slow = single_slips(text, n), single_slips_slow(text, n)
    assert fast.slips == slow.slips and fast.where == slow.where
    assert (fast.states == slow.states).all()


def test_a_face_seen_edge_on_is_not_read():
    # C at the centre; the white/yellow face (C, P2, P1, P6) squashed to a sliver.
    kp = np.float32([[100, 95], [60, 90], [100, 30], [160, 60], [160, 160], [100, 190], [100, 100]])
    kp[0], kp[1], kp[5] = [100, 92], [96, 96], [104, 96]  # P1, P2, P6 close to C
    r = readable(kp, 3, (300, 300))
    assert not r[:9].any() and r[9:].all()
