"""Skewb, pyraminx and FTO: shapes.py's tables and poly_check, with a perfect sticker reader.

Each fixture is the labels of one training render (tests/fixtures/poly/<event>.json, no
image). The puzzle shapes come from cubecheck/puzzle_data.
"""

import json
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "poly"
EVENTS = ("skewb", "pyram", "fto")
# corners, labellings, faces in view, stickers per face
TABLE = {"skewb": (7, 24, 3, 5), "pyram": (4, 12, 3, 9), "fto": (5, 24, 4, 9)}


def load(event: str):
    from cubecheck.shapes import shape
    return shape(event), json.loads((FIXTURES / f"{event}.json").read_text())


def perfect(sample: dict, sh, view: int):
    """The labelled colours as reader output, and which cells have a label."""
    from cubecheck.poly_reader import cell_labels
    y = cell_labels(sample["labels"], view, sh)
    probs = np.zeros((len(y), sh.colours + 1))
    probs[np.arange(len(y)), np.where(y < 0, 0, y)] = 1.0
    return probs, y >= 0


@pytest.mark.parametrize("event", EVENTS)
def test_tables(event):
    sh, _ = load(event)
    assert (sh.keypoints, len(sh.views), sh.n_faces, sh.per_face) == TABLE[event]
    assert sh.cells.shape == (len(sh.views), sh.n_faces * sh.per_face)
    for view, cells in zip(sh.views, sh.cells):
        assert len(set(view)) == len(view) and len(set(cells.tolist())) == len(cells)
    # Every sticker place is seen from some side, and a mirrored ring keeps its first corner and C.
    assert set(sh.cells.ravel().tolist()) == set(range(sh.puzzle.size))
    assert sh.flip[0] == 0 and sh.flip[-1] == sh.keypoints - 1 and sorted(sh.flip) == list(range(sh.keypoints))


@pytest.mark.parametrize("event", EVENTS)
def test_render_matches_the_model(event):
    sh, s = load(event)
    lab = s["labels"]
    p = sh.puzzle
    assert lab["faces"] == p.face_keys
    assert np.array_equal(p.apply(p.solved, s["scramble"].split()), lab["state"])
    views = sh.labellings(lab)
    assert len(views) == {"skewb": 3, "pyram": 6, "fto": 4}[event]  # the pyraminx fixture shows two faces
    xy = np.array([[t["x"], t["y"]] for t in lab["stickers"]])
    for v in views:
        kp = sh.keypoints_of(lab, v)[:, :2]
        centres = sh.cell_centres(kp, (s["height"], s["width"]))
        front = np.repeat(np.array(lab["facing"])[sh.view_faces[v]] > 0.1, sh.per_face)
        off = np.linalg.norm(centres - xy[sh.cells[v]], axis=1)[front] / sh.size(kp)
        assert off.max() < 0.03  # within 3% of an edge of where the generator put each sticker
        assert (sh.windings(kp)[np.array(lab["facing"])[sh.view_faces[v]] > 0.1] < 0).all()


@pytest.mark.parametrize("event", EVENTS)
def test_check_with_a_perfect_reader(event):
    from cubecheck.poly_check import check, readable, single_slips
    sh, s = load(event)
    p, moves = sh.puzzle, s["scramble"].split()
    for v in sh.labellings(s["labels"]):
        kp = sh.keypoints_of(s["labels"], v)[:, :2]
        probs, labelled = perfect(s, sh, v)
        inside = readable(kp, sh, (s["height"], s["width"])) & labelled
        right = check(probs, inside, sh, p.apply(p.solved, moves), single_slips(sh, " ".join(moves)))
        assert right.status == "correct" and not right.mismatches
        wrong = moves[:3] + moves[4:]  # the fourth move left out: the photo shows a slip from this claim
        v2 = check(probs, inside, sh, p.apply(p.solved, wrong), single_slips(sh, " ".join(wrong)))
        assert v2.status == "mis-scramble"


def test_slips_are_one_move_away():
    from cubecheck.poly_check import single_slips
    sh, s = load("skewb")
    p, moves = sh.puzzle, s["scramble"].split()
    alts = single_slips(sh, " ".join(moves))
    states = {st.tobytes() for st in alts.states}
    assert p.apply(p.solved, moves).tobytes() not in states
    assert p.apply(p.solved, moves[1:]).tobytes() in states  # first move skipped
    assert p.apply(p.solved, moves + ["R"]).tobytes() in states  # an extra turn at the end
    assert len(alts.slips) == len(alts.states) == len(alts.where)


@pytest.mark.parametrize("event", ("pyram", "fto"))
def test_too_few_faces_is_never_called_correct(event):
    """Held so that fewer faces show than the puzzle needs: the right scramble gets a hint
    to turn it, not "correct"; a slip that shows on the faces left is still reported."""
    from cubecheck.poly_check import check, readable, single_slips
    from cubecheck.shapes import HOLD_HINT, MIN_FACES

    sh, s = load(event)
    p = sh.puzzle
    view = sh.labellings(s["labels"])[0]
    kp = sh.keypoints_of(s["labels"], view)[:, :2]
    probs, labelled = perfect(s, sh, view)
    inside = readable(kp, sh, (s["height"], s["width"])) & labelled
    faces = np.flatnonzero(inside.reshape(sh.n_faces, -1).any(axis=1))
    assert len(faces) >= MIN_FACES[event]
    kept = inside.copy()
    kept.reshape(sh.n_faces, -1)[faces[MIN_FACES[event] - 1:]] = False  # one face fewer than needed
    moves = s["scramble"].split()
    state = p.apply(p.solved, moves)

    v = check(probs, kept, sh, state, single_slips(sh, " ".join(moves)), min_known=1)
    assert v.status == "re-check" and HOLD_HINT[event] in v.reason

    # A claim that differs on the faces still in view: caught all the same.
    for turn in p.turns:
        wrong = moves + [turn]
        cells = sh.cells[v.hypothesis] if v.hypothesis is not None else sh.cells[view]
        if (p.apply(p.solved, wrong)[cells][kept] != state[cells][kept]).sum() >= 4:
            v2 = check(probs, kept, sh, p.apply(p.solved, wrong), single_slips(sh, " ".join(wrong)), min_known=1)
            assert v2.status == "mis-scramble"
            break
    else:
        pytest.skip("no turn changes enough of the faces left in this fixture")


def _area(poly) -> float:
    q = np.asarray(poly)
    return 0.5 * abs(float(np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1])))


@pytest.mark.parametrize("event", EVENTS)
def test_flat_picture_has_every_sticker_once(event):
    """poly_net: one outline per sticker place, inside its own face, and no two faces on top of each other."""
    import cv2
    from cubecheck.poly_net import net
    from cubecheck.shapes import shape

    sh, n = shape(event), net(event)
    p = sh.puzzle
    assert len(n["polys"]) == p.size and len(n["faces"]) == len(p.face_keys)
    for place, poly in enumerate(n["polys"]):
        face = np.float32(n["faces"][int(p.solved[place])])
        middle = np.mean(poly, axis=0)
        assert cv2.pointPolygonTest(face, (float(middle[0]), float(middle[1])), False) > 0, place
        assert all(-1e-6 <= x <= n["width"] + 1e-6 and -1e-6 <= y <= n["height"] + 1e-6 for x, y in poly)
    # Faces don't overlap: no face's middle lies inside another face.
    for f, face in enumerate(n["faces"]):
        m = np.mean(face, axis=0)
        inside = [g for g, other in enumerate(n["faces"]) if cv2.pointPolygonTest(np.float32(other), (float(m[0]), float(m[1])), False) > 0]
        assert inside == [f]
    if event != "fto":  # a true unfolding keeps every sticker's size
        for place, poly in enumerate(n["polys"]):
            v = p.sticker_vertices[place]
            real = 0.5 * sum(np.linalg.norm(np.cross(v[i] - v[0], v[i + 1] - v[0])) for i in range(1, len(v) - 1))
            assert abs(_area(poly) - real) < 1e-2 * real + 1e-3  # outlines are rounded to 3 decimals
