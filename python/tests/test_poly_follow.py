"""Follow-along for skewb, pyraminx and FTO on simulated looks: the reader's probabilities made
from the true puzzle, turned a random way round, with covered and misread stickers.

"""

import numpy as np
import pytest

EVENTS = ("skewb", "pyram", "fto")


def setup(event, scramble=None, seed=0, **settings):
    from cubecheck.poly_follow import PolyFollower
    from cubecheck.shapes import random_scramble, shape
    rng = np.random.default_rng(seed)
    sh = shape(event)
    moves = scramble.split() if scramble else random_scramble(event, rng)
    return sh, sh.puzzle, moves, PolyFollower(" ".join(moves), event, **settings), rng


def look_at(state, sh, rng, covered=0.1, misread=0.02, faces=None):
    """What the reader would give for this puzzle, seen a random way round (`faces`: only that many in view)."""
    truth = state[sh.cells[int(rng.integers(len(sh.cells)))]]
    probs = np.full((len(truth), sh.colours + 1), 0.004)
    for i, colour in enumerate(truth):
        if rng.random() < covered:
            probs[i, sh.colours] = 0.95
        elif rng.random() < misread:
            probs[i, (colour + int(rng.integers(1, sh.colours))) % sh.colours] = 0.95
        else:
            probs[i, colour] = 0.95
    inside = np.ones(len(truth), bool)
    if faces is not None:
        inside[faces * sh.per_face:] = False
    return probs, inside


def after(p, moves):
    return p.apply(p.solved, moves)


@pytest.mark.parametrize("event", EVENTS)
def test_follows_a_scramble_with_jumps(event):
    sh, p, moves, f, rng = setup(event, seed=1)
    k, right, looks = 0, 0, 0
    while k < len(moves):
        k = min(len(moves), k + int(rng.integers(0, 4)))  # 0-3 moves between still looks
        r = f.look(*look_at(after(p, moves[:k]), sh, rng))
        right += r["position"] == k
        looks += 1
    assert right >= looks - 1
    assert r["status"] == "done" and not r["half"]


@pytest.mark.parametrize("event", EVENTS)
def test_a_wrong_move_is_caught_with_the_way_back(event):
    sh, p, moves, f, rng = setup(event, seed=3)
    for k in range(0, 5, 2):
        assert f.look(*look_at(after(p, moves[:k]), sh, rng))["position"] == k
    wrong = next(t for t in p.turns if t.rstrip("'") not in (moves[3].rstrip("'"), moves[4].rstrip("'"), moves[5].rstrip("'")) and t[0].isupper())
    done = moves[:4] + [wrong] + moves[5:7]  # move 5 done wrong, then two more moves
    for _ in range(2):
        r = f.look(*look_at(after(p, done), sh, rng))
    assert r["status"] == "mistake"
    assert r["mistake"] == {"kind": "wrong", "move": 4, "done": wrong, "after": 2}
    fixed = p.apply(after(p, done), r["undo"] + r["then"])
    assert np.array_equal(fixed, after(p, moves[:r["resume"]]))
    # Undo it as shown: the puzzle the way back leads to clears the mistake at the first look.
    r = f.look(*look_at(fixed, sh, rng))
    assert r["status"] == "following" and r["position"] == 5


@pytest.mark.parametrize("event", EVENTS)
def test_the_way_back_is_right_and_short_for_any_slip(event):
    for seed in range(8):
        sh, p, moves, f, rng = setup(event, seed=30 + seed)
        k = int(rng.integers(0, len(moves)))
        for m in f.mistakes_near(k):
            way = f.fix(m)
            turns = way["undo"] + way["then"]
            assert np.array_equal(p.apply(m.state, turns), f.points[way["resume"]])
            assert f.shorten(turns) == turns


def test_shorten():
    f = setup("pyram", "U L R B u")[3]
    assert f.shorten("R R".split()) == ["R'"]
    assert f.shorten("R R'".split()) == []
    assert f.shorten("R' R'".split()) == ["R"]
    assert f.shorten("u L u'".split()) == ["L"]  # a tip and anything else don't disturb each other
    assert f.shorten("U L U'".split()) == ["U", "L", "U'"]
    assert f.shorten("R U U' R".split()) == ["R'"]  # U U' cancels, so the two Rs meet
    g = setup("fto", "U D L")[3]
    assert g.shorten("U D U".split()) == ["U'", "D"]  # opposite faces
    assert g.shorten("U L U".split()) == ["U", "L", "U"]


PYRAM = "U L' R B' U R' L B U' L R' u l' b"  # ends with three tips


def test_a_tip_done_early_is_not_a_mistake():
    sh, p, moves, f, rng = setup("pyram", PYRAM, seed=4)
    for k in (0, 2, 4):
        f.look(*look_at(after(p, moves[:k]), sh, rng))
    for _ in range(3):  # tip b, the last move, turned now: nothing wrong, and nothing done either
        r = f.look(*look_at(after(p, moves[:4] + ["b"]), sh, rng))
        assert r["status"] == "following" and r["position"] == 4 and r["ahead"]
    for k in (6, 9, 11, 13):  # carried on, with b already done: the last look is the finished puzzle
        r = f.look(*look_at(after(p, moves[:k] + ["b"]), sh, rng))
        assert r["status"] != "mistake"
    for _ in range(2):
        r = f.look(*look_at(after(p, moves), sh, rng))
    assert r["status"] == "done"


def test_on_track_is_never_called_ahead():
    """Seen from two faces, a tip on the far side is out of sight: the look fits the point with and
    without that tip turned equally, and that isn't "a later move is already done"."""
    sh, p, moves, f, rng = setup("pyram", PYRAM, seed=9)
    for k in (0, 2, 4, 4, 6, 6, 9, 9, 11, 11):
        r = f.look(*look_at(after(p, moves[:k]), sh, rng, misread=0, faces=2))
        assert r["position"] == k and not r["ahead"]


def test_a_tip_turned_the_wrong_way_is_put_right_with_one_turn():
    sh, p, moves, f, rng = setup("pyram", PYRAM, seed=5)
    for k in (0, 3, 6, 9, 11):
        f.look(*look_at(after(p, moves[:k]), sh, rng))
    done = moves[:11] + ["u'", "l'", "b"]  # u' for u, then the other two tips
    for _ in range(3):
        r = f.look(*look_at(after(p, done), sh, rng))
    assert r["status"] == "mistake"
    assert (r["mistake"]["kind"], r["mistake"]["move"], r["mistake"]["done"]) == ("wrong", 11, "u'")
    assert r["undo"] + r["then"] == ["u'"] and r["resume"] == 14  # not: undo b' and l, then u', and do them again
    r = f.look(*look_at(after(p, moves), sh, rng))
    assert r["status"] == "done"


def test_opposite_faces_of_an_fto_in_either_order():
    sh, p, moves, f, rng = setup("fto", "R L' U D F B' R' BL U'", seed=6)
    for k in (0, 2):
        f.look(*look_at(after(p, moves[:k]), sh, rng))
    for _ in range(3):  # D before U
        r = f.look(*look_at(after(p, moves[:2] + ["D"]), sh, rng))
        assert r["status"] == "following" and r["position"] == 2
    r = f.look(*look_at(after(p, moves[:4]), sh, rng))
    assert r["status"] == "following" and r["position"] == 4


@pytest.mark.parametrize("event", EVENTS)
def test_a_look_at_one_face_changes_nothing(event):
    sh, p, moves, f, rng = setup(event, seed=7, min_faces=2)
    f.look(*look_at(after(p, moves[:3]), sh, rng))
    r = f.look(*look_at(after(p, moves[:8]), sh, rng, faces=1))
    assert r["position"] == 3 and not r["clear"]


@pytest.mark.parametrize("event", EVENTS)
def test_one_look_is_not_a_mistake_and_solved_goes_back_to_the_start(event):
    sh, p, moves, f, rng = setup(event, seed=8)
    for k in (0, 2, 4):
        f.look(*look_at(after(p, moves[:k]), sh, rng))
    wrong = next(t for t in p.turns if t.rstrip("'") != moves[4].rstrip("'") and t[0].isupper())
    assert f.look(*look_at(after(p, moves[:4] + [wrong]), sh, rng))["status"] != "mistake"  # needs a second look
    f.look(*look_at(after(p, moves[:6]), sh, rng))
    assert f.look(*look_at(p.solved, sh, rng))["position"] == 0  # the next puzzle of the group
