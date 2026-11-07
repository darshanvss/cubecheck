"""Follow-along on simulated looks: the reader's probabilities made from the true cube, with
covered and misread stickers and a random orientation each time."""

import numpy as np
import pytest

from cubecheck.follow import Follower, inverse, shorten
from cubecheck.nxn import cube, random_scramble
from cubecheck.nxn_check import cells


def look_at(state, n, rng, covered=0.15, misread=0.03):
    """What the reader would give for this cube, seen in a random orientation."""
    o = int(rng.integers(24))
    truth = state[cells(n)[o]]
    probs = np.full((3 * n * n, 7), 0.015)
    for i, colour in enumerate(truth):
        if rng.random() < covered:
            probs[i, 6] = 0.9
        elif rng.random() < misread:
            probs[i, (colour + int(rng.integers(1, 6))) % 6] = 0.9
        else:
            probs[i, colour] = 0.9
    return probs, np.ones(3 * n * n, bool)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_follows_a_scramble_with_jumps(n):
    rng = np.random.default_rng(n)
    c = cube(n)
    moves = random_scramble(n, rng)
    f = Follower(" ".join(moves), n)
    k, right, looks = 0, 0, 0
    while k < len(moves):
        k = min(len(moves), k + int(rng.integers(0, 5)))  # 0-4 moves between still looks
        r = f.look(*look_at(c.apply(c.solved, moves[:k]), n, rng))
        right += r["position"] == k
        looks += 1
    assert right >= looks - 1
    assert r["status"] == "done"


def test_a_wrong_move_is_caught_with_the_way_back():
    rng = np.random.default_rng(7)
    c = cube(3)
    moves = random_scramble(3, rng)
    f = Follower(" ".join(moves), 3)
    for k in range(0, 7, 2):
        assert f.look(*look_at(c.apply(c.solved, moves[:k]), 3, rng))["position"] == k
    wrong = next(t for t in c.event_turns if t[0] != moves[6][0])
    done = moves[:6] + [wrong] + moves[7:9]  # move 7 done wrong, then two more moves
    for _ in range(2):
        r = f.look(*look_at(c.apply(c.solved, done), 3, rng))
    assert r["status"] == "mistake"
    assert r["mistake"] == {"kind": "wrong", "move": 6, "done": wrong, "after": 2}
    fixed = c.apply(c.solved, done + r["undo"] + r["then"])
    assert np.array_equal(fixed, c.apply(c.solved, moves[:r["resume"]]))
    # Undo it as shown: the cube the way back leads to clears the mistake at the first look.
    r = f.look(*look_at(fixed, 3, rng))
    assert r["status"] == "following" and r["position"] == 7


def test_the_way_back_is_followed_turn_by_turn():
    """Each turn of the way back is taken at the first look: the shown way back never still
    has a move in it that was already undone, and it never says the cube is somewhere it isn't."""
    rng = np.random.default_rng(41)
    c = cube(3)
    moves = random_scramble(3, rng)
    f = Follower(" ".join(moves), 3)
    for k in range(0, 9, 2):
        f.look(*look_at(c.apply(c.solved, moves[:k]), 3, rng))
    wrong = next(t for t in c.event_turns if t[0] not in (moves[8][0], moves[9][0], moves[7][0]))
    state = c.apply(c.solved, moves[:8] + [wrong] + moves[9:12])
    for _ in range(2):
        r = f.look(*look_at(state, 3, rng))
    assert r["status"] == "mistake"
    way, resume = r["undo"] + r["then"], r["resume"]
    for i, m in enumerate(way):
        state = c.apply(state, [m])
        r = f.look(*look_at(state, 3, rng))
        if r["status"] == "mistake":
            assert r["undo"] + r["then"] == way[i + 1:]
        else:  # partway back, the cube can be on the scramble again (an undone move it had in it)
            assert r["status"] == "following" and np.array_equal(state, c.apply(c.solved, moves[:r["position"]]))
    assert r["status"] == "following" and r["position"] == resume


def test_an_unrelated_point_still_takes_two_looks_to_clear():
    rng = np.random.default_rng(42)
    c = cube(3)
    moves = random_scramble(3, rng)
    f = Follower(" ".join(moves), 3)
    for k in range(0, 9, 2):
        f.look(*look_at(c.apply(c.solved, moves[:k]), 3, rng))
    wrong = next(t for t in c.event_turns if t[0] not in (moves[8][0], moves[7][0]))
    for _ in range(2):
        r = f.look(*look_at(c.apply(c.solved, moves[:8] + [wrong]), 3, rng))
    assert r["status"] == "mistake"
    # A solved cube (the next one of the group, say) isn't on the way back: one look isn't enough.
    assert f.look(*look_at(c.solved, 3, rng))["status"] == "mistake"
    assert f.look(*look_at(c.solved, 3, rng))["status"] == "following"


@pytest.mark.parametrize("way", ["", "'"])
def test_a_quarter_of_a_half_turn_is_halfway_not_wrong(way):
    rng = np.random.default_rng(11)
    c = cube(3)
    moves = "R U F2 D L' B R2 U'".split()
    f = Follower(" ".join(moves), 3)
    for k in (0, 1, 2):
        f.look(*look_at(c.apply(c.solved, moves[:k]), 3, rng))
    for _ in range(3):  # held there: R2 was started with F (or F'), not done wrong
        r = f.look(*look_at(c.apply(c.solved, moves[:2] + ["F" + way]), 3, rng))
        assert r["status"] == "following" and r["position"] == 2 and r["half"]
    for _ in range(2):  # a look can misread a middle sticker; the next one settles it
        r = f.look(*look_at(c.apply(c.solved, moves[:3]), 3, rng))
    assert r["position"] == 3 and not r["half"]


def test_a_quarter_turn_then_carrying_on_is_still_wrong():
    rng = np.random.default_rng(12)
    c = cube(3)
    moves = "R U F2 D L' B R2 U'".split()
    f = Follower(" ".join(moves), 3)
    for k in (0, 2):
        f.look(*look_at(c.apply(c.solved, moves[:k]), 3, rng))
    done = moves[:2] + ["F", "D", "L'"]  # F instead of F2, then two more moves
    for _ in range(2):
        r = f.look(*look_at(c.apply(c.solved, done), 3, rng))
    assert r["status"] == "mistake" and r["mistake"]["move"] == 2


def test_one_unclear_look_is_not_a_mistake():
    rng = np.random.default_rng(3)
    c = cube(3)
    moves = random_scramble(3, rng)
    f = Follower(" ".join(moves), 3)
    f.look(*look_at(c.apply(c.solved, moves[:5]), 3, rng))
    wrong = next(t for t in c.event_turns if t[0] != moves[5][0])
    r = f.look(*look_at(c.apply(c.solved, moves[:5] + [wrong]), 3, rng))
    assert r["status"] != "mistake"  # needs a second look first


def test_a_solved_cube_goes_back_to_the_start():
    rng = np.random.default_rng(5)
    c = cube(3)
    moves = random_scramble(3, rng)
    f = Follower(" ".join(moves), 3)
    for k in (3, 8, 12):
        f.look(*look_at(c.apply(c.solved, moves[:k]), 3, rng))
    assert f.look(*look_at(c.solved, 3, rng))["position"] == 0  # the next cube of the group


def test_mostly_covered_looks_change_nothing():
    rng = np.random.default_rng(9)
    c = cube(3)
    moves = random_scramble(3, rng)
    f = Follower(" ".join(moves), 3)
    f.look(*look_at(c.apply(c.solved, moves[:4]), 3, rng))
    r = f.look(*look_at(c.apply(c.solved, moves[:20]), 3, rng, covered=0.85))
    assert r["position"] == 4 and not r["clear"]


def test_inverse():
    assert [inverse(m) for m in ("R", "R'", "R2", "Rw", "3Rw'", "x")] == ["R'", "R", "R2", "Rw'", "3Rw", "x'"]


SCRAMBLE = "R2 U2 L F2 L F2 L2 B2 D' B F2 R' F L2 D' U F2 R' D"  # ends in D, so U and D' turns there commute with it


def finished(rng, done):
    """A tracker that followed SCRAMBLE to the end, then two looks at the cube after `done`."""
    c = cube(3)
    moves = SCRAMBLE.split()
    f = Follower(SCRAMBLE, 3)
    for k in range(0, len(moves) + 1, 3):
        f.look(*look_at(c.apply(c.solved, moves[:k]), 3, rng))
    for _ in range(2):
        r = f.look(*look_at(c.apply(c.solved, moves), 3, rng))
    assert r["status"] == "done"
    for _ in range(2):
        r = f.look(*look_at(c.apply(c.solved, done), 3, rng))
    return r


@pytest.mark.parametrize("extra, back", [("L", ["L'"]), ("U'", ["U"]), ("R2", ["R2"])])
def test_an_extra_turn_after_the_last_move(extra, back):
    r = finished(np.random.default_rng(21), SCRAMBLE.split() + [extra])
    assert r["status"] == "mistake"
    assert r["mistake"]["kind"] == "extra" and r["mistake"]["move"] == 19
    assert r["undo"] == back and r["then"] == [] and r["resume"] == 19  # U' after D: just U, not D' U D


def test_a_half_turn_for_a_quarter_turn_is_fixed_with_one_turn():
    r = finished(np.random.default_rng(22), SCRAMBLE.split()[:-1] + ["D2"])
    assert r["status"] == "mistake"
    assert r["undo"] + r["then"] == ["D'"]  # not D2, then D


def test_the_way_back_is_right_and_short_for_any_slip():
    rng = np.random.default_rng(23)
    c = cube(3)
    for _ in range(30):
        moves = random_scramble(3, rng)
        f = Follower(" ".join(moves), 3)
        k = int(rng.integers(0, len(moves) - 3))
        for m in f.mistakes_near(k):
            way = m.fix(moves)
            assert np.array_equal(c.apply(m.state, way["undo"] + way["then"]), c.apply(c.solved, moves[:way["resume"]]))
            assert shorten(way["undo"] + way["then"]) == way["undo"] + way["then"]


def test_shorten():
    assert shorten("D2 D".split()) == ["D'"]
    assert shorten("D' U D".split()) == ["U"]
    assert shorten("R R'".split()) == []
    assert shorten("R U U' R".split()) == ["R2"]  # U U' cancels, so the two Rs meet
    assert shorten("Rw R' L".split()) == ["Rw", "R'", "L"]  # one axis, different layers: kept
    assert shorten("R U R'".split()) == ["R", "U", "R'"]


@pytest.mark.parametrize("k, more", [(4, 2), (4, 3), (9, 1), (9, 2), (13, 1), (13, 3)])
def test_a_skipped_move_is_called_skipped(k, more):
    """Skipping move k and doing move k+1 leaves the cube an extra turn would (or, here, a wrong
    one two moves back, or an extra B' that meets B through F2): it's still told as the skip,
    undone back to move k. (Skipping move 5, L, and doing F2 is no slip at all to the camera:
    L F2 F2 is the cube after move 3.)"""
    rng = np.random.default_rng(k * 10 + more)
    c = cube(3)
    moves = SCRAMBLE.split()
    f = Follower(SCRAMBLE, 3)
    for j in range(0, k + 1, 2):
        f.look(*look_at(c.apply(c.solved, moves[:j]), 3, rng))
    f.look(*look_at(c.apply(c.solved, moves[:k]), 3, rng))
    done = moves[:k] + moves[k + 1:k + 1 + more]
    for _ in range(4):  # two looks in a row; a look with a few misreads can put it off by one
        r = f.look(*look_at(c.apply(c.solved, done), 3, rng))
        if r["status"] == "mistake":
            break
    assert r["status"] == "mistake"
    assert (r["mistake"]["kind"], r["mistake"]["move"]) == ("skipped", k)
    assert r["resume"] == k and r["then"] == []
    assert np.array_equal(c.apply(c.solved, done + r["undo"]), c.apply(c.solved, moves[:k]))


def test_two_moves_swapped_are_called_swapped():
    rng = np.random.default_rng(31)
    c = cube(3)
    moves = SCRAMBLE.split()  # moves 10 and 11 are B F2: they commute, so 11 and 12 (F2 R') are swapped
    f = Follower(SCRAMBLE, 3)
    for j in range(0, 11, 2):
        f.look(*look_at(c.apply(c.solved, moves[:j]), 3, rng))
    done = moves[:10] + [moves[11], moves[10]] + moves[12:13]
    for _ in range(4):
        r = f.look(*look_at(c.apply(c.solved, done), 3, rng))
        if r["status"] == "mistake":
            break
    assert (r["mistake"]["kind"], r["mistake"]["move"]) == ("swapped", 10)
    assert np.array_equal(c.apply(c.solved, done + r["undo"] + r["then"]), c.apply(c.solved, moves[:r["resume"]]))
