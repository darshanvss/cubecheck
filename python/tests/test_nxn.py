import numpy as np
import pytest

from cubecheck import cube as c3
from cubecheck.nxn import EVENT_FAMILIES, EVENT_LENGTH, cube, random_scramble


def test_three_by_three_is_exactly_cube_py():
    c = cube(3)
    assert (c.solved == c3.SOLVED).all()
    for name in [f + a for f in "URFDLB" for a in ("", "'", "2")]:
        assert (c.move(name) == c3.MOVES[name]).all(), name
    scramble = "U R2 F B R B2 R U2 L B2 R U' D' R2 F R' L B2 U2 F2"
    assert c.to_string(c.after(scramble)) == c3.facelets_after(scramble)


@pytest.mark.parametrize("n", range(2, 8))
def test_orders_and_inverses(n):
    c = cube(n)
    for name in ["R", "U", "F", "L", "D", "B"] + (["Rw", "3Fw"] if n >= 4 else []) + (["Uw"] if n >= 3 else []):
        assert (c.apply(c.solved, [name] * 4) == c.solved).all(), name
        assert (c.apply(c.solved, [name, name + "'"]) == c.solved).all(), name
        assert not (c.apply(c.solved, [name]) == c.solved).all(), name
    sexy = "R U R' U'".split()
    assert (c.apply(c.solved, sexy * 6) == c.solved).all()


@pytest.mark.parametrize("n", range(2, 8))
def test_rotations_and_wide_moves(n):
    c = cube(n)
    # a whole-cube turn keeps every face one colour
    s = c.apply(c.solved, "x y z")
    for f in "URFDLB":
        assert len(set(s[c.face_slice(f)])) == 1
    # Rw on an N>=3 cube is R plus the next layer in: R' then Rw moves only that second layer
    if n >= 4:
        moved = c.apply(c.solved, "R' Rw") != c.solved
        assert moved.sum() == 4 * n  # one inner slice: N stickers on each of 4 faces


def test_wca_moves_only():
    c = cube(4)
    with pytest.raises(ValueError):
        c.parse("R 4Rw")  # depth must be below N
    with pytest.raises(ValueError):
        c.parse("3R")  # slice moves aren't WCA scramble notation
    with pytest.raises(ValueError):
        cube(2).parse("Rw")
    assert c.parse("R  3Rw’ Uw2") == ["R", "3Rw'", "Uw2"]


@pytest.mark.parametrize("n", [2])
def test_two_by_two_scrambles_never_move_the_dbl_corner(n):
    c = cube(n)
    rng = np.random.default_rng(n)
    s = c.apply(c.solved, random_scramble(n, rng))
    h = n - 1
    for pos, normal in [((-h, -h, -h), (0, -1, 0)), ((-h, -h, -h), (-1, 0, 0)), ((-h, -h, -h), (0, 0, -1))]:
        i = c.index[(pos, normal)]
        assert s[i] == c.solved[i]


@pytest.mark.parametrize("n", range(2, 8))
def test_random_scramble_follows_the_event(n):
    moves = random_scramble(n, np.random.default_rng(0))
    assert len(moves) == EVENT_LENGTH[n]
    fams = {m.rstrip("2'") for m in moves}
    assert fams <= set(EVENT_FAMILIES[n])
    cube(n).parse(" ".join(moves))


def test_tnoodle_scrambles_parse(tmp_path):
    # A sample of real TNoodle scrambles (data/tnoodle/test-comp), one per size.
    samples = {
        4: "U Bw' U2 B' R2 U' Fw' Rw' Uw B'",
        6: "3Rw2 Uw' 3Fw Lw2 R B' 3Uw' Dw2",
        7: "U Bw' U2 B' R2 U' 3Dw' 3Rw' 3Uw 3Bw B' 3Fw' 3Dw2 F' 3Fw U' 3Dw F2 Bw L'",
    }
    for n, text in samples.items():
        cube(n).after(text)


def test_four_by_four_mostly_opens_like_a_three_by_three():
    # TNoodle's 4x4 scrambles start with 15-21 outer turns, then mix in wide turns;
    # one in four here use wide turns from the start instead.
    rng = np.random.default_rng(1)
    shaped, wide_later = 0, []
    for _ in range(400):
        moves = random_scramble(4, rng)
        first_wide = next(i for i, m in enumerate(moves) if "w" in m)
        if 15 <= first_wide <= 21:
            shaped += 1
            wide_later.append(sum("w" in m for m in moves[first_wide:]) / len(moves[first_wide:]))
    assert 0.68 < shaped / 400 < 0.82
    assert 0.3 < np.mean(wide_later) < 0.5
    for n in (5, 6, 7):  # the bigger cubes use wide turns from the start
        assert any("w" in m for m in random_scramble(n, rng)[:8])
