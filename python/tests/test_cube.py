from cubecheck.cube import SOLVED, apply, facelets_after, to_string


def test_single_moves_match_known_facelet_strings():
    assert facelets_after("R") == "UUFUUFUUFRRRRRRRRRFFDFFDFFDDDBDDBDDBLLLLLLLLLUBBUBBUBB"
    assert facelets_after("U") == "UUUUUUUUUBBBRRRRRRRRRFFFFFFDDDDDDDDDFFFLLLLLLLLLBBBBBB"


def test_superflip():
    superflip = "U R2 F B R B2 R U2 L B2 R U' D' R2 F R' L B2 U2 F2"
    assert facelets_after(superflip) == "UBULURUFURURFRBRDRFUFLFRFDFDFDLDRDBDLULBLFLDLBUBRBLBDB"


def test_inverses_and_orders():
    for face in "URFDLB":
        assert (apply(SOLVED, [face] * 4) == SOLVED).all()
        assert (apply(SOLVED, [face, face + "'"]) == SOLVED).all()
        assert (apply(SOLVED, [face + "2", face + "2"]) == SOLVED).all()
    sexy = "R U R' U'".split()
    assert (apply(SOLVED, sexy * 6) == SOLVED).all()
    assert not (apply(SOLVED, sexy * 3) == SOLVED).all()


def test_centres_never_move():
    state = facelets_after("R U2 F' L D B2 R' F2 U' L2 D' B")
    assert [state[i * 9 + 4] for i in range(6)] == list("URFDLB")
    assert to_string(SOLVED) == "".join(f * 9 for f in "URFDLB")
