import numpy as np

from cubecheck.cube import SOLVED, apply
from cubecheck.verify import CELLS, single_slips


def test_every_reading_covers_27_distinct_stickers_with_three_centres():
    for cells in CELLS:
        assert len(set(cells.tolist())) == 27
        assert {cells[4], cells[13], cells[22]} <= {4, 13, 22, 31, 40, 49}


def test_single_slips_include_common_mistakes():
    scramble = "R U2 F' L D B2 R' U"
    alts = single_slips(scramble)
    states = {s.tobytes() for s in alts.states}
    for wrong in ("R U2 F L D B2 R' U", "R U2 F' L D B2 R'", "R U F' L D B2 R' U", "R U2 F' L D L2 R' U"):
        assert apply(SOLVED, wrong.split()).tobytes() in states
    assert apply(SOLVED, scramble.split()).tobytes() not in states


def test_legal_pieces_match_cube_rules():
    from collections import Counter

    from cubecheck.verify import LEGAL

    # 8 corner slots and 12 edge slots, each with 24 possible colourings.
    assert Counter((len(s), len(v)) for s, v in LEGAL.items()) == Counter({(3, 24): 8, (2, 24): 12})
    for colours in set().union(*LEGAL.values()):
        assert len(set(colours)) == len(colours)  # no piece shows a colour twice
        opposite = {(0, 3), (1, 4), (2, 5)}  # U-D, R-L, F-B in URFDLB order
        assert not any((a, b) in opposite or (b, a) in opposite for a in colours for b in colours)


def test_unique_rule_never_shows_a_piece_twice():
    from cubecheck.verify import CELLS, decode_pieces, SLOT_OF

    rng = np.random.default_rng(0)
    for _ in range(50):
        cells = CELLS[rng.integers(24)]
        dist6 = rng.uniform(0, 30, (27, 6))  # random readings: rules must still give a possible cube
        colours = decode_pieces(cells, dist6, np.ones(27, bool))
        seen = {}
        for n, f in enumerate(cells.tolist()):
            if f in SLOT_OF:
                seen.setdefault(SLOT_OF[f], {})[f] = int(colours[n])
        pieces = [frozenset(v.values()) for s, v in seen.items() if len(v) == len(s)]
        assert len(pieces) == len(set(pieces))
        assert max(np.bincount(colours, minlength=6)) <= 9


def test_pieces_rule_fixes_a_corner_from_its_other_stickers():
    from cubecheck.cube import FACELETS
    from cubecheck.verify import CELLS, decode_pieces

    # Solved cube, one corner sticker looks slightly more like a wrong colour:
    # the corner's other stickers say what it must be.
    cells = CELLS[0]
    truth = SOLVED[cells]
    dist6 = np.full((27, 6), 20.0)
    dist6[np.arange(27), truth] = 5.0
    n = next(n for n in range(27) if sum(map(abs, FACELETS[cells[n]][0])) == 3 and n % 9 != 4)
    wrong = (truth[n] + 1) % 6
    dist6[n, wrong], dist6[n, truth[n]] = 4.0, 9.0
    assert decode_pieces(cells, dist6, np.ones(27, bool))[n] == truth[n]
