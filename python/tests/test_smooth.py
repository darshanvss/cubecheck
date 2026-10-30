import numpy as np

from cubecheck.smooth import ROTATIONS, CornerSmoother, align

KP = np.array([[500.0, 200.0], [700.0, 260.0], [690.0, 480.0], [520.0, 520.0], [400.0, 420.0], [420.0, 250.0], [560.0, 360.0]])


def test_align_undoes_a_different_starting_face():
    assert np.allclose(align(KP[ROTATIONS[2]], KP), KP)


def test_jitter_shrinks_when_held_still():
    rng = np.random.default_rng(0)
    s = CornerSmoother()
    raw, smooth = [], []
    for n in range(60):
        frame = KP + rng.normal(0, 2, KP.shape)
        raw.append(frame)
        smooth.append(s.update(frame, n / 15))
    assert np.std(np.array(smooth[20:]), axis=0).mean() < 0.5 * np.std(np.array(raw[20:]), axis=0).mean()


def test_keeps_up_with_a_moving_cube():
    s = CornerSmoother()
    for n in range(30):  # one cube edge (~200 px) per second, sideways
        out = s.update(KP + [n * 200 / 15, 0], n / 15)
    assert abs(out[6, 0] - (KP[6, 0] + 29 * 200 / 15)) < 25  # trails by well under a sticker


def test_starts_over_after_a_gap():
    s = CornerSmoother()
    s.update(KP, 0.0)
    assert np.allclose(s.update(KP + 40, 2.0), KP + 40)
