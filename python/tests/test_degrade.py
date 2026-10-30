import numpy as np

from cubecheck.degrade import PIPELINE, degrade


def _image():
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, (90, 120, 3), dtype=np.uint8)


def test_every_step_keeps_size_and_type():
    img = _image()
    for name, fn, _ in PIPELINE:
        out, params = fn(img, np.random.default_rng(0))
        assert out.shape == img.shape, name
        assert out.dtype == np.uint8, name
        assert isinstance(params, dict), name


def test_same_seed_same_result():
    img = _image()
    a, applied_a = degrade(img, np.random.default_rng([0, 42]))
    b, applied_b = degrade(img, np.random.default_rng([0, 42]))
    assert applied_a == applied_b
    assert np.array_equal(a, b)
