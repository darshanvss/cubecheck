import copy
import json
from pathlib import Path

from cubecheck.samples import check_sample

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "sample_handheld.json").read_text())


def test_generated_sample_is_consistent():
    # The fixture is one label file from the dataset; its facelets must match the
    # Python simulator for the same scramble, among the other checks.
    assert check_sample(FIXTURE) == []


def test_wrong_state_is_caught():
    broken = copy.deepcopy(FIXTURE)
    f = list(broken["labels"]["facelets"])
    f[0], f[1] = f[1], f[0]
    broken["labels"]["facelets"] = "".join(f)
    if f[0] != f[1]:
        assert any("facelets" in p for p in check_sample(broken))


def test_swapped_keypoints_are_caught():
    broken = copy.deepcopy(FIXTURE)
    kp = broken["labels"]["keypoints"]
    kp[1], kp[5] = kp[5], kp[1]  # mirror order: anticlockwise
    assert any("clockwise" in p for p in check_sample(broken))
