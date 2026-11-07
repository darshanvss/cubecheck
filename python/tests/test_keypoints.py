"""The 7-corner layout in corner views and in exactly-two-face views (camera past one face's plane)."""

import itertools

import numpy as np

from cubecheck.keypoints import alternative_keypoints, keypoints_from_corners, seen_from_behind
from cubecheck.nxn_check import readable


def view(camera) -> list[dict]:
    """A unit cube's 8 corners seen by a pinhole camera at `camera`, looking at the centre, y up."""
    cam = np.array(camera, float)
    fwd = -cam / np.linalg.norm(cam)
    right = np.cross(fwd, [0, 1, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    corners = []
    for x, y, z in itertools.product((1, -1), repeat=3):
        d = np.array([x, y, z], float) - cam
        u, v, w = d @ right, d @ up, d @ fwd
        name = ("U" if y > 0 else "D") + ("F" if z > 0 else "B") + ("R" if x > 0 else "L")
        corners.append({"name": name, "x": 320 + 400 * u / w, "y": 240 - 400 * v / w, "inFrame": True, "occluded": False})
    return corners


def labels(camera) -> dict:
    corners = view(camera)
    near = tuple(int(np.sign(c)) for c in camera)
    return {"keypoints": keypoints_from_corners(corners, near), "corners": corners}


def test_corner_view_has_three_front_faces_and_no_alternative():
    lab = labels([5, 4, 6])
    kp = np.array(lab["keypoints"])[:, :2]
    assert seen_from_behind(kp.tolist()) == [False, False, False]
    assert alternative_keypoints(lab) is None
    assert readable(kp, 3, (480, 640)).all()


def test_two_face_view_skips_the_face_seen_from_behind():
    # Camera level with the cube's middle, looking at the F-R edge: U is seen from behind.
    lab = labels([6, 0.3, 6])
    kp = np.array(lab["keypoints"])[:, :2]
    behind = seen_from_behind(kp.tolist())
    assert sum(behind) == 1
    read = readable(kp, 3, (480, 640)).reshape(3, 9).all(axis=1)
    assert read.tolist() == [not b for b in behind]


def test_alternative_puts_c_at_the_other_end_of_the_shared_edge():
    above, below = labels([6, 0.3, 6]), labels([6, -0.3, 6])
    alt = alternative_keypoints(above)
    assert alt is not None
    # The alternative names the corners the labels would name with the camera just below the middle.
    def names(lab, kps):
        return [min(lab["corners"], key=lambda c: (c["x"] - k[0]) ** 2 + (c["y"] - k[1]) ** 2)["name"] for k in kps]

    assert names(above, alt) == names(below, below["keypoints"])
    assert names(above, alt)[6] != names(above, above["keypoints"])[6]
    assert sum(seen_from_behind([k[:2] for k in alt])) == 1


def test_a_two_by_two_showing_two_faces_asks_for_a_corner():
    from cubecheck.nxn import cube
    from cubecheck.nxn_check import check

    kp = np.array(labels([6, 0.3, 6])["keypoints"])[:, :2]
    inside = readable(kp, 2, (480, 640))
    probs = np.full((12, 7), 0.01)
    probs[:, 0] = 0.94
    v = check(probs, inside, 2, cube(2).solved)
    assert v.status == "re-check" and "corner" in v.reason


def test_shape_check_accepts_two_faces_only_when_asked():
    from cubecheck.corners import cube_shaped

    corner = np.array(labels([5, 4, 6])["keypoints"])[:, :2]
    edge = np.array(labels([6, 0.3, 6])["keypoints"])[:, :2]
    assert cube_shaped(corner) and cube_shaped(corner, two_faces=True)
    assert not cube_shaped(edge)  # older models: a face seen from behind isn't a cube
    assert cube_shaped(edge, two_faces=True)
