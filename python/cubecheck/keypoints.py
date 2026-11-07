"""The 7-keypoint convention shared by the generator, training and inference.

C is the cube corner nearest the camera. P1 is the corner of the white-or-yellow
face diagonally opposite C (in any corner view exactly one of those two faces is
visible). P2..P6 follow clockwise around the silhouette in the image.
"""

KEYPOINT_NAMES = ["P1", "P2", "P3", "P4", "P5", "P6", "C"]

# A horizontal flip turns clockwise into anticlockwise: P1, P4 and C stay, P2<->P6, P3<->P5.
FLIP_IDX = [0, 5, 4, 3, 2, 1, 6]

# Lines to draw: the silhouette hexagon and the three edges out of C.
SKELETON = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (6, 1), (6, 3), (6, 5)]

# Visible faces in image-clockwise order starting with the white/yellow face,
# each as a quad of keypoint indices: C first, then the face's other corners
# anticlockwise in the image (matches the "visibleFaces" label order).
FACE_QUADS = [(6, 1, 0, 5), (6, 3, 2, 1), (6, 5, 4, 3)]

VISIBLE, OCCLUDED, OUTSIDE = 2, 1, 0


def face_windings(keypoints) -> list[float]:
    """Signed area (pixels², image coordinates) of each face quad in FACE_QUADS.

    Faces the camera sees from the front all wind the same way. A face seen from
    behind (the camera is past its plane: an exactly-two-face view) winds the other
    way, so its sign is opposite; a face seen exactly edge-on has about 0.
    """
    out = []
    for quad in FACE_QUADS:
        p = [keypoints[i] for i in quad]
        out.append(0.5 * sum(p[i][0] * p[(i + 1) % 4][1] - p[(i + 1) % 4][0] * p[i][1] for i in range(4)))
    return out


def seen_from_behind(keypoints) -> list[bool]:
    """Which faces wind against the largest face (see face_windings)."""
    w = face_windings(keypoints)
    big = max(w, key=abs)
    return [x * big < 0 for x in w]


_AXES = {"U": (1, 1), "D": (1, -1), "F": (2, 1), "B": (2, -1), "R": (0, 1), "L": (0, -1)}


def _signs(name: str) -> tuple[int, int, int]:
    s = [0, 0, 0]
    for ch in name:
        axis, sign = _AXES[ch]
        s[axis] = sign
    return tuple(s)


def _quad_area(q) -> float:
    return 0.5 * sum(q[i]["x"] * q[(i + 1) % 4]["y"] - q[(i + 1) % 4]["x"] * q[i]["y"] for i in range(4))


def keypoints_from_corners(corners: list[dict], near: tuple[int, int, int]) -> list[list[float]]:
    """The 7 keypoints (x, y, visibility) with `near` as C, exactly as the generator's labels.ts orders them:
    P1..P6 follow the cube's ring of corners around C, in the direction that runs clockwise in the image."""
    by = {_signs(c["name"]): c for c in corners}
    x, y, z = near
    ring = [by[(-x, y, -z)], by[(x, y, -z)], by[(x, -y, -z)], by[(x, -y, z)], by[(-x, -y, z)], by[(-x, y, z)]]
    c0 = by[near]
    faces = [_quad_area([c0, ring[1], ring[0], ring[5]]), _quad_area([c0, ring[3], ring[2], ring[1]]), _quad_area([c0, ring[5], ring[4], ring[3]])]
    if max(faces, key=abs) > 0:
        ring = [ring[0]] + ring[1:][::-1]
    return [[c["x"], c["y"], 1 if c["occluded"] else 2] if c["inFrame"] else [0, 0, 0] for c in ring + [c0]]


def alternative_keypoints(labels: dict) -> list[list[float]] | None:
    """The other valid labelling of an exactly-two-face view, or None for a corner view.

    When the camera is past one face's plane, both ends of the edge the two visible faces
    share are equally "the corner nearest the camera" near the middle: the label switches
    from one end to the other as the camera moves past the cube's middle. Training and
    scoring accept either. Needs the label's 8 corners (generator labels have them).
    """
    kp = labels["keypoints"]
    if any(v[2] == 0 for v in kp) or "corners" not in labels:
        return None
    behind = seen_from_behind([k[:2] for k in kp])
    if not any(behind):
        return None
    hidden = (0, 2, 4)[behind.index(True)]  # the face's corner that belongs to it alone
    def name_at(p):  # the labelled corner at this keypoint
        return min(labels["corners"], key=lambda c: (c["x"] - p[0]) ** 2 + (c["y"] - p[1]) ** 2)["name"]

    near, own = _signs(name_at(kp[6])), _signs(name_at(kp[hidden]))
    axis = next(a for a in range(3) if own[a] == near[a])  # the hidden face's axis
    flipped = tuple(-v if a == axis else v for a, v in enumerate(near))
    return keypoints_from_corners(labels["corners"], flipped)
