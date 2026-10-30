"""The 7-keypoint convention shared by the dataset labels, training and inference.

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
