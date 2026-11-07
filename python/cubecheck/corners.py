"""Our own corner model: finds the 7 cube corners without Ultralytics.

Two passes of the same small network design, trained separately:

1. The finder sees the whole frame shrunk to 256 x 256 and answers roughly
   where the 7 corners are, plus whether there's a cube at all.
2. The refiner sees a 256 x 256 crop around the cube, cut from the full-quality
   frame, and places the corners precisely. It runs twice: the second time on
   a crop centred with its own first answer.

Each answer is a heatmap per corner: a 64 x 64 grid of scores, one blob where
the corner is. Softmax turns the scores into weights that add up to 1, and the
corner is the weighted average position (soft-argmax), which lands between
grid cells, so it is finer than the 4-pixel grid. See docs/how-it-works.md.

Is there a cube at all? Both networks also give a cube-present score, from 0
to 1. Since c2 both learned it from renders of the same rooms with the cube
hidden (generator --empty), so a cube counts when the finder says so for the
whole frame, the refiner agrees on its close-up crop, and the three faces come
out as proper four-sided shapes. On the test renders that finds 99.6% of cubes
and calls none in 300 frames without one. The finder's bar is higher (0.75):
real frames showed it half-believing a patterned curtain (0.5 to 0.7), which
the refiner then confirmed, while on the test renders a bar up to 0.8 loses
no cubes. (c1 learned "no cube" only from
zoomed-in patches, so it said "cube" for any whole frame, 1.00 even for plain
grey; its refiner's score means nothing.)
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .keypoints import FACE_QUADS, FLIP_IDX

IN_PX = 256  # network input, square
HEAT_PX = 64  # heatmap grid: one cell per 4 input pixels
REFINE_MARGIN = 1.35  # refiner crop side = this x the corners' bounding box
MIN_PRESENT = 0.5  # cube-present score, from the refiner
MIN_FIND = 0.75  # the finder's own bar, before the refiner gets a look


def _conv(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class _Down(nn.Module):
    """Halve the size: a strided convolution, then a residual pair."""

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.down = _conv(cin, cout, 2)
        self.body = nn.Sequential(_conv(cout, cout), nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return F.relu(x + self.body(x))


class CornerNet(nn.Module):
    """256 x 256 image -> 7 heatmaps (64 x 64) and a cube-present score.

    A small U-Net: the image is shrunk step by step to 8 x 8 (wide view: which
    blob is which corner), then grown back to 64 x 64, mixing in the earlier,
    sharper layers (exact position).
    """

    def __init__(self, widths: tuple[int, ...] = (24, 48, 96, 160, 224), up: int = 96) -> None:
        super().__init__()
        w = widths
        self.stem = nn.Sequential(_conv(3, w[0], 2), _conv(w[0], w[0]))  # 128
        self.d1, self.d2, self.d3, self.d4 = _Down(w[0], w[1]), _Down(w[1], w[2]), _Down(w[2], w[3]), _Down(w[3], w[4])  # 64, 32, 16, 8
        self.lat = nn.ModuleList([nn.Conv2d(c, up, 1) for c in (w[4], w[3], w[2], w[1])])
        self.smooth = nn.ModuleList([_conv(up, up) for _ in range(3)])
        self.head = nn.Sequential(_conv(up, up), nn.Conv2d(up, 7, 1))
        self.present = nn.Linear(w[4], 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = self.d1(self.stem(x))
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)
        y = self.lat[0](x4)
        for lat, smooth, skip in zip(self.lat[1:], self.smooth, (x3, x2, x1)):
            y = smooth(F.interpolate(y, scale_factor=2, mode="nearest") + lat(skip))
        return self.head(y), self.present(x4.mean((2, 3))).squeeze(1)


def soft_argmax(heat: torch.Tensor) -> torch.Tensor:
    """(B, 7, H, W) scores -> (B, 7, 2) positions (x, y) in 0..1 of the input."""
    b, k, h, w = heat.shape
    p = heat.flatten(2).softmax(-1).view(b, k, h, w)
    xs = (torch.arange(w, device=heat.device, dtype=heat.dtype) + 0.5) / w
    ys = (torch.arange(h, device=heat.device, dtype=heat.dtype) + 0.5) / h
    return torch.stack([(p.sum(2) * xs).sum(-1), (p.sum(3) * ys).sum(-1)], dim=-1)


def cube_shaped(kp: np.ndarray, two_faces: bool = False) -> bool:
    """The three faces are convex quads, all turning the same way, as any corner view of a real cube gives.

    two_faces also accepts a view of exactly two faces (looking at an edge): those two convex
    and turning the same way, the third seen from behind (turning the other way) or edge-on
    (almost no area). Only for corner models trained on such views (CornerFinder.two_faces).
    """
    turns = []
    for quad in FACE_QUADS:
        pts = kp[list(quad)]
        edges = np.roll(pts, -1, axis=0) - pts
        nxt = np.roll(edges, -1, axis=0)
        turns.append(edges[:, 0] * nxt[:, 1] - edges[:, 1] * nxt[:, 0])
    if np.all(np.concatenate(turns) > 0) or np.all(np.concatenate(turns) < 0):
        return True
    if not two_faces:
        return False
    sign = [1 if np.all(t > 0) else -1 if np.all(t < 0) else 0 for t in turns]
    area = [abs(cv2.contourArea(kp[list(q)].astype(np.float32))) for q in FACE_QUADS]
    for third in range(3):
        a, b = [f for f in range(3) if f != third]
        if sign[a] != 0 and sign[a] == sign[b] and sign[third] != sign[a] and min(area[a], area[b]) > 0.2 * max(area):
            return True
    return False


# --- geometry: every crop is an affine map from the image to the 256 x 256 input ---

def _mat(a: np.ndarray) -> np.ndarray:
    return np.vstack([a, [0, 0, 1]])


def view_matrix(centre: np.ndarray, side: float, angle: float = 0.0, flip: bool = False) -> np.ndarray:
    """2 x 3 affine map: a square of `side` pixels around `centre`, turned by `angle` degrees, onto the input."""
    s = IN_PX / side
    rot = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), angle, s)
    rot[:, 2] += IN_PX / 2 - centre * 1.0  # centre lands in the middle of the input
    m = _mat(rot)
    if flip:
        m = _mat(np.array([[-1, 0, IN_PX], [0, 1, 0]], float)) @ m
    return m[:2]


def full_view(shape: tuple[int, ...]) -> np.ndarray:
    """The whole frame, letterboxed into the input."""
    h, w = shape[:2]
    return view_matrix(np.array([w / 2, h / 2]), max(w, h))


def refine_view(keypoints: np.ndarray, margin: float = REFINE_MARGIN) -> np.ndarray:
    """A square around the cube's corners, with some room around them."""
    lo, hi = keypoints.min(0), keypoints.max(0)
    return view_matrix((lo + hi) / 2, float(max(hi - lo)) * margin)


def warp(img: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Cut the view out of the image. Big shrinks are done with a proper area resize
    first, since an affine warp alone would skip pixels and turn edges jagged."""
    scale = float(np.hypot(m[0, 0], m[0, 1]))
    if scale < 0.75:
        f = scale * 1.25
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        m = np.hstack([m[:, :2] / f, m[:, 2:]])
    return cv2.warpAffine(img, m, (IN_PX, IN_PX), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))


def apply(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ m[:, :2].T + m[:, 2]


def invert(m: np.ndarray) -> np.ndarray:
    return cv2.invertAffineTransform(m)


def to_input(img_bgr: np.ndarray) -> torch.Tensor:
    """(..., H, W, 3) uint8 BGR -> (..., 3, H, W) float RGB in 0..1."""
    return torch.from_numpy(np.ascontiguousarray(img_bgr[..., ::-1])).movedim(-1, -3).float() / 255.0


FLIP = np.array(FLIP_IDX)


# --- inference ---

class CornerFinder:
    """Finder + refiner together: image in, 7 corners out."""

    def __init__(self, finder: CornerNet, refiner: CornerNet, device: str = "cpu", passes: int = 2, two_faces: bool = False) -> None:
        self.finder, self.refiner, self.device, self.passes = finder.eval(), refiner.eval(), device, passes
        self.two_faces = two_faces  # the refiner was trained on views of exactly two faces (cube_shaped)

    @classmethod
    def load(cls, finder_path: str, refiner_path: str, device: str = "cpu") -> CornerFinder:
        nets = []
        for path in (finder_path, refiner_path):
            ckpt = torch.load(path, map_location=device)
            net = CornerNet(**ckpt["config"])
            net.load_state_dict(ckpt["model"])
            nets.append(net.to(device))
        return cls(*nets, device=device, two_faces=bool(ckpt.get("two_faces", False)))

    @torch.no_grad()
    def _run(self, net: CornerNet, img: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, float]:
        """Corners in the image and the cube-present score."""
        heat, present = net(to_input(warp(img, m))[None].to(self.device))
        xy = soft_argmax(heat)[0].cpu().numpy() * IN_PX
        return apply(invert(m), xy), float(torch.sigmoid(present)[0])

    def _checked(self, kp: np.ndarray, present: float, min_present: float) -> tuple[np.ndarray, float] | None:
        return (kp, present) if present >= min_present and cube_shaped(kp, self.two_faces) else None

    def find(self, img_bgr: np.ndarray, min_present: float = MIN_PRESENT,
             steps: list[dict] | None = None, min_find: float = MIN_FIND, zoom: float = 0.0) -> tuple[np.ndarray, float] | None:
        """(7, 2) corner positions in the image and the refiner's cube-present score, or None when there's no cube.

        Pass a list as steps to get each network's view of the image (its 4 corners in the
        image), corners and cube-present score, to show how the answer was reached.
        min_find is the finder's own bar on the whole frame; min_present the refiner's.
        zoom (e.g. 2): when the finder is under its bar, look again at a square 1/zoom of the
        frame's short side around its rough guess. A cube small in the frame is only ~20 px
        across in the finder's whole-frame view; on real frames a 2x closer look raised its
        score from 0.34 to 0.93 (median).
        """
        views = [full_view(img_bgr.shape)]
        kp, present = self._run(self.finder, img_bgr, views[0])
        if zoom and present < min_find:
            closer = view_matrix(kp.mean(0), min(img_bgr.shape[:2]) / zoom)
            kp2, present2 = self._run(self.finder, img_bgr, closer)
            if present2 > present:
                views[0], kp, present = closer, kp2, present2
        for n in range(self.passes + 1):
            if steps is not None:
                square = apply(invert(views[-1]), np.float32([[0, 0], [IN_PX, 0], [IN_PX, IN_PX], [0, IN_PX]]))
                steps.append({"net": "refiner" if n else "finder", "view": square, "corners": kp, "present": present})
            if n == 0 and present < min_find:
                return None
            if present < min_present or n == self.passes:
                break
            views.append(refine_view(kp))
            kp, present = self._run(self.refiner, img_bgr, views[-1])
        return self._checked(kp, present, min_present)

    def track(self, img_bgr: np.ndarray, previous: np.ndarray,
              min_present: float = MIN_PRESENT) -> tuple[np.ndarray, float] | None:
        """Live video: refine around last frame's corners, skipping the finder. None when the cube has left the crop."""
        kp, present = self._run(self.refiner, img_bgr, refine_view(previous))
        return self._checked(kp, present, min_present)
