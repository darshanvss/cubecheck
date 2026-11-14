"""The sticker reader for skewb, pyraminx and FTO (experimental): reader.py for faces that aren't grids.

As for the cubes, every visible face is straightened to a fixed picture, so each sticker
sits in a fixed spot, and one network reads all of them at once: per sticker, the
probability of each of the puzzle's colours and of "covered". The differences:

- A skewb face is straightened to a square like a cube's. A triangle face (pyraminx,
  FTO) is straightened to an upright triangle with C at the top, using the fit of the
  whole shape (shapes.Shape.pose), since 3 corners don't fix the perspective.
- The stickers aren't a grid, so a sticker's features aren't one cell of the last
  layer: they are that layer averaged over the sticker's own outline (its "slot").
- Colours are the puzzle's own: 6 for a skewb, 4 for a pyraminx, 8 for an FTO, in the
  order of the puzzle model's faces.

The convolution and mixing layers are r4's (the cube reader), so training starts from a
network that already reads stickers. See docs/other-puzzles.md.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn

from .reader import StickerReader
from .shapes import Shape, shape

IGNORE = -1  # no label: out of frame, on a face turned away, or half covered
STRIDE = 8  # the convolution layers shrink a face 8 times


def face_px(sh: Shape) -> int:
    return 72 if sh.quads else 96


def map_matrix(sh: Shape) -> np.ndarray:
    """3 x 3 map from a face's (u, v) to its straightened picture."""
    s = face_px(sh)
    if sh.quads:  # C top left, as reader.face_maps
        return np.diag([s, s, 1.0])
    c, nxt, prev = np.array([s / 2, 7.0]), np.array([4.7, 82.0]), np.array([s - 4.7, 82.0])
    return np.array([[nxt[0] - c[0], prev[0] - c[0], c[0]], [nxt[1] - c[1], prev[1] - c[1], c[1]], [0, 0, 1]])


def face_maps(img_bgr: np.ndarray, matrices: list[np.ndarray], sh: Shape) -> np.ndarray:
    """The visible faces straightened, (faces, size, size, 3) uint8 RGB. matrices: shapes.Shape.face_matrices."""
    s, to_map = face_px(sh), map_matrix(sh)
    out = []
    for m in matrices:
        h = to_map @ np.linalg.inv(m)
        face = cv2.warpPerspective(img_bgr, h, (s, s), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
        out.append(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    return np.stack(out)


def slot_masks(sh: Shape, shrink: float = 0.8) -> np.ndarray:
    """(slots, size/8, size/8) weights: how much of each cell of the last layer lies inside each sticker
    (drawn a little small, so a neighbour's colour doesn't leak in). Each sticker's weights add up to 1."""
    s, to_map = face_px(sh), map_matrix(sh)
    out = []
    for poly in sh.slots:
        poly = poly.mean(axis=0) + (poly - poly.mean(axis=0)) * shrink
        pts = cv2.perspectiveTransform(poly[None].astype(np.float64), to_map)[0]
        mask = np.zeros((s * 4, s * 4), np.uint8)
        cv2.fillPoly(mask, [np.round(pts * 4).astype(np.int32)], 255)
        cells = cv2.resize(mask, (s // STRIDE, s // STRIDE), interpolation=cv2.INTER_AREA).astype(np.float32)
        out.append(cells / max(cells.sum(), 1e-6))
    return np.stack(out)


def cell_labels(labels: dict, view: int, sh: Shape) -> np.ndarray:
    """What the reader should answer for each (face, slot) cell under this labelling of the render."""
    covered = sh.colours
    out = np.full(sh.n_faces * sh.per_face, IGNORE)
    for cell, place in enumerate(sh.cells[view]):
        st = labels["stickers"][place]
        if labels["facing"][sh.view_faces[view][cell // sh.per_face]] < 0.1 or not st["inFrame"]:
            continue
        if st["visibleFraction"] >= 0.6:
            out[cell] = st["color"]
        elif st["visibleFraction"] < 0.4:
            out[cell] = covered
    return out


class PolyReader(nn.Module):
    def __init__(self, event: str, width: int = 128, layers: int = 2) -> None:
        super().__init__()
        sh = shape(event)
        self.event = event
        base = StickerReader(width, layers)
        self.cnn = base.cnn[:-1]  # without the last pooling step: one feature vector per 8 x 8 pixels
        self.mix = base.mix
        self.head = nn.Linear(width, sh.colours + 1)
        self.register_buffer("masks", torch.from_numpy(slot_masks(sh)), persistent=False)  # rebuilt from the shape
        self.position = nn.Parameter(torch.zeros(sh.n_faces * sh.per_face, width))  # which face, which sticker of it

    def forward(self, faces: torch.Tensor) -> torch.Tensor:
        """faces: (B, F, 3, H, W) floats in 0..1 -> logits (B, F x slots, colours + 1)."""
        b, f = faces.shape[:2]
        x = self.cnn(faces.flatten(0, 1))  # (B*F, C, h, w)
        x = torch.einsum("nchw,shw->nsc", x, self.masks.to(x.dtype))  # each sticker: the average over its outline
        x = x.reshape(b, f * self.masks.shape[0], -1)
        return self.head(self.mix(x + self.position))

    @classmethod
    def from_cube_reader(cls, event: str, path: str, device: str = "cpu") -> PolyReader:
        """Start from a trained cube reader (r4): its convolution and mixing layers carry over, and for a
        skewb, whose colours are a cube's, the answer layer too."""
        ckpt = torch.load(path, map_location=device)
        model = cls(event, **ckpt["config"])
        state = {k: v for k, v in ckpt["model"].items() if k.startswith(("cnn.", "mix."))}
        if event == "skewb":
            order = ["URFDLB".index(f) for f in shape(event).puzzle.face_keys] + [6]
            state["head.weight"], state["head.bias"] = ckpt["model"]["head.weight"][order], ckpt["model"]["head.bias"][order]
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected and all(k.startswith(("position", "head.")) for k in missing), (missing, unexpected)
        return model.to(device)


def load_poly_reader(path: str, device: str = "cpu") -> PolyReader:
    ckpt = torch.load(path, map_location=device)
    model = PolyReader(ckpt["event"], **ckpt["config"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def to_tensor(maps: np.ndarray) -> torch.Tensor:
    """(..., F, H, W, 3) uint8 face maps -> (..., F, 3, H, W) float tensor."""
    return torch.from_numpy(maps).movedim(-1, -3).float() / 255.0


@torch.no_grad()
def read_stickers(model: PolyReader, img_bgr: np.ndarray, keypoints: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Probabilities (F x slots, colours + 1) for one image and its corners."""
    sh = shape(model.event)
    x = to_tensor(face_maps(img_bgr, sh.face_matrices(keypoints, img_bgr.shape), sh))[None].to(device)
    return torch.softmax(model(x), dim=-1)[0].cpu().numpy()
