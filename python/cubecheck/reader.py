"""The learned sticker reader: a small network that reads all 27 visible stickers at once.

Input: the three visible faces, each straightened into a square with the same
homography as cell_centres(), so every sticker sits in a fixed spot. Output:
for each of the 27 stickers, 7 probabilities: the 6 colours (URFDLB order, as
in cube.py) and "covered" (a finger, or something else on top of it).

Each face goes through the same small convolutional network, which turns
every sticker into a list of 128 numbers describing it. Then a transformer
layer lets the 27 stickers look at each other before deciding, which is how
it can tell "redder than that one" instead of guessing a colour on its own.
See docs/how-it-works.md.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn

from .cube import FACES
from .keypoints import FACE_QUADS
from .verify import cell_centres, cube_size

FACE_PX = 72  # each face is straightened to 72 x 72 pixels: 24 per sticker
COVERED = 6  # class index after the 6 colours
IGNORE = -1  # no label: out of frame, seen edge-on, or half covered
CLASSES = FACES + "x"  # one letter per class, x = covered


def face_maps(img_bgr: np.ndarray, keypoints: np.ndarray, size: int = FACE_PX) -> np.ndarray:
    """The three visible faces straightened into squares, (3, size, size, 3) uint8 RGB."""
    square = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    out = []
    for quad in FACE_QUADS:
        H = cv2.getPerspectiveTransform(keypoints[list(quad)].astype(np.float32), square)
        face = cv2.warpPerspective(img_bgr, H, (size, size), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
        out.append(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    return np.stack(out)


def cell_labels(sample: dict, keypoints: np.ndarray, n: int = 3) -> np.ndarray:
    """What the reader should answer for each of the 3N² cells, from the dataset's sticker labels."""
    from .nxn_check import cell_centres as grid_centres

    stickers = [s for s in sample["labels"]["stickers"] if s["facing"] > 0]
    xy = np.array([[s["x"], s["y"]] for s in stickers])
    reach = 0.4 * cube_size(keypoints) / n  # under half a sticker from the cell centre
    out = np.full(3 * n * n, IGNORE)
    centres = cell_centres(keypoints) if n == 3 else grid_centres(keypoints, n)
    for cell, c in enumerate(centres):
        d = np.linalg.norm(xy - c, axis=1)
        k = int(d.argmin())
        s = stickers[k]
        if d[k] > reach or not s["inFrame"] or s["facing"] < 0.1:
            continue
        if s["visibleFraction"] >= 0.6:
            out[cell] = FACES.index(s["color"])
        elif s["visibleFraction"] < 0.4:
            out[cell] = COVERED
    return out


def _block(cin: int, cout: int, stride: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class StickerReader(nn.Module):
    def __init__(self, width: int = 128, layers: int = 2) -> None:
        super().__init__()
        # 72 -> 36 -> 18 -> 9 pixels, then 3 x 3: one feature vector per sticker.
        self.cnn = nn.Sequential(
            _block(3, 32, 1), _block(32, 32, 2), _block(32, 64, 1), _block(64, 64, 2),
            _block(64, width, 1), _block(width, width, 2), _block(width, width, 1), nn.AvgPool2d(3),
        )
        self.position = nn.Parameter(torch.zeros(27, width))  # where each sticker sits (centre, edge, corner, which face)
        layer = nn.TransformerEncoderLayer(width, 4, 2 * width, dropout=0.1, batch_first=True)
        self.mix = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(width, len(CLASSES))

    def forward(self, faces: torch.Tensor) -> torch.Tensor:
        """faces: (B, 3, 3, H, W) floats in 0..1 -> logits (B, 27, 7)."""
        b = faces.shape[0]
        x = self.cnn(faces.flatten(0, 1))  # (B*3, C, 3, 3)
        x = x.flatten(2).transpose(1, 2).reshape(b, 27, -1)  # face k, row j, column i: same order as cell_centres
        x = self.mix(x + self.position)
        return self.head(x)


def to_tensor(maps: np.ndarray) -> torch.Tensor:
    """(..., 3, H, W, 3) uint8 face maps -> (..., 3, 3, H, W) float tensor."""
    return torch.from_numpy(maps).movedim(-1, -3).float() / 255.0


class GridReader(nn.Module):
    """The sticker reader for any N×N cube (2 to 7).

    The same network as StickerReader, with one change. Each face is straightened
    to N×24 pixels (24 per sticker, as before), and the same convolution layers
    shrink it 24 times, so they give one feature vector per sticker at any size;
    r2's layers load unchanged. StickerReader learned 27 fixed "where am I"
    vectors, one per 3x3 cell. Here a small network computes that from where the
    sticker is: which face, how far along and down it (0..1), whether it's on the
    face's edge or corner, and whether it's an odd cube's fixed middle sticker.
    """

    def __init__(self, width: int = 128, layers: int = 2) -> None:
        super().__init__()
        r2 = StickerReader(width, layers)
        self.cnn, self.mix, self.head = r2.cnn, r2.mix, r2.head
        self.place = nn.Sequential(nn.Linear(len(self.places(3)[0]), width), nn.GELU(), nn.Linear(width, width))
        self._cache: dict[tuple[int, str], torch.Tensor] = {}

    @staticmethod
    def places(n: int) -> list[list[float]]:
        """What the position network is told about each of the 3N² cells (face k, row j, column i)."""
        out = []
        for k in range(3):
            for j in range(n):
                for i in range(n):
                    u, v = (i + 0.5) / n, (j + 0.5) / n
                    waves = [f(np.pi * m * t) for m in (1, 2, 3) for t in (u, v) for f in (np.sin, np.cos)]
                    edges = [i == 0, i == n - 1, j == 0, j == n - 1]
                    middle = n % 2 == 1 and i == n // 2 and j == n // 2
                    out.append([float(k == 0), float(k == 1), float(k == 2), u, v, *waves, *map(float, edges), float(middle), 1 / n])
        return out

    def forward(self, faces: torch.Tensor) -> torch.Tensor:
        """faces: (B, 3, 3, N*24, N*24) floats in 0..1 -> logits (B, 3N², 7)."""
        b, n = faces.shape[0], faces.shape[-1] // 24
        x = self.cnn(faces.flatten(0, 1))  # (B*3, C, N, N)
        x = x.flatten(2).transpose(1, 2).reshape(b, 3 * n * n, -1)
        key = (n, str(faces.device))
        if key not in self._cache:
            self._cache[key] = torch.tensor(self.places(n), dtype=faces.dtype, device=faces.device)
        x = self.mix(x + self.place(self._cache[key]))
        return self.head(x)

    @classmethod
    def from_r2(cls, path: str, device: str = "cpu") -> GridReader:
        """Start from a trained StickerReader: its convolution, mixing and answer layers carry over."""
        ckpt = torch.load(path, map_location=device)
        model = cls(**ckpt["config"])
        missing, unexpected = model.load_state_dict({k: v for k, v in ckpt["model"].items() if k != "position"}, strict=False)
        assert not unexpected and all(k.startswith("place.") for k in missing), (missing, unexpected)
        return model.to(device)


def load_reader(path: str, device: str = "cpu") -> StickerReader | GridReader:
    ckpt = torch.load(path, map_location=device)
    model = (GridReader if ckpt.get("grid") else StickerReader)(**ckpt["config"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


@torch.no_grad()
def read_stickers(model: StickerReader, img_bgr: np.ndarray, keypoints: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Probabilities (27, 7) for one image and its 7 corners."""
    x = to_tensor(face_maps(img_bgr, keypoints))[None].to(device)
    return torch.softmax(model(x), dim=-1)[0].cpu().numpy()


@torch.no_grad()
def read_grid(model: GridReader, img_bgr: np.ndarray, keypoints: np.ndarray, n: int, device: str = "cpu") -> np.ndarray:
    """Probabilities (3N², 7) for one image of an N×N cube and its 7 corners."""
    x = to_tensor(face_maps(img_bgr, keypoints, 24 * n))[None].to(device)
    return torch.softmax(model(x), dim=-1)[0].cpu().numpy()
