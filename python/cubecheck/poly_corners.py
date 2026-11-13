"""The corner model for skewb, pyraminx and FTO (experimental): corners.CornerFinder with a
shape's own corners (shapes.py) instead of a cube's 7.

The network, the finder-then-refiner passes and the second closer look are the cube's.
Only two things differ: how many corners the network gives (CornerNet's k: 7 for a
skewb, which is a cube; 4 for a pyraminx; 5 for an FTO), and what counts as a properly
shaped answer (shapes.Shape.well_formed in place of corners.cube_shaped).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .corners import MIN_PRESENT, CornerFinder, CornerNet
from .shapes import FOUND_FACES, shape


class PolyCornerFinder(CornerFinder):
    def __init__(self, event: str, finder: CornerNet, refiner: CornerNet, device: str = "cpu", passes: int = 2) -> None:
        super().__init__(finder, refiner, device, passes)
        self.event, self.shape = event, shape(event)

    @classmethod
    def load(cls, event: str, folder: Path, device: str = "cpu") -> PolyCornerFinder:
        nets = []
        for stage in ("finder", "refiner"):
            ckpt = torch.load(str(folder / f"{stage}.pt"), map_location=device)
            net = CornerNet(**ckpt["config"])
            net.load_state_dict(ckpt["model"])
            nets.append(net.to(device))
        return cls(event, *nets, device=device)

    def _checked(self, kp: np.ndarray, present: float, min_present: float = MIN_PRESENT):
        return (kp, present) if present >= min_present and self.shape.well_formed(kp, FOUND_FACES[self.event]) else None
