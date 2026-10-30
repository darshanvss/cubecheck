"""Export the finder, refiner and sticker reader to ONNX for the browser, and check they still agree.

    uv run --extra export python scripts/export_onnx.py ../runs/corners/c2 ../runs/reader/r2/best.pt \
        --out ../web/public/models --check ../data/raw/test-v1

ONNX is a file format for trained networks that many programs can run; the
website runs these files with ONNX Runtime Web, on the visitor's own device.
The exported networks do a little more than the PyTorch ones, so the browser
code stays simple:

- finder.onnx, refiner.onnx: a 256 x 256 RGB picture (0..1) in; the 7 corners
  (0..1 of the picture, after softmax and soft-argmax) and the cube-present
  probability out.
- reader.onnx: the 3 straightened faces, 72 x 72 RGB (0..1) in; 27 x 7
  probabilities out (6 colours, then covered).

meta.json records the sizes and settings the browser needs. With --check, every
network runs in PyTorch and in ONNX Runtime on real test renders, and the
largest difference is printed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from cubecheck.corners import IN_PX, MIN_PRESENT, REFINE_MARGIN, CornerFinder, full_view, refine_view, soft_argmax, to_input, warp
from cubecheck.reader import CLASSES, FACE_PX, face_maps, load_reader, read_stickers, to_tensor
from cubecheck.samples import load_samples


class Corners(nn.Module):
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        heat, present = self.net(image)
        return soft_argmax(heat), torch.sigmoid(present)


class Reader(nn.Module):
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, faces: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(faces), dim=-1)


def export(model: nn.Module, example: torch.Tensor, path: Path, inputs: list[str], outputs: list[str]) -> None:
    # Fixed shapes (one picture at a time, as the browser runs them): ONNX Runtime's WebGPU
    # backend mislaid the reader's channels when the batch size was left open.
    torch.onnx.export(model.eval(), (example,), str(path), input_names=inputs, output_names=outputs,
                      opset_version=17, dynamo=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corners", type=Path, help="folder with finder.pt and refiner.pt")
    ap.add_argument("reader", type=Path, help="sticker reader weights")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", type=Path, help="test renders to compare PyTorch and ONNX on")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    finder = CornerFinder.load(str(args.corners / "finder.pt"), str(args.corners / "refiner.pt"), "cpu")
    reader = load_reader(str(args.reader), "cpu")
    torch.backends.mha.set_fastpath_enabled(False)  # the fused attention kernel has no ONNX form
    image, faces = torch.rand(1, 3, IN_PX, IN_PX), torch.rand(1, 3, 3, FACE_PX, FACE_PX)
    nets = {"finder": Corners(finder.finder), "refiner": Corners(finder.refiner), "reader": Reader(reader)}
    for name in ("finder", "refiner"):
        export(nets[name], image, args.out / f"{name}.onnx", ["image"], ["corners", "present"])
    export(nets["reader"], faces, args.out / "reader.onnx", ["faces"], ["probs"])
    meta = {"corners": args.corners.name, "reader": args.reader.parent.name, "in_px": IN_PX, "face_px": FACE_PX,
            "refine_margin": REFINE_MARGIN, "passes": finder.passes, "min_present": MIN_PRESENT, "classes": CLASSES}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1))
    for f in sorted(args.out.glob("*.onnx")):
        print(f"{f.name}: {f.stat().st_size / 1e6:.1f} MB")

    if args.check:
        import onnxruntime as ort

        sessions = {name: ort.InferenceSession(str(args.out / f"{name}.onnx")) for name in nets}
        worst = {"corners (pixels of 256)": 0.0, "present": 0.0, "sticker probability": 0.0}
        for s in load_samples(args.check)[: args.n]:
            img = cv2.imread(str(args.check / s["image"]))
            found = finder.find(img)
            if found is None:
                continue
            views = {"finder": full_view(img.shape), "refiner": refine_view(found[0])}
            for name, m in views.items():
                x = to_input(warp(img, m))[None]
                with torch.no_grad():
                    want = [t.numpy() for t in nets[name](x)]
                got = sessions[name].run(None, {"image": x.numpy()})
                worst["corners (pixels of 256)"] = max(worst["corners (pixels of 256)"], float(np.abs(got[0] - want[0]).max() * IN_PX))
                worst["present"] = max(worst["present"], float(np.abs(got[1] - want[1]).max()))
            faces = to_tensor(face_maps(img, found[0]))[None]  # RGB, as the browser sends them
            with torch.no_grad():
                want = read_stickers(reader, img, found[0])  # the reader exactly as Python uses it
            got = sessions["reader"].run(None, {"faces": faces.numpy()})[0][0]
            worst["sticker probability"] = max(worst["sticker probability"], float(np.abs(got - want).max()))
        print("largest difference, PyTorch vs ONNX:", {k: round(v, 6) for k, v in worst.items()})


if __name__ == "__main__":
    main()
