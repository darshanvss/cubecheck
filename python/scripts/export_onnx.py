"""Export the finder, refiner and sticker reader to ONNX for the browser, and check they still agree.

    uv run --extra export python scripts/export_onnx.py ../runs/corners/c7 ../runs/reader/r4/best.pt \
        --out ../web/public/models --check ../data/raw/test-v1

ONNX is a file format for trained networks that many programs can run; the
website runs these files with ONNX Runtime Web, on the visitor's own device.
The exported networks do a little more than the PyTorch ones, so the browser
code stays simple:

- finder.onnx, refiner.onnx: a 256 x 256 RGB picture (0..1) in; the 7 corners
  (0..1 of the picture, after softmax and soft-argmax) and the cube-present
  probability out.
- reader-N.onnx, one per cube size N (2 to 7): the 3 straightened faces,
  24N x 24N RGB (0..1) in; 3N² x 7 probabilities out (6 colours, then covered).
  The page loads only the size it needs. One file can't serve every size: the
  reader's attention layers are traced with a fixed number of stickers. (An
  older StickerReader, r2, is exported as reader-3.onnx only.)

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
from cubecheck.reader import CLASSES, FACE_PX, GridReader, face_maps, load_reader, read_grid, read_stickers, to_tensor
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
    ap.add_argument("--min-find", type=float, default=0.5, help="the finder's own cube-present bar (corners.find)")
    ap.add_argument("--zoom", type=float, default=2.0, help="the finder's second look, this much closer, when it's unsure")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    finder = CornerFinder.load(str(args.corners / "finder.pt"), str(args.corners / "refiner.pt"), "cpu")
    reader = load_reader(str(args.reader), "cpu")
    torch.backends.mha.set_fastpath_enabled(False)  # the fused attention kernel has no ONNX form
    grid = isinstance(reader, GridReader)
    sizes = [2, 3, 4, 5, 6, 7] if grid else [3]
    nets = {"finder": Corners(finder.finder), "refiner": Corners(finder.refiner), "reader": Reader(reader)}
    for name in ("finder", "refiner"):
        export(nets[name], torch.rand(1, 3, IN_PX, IN_PX), args.out / f"{name}.onnx", ["image"], ["corners", "present"])
    for n in sizes:
        export(nets["reader"], torch.rand(1, 3, 3, 24 * n, 24 * n), args.out / f"reader-{n}.onnx", ["faces"], ["probs"])
    (args.out / "reader.onnx").unlink(missing_ok=True)  # the single 3x3 reader of earlier exports
    meta = {"corners": args.corners.name, "reader": args.reader.parent.name, "in_px": IN_PX, "face_px": FACE_PX,
            "sticker_px": FACE_PX // 3, "grid": grid, "sizes": sizes,
            "refine_margin": REFINE_MARGIN, "passes": finder.passes, "min_present": MIN_PRESENT,
            "min_find": args.min_find, "zoom": args.zoom, "two_faces": finder.two_faces, "classes": CLASSES}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1))
    for f in sorted(args.out.glob("*.onnx")):
        print(f"{f.name}: {f.stat().st_size / 1e6:.1f} MB")

    if args.check:
        import onnxruntime as ort

        sessions = {name: ort.InferenceSession(str(args.out / f"{name}.onnx")) for name in ("finder", "refiner")}
        sessions |= {f"reader-{n}": ort.InferenceSession(str(args.out / f"reader-{n}.onnx")) for n in sizes}
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
            for n in sizes:
                faces = to_tensor(face_maps(img, found[0], 24 * n))[None]  # RGB, as the browser sends them
                with torch.no_grad():  # the reader exactly as Python uses it
                    want = read_grid(reader, img, found[0], n) if grid else read_stickers(reader, img, found[0])
                got = sessions[f"reader-{n}"].run(None, {"faces": faces.numpy()})[0][0]
                worst["sticker probability"] = max(worst["sticker probability"], float(np.abs(got - want).max()))
        print("largest difference, PyTorch vs ONNX:", {k: round(v, 6) for k, v in worst.items()})


if __name__ == "__main__":
    main()
