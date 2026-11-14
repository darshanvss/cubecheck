"""Export the skewb, pyraminx and FTO models to ONNX for the browser (experimental), and check they agree.

    uv run --extra export python scripts/export_poly_onnx.py --out ../web/public/models \
        --check skewb=../data/poly/skewb/test --check pyram=... --check fto=...

For each puzzle (shapes.MODELS names its corner model in runs/corners/ and its reader in
runs/reader/), into <out>/<event>/:

- finder.onnx, refiner.onnx: as export_onnx.py's, with the puzzle's own corners (7 for a
  skewb, 4 for a pyraminx, 5 for an FTO).
- reader.onnx: the visible faces straightened (3 squares of 72 px for a skewb, 3 or 4
  upright triangles in 96 px squares for a pyraminx or FTO), RGB 0..1 in; per sticker the
  probability of each of the puzzle's colours and of "covered" out.
- meta.json: the settings the browser needs.

The shapes themselves (corners, sticker places, moves) go to the website separately, as
numbers: scripts/poly_web_data.py. With --check, every network runs in PyTorch and in ONNX
Runtime on a few test pictures and the largest difference is printed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from cubecheck.corners import IN_PX, MIN_PRESENT, REFINE_MARGIN, full_view, refine_view, to_input, warp
from cubecheck.poly_corners import PolyCornerFinder
from cubecheck.poly_reader import face_maps, face_px, load_poly_reader, read_stickers, to_tensor
from cubecheck.shapes import EVENTS, MODELS, shape

from export_onnx import Corners, export

RUNS = Path(__file__).resolve().parents[2] / "runs"


class PolyReaderOnnx(torch.nn.Module):
    """PolyReader.forward with its einsum written as a matrix product, then softmax. ONNX Runtime's
    WebGPU backend got the einsum wrong (stickers read differently from the CPU on 14 Nov 2026)."""

    def __init__(self, net) -> None:
        super().__init__()
        self.net = net

    def forward(self, faces: torch.Tensor) -> torch.Tensor:
        net = self.net
        b, f = faces.shape[:2]
        x = net.cnn(faces.flatten(0, 1))  # (B*F, C, h, w)
        masks = net.masks.to(x.dtype).flatten(1)  # (slots, h*w)
        x = torch.matmul(x.flatten(2), masks.t()).transpose(1, 2)  # (B*F, slots, C): each sticker's average
        x = x.reshape(b, f * masks.shape[0], -1)
        return torch.softmax(net.head(net.mix(x + net.position)), dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--events", nargs="*", default=list(EVENTS), choices=list(EVENTS))
    ap.add_argument("--check", action="append", default=[], help="event=folder of test pictures (*.jpg)")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--min-find", type=float, default=0.5)
    ap.add_argument("--zoom", type=float, default=2.0)
    args = ap.parse_args()
    checks = dict(c.split("=", 1) for c in args.check)
    torch.backends.mha.set_fastpath_enabled(False)  # the fused attention kernel has no ONNX form

    for event in args.events:
        sh, (corners, reader_name) = shape(event), MODELS[event]
        out = args.out / event
        out.mkdir(parents=True, exist_ok=True)
        finder = PolyCornerFinder.load(event, RUNS / "corners" / corners, "cpu")
        reader = load_poly_reader(str(RUNS / "reader" / reader_name / "best.pt"), "cpu")
        nets = {"finder": Corners(finder.finder), "refiner": Corners(finder.refiner), "reader": PolyReaderOnnx(reader)}
        for name in ("finder", "refiner"):
            export(nets[name], torch.rand(1, 3, IN_PX, IN_PX), out / f"{name}.onnx", ["image"], ["corners", "present"])
        s = face_px(sh)
        export(nets["reader"], torch.rand(1, sh.n_faces, 3, s, s), out / "reader.onnx", ["faces"], ["probs"])
        meta = {"event": event, "corners": corners, "reader": reader_name, "in_px": IN_PX, "face_px": s,
                "keypoints": sh.keypoints, "refine_margin": REFINE_MARGIN, "passes": finder.passes, "min_present": MIN_PRESENT,
                "min_find": args.min_find, "zoom": args.zoom, "colours": sh.colours}
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        for f in sorted(out.glob("*.onnx")):
            print(f"{event}/{f.name}: {f.stat().st_size / 1e6:.1f} MB")

        if event in checks:
            import onnxruntime as ort

            sessions = {name: ort.InferenceSession(str(out / f"{name}.onnx")) for name in nets}
            worst = {"corners (pixels of 256)": 0.0, "present": 0.0, "sticker probability": 0.0}
            pictures = sorted(Path(checks[event]).glob("*.jpg"))[: args.n]
            found_any = 0
            for path in pictures:
                img = cv2.imread(str(path))
                found = finder.find(img, min_find=args.min_find, zoom=args.zoom)
                if found is None:
                    continue
                found_any += 1
                for name, m in {"finder": full_view(img.shape), "refiner": refine_view(found[0])}.items():
                    x = to_input(warp(img, m))[None]
                    with torch.no_grad():
                        want = [t.numpy() for t in nets[name](x)]
                    got = sessions[name].run(None, {"image": x.numpy()})
                    worst["corners (pixels of 256)"] = max(worst["corners (pixels of 256)"], float(np.abs(got[0] - want[0]).max() * IN_PX))
                    worst["present"] = max(worst["present"], float(np.abs(got[1] - want[1]).max()))
                faces = to_tensor(face_maps(img, sh.face_matrices(found[0], img.shape), sh))[None]
                want = read_stickers(reader, img, found[0])
                got = sessions["reader"].run(None, {"faces": faces.numpy()})[0][0]
                worst["sticker probability"] = max(worst["sticker probability"], float(np.abs(got - want).max()))
            print(f"{event}: found in {found_any} of {len(pictures)}; largest difference, PyTorch vs ONNX:",
                  {k: round(v, 6) for k, v in worst.items()})


if __name__ == "__main__":
    main()
