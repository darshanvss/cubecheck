"""Train our own corner model (cubecheck/corners.py): the finder or the refiner.

    uv run python scripts/train_corners.py finder --train ../data/raw/<sets...> --test ../data/raw/<test sets...> --name c7
    uv run python scripts/train_corners.py refiner --train ../data/raw/<sets...> --test ../data/raw/<test sets...> --name c7

(docs/training.md has the exact commands for the published models.)

Each time a render is used it gets a fresh camera degradation (degrade.py) and
a fresh random view: zoomed, shifted, turned up to 30 degrees and mirrored
half the time. The corner labels move with the picture, and mirroring swaps
P2<->P6 and P3<->P5 (FLIP_IDX), because left and right swap places.

- finder: the whole frame, zoomed in or out a little. One time in twelve it
  gets a patch of background with no cube, so it learns to say "no cube".
- refiner: a crop around the cube, sometimes tighter, sometimes looser and
  off-centre, the way the finder's rough answer will be.

--fingers P paints, with probability P, one to three finger shapes over one or
two of the cube's corners (the labels stay where the corners really are). On
real photos c4 put a corner hidden by fingers where the fingers begin, not
where the cube's shape says it is; the renders' own fingers lie on the faces
and seldom cover a corner the way a real grip does.

--edge-grip P paints, with probability P (instead of --fingers' shapes), a row
of 3-4 fingers across one whole outer edge of the silhouette, both its corners
included: about a sticker row deep, sticking out past the edge toward a palm
(a side grip, or four fingers under the bottom edge). Half the time it's the
lowest edge. c5b moved those two corners 7-11% of the edge toward the fingers; the labels stay where the corners are.

Renders with no cube in the scene (the dataset's empty-* folders) teach
both stages to say "no cube" for whole frames and for crops: without them the
finder learned that every whole frame has a cube. Test them with --test-empty.

--augment gpu (the default) leaves the data workers only the cheap part. The
camera effects and the final view are done a whole batch at a time on the GPU
(gpu_augment.py), since on the CPU they kept the GPU waiting. The workers
still decode the JPEGs: --decode gpu moves that to the GPU too, but on an A40
(no hardware JPEG decoder) it was the slowest step, 147 ms per batch against
93 ms for everything else. Test renders always use the CPU
path (degrade.py), so scores stay comparable between runs.

The model doesn't have to know which face is white or yellow: the corners
may start at any of the three faces (the check tries all three, and the
sticker reader was trained on rotated faces). So every loss and score uses
whichever of the 3 rotations of the labels fits best. In a view showing exactly
two faces, the corner nearest the camera can be either end of the edge they
share (keypoints.alternative_keypoints), so those renders also accept that
labelling.

Scored on the test renders (fixed degradations, no random view) as corner
error in share of the cube's edge, like evaluate.py. Best epoch goes to
runs/corners/<name>/<stage>.pt.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, default_collate

from cubecheck import gpu_augment
from cubecheck.corners import FLIP, IN_PX, REFINE_MARGIN, CornerNet, apply, invert, soft_argmax, to_input, view_matrix, warp
from cubecheck.degrade import degrade
from cubecheck.keypoints import FACE_QUADS, alternative_keypoints
from cubecheck.samples import load_samples
from cubecheck.verify import cube_size


# One thread per data worker: OpenCV otherwise starts a thread per host core
# (96 on a RunPod machine allowed ~8), and they fight over the allowance.
cv2.setNumThreads(1)

# The renders' skin tones, light to dark, as BGR.
SKIN = np.array([[(c >> 0) & 255, (c >> 8) & 255, (c >> 16) & 255] for c in
                 (0xf1d3c2, 0xe8b99c, 0xd9a07e, 0xc68863, 0xa86b4a, 0x8a5536, 0x6b3f26, 0x4a2a1a)], np.float32)


def paint_fingers(img: np.ndarray, kp: np.ndarray, vis: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One to three shaded finger shapes across one or two of the cube's corners (in place)."""
    inframe = vis > 0
    if inframe.sum() < 4:
        return img
    edge = float(np.ptp(kp[inframe], axis=0).max()) / 1.6
    tone = SKIN[rng.integers(len(SKIN))] * rng.uniform(0.7, 1.15) * rng.uniform(0.96, 1.04, 3)  # lighting, slight tint
    pick = rng.choice(np.flatnonzero(inframe), size=min(int(rng.integers(1, 3)), int(inframe.sum())), replace=False)
    for c in pick:
        a = rng.uniform(0, 2 * np.pi)
        d, perp = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
        n = int(rng.integers(1, 4))
        width = edge * rng.uniform(0.2, 0.35)  # a finger is about a third of a cube's edge
        for f in range(n):
            centre = kp[c] + perp * (f - (n - 1) / 2) * width * 1.05 + d * rng.uniform(-0.25, 0.25) * edge
            length = edge * rng.uniform(0.5, 1.1)
            p0 = centre - d * length * rng.uniform(0.3, 0.7)
            p1 = p0 + d * length
            stroke(img, p0, p1, width, tone * rng.uniform(0.85, 1.1))
    return img


def stroke(img: np.ndarray, p0: np.ndarray, p1: np.ndarray, width: float, colour: np.ndarray,
           palm: tuple[float, float, float] | None = None) -> None:
    """One soft-edged finger from p0 to p1 (in place), darker towards its sides.
    With palm = (half length, half width, angle in degrees) it's an ellipse round p0 instead."""
    h, w = img.shape[:2]
    r = int((max(palm[:2]) if palm else width / 2)) + 4
    x0, y0 = np.clip(np.floor(np.minimum(p0, p1)) - r, 0, [w, h]).astype(int)
    x1, y1 = np.clip(np.ceil(np.maximum(p0, p1)) + r, 0, [w, h]).astype(int)
    if x1 <= x0 or y1 <= y0:
        return
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    if palm:
        cv2.ellipse(mask, tuple((p0 - [x0, y0]).astype(int)), (int(palm[0]), int(palm[1])), palm[2], 0, 360, 255, -1, cv2.LINE_AA)
    else:
        cv2.line(mask, tuple((p0 - [x0, y0]).astype(int)), tuple((p1 - [x0, y0]).astype(int)), 255, max(2, int(width)), cv2.LINE_AA)
    alpha = cv2.GaussianBlur(mask, (0, 0), max(1.0, width / 15)).astype(np.float32)[..., None] / 255
    shade = colour * (0.65 + 0.35 * alpha)
    roi = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = np.clip(roi * (1 - alpha) + shade * alpha, 0, 255).astype(np.uint8)


def paint_edge_grip(img: np.ndarray, kp: np.ndarray, vis: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """3-4 fingers side by side across one whole outer edge, a sticker row deep, reaching back past it (in place)."""
    inframe = vis > 0
    edges = [(i, (i + 1) % 6) for i in range(6) if inframe[i] and inframe[(i + 1) % 6]]
    lengths = [float(np.linalg.norm(kp[b] - kp[a])) for a, b in edges]
    edges = [e for e, length in zip(edges, lengths) if length > 0.4 * max(lengths, default=0)]  # not a sliver's edge
    if not edges:
        return img
    if rng.random() < 0.5:  # fingers under the cube
        edge = max(edges, key=lambda e: kp[e[0], 1] + kp[e[1], 1])
    else:
        edge = edges[int(rng.integers(len(edges)))]
    a, b = kp[edge[0]], kp[edge[1]]
    along = (b - a) / (np.linalg.norm(b - a) + 1e-9)
    inward = np.array([-along[1], along[0]])
    if inward @ (kp[inframe].mean(0) - (a + b) / 2) < 0:  # toward the cube
        inward = -inward
    length = float(np.linalg.norm(b - a))
    quad = kp[list(next(q for q in FACE_QUADS if edge[0] in q and edge[1] in q))]
    row = max(abs(float((quad.mean(0) - (a + b) / 2) @ inward)) * 2 / n, 0.5 * length / n)  # one sticker row, across the edge
    tone = SKIN[rng.integers(len(SKIN))] * rng.uniform(0.7, 1.15) * rng.uniform(0.96, 1.04, 3)
    count = int(rng.integers(3, 5))
    span = rng.uniform(1.0, 1.15)  # end to end, both corners under a finger
    width = length * span / count * rng.uniform(1.0, 1.2)
    depth = row * rng.uniform(0.7, 1.5)
    reach = width * rng.uniform(1.5, 3.0)  # how far back the fingers go before the palm
    if rng.random() < 0.6:
        centre = (a + b) / 2 - inward * (reach + width)
        angle = float(np.degrees(np.arctan2(along[1], along[0])))
        stroke(img, centre, centre, width, tone * rng.uniform(0.8, 0.95), palm=(length * span / 2, width * 1.3, angle))
    for j in range(count):
        t = (j + 0.5) / count * span - (span - 1) / 2 + rng.uniform(-0.03, 0.03)
        base = a + (b - a) * t
        tilt = np.radians(rng.uniform(-15, 15))
        d = inward * np.cos(tilt) + along * np.sin(tilt)
        tip = base + d * depth * rng.uniform(0.85, 1.15)
        stroke(img, base - d * reach, tip, width * rng.uniform(0.85, 1.05), tone * rng.uniform(0.85, 1.1))
    return img


class Views(Dataset):
    def __init__(self, raws: list[Path], stage: str, train: bool, fraction: float = 1.0, gpu: bool = False,
                 gpu_decode: bool = False, fingers: float = 0.0, edge_grip: float = 0.0) -> None:
        pairs = []
        for raw in raws:
            share = fraction
            if "@" in raw.name:  # "<set>@0.2": only this share of that set
                raw, share = raw.with_name(raw.name.split("@")[0]), fraction * float(raw.name.split("@")[1])
            found = load_samples(raw)
            pairs += [(raw, s) for s in found[: max(1, int(len(found) * share))]]
        if stage == "refiner":
            pairs = [(raw, s) for raw, s in pairs if s.get("empty") or all(k[2] > 0 for k in s["labels"]["keypoints"])]
        samples = [s for _, s in pairs]
        self.paths = [str(raw / s["image"]) for raw, s in pairs]
        self.kps = np.array([s["labels"]["keypoints"] for s in samples], np.float32)  # (N, 7, 3): x, y, visibility
        # The other valid labelling of exactly-two-face views; the labels themselves when there's none.
        alts = [None if s.get("empty") else alternative_keypoints(s["labels"]) for s in samples]
        self.alts = np.array([a if a is not None else k for a, k in zip(alts, self.kps.tolist())], np.float32)
        self.empty = np.array([bool(s.get("empty")) for s in samples])
        self.sizes = [(s["width"], s["height"]) for s in samples]
        self.n = [s.get("cube", {}).get("size", 3) for s in samples]
        self.stage, self.train, self.gpu = stage, train, gpu and train
        self.gpu_decode = self.gpu and gpu_decode
        self.fingers = fingers if train else 0.0
        self.edge_grip = edge_grip if train else 0.0

    def paint(self, img: np.ndarray, kp: np.ndarray, vis: np.ndarray, i: int, rng: np.random.Generator) -> np.ndarray:
        """A whole-edge grip (--edge-grip), else fingers over corners (--fingers), else nothing."""
        r = rng.random()
        if r < self.edge_grip:
            return paint_edge_grip(img, kp, vis, self.n[i], rng)
        if r < self.edge_grip + self.fingers:
            return paint_fingers(img, kp, vis, rng)
        return img

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> dict:
        rng = np.random.default_rng() if self.train else np.random.default_rng([999, i])
        kp, vis = self.kps[i, :, :2], self.kps[i, :, 2]
        alt, alt_vis = self.alts[i, :, :2], self.alts[i, :, 2]
        empty = bool(self.empty[i])
        if self.gpu_decode:  # the GPU decodes it; only its size is needed here
            img, (w, h) = None, self.sizes[i]
        elif self.stage == "finder":
            # The finder sees the frame shrunk ~5x, so decoding at half size loses nothing it uses.
            img = cv2.imread(self.paths[i], cv2.IMREAD_REDUCED_COLOR_2)
            kp, alt = kp / 2, alt / 2
            if not empty:
                img = self.paint(img, kp, vis, i, rng)
            if not self.gpu:
                img, _ = degrade(img, rng)
        else:
            img = cv2.imread(self.paths[i])
            if not empty:
                img = self.paint(img, kp, vis, i, rng)
        if img is not None:
            h, w = img.shape[:2]
        inframe = vis > 0
        lo, hi = (kp[inframe].min(0), kp[inframe].max(0)) if inframe.any() else (np.zeros(2), np.zeros(2))
        present = 0.0 if empty else 1.0
        if self.stage == "finder":
            centre, side = np.array([w / 2, h / 2]), float(max(w, h))
            if self.train:
                side *= rng.uniform(0.75, 1.25)
                centre = centre + rng.uniform(-0.15, 0.15, 2) * side
                if not empty and rng.random() < 1 / 12:  # background only: no cube in view
                    for _ in range(20):
                        side = rng.uniform(0.25, 0.6) * h
                        centre = rng.uniform([side / 2, side / 2], [w - side / 2, h - side / 2])
                        if np.any(np.abs(centre - (lo + hi) / 2) > (hi - lo) / 2 + side / 2):
                            present = 0.0
                            break
        elif empty:  # a crop the size a cube's would be, anywhere in a scene without one
            side = rng.uniform(0.08, 0.6) * h * REFINE_MARGIN
            centre = rng.uniform([0, 0], [w, h])
        else:
            centre, side = (lo + hi) / 2, float(max(hi - lo))
            if self.train:
                side *= rng.uniform(1.15, 1.7)
                centre = centre + rng.uniform(-0.1, 0.1, 2) * side
            else:
                side *= REFINE_MARGIN
        angle = rng.uniform(-30, 30) if self.train else 0.0
        flip = bool(self.train and rng.random() < 0.5)
        m = view_matrix(centre, side, angle, flip)
        item: dict = {}
        if self.gpu:
            if self.gpu_decode:
                info = gpu_augment.canvas_box(m, w, h)
                item["jpeg"] = torch.from_numpy(np.fromfile(self.paths[i], np.uint8))
            else:
                canvas, info = gpu_augment.cut_canvas(img, m)
                item["canvas"] = torch.from_numpy(canvas)
            item.update({k: torch.from_numpy(np.asarray(v)) for k, v in info.items()})
            item.update({f"fx_{k}": torch.from_numpy(v) for k, v in gpu_augment.draw_effects(rng, float(info["k"])).items()})
            kp_view, alt_view = kp, alt
        elif self.stage == "refiner":
            # Degrade only the part of the frame this view shows (plus a little room for blur).
            seen = apply(invert(m), np.array([[0, 0], [IN_PX, 0], [IN_PX, IN_PX], [0, IN_PX]], float))
            x0, y0 = np.clip(np.floor(seen.min(0)) - 16, 0, [w - 1, h - 1]).astype(int)
            x1, y1 = np.clip(np.ceil(seen.max(0)) + 16, [x0 + 1, y0 + 1], [w, h]).astype(int)
            img, _ = degrade(img[y0:y1, x0:x1], rng, frame=(int(x0), int(y0), w, h))
            m = np.hstack([m[:, :2], (m[:, :2] @ [x0, y0] + m[:, 2])[:, None]])
            kp_view, alt_view = kp - [x0, y0], alt - [x0, y0]
            item["x"] = to_input(warp(img, m))
        else:
            kp_view, alt_view = kp, alt
            item["x"] = to_input(warp(img, m))
        target, alt_target = apply(m, kp_view), apply(m, alt_view)
        if flip:
            target, vis, alt_target, alt_vis = target[FLIP], vis[FLIP], alt_target[FLIP], alt_vis[FLIP]
        mask = (vis > 0) & np.all((target >= 0) & (target < IN_PX), axis=1) & (present > 0)
        alt_mask = (alt_vis > 0) & np.all((alt_target >= 0) & (alt_target < IN_PX), axis=1) & (present > 0)
        item.update(target=torch.from_numpy(target / IN_PX).float(), mask=torch.from_numpy(mask), present=torch.tensor(present),
                    alt_target=torch.from_numpy(alt_target / IN_PX).float(), alt_mask=torch.from_numpy(alt_mask),
                    m=torch.from_numpy(m).float(), kp=torch.from_numpy(kp_view.astype(np.float32)),
                    alt_kp=torch.from_numpy(alt_view.astype(np.float32)),
                    edge=torch.tensor(cube_size(kp) if inframe.all() else np.nan))
        return item


# The 3 ways to start the corner order: shift P1..P6 by two places (one face), C stays.
ROTATIONS = [[(i + 2 * r) % 6 for i in range(6)] + [6] for r in range(3)]


def heat_loss(heat: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    """Cross-entropy between each predicted heatmap and a small blob at its corner: (B,) per render."""
    b, k, h, w = heat.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=heat.device), torch.arange(w, device=heat.device), indexing="ij")
    cx, cy = target[..., 0, None, None] * w - 0.5, target[..., 1, None, None] * h - 0.5
    blob = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2)).flatten(2)
    blob = blob / blob.sum(-1, keepdim=True).clamp_min(1e-9)
    ce = -(blob * heat.flatten(2).log_softmax(-1)).sum(-1)
    return (ce * mask).sum(1) / mask.sum(1).clamp_min(1)


def corner_loss(heat: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                alt_target: torch.Tensor | None = None, alt_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Heatmap + position loss under the best of the 3 label rotations (and of the two
    labellings of an exactly-two-face view), averaged over the batch."""
    xy = soft_argmax(heat)
    losses = []
    labellings = [(target, mask)] + ([(alt_target, alt_mask)] if alt_target is not None else [])
    for tt, mm in labellings:
        for rot in ROTATIONS:
            t, m = tt[:, rot], mm[:, rot]
            coord = ((xy - t).abs().sum(-1) * m).sum(1) / m.sum(1).clamp_min(1)
            losses.append(heat_loss(heat, t, m) + 100 * coord)
    return torch.stack(losses).min(0).values.mean()


@torch.no_grad()
def evaluate(net: CornerNet, batches: list[dict], device: str) -> dict:
    """Corner error on renders with a cube, and how often it says there's no cube there."""
    net.eval()
    errors, present = [], []
    for b in batches:
        heat, logit = net(b["x"].to(device).float() / 255)
        present.append(torch.sigmoid(logit).cpu())
        xy = soft_argmax(heat.float()).cpu().numpy() * IN_PX
        for j in range(len(xy)):
            if np.isnan(float(b["edge"][j])):
                continue
            pred = apply(invert(b["m"][j].numpy().astype(np.float64)), xy[j])
            errs = [np.linalg.norm(pred - b[key][j].numpy()[rot], axis=1)[b[mk][j].numpy()[rot]] / float(b["edge"][j])
                    for key, mk in (("kp", "mask"), ("alt_kp", "alt_mask")) for rot in ROTATIONS]
            errors.extend(min(errs, key=np.mean))
    e = np.array(errors) * 100
    return {"median": float(np.median(e)), "p95": float(np.percentile(e, 95)), "over5": float(np.mean(e > 5) * 100),
            "missed": float((torch.cat(present) < 0.5).float().mean() * 100)}


@torch.no_grad()
def false_cubes(net: CornerNet, batches: list[dict], device: str) -> float:
    """Share of views without a cube where it says there is one, in %."""
    net.eval()
    said = torch.cat([torch.sigmoid(net(b["x"].to(device).float() / 255)[1]).cpu() for b in batches])
    return float((said >= 0.5).float().mean() * 100)


def collate(items: list[dict]) -> dict:
    """Like the default, but JPEG files differ in length, so they stay a list."""
    jpegs = [it.pop("jpeg") for it in items] if "jpeg" in items[0] else None
    batch = default_collate(items)
    if jpegs is not None:
        batch["jpeg"] = jpegs
    return batch


def network_input(batch: dict, device: str, gpu: bool) -> tuple[torch.Tensor, dict]:
    """A batch from the loader -> network input on the device (and the rest of the batch there too)."""
    jpegs, box = batch.pop("jpeg", None), batch.pop("box", None)
    b = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    if not gpu:
        return b["x"], b
    if jpegs is not None:
        x = gpu_augment.cut_canvas_gpu(gpu_augment.decode_gpu(jpegs, device), box)
    else:
        x = b["canvas"].permute(0, 3, 1, 2).float() / 255
    return gpu_augment.render(x, b), b


def cache(loader: DataLoader) -> list[dict]:
    """Test views never change, so make them once (as bytes, to save memory)."""
    return [{**b, "x": (b["x"] * 255).round().to(torch.uint8)} for b in loader]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["finder", "refiner"])
    ap.add_argument("--train", type=Path, nargs="+", required=True, help="one or more render folders (<folder>@0.3: a share of it)")
    ap.add_argument("--test", type=Path, nargs="+", required=True)
    ap.add_argument("--test-empty", type=Path, nargs="*", default=[], help="renders without a cube")
    ap.add_argument("--augment", choices=["gpu", "cpu"], default="gpu", help="where the camera effects run")
    ap.add_argument("--decode", choices=["gpu", "cpu"], default="cpu", help="where JPEGs are decoded (gpu: NVIDIA only)")
    ap.add_argument("--init", type=Path, help="start from these weights instead of from scratch")
    ap.add_argument("--max-batches", type=int, default=0, help="stop each epoch early (speed checks)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fraction", type=float, default=1.0, help="share of the training renders (quick checks)")
    ap.add_argument("--fingers", type=float, default=0.0, help="share of training views with finger shapes painted over corners")
    ap.add_argument("--edge-grip", type=float, default=0.0, help="share of training views with fingers painted across one whole outer edge")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("../runs/corners"))
    ap.add_argument("--name", default="corners")
    args = ap.parse_args()

    gpu = args.augment == "gpu"
    gpu_decode = gpu and args.decode == "gpu" and args.device == "cuda"
    train = Views(args.train, args.stage, True, args.fraction, gpu=gpu, gpu_decode=gpu_decode, fingers=args.fingers,
                  edge_grip=args.edge_grip)
    two_faces = bool((train.alts != train.kps).any())  # trained on exactly-two-face views: CornerFinder accepts that layout
    kw = dict(batch_size=args.batch, num_workers=args.workers, persistent_workers=args.workers > 0,
              pin_memory=args.device == "cuda")
    train_loader = DataLoader(train, shuffle=True, drop_last=True, collate_fn=collate, **kw)
    test = cache(DataLoader(Views(args.test, args.stage, False), shuffle=False, **kw))
    test_empty = cache(DataLoader(Views(args.test_empty, args.stage, False), shuffle=False, **kw)) if args.test_empty else []

    config = {}
    net = CornerNet(**config).to(args.device)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=args.device)["model"])
    amp = args.device == "cuda"  # bfloat16 for the network on NVIDIA GPUs; losses stay in float32
    torch.backends.cudnn.benchmark = True
    steps = min(len(train_loader), args.max_batches or len(train_loader))
    print(f"{args.stage}: {len(train)} training renders ({int(train.empty.sum())} without a cube), {sum(len(b['x']) for b in test)} test views, "
          f"{sum(p.numel() for p in net.parameters()):,} numbers to learn, camera effects on the {args.augment.upper()}"
          f"{', JPEGs decoded on the GPU' if gpu_decode else ''}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * steps)

    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    best, log = float("inf"), []
    for epoch in range(1, args.epochs + 1):
        net.train()
        t0, total, waited = time.perf_counter(), torch.zeros((), device=args.device), 0.0
        batches = iter(train_loader)
        for _ in range(steps):
            t_wait = time.perf_counter()
            batch = next(batches)
            waited += time.perf_counter() - t_wait
            x, b = network_input(batch, args.device, gpu)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                heat, logit = net(x)
            loss = corner_loss(heat.float(), b["target"], b["mask"], b["alt_target"], b["alt_mask"])
            loss = loss + F.binary_cross_entropy_with_logits(logit.float(), b["present"].float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.detach()
        train_seconds = time.perf_counter() - t0
        scores = evaluate(net, test, args.device)
        if test_empty:
            scores["false_cube"] = false_cubes(net, test_empty, args.device)
        row = {"epoch": epoch, "loss": round(float(total) / steps, 4), **{k: round(v, 3) for k, v in scores.items()},
               "seconds": round(time.perf_counter() - t0), "train_seconds": round(train_seconds, 1),
               "waiting_for_data": round(waited, 1)}
        log.append(row)
        print(json.dumps(row), flush=True)
        # corner accuracy, plus a penalty for each % of missed cubes or cubes seen where there are none
        score = scores["median"] + scores["p95"] / 4 + (scores["missed"] + scores.get("false_cube", 0)) / 10
        if score < best:
            best = score
            torch.save({"model": net.state_dict(), "config": config, "epoch": epoch, "scores": scores,
                        "two_faces": two_faces}, out / f"{args.stage}.pt")
        (out / f"{args.stage}_log.json").write_text(json.dumps(log, indent=1))
    print(f"best epoch saved -> {out / (args.stage + '.pt')}")


if __name__ == "__main__":
    main()
