"""Train the corner model (cubecheck/corners.py): the finder or the refiner.

    uv run python scripts/train_corners.py finder --train ../data/raw/v1 ../data/raw/easy-v1 --test ../data/raw/test-v1 --name c1
    uv run python scripts/train_corners.py refiner --train ../data/raw/v1 ../data/raw/easy-v1 --test ../data/raw/test-v1 --name c1

Each time a render is used it gets a fresh camera degradation (degrade.py) and
a fresh random view: zoomed, shifted, turned up to 30 degrees and mirrored
half the time. The corner labels move with the picture, and mirroring swaps
P2<->P6 and P3<->P5 (FLIP_IDX), because left and right swap places.

- finder: the whole frame, zoomed in or out a little. One time in twelve it
  gets a patch of background with no cube, so it learns to say "no cube".
- refiner: a crop around the cube, sometimes tighter, sometimes looser and
  off-centre, the way the finder's rough answer will be.

Renders with no cube in the scene (the dataset's empty-* folders) teach
both stages to say "no cube" for whole frames and for crops: without them the
finder learned that every whole frame has a cube. Test them with --test-empty.

--augment gpu (the default) leaves the data workers only the cheap part. The
camera effects and the final view are done a whole batch at a time on the GPU
(gpu_augment.py), since on the CPU they kept the GPU waiting. The workers
still decode the JPEGs: --decode gpu moves that to the GPU too, but on an A40
(no hardware JPEG decoder) it was the slowest step, 147 ms per batch against
93 ms for everything else (docs/training.md). Test renders always use the CPU
path (degrade.py), so scores stay comparable between runs.

The model doesn't have to know which face is white or yellow: the corners
may start at any of the three faces (the check tries all three, and the
sticker reader was trained on rotated faces). So every loss and score uses
whichever of the 3 rotations of the labels fits best.

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
from cubecheck.samples import load_samples
from cubecheck.verify import cube_size


# One thread per data worker: OpenCV otherwise starts a thread per host core
# (96 on a RunPod machine allowed ~8), and they fight over the allowance.
cv2.setNumThreads(1)

class Views(Dataset):
    def __init__(self, raws: list[Path], stage: str, train: bool, fraction: float = 1.0, gpu: bool = False,
                 gpu_decode: bool = False) -> None:
        pairs = []
        for raw in raws:
            found = load_samples(raw)
            pairs += [(raw, s) for s in found[: max(1, int(len(found) * fraction))]]
        if stage == "refiner":
            pairs = [(raw, s) for raw, s in pairs if s.get("empty") or all(k[2] > 0 for k in s["labels"]["keypoints"])]
        samples = [s for _, s in pairs]
        self.paths = [str(raw / s["image"]) for raw, s in pairs]
        self.kps = np.array([s["labels"]["keypoints"] for s in samples], np.float32)  # (N, 7, 3): x, y, visibility
        self.empty = np.array([bool(s.get("empty")) for s in samples])
        self.sizes = [(s["width"], s["height"]) for s in samples]
        self.stage, self.train, self.gpu = stage, train, gpu and train
        self.gpu_decode = self.gpu and gpu_decode

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> dict:
        rng = np.random.default_rng() if self.train else np.random.default_rng([999, i])
        kp, vis = self.kps[i, :, :2], self.kps[i, :, 2]
        empty = bool(self.empty[i])
        if self.gpu_decode:  # the GPU decodes it; only its size is needed here
            img, (w, h) = None, self.sizes[i]
        elif self.stage == "finder":
            # The finder sees the frame shrunk ~5x, so decoding at half size loses nothing it uses.
            img = cv2.imread(self.paths[i], cv2.IMREAD_REDUCED_COLOR_2)
            if not self.gpu:
                img, _ = degrade(img, rng)
            kp = kp / 2
        else:
            img = cv2.imread(self.paths[i])
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
            kp_view = kp
        elif self.stage == "refiner":
            # Degrade only the part of the frame this view shows (plus a little room for blur).
            seen = apply(invert(m), np.array([[0, 0], [IN_PX, 0], [IN_PX, IN_PX], [0, IN_PX]], float))
            x0, y0 = np.clip(np.floor(seen.min(0)) - 16, 0, [w - 1, h - 1]).astype(int)
            x1, y1 = np.clip(np.ceil(seen.max(0)) + 16, [x0 + 1, y0 + 1], [w, h]).astype(int)
            img, _ = degrade(img[y0:y1, x0:x1], rng, frame=(int(x0), int(y0), w, h))
            m = np.hstack([m[:, :2], (m[:, :2] @ [x0, y0] + m[:, 2])[:, None]])
            kp_view = kp - [x0, y0]
            item["x"] = to_input(warp(img, m))
        else:
            kp_view = kp
            item["x"] = to_input(warp(img, m))
        target = apply(m, kp_view)
        if flip:
            target, vis = target[FLIP], vis[FLIP]
        mask = (vis > 0) & np.all((target >= 0) & (target < IN_PX), axis=1) & (present > 0)
        item.update(target=torch.from_numpy(target / IN_PX).float(), mask=torch.from_numpy(mask), present=torch.tensor(present),
                    m=torch.from_numpy(m).float(), kp=torch.from_numpy(kp_view.astype(np.float32)),
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


def corner_loss(heat: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Heatmap + position loss under the best of the 3 label rotations, averaged over the batch."""
    xy = soft_argmax(heat)
    losses = []
    for rot in ROTATIONS:
        t, m = target[:, rot], mask[:, rot]
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
            ok = b["mask"][j].numpy()
            errs = [np.linalg.norm(pred - b["kp"][j].numpy()[rot], axis=1)[ok[rot]] / float(b["edge"][j]) for rot in ROTATIONS]
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
    ap.add_argument("--train", type=Path, nargs="+", required=True, help="one or more dataset folders")
    ap.add_argument("--test", type=Path, nargs="+", required=True)
    ap.add_argument("--test-empty", type=Path, nargs="*", default=[], help="renders without a cube (empty-* folders)")
    ap.add_argument("--augment", choices=["gpu", "cpu"], default="gpu", help="where the camera effects run")
    ap.add_argument("--decode", choices=["gpu", "cpu"], default="cpu", help="where JPEGs are decoded (gpu: NVIDIA only)")
    ap.add_argument("--init", type=Path, help="start from these weights instead of from scratch")
    ap.add_argument("--max-batches", type=int, default=0, help="stop each epoch early (speed checks)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fraction", type=float, default=1.0, help="share of the training renders (quick checks)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("../runs/corners"))
    ap.add_argument("--name", default="corners")
    args = ap.parse_args()

    gpu = args.augment == "gpu"
    gpu_decode = gpu and args.decode == "gpu" and args.device == "cuda"
    train = Views(args.train, args.stage, True, args.fraction, gpu=gpu, gpu_decode=gpu_decode)
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
            loss = corner_loss(heat.float(), b["target"], b["mask"])
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
            torch.save({"model": net.state_dict(), "config": config, "epoch": epoch, "scores": scores}, out / f"{args.stage}.pt")
        (out / f"{args.stage}_log.json").write_text(json.dumps(log, indent=1))
    print(f"best epoch saved -> {out / (args.stage + '.pt')}")


if __name__ == "__main__":
    main()
