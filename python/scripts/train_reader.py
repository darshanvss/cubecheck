"""Train the sticker reader (cubecheck/reader.py) on the synthetic renders.

    uv run python scripts/train_reader.py --train ../data/raw/v1 ../data/raw/easy-v1 --test ../data/raw/test-v1 --name r2

Runs on the Mac (mps). Every time a render is used it gets a fresh random
camera degradation (degrade.py) and its corners are nudged a little, the way
the corner model's answers are a little off, before the faces are
straightened. So the reader never sees exactly the same picture twice.

The faces are also rotated one step around the near corner at random, so the
reader can't learn shortcuts like "the first face's centre is white or yellow".

The test renders are scored with fixed degradations and the labelled corners;
the best epoch is saved to runs/reader/<name>/best.pt.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cubecheck.degrade import degrade
from cubecheck.reader import COVERED, IGNORE, StickerReader, cell_labels, face_maps, to_tensor
from cubecheck.samples import load_samples
from cubecheck.verify import cube_size


# One thread per data worker: OpenCV otherwise starts a thread per host core
# (96 on a RunPod machine allowed ~8), and they fight over the allowance.
cv2.setNumThreads(1)

class Maps(Dataset):
    def __init__(self, raws: list[Path], train: bool, jitter: float, fraction: float = 1.0) -> None:
        pairs = []
        for raw in raws:
            found = [s for s in load_samples(raw) if all(k[2] > 0 for k in s["labels"]["keypoints"])]  # all 7 corners in frame
            pairs += [(raw, s) for s in found[: max(1, int(len(found) * fraction))]]
        samples = [s for _, s in pairs]
        self.paths = [str(raw / s["image"]) for raw, s in pairs]
        self.kps = np.array([np.array(s["labels"]["keypoints"], np.float32)[:, :2] for s in samples])
        self.labels = np.array([cell_labels(s, kp) for s, kp in zip(samples, self.kps)])
        self.train, self.jitter = train, jitter

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        rng = np.random.default_rng() if self.train else np.random.default_rng([7, i])
        img = cv2.imread(self.paths[i])
        kp, y = self.kps[i], self.labels[i]
        if self.train:
            kp = kp + rng.normal(0, self.jitter * cube_size(kp), kp.shape).astype(np.float32)
        # Degrade only the cube's part of the frame (plus room for blur): the same result
        # there as degrading the whole frame, at a fraction of the cost.
        h, w = img.shape[:2]
        x0, y0 = np.clip(np.floor(kp.min(0)) - 24, 0, [w - 1, h - 1]).astype(int)
        x1, y1 = np.clip(np.ceil(kp.max(0)) + 24, [x0 + 1, y0 + 1], [w, h]).astype(int)
        img, _ = degrade(img[y0:y1, x0:x1], rng, frame=(int(x0), int(y0), w, h))
        maps = face_maps(img, kp - [x0, y0])
        if self.train:
            r = int(rng.integers(3))
            maps, y = np.roll(maps, -r, axis=0), np.roll(y, -9 * r)
        return to_tensor(maps), torch.from_numpy(y.astype(np.int64))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict:
    model.eval()
    right = total = covered_right = covered_total = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(-1).cpu()
        colour = (y >= 0) & (y < COVERED)
        right += int((pred[colour] == y[colour]).sum())
        total += int(colour.sum())
        cov = y == COVERED
        covered_right += int((pred[cov] == COVERED).sum())
        covered_total += int(cov.sum())
    return {"stickers_right": right / max(total, 1), "covered_found": covered_right / max(covered_total, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, nargs="+", required=True, help="one or more dataset folders")
    ap.add_argument("--test", type=Path, nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--jitter", type=float, default=0.015, help="corner nudge, as a share of the cube's size")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fraction", type=float, default=1.0, help="share of the training renders (quick checks)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--name", default="reader")
    args = ap.parse_args()

    t0 = time.perf_counter()
    train = Maps(args.train, True, args.jitter, args.fraction)
    test = Maps(args.test, False, 0.0)
    counts = np.bincount(train.labels[train.labels != IGNORE], minlength=7)
    print(f"{len(train)} training renders, {len(test)} test renders, labels ready in {time.perf_counter() - t0:.0f}s")
    print("training stickers per class (URFDLB, covered):", counts.tolist())

    kw = dict(batch_size=args.batch, num_workers=args.workers, persistent_workers=args.workers > 0)
    train_loader = DataLoader(train, shuffle=True, drop_last=True, **kw)
    test_loader = DataLoader(test, shuffle=False, **kw)

    config = {"width": 128, "layers": 2}
    model = StickerReader(**config).to(args.device)
    print(f"{sum(p.numel() for p in model.parameters()):,} numbers to learn")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(train_loader))
    # Label smoothing keeps the probabilities honest: "certain" only when it should be.
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE, label_smoothing=0.05)

    out = Path("../runs/reader") / args.name
    out.mkdir(parents=True, exist_ok=True)
    best, log = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total = time.perf_counter(), 0.0
        for x, y in train_loader:
            logits = model(x.to(args.device))
            loss = loss_fn(logits.flatten(0, 1), y.to(args.device).flatten())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        scores = evaluate(model, test_loader, args.device)
        row = {"epoch": epoch, "loss": round(total / len(train_loader), 4), **{k: round(v, 4) for k, v in scores.items()},
               "seconds": round(time.perf_counter() - t0)}
        log.append(row)
        print(json.dumps(row), flush=True)
        if scores["stickers_right"] > best:
            best = scores["stickers_right"]
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch, "scores": scores}, out / "best.pt")
    (out / "log.json").write_text(json.dumps(log, indent=1))
    print(f"best: {best:.4f} stickers right -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
