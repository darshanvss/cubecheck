"""Train the size-aware sticker reader (reader.GridReader) on renders of every cube size.

    uv run python scripts/train_grid_reader.py --init ../runs/reader/r2/best.pt --name r4 \
        --train ../data/raw/<sets of every size...> --test ../data/raw/<test sets...>

Like train_reader.py (fresh camera degradation and nudged corners every time, faces
rotated one step at random), with three differences:

- A render of an N×N cube gives 3N² stickers, so each batch holds one size, with
  fewer renders for bigger cubes (about 1,150 stickers a batch whatever the size).
- --init starts from a trained 3x3 reader (r2): its layers carry over, only the
  new position network starts from scratch.
- Test scores are given per cube size and per scramble length (solved, 1-4 moves,
  in between, full), and r2 is scored the same way on the 3x3 tests to compare.

Labels are worked out once and cached next to the run (labels-*.npz).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from cubecheck.degrade import degrade
from cubecheck.nxn import EVENT_LENGTH
from cubecheck.reader import COVERED, IGNORE, GridReader, cell_labels, face_maps, load_reader, to_tensor
from cubecheck.verify import cube_size

cv2.setNumThreads(1)
STICKERS_PER_BATCH = 128 * 27 // 3  # r2's batch of 128 3x3 renders held 1,152 stickers
BANDS = ("solved", "1-4", "between", "full")


def band(scramble: str, n: int) -> str:
    k = len(scramble.split())
    return "solved" if k == 0 else "1-4" if k <= 4 else "full" if k >= EVENT_LENGTH[n] else "between"


def read_set(raw: Path, cache: Path) -> dict:
    """Paths, corners, labels, size and scramble band of every render with all 7 corners in frame."""
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    paths, kps, labels, sizes, bands = [], [], [], [], []
    for f in sorted((raw / "labels").glob("*.json")):
        s = json.loads(f.read_text())
        if not all(k[2] > 0 for k in s["labels"]["keypoints"]):
            continue
        n = s.get("cube", {}).get("size", 3)
        kp = np.array(s["labels"]["keypoints"], np.float32)[:, :2]
        paths.append(str(raw / s["image"]))
        kps.append(kp)
        labels.append(cell_labels(s, kp, n).astype(np.int8))
        sizes.append(n)
        bands.append(band(s["scramble"], n))
    out = {"paths": np.array(paths), "kps": np.array(kps), "labels": np.array(labels, dtype=object),
           "sizes": np.array(sizes), "bands": np.array(bands)}
    np.savez(cache, **out)
    return out


class Faces(Dataset):
    def __init__(self, raws: list[Path], cache_dir: Path, train: bool, jitter: float, fraction: float = 1.0) -> None:
        parts = [read_set(raw, cache_dir / f"labels-{raw.parent.name}-{raw.name}.npz") for raw in raws]
        keep = [np.arange(len(p["paths"]))[: max(1, int(len(p["paths"]) * fraction))] for p in parts]
        self.paths = np.concatenate([p["paths"][k] for p, k in zip(parts, keep)])
        self.kps = np.concatenate([p["kps"][k] for p, k in zip(parts, keep)])
        self.labels = [lab for p, k in zip(parts, keep) for lab in p["labels"][k]]
        self.sizes = np.concatenate([p["sizes"][k] for p, k in zip(parts, keep)])
        self.bands = np.concatenate([p["bands"][k] for p, k in zip(parts, keep)])
        self.train, self.jitter = train, jitter

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        rng = np.random.default_rng() if self.train else np.random.default_rng([7, i])
        img = cv2.imread(self.paths[i])
        kp, y, n = self.kps[i], np.asarray(self.labels[i], np.int64), int(self.sizes[i])
        if self.train:
            kp = kp + rng.normal(0, self.jitter * cube_size(kp), kp.shape).astype(np.float32)
        h, w = img.shape[:2]
        x0, y0 = np.clip(np.floor(kp.min(0)) - 24, 0, [w - 1, h - 1]).astype(int)
        x1, y1 = np.clip(np.ceil(kp.max(0)) + 24, [x0 + 1, y0 + 1], [w, h]).astype(int)
        img, _ = degrade(img[y0:y1, x0:x1], rng, frame=(int(x0), int(y0), w, h))
        maps = face_maps(img, kp - [x0, y0], 24 * n)
        if self.train:
            r = int(rng.integers(3))
            maps, y = np.roll(maps, -r, axis=0), np.roll(y, -n * n * r)
        return to_tensor(maps), torch.from_numpy(y), i


class BySize(Sampler):
    """Batches of one cube size, about STICKERS_PER_BATCH stickers each, in random order."""

    def __init__(self, sizes: np.ndarray, shuffle: bool) -> None:
        self.sizes, self.shuffle = sizes, shuffle

    def batches(self) -> list[list[int]]:
        out = []
        for n in np.unique(self.sizes):
            idx = np.flatnonzero(self.sizes == n)
            if self.shuffle:
                idx = np.random.permutation(idx)
            per = max(8, STICKERS_PER_BATCH // (3 * n * n))
            out += [idx[k:k + per].tolist() for k in range(0, len(idx), per)]
        if self.shuffle:
            out = [out[k] for k in np.random.permutation(len(out))]
        return out

    def __iter__(self):
        return iter(self.batches())

    def __len__(self) -> int:
        return len(self.batches())


@torch.no_grad()
def evaluate(model: nn.Module, data: Faces, loader: DataLoader, device: str, only3: bool = False) -> dict:
    """Stickers read right, per cube size and per scramble band, and covered stickers found."""
    model.eval()
    right, total = defaultdict(int), defaultdict(int)
    cov_right = cov_total = 0
    for x, y, idx in loader:
        n = int(data.sizes[idx[0]])
        if only3 and n != 3:
            continue
        pred = model(x.to(device)).argmax(-1).cpu()
        colour = (y >= 0) & (y < COVERED)
        ok = (pred == y) & colour
        for b, i in enumerate(idx.tolist()):
            b_name = str(data.bands[i])  # a plain str: numpy strings in the checkpoint would stop it loading with weights_only
            for key in ("all", f"{n}x{n}", b_name, f"{n}x{n} {b_name}"):
                right[key] += int(ok[b].sum())
                total[key] += int(colour[b].sum())
        cov = y == COVERED
        cov_right += int((pred[cov] == COVERED).sum())
        cov_total += int(cov.sum())
    out = {k: round(float(right[k]) / total[k], 4) for k in total}  # plain floats: the checkpoint must load with weights_only
    out["covered_found"] = round(cov_right / max(cov_total, 1), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, nargs="+", required=True)
    ap.add_argument("--test", type=Path, nargs="+", required=True)
    ap.add_argument("--init", type=Path, help="a trained StickerReader (r2) to start from")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--jitter", type=float, default=0.015, help="corner nudge, as a share of the cube's size")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fraction", type=float, default=1.0, help="share of the training renders (quick checks)")
    ap.add_argument("--max-batches", type=int, default=0, help="stop each epoch after this many batches (timing)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--name", default="grid")
    args = ap.parse_args()

    out = Path(__file__).resolve().parents[2] / "runs" / "reader" / args.name
    out.mkdir(parents=True, exist_ok=True)
    cache = out.parent / "label-cache"
    cache.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    train = Faces(args.train, cache, True, args.jitter, args.fraction)
    test = Faces(args.test, cache, False, 0.0)
    print(f"{len(train)} training renders, {len(test)} test renders, labels ready in {time.perf_counter() - t0:.0f}s", flush=True)
    print("training renders per size:", {f"{n}x{n}": int((train.sizes == n).sum()) for n in np.unique(train.sizes)}, flush=True)

    kw = dict(num_workers=args.workers, persistent_workers=args.workers > 0)
    train_loader = DataLoader(train, batch_sampler=BySize(train.sizes, True), **kw)
    test_loader = DataLoader(test, batch_sampler=BySize(test.sizes, False), **kw)

    if args.init:
        r2 = load_reader(str(args.init), args.device)
        print("start (r2, 3x3 tests):", json.dumps(evaluate(r2, test, test_loader, args.device, only3=True)), flush=True)
        model = GridReader.from_r2(str(args.init), args.device)
    else:
        model = GridReader().to(args.device)
    config = {"width": 128, "layers": 2}
    print(f"{sum(p.numel() for p in model.parameters()):,} numbers to learn", flush=True)
    steps = args.epochs * (min(len(train_loader), args.max_batches) if args.max_batches else len(train_loader))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE, label_smoothing=0.05)

    best, log = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, seen = time.perf_counter(), 0.0, 0
        for x, y, _ in train_loader:
            logits = model(x.to(args.device))
            loss = loss_fn(logits.flatten(0, 1), y.to(args.device).flatten())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total, seen = total + loss.item(), seen + 1
            if args.max_batches and seen >= args.max_batches:
                break
        train_s = time.perf_counter() - t0
        scores = evaluate(model, test, test_loader, args.device)
        row = {"epoch": epoch, "loss": round(total / seen, 4), "seconds": round(time.perf_counter() - t0), "train_seconds": round(train_s),
               "batches": seen, **scores}
        log.append(row)
        print(json.dumps(row), flush=True)
        (out / "log.json").write_text(json.dumps(log, indent=1))
        if scores["all"] > best:
            best = scores["all"]
            torch.save({"model": model.state_dict(), "config": config, "grid": True, "epoch": epoch, "scores": scores}, out / "best.pt")
    print(f"best: {best:.4f} stickers right -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
