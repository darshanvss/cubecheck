"""How long each part of a corner-model training step takes, to size a RunPod machine.

    uv run python scripts/bench_corners.py refiner ../data/raw/v1 ../data/raw/empty-v1

1. CPU, per picture, in one process, three ways: everything on the CPU (old);
   decode + cut-out on the CPU, effects on the GPU; only reading the file, with
   the GPU decoding too (NVIDIA only). A data worker uses one core, so pictures
   per second = cores x 1000 / ms.
2. GPU, per batch: decoding + cut-out (NVIDIA only), camera effects + view
   (gpu_augment.render), then the network's forward and backward pass.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cubecheck import gpu_augment
from cubecheck.corners import CornerNet

import train_corners as T


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["finder", "refiner"])
    ap.add_argument("raw", type=Path, nargs="+")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n", type=int, default=150, help="pictures for the CPU timing")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    T.cv2.setNumThreads(1)

    cpu = T.Views(args.raw, args.stage, True)
    gpu = T.Views(args.raw, args.stage, True, gpu=True)
    files = T.Views(args.raw, args.stage, True, gpu=True, gpu_decode=True)
    idx = np.random.default_rng(0).choice(len(cpu), args.n, replace=False)
    for name, ds in (("everything on the CPU", cpu), ("effects on the GPU", gpu), ("decoding on the GPU too", files)):
        t0 = time.perf_counter()
        for i in idx:
            ds[int(i)]
        print(f"CPU per picture, {name:<28} {1000 * (time.perf_counter() - t0) / args.n:6.1f} ms")

    cuda = args.device == "cuda"
    batch = T.collate([(files if cuda else gpu)[int(i)] for i in idx[: args.batch]])
    jpegs, box = batch.pop("jpeg", None), batch.pop("box")
    b = {k: v.to(args.device) for k, v in batch.items()}
    net = CornerNet().to(args.device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    timings = {"decode + cut-out": [], "effects + view": [], "network step": []}
    for rep in range(12):
        sync(args.device)
        tc = time.perf_counter()
        if cuda:
            canvas = gpu_augment.cut_canvas_gpu(gpu_augment.decode_gpu(jpegs, args.device), box)
        else:
            canvas = b["canvas"].permute(0, 3, 1, 2).float() / 255
        sync(args.device)
        t0 = time.perf_counter()
        x = gpu_augment.render(canvas, b)
        sync(args.device)
        t1 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
            heat, logit = net(x)
        loss = T.corner_loss(heat.float(), b["target"], b["mask"]) + F.binary_cross_entropy_with_logits(logit.float(), b["present"].float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        sync(args.device)
        t2 = time.perf_counter()
        if rep >= 2:  # skip warm-up
            if cuda:
                timings["decode + cut-out"].append(t0 - tc)
            timings["effects + view"].append(t1 - t0)
            timings["network step"].append(t2 - t1)
    for name, ts in timings.items():
        if ts:
            print(f"GPU per batch of {args.batch} ({args.device}), {name:<15} {1000 * np.median(ts):6.1f} ms")


if __name__ == "__main__":
    main()
