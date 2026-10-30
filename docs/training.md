# Training

All three networks train from scratch on the synthetic dataset ([dataset.md](dataset.md)). Commands run from `python/`, with the dataset unpacked in `data/raw/` at the repository root.

## Sticker reader

Small enough for a laptop GPU: about 75 minutes on an Apple M-series Mac.

```sh
uv run python scripts/train_reader.py --train ../data/raw/v1 ../data/raw/easy-v1 --test ../data/raw/test-v1 --name r2
```

## Corner model (finder, then refiner)

```sh
common="--train ../data/raw/v1 ../data/raw/easy-v1 ../data/raw/empty-v1 --test ../data/raw/test-v1 ../data/raw/test-easy-v1 --test-empty ../data/raw/test-empty-v1"
uv run python scripts/train_corners.py finder $common --epochs 60 --name c2
uv run python scripts/train_corners.py refiner $common --epochs 100 --name c2
```

Version c2 was trained on a rented NVIDIA A40 starting from an earlier version's weights (`--init`): 48 s per epoch for each stage, about 2.2 hours in all.

## Feeding the GPU

Every time a render is used it gets a fresh random view (zoomed, shifted, turned up to 30°, mirrored half the time) and fresh camera effects (white balance, exposure, blur, noise, sharpening, vignette, low resolution, JPEG). Done on the CPU, this cost 12–20 ms per picture. A cloud machine allowed about 8 CPU cores, so the GPU sat mostly idle.

`--augment gpu` (the default) moves the effects and the final view onto the GPU, a whole batch at a time (`cubecheck/gpu_augment.py`). Each effect was checked against the CPU version (`degrade.py`) to within about 1 brightness level of 255. The CPU workers only decode the JPEGs and draw the random settings.

GPU time per batch of 64 on an A40, measured with `scripts/bench_corners.py`:

| Step | Time |
| --- | --- |
| JPEG decoding + cut-out on the GPU | 147 ms |
| Camera effects + view | 48 ms |
| Network step (bfloat16) | 45 ms |

The A40 has no hardware JPEG decoder, so decoding on the GPU was the slowest step. Decoding stays in the CPU workers by default (`--decode cpu`), which keeps up with the GPU's 93 ms per batch. With that, the GPU was 90–95% busy. Cards with a hardware decoder (A100, H100) may do better with `--decode gpu`: benchmark first.

## Evaluation

To evaluate the published models instead of your own, `uv run --extra models python scripts/download_models.py` puts them in `runs/corners/c2/` and `runs/reader/r2/`.

```sh
uv run python scripts/evaluate_corners.py ../runs/corners/c2 ../data/raw/test-v1 ../data/raw/test-easy-v1
uv run python scripts/evaluate_reader.py ../runs/corners/c2 ../runs/reader/r2/best.pt ../data/raw/test-v1
```

Every test render gets a fixed random camera degradation, so scores are comparable between runs.
