# Training

The networks train on the synthetic dataset ([dataset.md](dataset.md)). Commands run from `python/`, with the dataset unpacked in `data/raw/` at the repository root (`data/raw/<size>/<set>`).

Each published model was trained from the one before, so the history is a chain. The first trained from scratch; each later one starts from its parent's weights (`--init`) and only has to learn what's new.

| Model | From | What was added |
| --- | --- | --- |
| r2 | scratch | The 3×3 sticker reader |
| r3 | r2 | Any size (`GridReader`): r2's layers, plus a small network told where each sticker sits |
| **r4** | r3 | Two faces and a sliver, cluttered scenes |
| c2 | scratch | The 3×3 corner model |
| c3, c4 | c2 | Every size; two faces and a sliver, curtains and clutter, 3× longer |
| c5b | c4 | Exactly two faces; painted fingers over corners |
| **c7** | c5b | Fingers across a whole edge; a second set of two-face views |

## Sticker reader (r4)

Small enough for a laptop GPU: about 2 hours a stage on an Apple M-series Mac.

```sh
sizes="222 333 444 555 666 777"
train=$(for p in $sizes; do echo ../data/raw/$p/train-mixed-v2; done)
test=$(for p in $sizes; do echo ../data/raw/$p/test-mixed-v2; done)
uv run python scripts/train_grid_reader.py --init ../runs/reader/r2/best.pt --name r3 --train $train --test $test
uv run python scripts/train_grid_reader.py --init ../runs/reader/r3/best.pt --name r4 --epochs 15 \
    --train $train $(for p in $sizes; do echo ../data/raw/$p/train-shallow-v2; done) ../data/raw/333/train-busy-v2 --test $test
```

A batch holds one cube size (a 7×7 gives 147 stickers a render, a 2×2 gives 12), with fewer renders for bigger cubes. Test scores are given per size and per scramble length.

## Corner model (c7: finder, then refiner)

```sh
D=../data/raw
TRAIN=""; TEST=""
for p in 222 333 444 555 666 777; do
  TRAIN="$TRAIN $D/$p/train-mixed-v2 $D/$p/train-shallow-v2 $D/$p/train-edgeon-v2 $D/$p/train-shallow-v3 $D/$p/train-edgeon-v3"
  TEST="$TEST $D/$p/test-mixed-v2 $D/$p/test-shallow-v2 $D/$p/test-edgeon-v2"
done
TRAIN="$TRAIN $D/333/train-busy-v2 $D/none/train-v1 $D/none/train-v2"
EMPTY="$D/none/test-v1 $D/none/test-v2"
grips="--fingers 0.35 --edge-grip 0.25"
uv run python scripts/train_corners.py finder --init ../runs/corners/c5b/finder.pt --epochs 8 $grips --name c7 --train $TRAIN --test $TEST --test-empty $EMPTY
uv run python scripts/train_corners.py refiner --init ../runs/corners/c5b/refiner.pt --epochs 20 $grips --name c7 --train $TRAIN --test $TEST --test-empty $EMPTY
```

c7 trained on one rented RTX 4090 in about 1.5 hours ($1.79). Each refiner pass spent about 90 of its 210 seconds waiting for the CPU to paint the fingers. On a machine with few CPU cores, more workers (`--workers`), or a card with more CPU behind it, helps more than a faster GPU.

## Feeding the GPU

Every time a render is used it gets a fresh random view (zoomed, shifted, turned up to 30°, mirrored half the time) and fresh camera effects (white balance, exposure, blur, noise, sharpening, vignette, low resolution, JPEG). Done on the CPU, this cost 12–20 ms per picture. A cloud machine allowed about 8 CPU cores, so the GPU sat mostly idle.

`--augment gpu` (the default) moves the effects and the final view onto the GPU, a whole batch at a time (`cubecheck/gpu_augment.py`). Each effect was checked against the CPU version (`degrade.py`) to within about 1 brightness level of 255. The CPU workers only decode the JPEGs, paint fingers and draw the random settings.

GPU time per batch of 64 on an A40, measured with `scripts/bench_corners.py`:

| Step | Time |
| --- | --- |
| JPEG decoding + cut-out on the GPU | 147 ms |
| Camera effects + view | 48 ms |
| Network step (bfloat16) | 45 ms |

The A40 has no hardware JPEG decoder, so decoding on the GPU was the slowest step. Decoding stays in the CPU workers by default (`--decode cpu`), which keeps up with the GPU's 93 ms per batch. Cards with a hardware decoder (A100, H100) may do better with `--decode gpu`: benchmark first.

## Evaluation

To evaluate the published models instead of your own, `uv run --extra models python scripts/download_models.py` puts them in `runs/corners/c7/` and `runs/reader/r4/`.

```sh
# The whole check, every size, exactly as the website runs it (finder bar 0.5, second look 2x closer):
uv run python scripts/evaluate_nxn.py ../runs/corners/c7 ../runs/reader/r4/best.pt ../data/raw/{222,333,444,555,666,777}/test-mixed-v2 --limit 300
```

Every test render gets a fixed random camera degradation, so scores are comparable between runs.

## For the browser

```sh
uv run --extra export python scripts/export_onnx.py ../runs/corners/c7 ../runs/reader/r4/best.pt --out ../web/public/models --check ../data/raw/333/test-full-v1
uv run python scripts/web_reference.py ../runs/corners/c7 ../runs/reader/r4/best.pt \
    ../data/raw/333/test-full-v1 ../data/raw/{222,444,555,666,777}/test-mixed-v2 --out ../web/public/check
uv run python scripts/follow_reference.py --out ../web/tests/fixtures/follow.json
```

`export_onnx.py` writes `finder.onnx`, `refiner.onnx` and one `reader-N.onnx` per size, and checks each against PyTorch. `web_reference.py` writes Python's answers for the website's `/test` page and `web/tests/check.test.ts`, and `follow_reference.py` does the same for follow-along.
