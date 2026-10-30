# CubeCheck website

The check in the browser: paste the scrambles you're about to do, show the cube to the camera (or pick a photo), and see which scramble it matches. The three networks run on the visitor's own device with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/), on the GPU through WebGPU where the browser has it, otherwise on the CPU. Camera frames and scrambles never leave the device.

Next.js + shadcn, styled like the other cubing.in sites (shared cube icon and navy neutrals; CubeCheck's colour is cyan).

```sh
npm install          # also copies ONNX Runtime into public/ort/
npm run dev          # http://127.0.0.1:3005
npm test             # the TypeScript check against Python's verdicts
npm run build
```

The models aren't in git. `npm run build` (and `npm run fetch-models`) downloads them from [Hugging Face](https://huggingface.co/darshanvss/cubecheck-3x3) into `public/models/`, unless they're already there; to use your own, export them from `python/` (see below).

| | |
| --- | --- |
| `src/lib/vision.ts` | finder, refiner and sticker reader, step for step as in `python/cubecheck` |
| `src/lib/check.ts` | the cube, the 24 orientations, the cube rules and the slip test (`verify_probs`) |
| `src/lib/scrambles.ts` | a list of scrambles, and which one the cube matches |
| `src/lib/smooth.ts` | steadier corners in live video (One Euro filter) |
| `src/components/checker.tsx` | the check page |
| `src/app/test` | `/test`: the browser against Python on test renders (not linked) |

`/support` redirects to https://cubing.in/support; no payment code or keys live here.

## Checking the browser against Python

From `python/`:

```sh
uv run --extra export python scripts/export_onnx.py ../runs/corners/c2 ../runs/reader/r2/best.pt --out ../web/public/models --check ../data/raw/test-v1
uv run python scripts/web_reference.py ../runs/corners/c2 ../runs/reader/r2/best.pt ../data/raw/test-v1 --out ../web/public/check
```

Copy `public/check/reference.json` to `tests/fixtures/` (the test reads it from there), then run `npm test` here, and open http://127.0.0.1:3005/test (add `?backend=wasm` for the CPU).

Result on 28 October 2026 (Chrome, Apple M-series Mac, 20 renders):

| | |
| --- | --- |
| Stickers read the same as Python | 540 / 540 |
| Verdicts the same as Python (true scramble and a one-slip version) | 40 / 40 |
| Corners, distance from Python's | median 0.85 px, at most 4.4 px |
| Time per render (finder, 2 refiner passes, reader) | 40–90 ms on WebGPU, about 1 s on the CPU |

The small corner differences come from resizing: the browser's canvas and OpenCV shrink the frame slightly differently. In live video the finder runs only when the cube is lost; otherwise one refiner pass follows it from the last frame.
