# CubeCheck website

The check in the browser: pick the scrambles you're about to do (pasted, or from a TNoodle PDF), show the cube to the camera (or pick a photo), and see which scramble it matches. Any cube from 2×2 to 7×7, and (experimental) skewb, pyraminx and FTO. The networks run on the visitor's own device with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/), on the GPU through WebGPU where the browser has it, otherwise on the CPU. TNoodle PDFs are read with [pdf.js](https://mozilla.github.io/pdf.js/). Camera frames, scrambles and PDFs never leave the device.

Next.js + shadcn, styled like the other cubing.in sites (shared cube icon and navy neutrals; CubeCheck's colour is cyan).

```sh
npm install          # also copies ONNX Runtime and pdf.js into public/
npm run dev          # http://127.0.0.1:3005
npm test             # the TypeScript check, follow-along and PDF reading against Python's answers
npm run build
```

The models aren't in git. `npm run build` (and `npm run fetch-models`) downloads them from [Hugging Face](https://huggingface.co/darshanvss/cubecheck) into `public/models/`, unless they're already there; to use your own, export them from `python/` (see below).

| | |
| --- | --- |
| `src/lib/vision.ts` | finder (with its second, closer look), refiner and sticker reader, step for step as in `python/cubecheck`; one reader per cube size, loaded when needed |
| `src/lib/check.ts` | the 3×3 cube, the 24 orientations, the cube rules and the slip test (`verify_probs`) |
| `src/lib/nxn.ts`, `src/lib/nxn-check.ts` | the cube and the check for any size (`nxn.py`, `nxn_check.py`), and the size a scramble is for |
| `src/lib/scrambles.ts` | a list of scrambles, and which one the cube matches |
| `src/lib/tnoodle.ts` | scramble sets out of TNoodle PDFs, by event, round and group |
| `src/lib/follow.ts` | follow-along (beta): where in the scramble the cube is, and the first slip (`follow.py`) |
| `src/lib/poly.ts`, `src/lib/poly-follow.ts` | skewb, pyraminx and FTO (experimental): the puzzles, the pose fit, the check and follow-along (`shapes.py`, `poly_check.py`, `poly_follow.py`); their shapes come from `poly-data.json` |
| `src/lib/smooth.ts` | steadier corners in live video (One Euro filter) |
| `src/components/checker.tsx` | the check page; `scramble-source.tsx` the paste / PDF picker, `follow-panel.tsx` follow-along, `poly-net.tsx` the other puzzles' flat pictures |
| `src/app/test` | `/test`: the browser against Python on test renders of every size (not linked) |

`/support` redirects to https://cubing.in/support; no payment code or keys live here.

## Checking the browser against Python

From `python/`:

```sh
uv run --extra export python scripts/export_onnx.py ../runs/corners/c7 ../runs/reader/r4/best.pt --out ../web/public/models --check ../data/raw/333/test-full-v1
uv run python scripts/web_reference.py ../runs/corners/c7 ../runs/reader/r4/best.pt \
    ../data/raw/333/test-full-v1 ../data/raw/{222,444,555,666,777}/test-mixed-v2 --out ../web/public/check
uv run python scripts/follow_reference.py --out ../web/tests/fixtures/follow.json
```

Copy `public/check/reference.json` to `tests/fixtures/` (the test reads it from there), then run `npm test` here, and open http://127.0.0.1:3005/test (add `?backend=wasm` for the CPU).

For skewb, pyraminx and FTO (their models in `runs/corners/` and `runs/reader/`, named in `cubecheck/shapes.py`'s `MODELS`; test renders as folders of `images/` and `labels/`):

```sh
uv run --extra export python scripts/export_poly_onnx.py --out ../web/public/models
uv run python scripts/poly_web_data.py --out ../web/src/lib/poly-data.json
uv run python scripts/poly_reference.py skewb=<renders> pyram=<renders> fto=<renders> --out ../web/public/check --fixtures ../web/tests/fixtures
```

The /test page then runs those too. On 14 November 2026 (Chrome, WebGPU, 30 renders: 6 skewb, 12 pyraminx, 12 FTO, some held any way round): every verdict the same as Python's (60 / 60), 838 of 846 stickers read the same, corners a median 0.8 px from Python's, about 40 ms a render.

The TNoodle tests run on PDFs built in the test. To also run them on a real competition, point `TNOODLE_FIXTURE` at its unzipped TNoodle output and give its PDF password: `TNOODLE_FIXTURE=<folder> TNOODLE_PASSWORD=<password> npm test`. Keep real competitions out of the repository: their scrambles are secret.

Result on 7 November 2026 (Chrome, Apple M-series Mac, 40 renders: 20 3×3 and 4 of every other size):

| | WebGPU | CPU |
| --- | --- | --- |
| Stickers read the same as Python | 2,098 / 2,100 | 2,098 / 2,100 |
| Verdicts the same as Python (true scramble and a one-slip version) | 80 / 80 | 80 / 80 |
| Corners, distance from Python's | median 0.72 px, at most 1.7 px | the same |
| Time per render (finder, 2 refiner passes, reader), median | 32 ms | 191 ms |

The small corner differences come from resizing: the browser's canvas and OpenCV shrink the frame slightly differently. In live video the finder runs only when the cube is lost; otherwise one refiner pass follows it from the last frame.
