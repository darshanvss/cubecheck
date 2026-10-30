# The dataset

Everything the models learn from is synthetic: 3D renders of a 3×3 in randomised rooms, with labels computed from the scene, so they're exact.

> The dataset is not published yet. It will be released once it covers more puzzles (4×4 and up). This page describes the version the published models were trained on.

## What's in it

| Part | Renders | What it's for |
| --- | --- | --- |
| `v1` | 20,000 | Scrambled cubes: handheld and on tables, any angle, fingers over stickers, many lighting setups |
| `easy-v1` | 6,000 | Near-solved cubes, 0–4 moves from solved, which look different from full scrambles |
| `empty-v1` | 5,000 | **No cube.** The same rooms, light, camera and hands as the first 5,000 of `v1`, with the cube hidden |
| `test-v1`, `test-easy-v1`, `test-empty-v1` | 1,000 / 500 / 300 | Held-out test renders from different seeds |

The "no cube" part is kept separate so it's easy to skip, but a model needs it: without frames that have no cube, the first corner model (c1) learned that every frame has one and drew a cube on whatever was in view.

## Labels, per render

- The scramble and the full cube state.
- The 7 visible corners (P1–P6 around the outside, C in the middle) in pixels, each marked visible, covered or outside the frame.
- Every visible sticker: its colour, its outline, and how much of it is covered.
- Camera, lighting and scene settings, so results can be broken down by condition.

Camera effects (blur, noise, JPEG, white balance, exposure) are **not** baked in: training adds a fresh random set each time a render is used (`python/cubecheck/degrade.py`).

## Where the pictures come from

- Cube, hands and fingers are rendered.
- Room backgrounds and table tops are CC0 photos from [Poly Haven](https://polyhaven.com) (indoor panoramas and surface textures). The source list will ship with the dataset.
- The training images are synthetic renders. The dataset contains no photographs of people and no camera photographs of real cubes. Real webcam photos used for testing are not published; only their scores are.
