# The dataset

Everything the models learn from is synthetic: 3D renders of 2×2 to 7×7 cubes in randomised rooms, with labels computed from the scene, so they're exact.

> The dataset is not published yet. It will be released once it covers more puzzles. This page describes the version the published models (c7 and r4) were trained on.

## What's in it

Sets are named by cube size (`222` … `777`, as WCA event ids) and kind, in `data/raw/<size>/<set>`. Numbers are renders per size.

| Set | Renders | What it's for |
| --- | --- | --- |
| `333/train-full-v1` | 20,000 | 3×3 full scrambles: handheld and on tables, any angle, fingers over stickers, many lighting setups |
| `333/train-easy-v1` | 6,000 | 3×3 cubes 0–4 moves from solved, which look different from full scrambles |
| `train-mixed-v2` | 10,000 per size | Every size, scrambles of every length (solved, a few moves, partway, full) |
| `train-shallow-v2`, `-v3` | 2,000 + 2,000 per size | **Two faces and a sliver**: the cube held toward the scrambler, the third face nearly edge-on |
| `train-edgeon-v2`, `-v3` | 2,000 + 2,000 per size | **Exactly two faces**: looking at an edge, the third face out of sight |
| `train-small-v2` | 2,000 per size | Cubes small in the frame, far from the camera |
| `333/train-busy-v2` | 5,000 | Cluttered scenes and patterned curtains, which an early finder took for cubes |
| `none/train-v1`, `-v2` | 5,000 + 10,000 | **No cube**: the same rooms, light, camera and hands, with the cube hidden |
| `test-mixed-v2` | 1,000 per size | Held-out tests from other seeds. Also `test-shallow-v2`, `test-edgeon-v2`, `test-small-v2` (300 per size), `333/test-full-v1` (1,000), `none/test-v1`, `-v2` (300, 1,000) |

The "no cube" sets are kept separate so they're easy to skip, but a model needs them: without frames that have no cube, the first corner model (c1) learned that every frame has one and drew a cube on whatever was in view.

## Labels, per render

- The cube size, the scramble and the full cube state (6N² stickers, Kociemba order U R F D L B).
- The 7 visible corners (P1–P6 around the outside, C in the middle) in pixels, each marked visible, covered or outside the frame. In an exactly-two-face view both ends of the shared edge are fair answers for C, and training accepts either.
- Every visible sticker: its colour, its outline, and how much of it is covered.
- Camera, lighting and scene settings, so results can be broken down by condition.

Camera effects (blur, noise, JPEG, white balance, exposure) are **not** baked in: training adds a fresh random set each time a render is used (`python/cubecheck/degrade.py`). The corner model's training also paints fingers on: one to three over a corner (`--fingers`), or a row of three or four across a whole outer edge (`--edge-grip`), the way a hand really grips a cube. The labels stay where the corners really are, so the model learns that a corner is where the cube's edges meet, not where the fingers start.

## Where the pictures come from

- Cubes, hands and fingers are rendered.
- Room backgrounds and table tops are CC0 photos from [Poly Haven](https://polyhaven.com) (indoor panoramas and surface textures). The source list will ship with the dataset.
- The training images are synthetic renders. The dataset contains no photographs of people and no camera photographs of real cubes. Real webcam photos used for testing are not published; only their scores are.
