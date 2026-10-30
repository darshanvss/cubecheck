# CubeCheck

**Check a 3×3 scramble with a camera.** Paste the scramble, show the scrambled cube with one corner pointing at the camera, and get **correct** or **mis-scramble**. No scanning face by face, no smart cube, and nothing leaves your device.

[![Try it](https://img.shields.io/badge/try%20it-check.cubing.in-06b6d4)](https://check.cubing.in)
[![Models on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20models-cubecheck--3x3-yellow)](https://huggingface.co/darshanvss/cubecheck-3x3)
[![CI](https://github.com/darshanvss/cubecheck/actions/workflows/ci.yml/badge.svg)](https://github.com/darshanvss/cubecheck/actions/workflows/ci.yml)
[![License: AGPL v3](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)

> CubeCheck helps the scrambler's own check. It is not an official WCA tool.

## How it works

```
camera frame ─► finder ─────────► refiner ──────► sticker reader ──► the check
                is there a cube?   the 7 corners    27 stickers:       cube rules + slip test:
                rough corners      precisely        6 colours or       correct / mis-scramble /
                                   (close-up)       "covered"          re-check
```

One view of a cube with a corner toward the camera shows three faces: 27 of its 54 stickers. That is enough, because a slipped move changes about 8 of the 24 non-centre stickers in view. In simulation over about 196,000 mis-scrambles, the three-face view missed none.

1. **Finder.** A small network looks at the whole frame, shrunk to 256 × 256, and answers whether there's a cube and roughly where its 7 visible corners are.
2. **Refiner.** The same kind of network looks at a close-up of the full-quality frame and places the 7 corners precisely. In live video it follows the cube from frame to frame.
3. **Sticker reader.** The corners give a 3 × 3 grid on each face. The three faces are straightened and a third network reads all 27 stickers together: one of the 6 colours, or *covered* (a finger is in the way).
4. **The check.** The scramble gives the expected cube. The cube rules (each piece exists exactly once) clean up the reading, then a slip test asks whether the picture fits the scramble as written better than any scramble with one move slipped.

More detail: [docs/how-it-works.md](docs/how-it-works.md).

## Results

On 1,000 held-out synthetic renders with harsh camera effects (single frames, no averaging):

| | |
| --- | --- |
| Cube found | 994 / 1,000 |
| Stickers read correctly | 99.0% |
| True scramble called correct | 99.2% (no false alarms; the rest say "re-check") |
| One slipped move caught | 99.1% |

Results on real photos will be added.

## Repository layout

| Folder | What's in it |
| --- | --- |
| [`web/`](web/) | The website (Next.js). Runs the three networks in the browser with ONNX Runtime Web |
| [`python/cubecheck/`](python/cubecheck/) | The models, the cube rules and the check, in Python |
| [`python/scripts/`](python/scripts/) | Training, evaluation, ONNX export and model download |
| `python/tests/`, `web/tests/` | Tests |
| [`docs/`](docs/) | How it works, how it was trained, what the training data looks like |

## Quick start

Run the website (Node 22):

```sh
cd web
npm install           # also copies ONNX Runtime into public/ort/
npm run fetch-models  # downloads the ONNX models from Hugging Face into public/models/
npm run dev           # http://127.0.0.1:3005
```

Run the check in Python (with [uv](https://docs.astral.sh/uv/)):

```sh
cd python
uv sync --extra train --extra models
uv run python scripts/download_models.py   # weights into runs/
uv run python -m pytest
```

## Models

The trained weights are on Hugging Face: [darshanvss/cubecheck-3x3](https://huggingface.co/darshanvss/cubecheck-3x3), as PyTorch weights (for Python and further training) and ONNX (for the browser), about 45 MB in all.

## Train on your own data

The training dataset is not published yet; it will be once it covers more puzzles (4×4 and up). All the training code is here, so the models can be trained on any data in the same format:

- a folder with `images/` (JPEG) and `labels/` (one JSON per image: the scramble, the 7 corner positions with visibility, and each visible sticker's colour and outline). [`python/tests/fixtures/sample_handheld.json`](python/tests/fixtures/sample_handheld.json) is a full example, and `scripts/check_samples.py <folder>` checks a folder for label mistakes;
- a few thousand frames *without* a cube, so the models learn to say "no cube".

Then follow [docs/training.md](docs/training.md). What the published models were trained on is described in [docs/dataset.md](docs/dataset.md).

## Privacy

The website runs the models on your device: camera frames and scrambles never leave it. The training images are synthetic renders and contain no photographs of people.

## Contributing

Issues and pull requests are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md). To report a security problem, see [SECURITY.md](SECURITY.md).

## Licence

Code and model weights: [AGPL-3.0-or-later](LICENSE). If you run a modified version as a website, share its source with your users. Many cubing tools are copyleft too: TNoodle is AGPL-3.0, and the WCA website and csTimer are GPL-3.0.
