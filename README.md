# CubeCheck

**Check a scramble with a camera, for any cube from 2×2 to 7×7.** Paste the scrambles or open the competition's TNoodle PDF, show the scrambled cube with one corner pointing at the camera, and get **correct** or **mis-scramble**. No scanning face by face, no smart cube, and nothing leaves your device.

[![Try it](https://img.shields.io/badge/try%20it-check.cubing.in-06b6d4)](https://check.cubing.in)
[![Models on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20models-cubecheck-yellow)](https://huggingface.co/darshanvss/cubecheck)
[![CI](https://github.com/darshanvss/cubecheck/actions/workflows/ci.yml/badge.svg)](https://github.com/darshanvss/cubecheck/actions/workflows/ci.yml)
[![License: AGPL v3](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)

> CubeCheck helps the scrambler's own check. It is not an official WCA tool.

## What it does

- **Every WCA cube event from 2×2 to 7×7.** The size is told from the moves (or picked by hand).
- **TNoodle PDFs.** Open a group's Computer Display PDF (with its passcode) or the printing PDF (with the competition password), then pick the event, round and group. The PDF is read in the browser and kept only in memory. A PDF that isn't a TNoodle scramble sheet is refused.
- **Checks against the whole list.** It finds which scramble the cube in view matches, or says it matches none.
- **Follow-along (beta).** It follows one scramble move by move from solved, says the moment a move goes wrong, and shows the way back. See [docs/follow-along.md](docs/follow-along.md).
- **Big cubes as they're really made.** A 6×6 or 7×7 has thicker outer layers; the sticker grid follows the measured widths, and an **Outer layers** setting covers other makes (4×4 to 7×7).
- **Skewb, pyraminx and FTO (experimental).** Checked and followed the same way, each with its own models. See [docs/other-puzzles.md](docs/other-puzzles.md).

## How it works

```
camera frame ─► finder ─────────► refiner ──────► sticker reader ──► the check
                is there a cube?   the 7 corners    3N² stickers:      slip test (+ cube rules on 3×3):
                rough corners      precisely        6 colours or       correct / mis-scramble /
                (2× closer if      (close-up)       "covered"          re-check
                 it's unsure)
```

One view of a cube with a corner toward the camera shows three faces: half its stickers. That is enough, because a slipped move changes many of them: on a 3×3, about 8 of the 24 non-centre stickers in view. In simulation over about 196,000 3×3 mis-scrambles, the three-face view missed none. Two faces work too, except on a 2×2.

1. **Finder.** A small network looks at the whole frame, shrunk to 256 × 256, and answers whether there's a cube and roughly where its 7 visible corners are. When it's unsure, it looks again 2× closer (a cube far from the camera is only ~20 px across in that view).
2. **Refiner.** The same kind of network looks at a close-up of the full-quality frame and places the 7 corners precisely. In live video it follows the cube from frame to frame. It was trained with painted fingers over corners and across whole edges, the way real hands hold a cube.
3. **Sticker reader.** The corners give an N × N grid on each face. The faces are straightened and a third network reads all the stickers together: one of the 6 colours, or *covered* (a finger is in the way). One reader works for every size; the browser loads it per size (about 3 MB each).
4. **The check.** The scramble gives the expected cube. A slip test asks whether the picture fits the scramble as written better than any scramble with one human slip (skipped, wrong, extra or swapped moves). On a 3×3 the cube rules (each piece exists exactly once) clean up the reading first.

More detail: [docs/how-it-works.md](docs/how-it-works.md).

## Results

Models c7 (corners) and r4 (sticker reader), on held-out synthetic renders with harsh camera effects, single frames, no averaging, exactly as the website runs them (`python/scripts/evaluate_nxn.py`). 300 renders per size, with mixed scramble lengths:

| | 2×2 | 3×3 | 4×4 | 5×5 | 6×6 | 7×7 |
| --- | --- | --- | --- | --- | --- | --- |
| True scramble called correct | 98.7% | 99.3% | 100% | 99.0% | 96.3% | 90.3% |
| False alarm (the rest say "re-check") | 0% | 0.7% | 0% | 0% | 0.7% | 0.3% |
| One slipped move caught | 96.7% | 100% | 100% | 99.0% | 96.7% | 90.7% |
| One slipped move missed | 2.0% | 0% | 0% | 0% | 0.3% | 0% |

On 1,000 full 3×3 scrambles: true scramble correct 99.9% with no false alarms, one slip caught 99.5% (missed 0.4%). The bigger cubes' gap is mostly "re-check": their stickers are small, so a frame has to be close and sharp.

On views with two faces and a sliver, the true scramble is called correct 86–95% of the time, with 0.3–2.7% false alarms. The live page waits for several frames to agree, which removes most single-frame false alarms. Results on real photos will be added.

## Repository layout

| Folder | What's in it |
| --- | --- |
| [`web/`](web/) | The website (Next.js). Runs the networks in the browser with ONNX Runtime Web, and TNoodle PDFs with pdf.js |
| [`python/cubecheck/`](python/cubecheck/) | The models, the cube for every size, the check and follow-along, in Python |
| [`python/scripts/`](python/scripts/) | Training, evaluation, ONNX export, model download, and the reference answers the website is tested against |
| `python/tests/`, `web/tests/` | Tests (the TypeScript is checked against Python's answers, look by look) |
| [`docs/`](docs/) | How it works, follow-along, the other puzzles, how it was trained, what the training data looks like |

## Quick start

Run the website (Node 22):

```sh
cd web
npm install           # also copies ONNX Runtime and pdf.js into public/
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

The trained weights are on Hugging Face: [darshanvss/cubecheck](https://huggingface.co/darshanvss/cubecheck), as PyTorch weights (for Python and further training) and ONNX (for the browser). Each release is a tag (`v0.1.0`: c2 + r2, 3×3 only; `v0.2.0`: c7 + r4, every size), and the code pins the one it was tested with. (The repository was called `cubecheck-3x3` until v0.2.0; the old name redirects.)

## Train on your own data

The renders the models were trained on are on Hugging Face: [darshanvss/cubecheck](https://huggingface.co/datasets/darshanvss/cubecheck) (180,200 renders of 2×2 to 7×7 cubes and rooms with no cube, CC BY 4.0; `python/scripts/download_dataset.py` unpacks them into `data/raw/`). The training code works on any data in the same format:

- a folder with `images/` (JPEG) and `labels/` (one JSON per image: the cube size, the scramble, the 7 corner positions with visibility, and each sticker's colour, centre and visibility). [`python/tests/fixtures/sample_handheld.json`](python/tests/fixtures/sample_handheld.json) is a full example, and `scripts/check_samples.py <folder>` checks a folder for label mistakes;
- a few thousand frames *without* a cube, so the models learn to say "no cube".

Then follow [docs/training.md](docs/training.md). What the published models were trained on is described in [docs/dataset.md](docs/dataset.md).

## Privacy

The website runs the models on your device: camera frames, scrambles and TNoodle PDFs never leave it. The training images are synthetic renders and contain no photographs of people.

## Contributing

Issues and pull requests are welcome: see [CONTRIBUTING.md](CONTRIBUTING.md). To report a security problem, see [SECURITY.md](SECURITY.md).

## Licence

Code and model weights: [AGPL-3.0-or-later](LICENSE). If you run a modified version as a website, share its source with your users. Many cubing tools are copyleft too: TNoodle is AGPL-3.0, and the WCA website and csTimer are GPL-3.0. pdf.js, used to read TNoodle PDFs, is Apache-2.0.
