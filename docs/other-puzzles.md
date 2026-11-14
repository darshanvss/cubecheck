# Skewb, pyraminx and FTO (experimental)

CubeCheck can check a skewb, a pyraminx or an FTO the way it checks a cube: from one camera view, against the scramble as written. Pick the puzzle under **Experimental** in the puzzle menu, or paste its scrambles (the moves usually say which puzzle it is). Check and follow-along both work. TNoodle PDFs give their Skewb and Pyraminx sets too; the FTO isn't a WCA event, so its scrambles are pasted.

This is experimental: the models were trained and measured on synthetic renders, and have seen only a handful of real puzzles. Treat a verdict as a second opinion, not a ruling.

## How to hold each one

| Puzzle | Hold it like this | What the camera then sees |
| --- | --- | --- |
| Skewb | a corner toward the camera, as a cube | 3 faces, 15 of its 30 stickers |
| Pyraminx | an edge toward the camera | 2 faces, 18 of 36 |
| FTO | a corner toward the camera | 4 faces, 36 of 72 |

Held another way (a pyraminx face-on, an FTO tilted to two or three faces), the puzzle is still found and read. A slip that shows is still a mis-scramble, but "correct" waits for the better view: the banner says *fits so far: turn a corner toward the camera*. From fewer faces too many slips change nothing in view (an FTO held with 2 faces in view hides about one slip in six).

## What's different from a cube

The pipeline is a cube's (finder, refiner, sticker reader, check, see [how it works](how-it-works.md)), with each puzzle's own shape:

- **Corners.** Each puzzle has its own corner model: 7 corners for a skewb (it's a cube), 4 for a pyraminx (its tips), 5 for an FTO (the corner nearest the camera and the four around it). A cube's corner model found a skewb in only half the renders, and placed a corner over 5% off in most of those.
- **Faces that are triangles.** A skewb face is a square, mapped from its 4 corners as a cube's. A triangle's 3 corners don't say how it's foreshortened (with them alone, stickers came out up to a fifth of an edge off), so the whole shape is fitted to its corners as a camera would see it: a pinhole camera through the middle of the picture, four zooms tried, the best fit kept. In the browser this is done without OpenCV (`web/src/lib/poly.ts`, `Poly.fit`): a first guess from a camera without perspective, then least squares (Levenberg-Marquardt). It puts the stickers within 0.003 px of Python's.
- **A guard on the fit.** Corners that no rigid pyraminx or FTO could have (one corner on the wrong spot) are not read at all: a fit worse than 5% of the puzzle's size gives "re-check". On renders held any way round, this took false alarms from 2.4% to 0.2% (FTO).
- **Stickers that aren't a grid.** Each face is straightened (a triangle upright, with the corner nearest the camera at the top), and the reader averages its features over each sticker's own outline. The convolution and mixing layers are the cube reader's, so training started from a network that already reads stickers.
- **Colours.** The puzzle's own: 6 on a skewb, 4 on a pyraminx, 8 on an FTO.
- **Which way round.** No sticker stays put the way a 3×3's centres do, so every way the puzzle can face the camera is tried: 24 for a skewb or an FTO, 12 for a pyraminx.
- **Thresholds.** A pyraminx tip turned wrong changes only 3 stickers, and an edge-on view shows 1 or 2 of them, so on a pyraminx one sticker the reader is sure about is enough to call a slip.

## Results

On held-out synthetic renders with harsh camera effects, single frames, the models as the website runs them, each puzzle held the way the table above says:

| | Skewb | Pyraminx | FTO |
| --- | --- | --- | --- |
| True scramble called correct | 98.3% | 96.7% | 98.3% |
| False alarm | 0.2% | 0% | 0% |
| One slipped move caught | 98.4% | 94.7% | 98.1% |

Held any way round (face-on, tilted, a corner toward the camera), most views become "re-check" with a hint to turn the puzzle, and false alarms stay low: 0.5% on a pyraminx, 0.2% on an FTO.

Follow-along, on about 360 test renders per puzzle, held the right way: the position right from one look 98.9 / 97.8 / 98.9% (skewb / pyraminx / FTO), a slip caught 98.0 / 94.9 / 98.6%, and no false "wrong move".

## Follow-along on these puzzles

The cube tracker ([follow-along](follow-along.md)) with these puzzles' moves. Three things are new:

- **Moves done early.** Some turns don't disturb each other: a pyraminx's tips and anything else, an FTO's opposite faces. Which ones is worked out from the moves themselves. Turning a pyraminx's tips out of order isn't a mistake: the puzzle is counted as the point before the tip that's still to come.
- **A shorter way back.** It leaves alone the moves it needn't undo: a tip turned the wrong way three moves ago is put right with that tip alone.
- **Turns add up in thirds.** Every turn is a third of a turn, so there are no half turns and no "halfway"; R R is R'.

A pyraminx needs two faces in view (an edge toward the camera) before a look counts.

## In the browser

Each puzzle has its own three networks (`models/<event>/finder.onnx`, `refiner.onnx`, `reader.onnx`, about 23 MB), loaded the first time the puzzle is chosen. The puzzles' shapes, moves and flat pictures come as numbers in `web/src/lib/poly-data.json`, written by `python/scripts/poly_web_data.py`, so the TypeScript doesn't redo the geometry. `web/tests/poly.test.ts` checks the pose fit, the sticker places, the verdicts and every follow-along report against Python's answers (`python/scripts/poly_reference.py`); the `/test` page does the same with the real networks in the browser.

The reader is exported with one operation written differently (`python/scripts/export_poly_onnx.py`): ONNX Runtime's WebGPU backend computed its `einsum` wrongly, so it's exported as a matrix product. With it, the browser on WebGPU gives Python's verdict on every test render.

## Not yet

- Megaminx and square-1.
- Real-puzzle results. The few real frames tried so far fit the shape to 0.5–4.9%, close to the 5% guard, because a real puzzle's corners are rounder than a render's and a close webcam bends the picture.
- The skewb's corner model knows only a corner toward the camera.
- The training renders for these puzzles aren't published yet (the models are, on [Hugging Face](https://huggingface.co/darshanvss/cubecheck) since `v0.3.0`).
