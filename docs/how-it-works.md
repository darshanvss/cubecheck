# How it works

The goal: pick a scramble (pasted, or from the competition's TNoodle PDF), show the scrambled cube to a camera once, and get **correct** or **mis-scramble** with no other interaction. Any WCA cube from 2×2 to 7×7.

## Why one view is enough

Hold the cube with one corner toward the camera and three faces show: 3N² of its 6N² stickers (27 of 54 on a 3×3). A slipped or wrong move moves many pieces at once, so it changes about 8 of the 24 non-centre stickers in view on a 3×3. In simulation over about 196,000 3×3 mis-scrambles, the three-face view missed none. One face alone misses 3–17%.

So the job isn't to recognise the whole cube. It's to **verify**: compute what the visible stickers should be, and check the photo against that.

Two faces work too on 3×3 and bigger, which is how a cube is often held toward the scrambler. A 2×2 needs all three: it shows only 8 stickers on two faces, and 40% of single slips change none of them, so with two faces it asks for a corner toward the camera.

## 1. Find the 7 corners: finder and refiner

The three visible faces meet in 7 corners: six around the outside (P1–P6) and one in the middle (C). Once those are known, every sticker's position follows, at any cube size.

Two small networks of the same design (U-Nets, 2.5 million numbers each) do this:

- **Finder.** Sees the whole frame shrunk to 256 × 256. Answers whether there's a cube at all (a score from 0 to 1) and roughly where the corners are. A cube far from the camera is only about 20 pixels across in that view, so when the finder is unsure it **looks again 2× closer** around its rough guess. On real webcam frames that second look raised its median score from 0.34 to 0.93.
- **Refiner.** Sees a 256 × 256 crop around those rough corners, cut from the full-quality frame. Places the corners precisely. It runs twice, re-centring on its own answer, and also says whether there's a cube in its crop.

A cube counts only when both say so and the faces come out as proper four-sided shapes: three faces turning the same way, or (a view of exactly two faces) two, with the third seen from behind or edge-on.

The current model, **c7**, practised on painted fingers: over a corner, and across a whole outer edge the way a hand grips a cube. With fingers hiding an edge it puts those corners half as far off as the model before it, and false alarms on a 7×7 held that way fell from 19% to 6%.

**How a network outputs positions.** For each corner it outputs a **heatmap**, a 64 × 64 grid of scores with a blob where the corner is. Softmax turns the scores into weights, and the corner is their weighted average position, which can land between grid cells.

**Which face comes first?** The model doesn't need to know which face is white or yellow. The check tries all three possible starting faces, so training scores each answer against whichever of the 3 rotations of the labels fits best.

**Live video.** Each frame starts the refiner from the last frame's corners and skips the finder, falling back to a full search when the cube is lost. A One Euro filter smooths the corners: it averages hard when the cube is still and follows quickly when it moves.

## 2. Read the stickers

The corners give an N × N grid on each face. Each face is straightened into a square of 24 pixels per sticker (72 × 72 on a 3×3, 168 × 168 on a 7×7), so every sticker sits in the same spot, and the **sticker reader** (r4) reads all three faces together:

- A small convolutional network describes each sticker.
- A small network tells it where each sticker sits (which face, how far along and down, on an edge or a corner, an odd cube's fixed middle), so the same reader works for every size.
- A transformer layer lets the stickers compare with each other. Under warm light a white sticker looks cream, which could pass for pale yellow. Next to the real yellows on the same cube it's clearly white.
- For each sticker it outputs 7 probabilities: the 6 colours, or **covered** (a finger is in the way).

Covered stickers are left out of the check, and so are stickers on a face seen nearly edge-on (a sliver a few pixels thin). Losing a few barely matters, because a mistake changes many.

In the browser there's one reader file per cube size (about 3 MB each), loaded the first time that size is checked.

### Thicker outer layers on big cubes

Big cubes aren't cut into equal layers: the outer layers are thicker, so the cube can be gripped and doesn't pop. Measured with a caliper, a 6×6 has 13.5 mm outer layers and 9.5 mm inner ones (1.42×), a 7×7 13 mm and 8.2 mm (1.59×). The training renders have equal layers, so on a real 7×7 an even grid puts the second ring's cells partly over the outer stickers next to them.

So when a face is straightened, each layer is stretched or squeezed to the same 24 pixels. The reader then sees every sticker in its own cell, as in training, and doesn't need retraining. In simulation (renders re-proportioned to the measured widths, read with true corners), the second ring of a 7×7 went from 94.2% of stickers read right with the even grid to 98.3% with the measured one, the same as on the renders themselves. On a 6×6 the gain is small (98.1% to 98.5% overall).

Other makes of cube have other proportions, so on 4×4 to 7×7 the page has an **Outer layers** setting next to the grid switch: even, 1.2×, 1.4×, 1.8×, or the measured ratio (the default on 6×6 and 7×7; 4×4 and 5×5 start even). The grid drawn on the cube follows it, so a wrong setting is easy to see. Python takes the same widths: `nxn_check.layer_edges(n, (outer, inner))`, passed to `reader.read_grid(..., edges=...)` and `nxn_check.cell_centres(..., edges)`.

## 3. Check against the scramble

1. **Expected cube.** Apply the scramble to a solved cube: all 6N² stickers.
2. **Orientation.** The cube can be held any way up, so all 24 orientations are tried. On odd cubes the middle stickers never move, so only orientations whose visible middles fit are considered, and of those the one the whole photo fits best. Even cubes (2×2, 4×4, 6×6) have no fixed middles, so the photo decides alone.
3. **Cube rules (3×3).** Real cubes have each corner and edge piece exactly once, so on a 3×3 the reader's probabilities are decoded as whole pieces, not single stickers. This fixes most leftover misreads (red vs orange, white vs yellow). Bigger cubes use the reader's own best guess per sticker.
4. **Slip test.** The reading is compared with the scramble as written and with every scramble that has one human slip: a move skipped, done wrong, an extra turn, or two moves swapped (687 cubes on a 3×3, about 11,000 on a 7×7, worked out once per scramble). If one of those fits clearly better, and at least 3 of the stickers it changes look wrong, it's a **mis-scramble**. If the scramble fits, it's **correct**. If the picture is too unclear to tell, it says **re-check**. The settings are tuned per size: a 2×2 shows 12 stickers, a 7×7 147.
5. **Over several frames.** Live, a verdict needs several agreeing frames, so one blurry frame can't flip it.

## Scrambles from a TNoodle PDF

TNoodle, the WCA's scramble program, writes each scramble into its PDFs as text, so the page can read them without any pictures. It runs **pdf.js** in the browser: the PDF is opened, unlocked with its passcode if it has one (a group's Computer Display PDF has its own passcode; the printing PDFs share the competition's password), and each page's title ("3x3x3 Round 1 Scramble Set A") and numbered scrambles are picked out by where they sit on the page. Long scrambles wrap over several lines, so each number is matched to the run of lines it's centred on.

A PDF with no page titled like a TNoodle scramble set is refused, and so is a file that isn't a PDF. The events found are listed in the WCA's order; the cube events from 2×2 to 7×7 can be checked (3×3 blindfolded, one-handed and Fewest Moves are 3×3 scrambles too), other puzzles show as "not yet". The PDF stays in the page's memory and is gone when the page closes.

## Follow-along (beta)

Instead of checking the finished cube, the page can follow one scramble move by move, from a solved cube, and say the moment a move goes wrong, with the way back. It's behind a switch that's off by default. See [follow-along.md](follow-along.md).

## Training data

All synthetic: 3D renders with exact labels, plus random camera effects and painted fingers at training time. See [dataset.md](dataset.md) and [training.md](training.md).
