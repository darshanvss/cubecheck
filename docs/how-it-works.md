# How it works

The goal: paste a 3×3 scramble, show the scrambled cube to a camera once, and get **correct** or **mis-scramble** with no other interaction.

## Why one view is enough

Hold the cube with one corner toward the camera and three faces show: 27 stickers, 24 of them not centres. A slipped or wrong move moves many pieces at once, so it changes about 8 of those 24 stickers on average. In simulation over about 196,000 mis-scrambles, the three-face view missed none. One face alone misses 3–17%.

So the job isn't to recognise the whole cube. It's to **verify**: compute what the 27 visible stickers should be, and check the photo against that.

## 1. Find the 7 corners: finder and refiner

The three visible faces meet in 7 corners: six around the outside (P1–P6) and one in the middle (C). Once those are known, every sticker's position follows.

Two small networks of the same design (U-Nets, 2.5 million numbers each) do this:

- **Finder.** Sees the whole frame shrunk to 256 × 256. Answers whether there's a cube at all (a score from 0 to 1) and roughly where the corners are.
- **Refiner.** Sees a 256 × 256 crop around those rough corners, cut from the full-quality frame. Places the corners precisely. It runs twice, re-centring on its own answer, and also says whether there's a cube in its crop.

A cube counts only when both say so and the three faces come out as proper four-sided shapes.

**How a network outputs positions.** For each corner it outputs a **heatmap**, a 64 × 64 grid of scores with a blob where the corner is. Softmax turns the scores into weights, and the corner is their weighted average position, which can land between grid cells.

**Which face comes first?** The model doesn't need to know which face is white or yellow. The check tries all three possible starting faces, so training scores each answer against whichever of the 3 rotations of the labels fits best.

**Live video.** Each frame starts the refiner from the last frame's corners and skips the finder, falling back to a full search when the cube is lost. A One Euro filter smooths the corners: it averages hard when the cube is still and follows quickly when it moves.

## 2. Read the 27 stickers

The corners give a 3 × 3 grid on each face. Each face is straightened into a 72 × 72 square, so every sticker sits in the same spot, and the **sticker reader** reads all three faces together:

- A small convolutional network describes each sticker.
- A transformer layer lets the 27 stickers compare with each other. Under warm light a white sticker looks cream, which could pass for pale yellow. Next to the real yellows on the same cube it's clearly white.
- For each sticker it outputs 7 probabilities: the 6 colours, or **covered** (a finger is in the way).

Covered stickers are simply left out of the check. Losing a few of 24 barely matters, because a mistake changes about 8.

## 3. Check against the scramble

1. **Expected cube.** Apply the scramble to a solved cube: all 54 stickers.
2. **Orientation.** The cube can be held any way up, so all 24 orientations are tried. Face turns never move the centres, so only orientations whose visible centres fit are considered, and of those the one the whole photo fits best.
3. **Cube rules.** Real cubes have each corner and edge piece exactly once, so the reader's probabilities are decoded as whole pieces, not single stickers. This fixes most leftover misreads (red vs orange, white vs yellow).
4. **Slip test.** The reading is compared with the scramble as written and with every scramble that has one move slipped or wrong. If one of those fits clearly better, it's a **mis-scramble**. If the scramble fits, it's **correct**. If the picture is too unclear to tell, it says **re-check**.
5. **Over several frames.** Live, a verdict needs several agreeing frames, so one blurry frame can't flip it.

## Training data

All synthetic: 3D renders with exact labels, plus random camera effects at training time. See [dataset.md](dataset.md) and [training.md](training.md).
