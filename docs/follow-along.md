# Follow-along (beta)

Follow-along watches the scrambler turn the cube, from solved, and keeps track of where in the scramble they are. When a move goes wrong it says so at once, names the likely slip, and shows the way back. It's off by default: turn it on with **Try follow-along (beta)** under the scramble list, then pick **Follow along** and the scramble to follow.

The code is `web/src/lib/follow.ts`, a port of `python/cubecheck/follow.py`. `web/tests/follow.test.ts` feeds both the same 329 simulated looks and checks the reports agree at every one.

## What it does with each look

Each look is the sticker reader's probabilities for the stickers in view. The tracker keeps a score for every point of the scramble (0 = solved, up to every move done) and updates it the way a hidden Markov model does:

1. **Before looking**: the scrambler is probably at the same point, or a few moves on. A jump anywhere is possible, but unlikely.
2. **After looking**: each point's score is multiplied by how well its stickers match the look, in whichever of the 24 orientations fits it best.

A half turn (R2) is done as two quarter turns, so the cube after the first one is **halfway**, not a mistake: the page says "Keep turning".

## Catching a slip

When no point of the scramble fits, it tries one slip near where the cube was last seen on track: at that move or the 2 before it, followed by up to **6** correct moves done before the camera saw it. A slip is a move done wrong, an extra turn, a skipped move, or two moves swapped. An extra turn after the last move counts too.

A slip is shown only when it fits clearly better than any point of the scramble (by the same margin as the check) in **two looks in a row**, so one blurry frame can't flash it. It clears after two looks back on track. Looks that see under 30% of the stickers change nothing.

**The way back** is shortened: turns of one axis don't disturb each other, so next to each other they add up (D2 then D is D') or cancel (D' U D is U). It's shown as "Undo", "Undo, in this order", or "To fix it" when the undoing and the right move add up into one turn. Each cube along the way back is accepted at the first look, so the page follows the undoing turn by turn instead of showing a stale way back.

**Same cube, different stories.** Different slips can leave the very same cube: skipping move k then doing move k+1 is the cube an extra turn of move k+1 would give. The way back is right either way. The name shown is the likelier story: not before where the cube was last seen on track, then the shortest way back, then a skip, a swap, a wrong move, an extra turn, in that order. The camera sees the cube, not the hands, so the name is a best guess.

## Why 6 moves, and why one slip

Slips are often noticed a few moves later. Trying 3 moves after a slip caught none of the slips seen 5 or more moves late; 6 caught about 9 in 10 at 5 moves late, with no false mistakes in simulation, at about 4 ms a look on a 3×3 (18 ms on a 7×7) next to 100+ ms for the networks.

Two slips at once are not tried. A slip at one move is one of about 2T + 1 (T = the size's turns: 18 on a 3×3), and a second one can follow anywhere in the next 6 moves:

| Size | One slip, a look | Two slips, a look | Three slips, a look |
| --- | --- | --- | --- |
| 3×3 | about 780 cubes | 173 thousand (31 ms) | 38 million (7 s) |
| 7×7 | about 2,300 cubes | 1.5 million (3.6 s) | 979 million (39 minutes) |

Beyond the time, more stories mean more ways for a misread to fit by luck. A slip while undoing (a second slip) shows as "can't see it" until the cube is back on the scramble.

## Things to know

- **Some slips can't be seen.** R then R', or skipping a move whose next two cancel (F2 F2), leave the cube on the scramble. The tracker follows it backwards; that's not a bug.
- **Some are seen by only 2 or 3 stickers** from one side. Showing another corner settles it. When every move is done, the normal check judges the finished cube.
- **Partway back, the cube can be on the scramble again** (an undone extra turn). Then the page shows it following at an earlier move, and the moves from there are the rest of the way back.
- **2×2 is the weakest**: only 12 stickers in view, so a finger hides a quarter of what it sees. **5×5 to 7×7** lose track more often when fingers hide a whole edge; c7 is much better at that, but it's the case to watch.
- **False mistakes were 0% in simulation, not yet measured on a real webcam.** Real misreads come in patches (glare on one face). If it says "can't see it clearly", hold still with a corner toward the camera.
