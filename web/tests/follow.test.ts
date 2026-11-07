// Follow-along in TypeScript must give Python's reports, look by look. Reference: python/scripts/follow_reference.py
// (random scrambles of 2x2 to 5x5, each with one slip, undone the way shown, then finished).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { Follower, inverse, shorten } from "../src/lib/follow.ts";
import { cube, guessSize, randomScramble } from "../src/lib/nxn.ts";
import { SOLVED, apply } from "../src/lib/check.ts";

const runs = JSON.parse(readFileSync(new URL("./fixtures/follow.json", import.meta.url), "utf8"));

test("every report matches Python's, look by look", () => {
  let looks = 0;
  for (const run of runs) {
    const f = new Follower(run.scramble, run.size), inside = Array(3 * run.size * run.size).fill(true);
    run.looks.forEach((look: { probs: number[]; report: Record<string, unknown> }, i: number) => {
      const got = f.look(Float32Array.from(look.probs), inside) as unknown as Record<string, unknown>;
      const label = `${run.size}x${run.size} ${run.slip} at ${run.at}, look ${i + 1}`;
      for (const [k, want] of Object.entries(look.report)) assert.deepEqual(got[k], want, `${label}: ${k}`);
      looks++;
    });
  }
  assert.ok(looks > 300);
});

test("the way back is shortened", () => {
  assert.deepEqual(shorten(["D2", "D"]), ["D'"]);
  assert.deepEqual(shorten(["D'", "U", "D"]), ["U"]);
  assert.deepEqual(shorten(["R", "R'"]), []);
  assert.deepEqual(shorten(["R", "U", "U'", "R"]), ["R2"]);
  assert.equal(inverse("3Rw'"), "3Rw");
  assert.equal(inverse("Rw2"), "Rw2");
});

test("the any-size cube is the 3x3 cube for N = 3", () => {
  const moves = randomScramble(3);
  assert.deepEqual(cube(3).after(moves), apply(SOLVED, moves));
});

test("a scramble's size is told from its moves", () => {
  for (const n of [2, 3, 4, 5, 6, 7]) for (let i = 0; i < 20; i++) assert.equal(guessSize(randomScramble(n).join(" ")), n);
  assert.equal(guessSize("R U X"), null);
});
