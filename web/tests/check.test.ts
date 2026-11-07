// The TypeScript check must give Python's verdicts. Reference: python/scripts/web_reference.py
// (20 3x3 test renders and 4 of every other size, each with its true scramble and a one-slip version).
//   node --test tests/
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { SOLVED, apply, expectedState, parseScramble, singleSlips, verifyProbs, type Point } from "../src/lib/check.ts";
import { checkN, readable, singleSlipsN } from "../src/lib/nxn-check.ts";
import { cube } from "../src/lib/nxn.ts";

const reference = JSON.parse(readFileSync(new URL("./fixtures/reference.json", import.meta.url), "utf8"));

test("a scramble and its inverse give the solved cube", () => {
  const moves = parseScramble("R U R' U' F2 D' L B2");
  const inverse = [...moves].reverse().map((m) => (m.endsWith("'") ? m[0] : m.endsWith("2") ? m : `${m}'`));
  assert.deepEqual(apply(apply(SOLVED, moves), inverse), SOLVED);
});

test("four quarter turns change nothing, and bad notation is refused", () => {
  assert.deepEqual(apply(SOLVED, ["R", "R", "R", "R"]), SOLVED);
  assert.throws(() => parseScramble("R U X2"), /X2/);
});

test("the stickers that can be read match Python", () => {
  for (const row of reference.filter((r: { corners: Point[] | null }) => r.corners)) {
    assert.deepEqual(readable(row.corners, row.size, row.width, row.height), row.inside, row.image);
  }
});

test("verdicts match Python, field by field, for every size", () => {
  let checked = 0;
  for (const row of reference.filter((r: { verdicts?: object }) => r.verdicts)) {
    const probs = Float32Array.from(row.probs.flat()), n: number = row.size;
    for (const [kind, want] of Object.entries<Record<string, unknown>>(row.verdicts)) {
      const claim = want.scramble as string;
      const alts = n === 3 ? singleSlips(claim) : singleSlipsN(claim, n);
      const got = n === 3 ? verifyProbs(probs, row.inside, expectedState(claim), alts) : checkN(probs, row.inside, n, cube(n).after(claim), alts);
      const label = `${row.image}, ${kind} scramble`;
      assert.equal(alts.states.length, want.slips, `${label}: one-slip cubes`);
      assert.equal(got.status, want.status, label);
      assert.equal(got.observed, want.observed, label);
      assert.deepEqual(got.mismatches, want.mismatches, label);
      assert.equal(got.hypothesis, want.hypothesis, label);
      assert.equal(got.slip, want.slip, label);
      checked++;
    }
  }
  assert.equal(checked, 2 * reference.length);
});
