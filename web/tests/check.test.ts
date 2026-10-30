// The TypeScript check must give Python's verdicts. Reference: python/scripts/web_reference.py
// (20 test renders, each with its true scramble and a one-slip version).
//   node --test tests/
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { SOLVED, apply, expectedState, insideImage, parseScramble, singleSlips, verifyProbs, type Point } from "../src/lib/check.ts";

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

test("sticker centres inside the frame match Python", () => {
  for (const row of reference.filter((r: { corners: Point[] | null }) => r.corners)) {
    assert.deepEqual(insideImage(1280, 720, row.corners), row.inside, row.image);
  }
});

test("verdicts match Python, field by field", () => {
  let checked = 0;
  for (const row of reference.filter((r: { verdicts?: object }) => r.verdicts)) {
    const probs = Float32Array.from(row.probs.flat());
    for (const [kind, want] of Object.entries<Record<string, unknown>>(row.verdicts)) {
      const claim = want.scramble as string;
      const got = verifyProbs(probs, row.inside, expectedState(claim), singleSlips(claim));
      const label = `${row.image}, ${kind} scramble`;
      assert.equal(got.status, want.status, label);
      assert.equal(got.observed, want.observed, label);
      assert.deepEqual(got.mismatches, want.mismatches, label);
      assert.equal(got.hypothesis, want.hypothesis, label);
      assert.equal(got.slip, want.slip, label);
      checked++;
    }
  }
  assert.equal(checked, 40);
});
