// Skewb, pyraminx and FTO: the TypeScript must place the stickers and give the verdicts Python does.
// Reference: python/scripts/poly_reference.py (test renders of each puzzle, the corners Python's models found,
// the same corners jittered, the reader's probabilities, the verdicts for the true and a slipped scramble).
//   node --test tests/
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import type { Point } from "../src/lib/check.ts";
import { POLY_EVENTS, poly, type PolyEvent } from "../src/lib/poly.ts";

interface Geometry {
  corners: Point[];
  fit: number;
  centres: Point[];
  inside: boolean[];
  wellFormed: boolean[];
}
interface Row {
  event: PolyEvent;
  image: string;
  width: number;
  height: number;
  scramble: string;
  found: Geometry | null;
  jittered?: Geometry[];
  probs?: number[][];
  verdicts?: Record<string, { scramble: string; status: string; reason: string; observed: string; mismatches: number[]; hypothesis: number; slip: string; slips: number }>;
}
const reference: Row[] = JSON.parse(readFileSync(new URL("./fixtures/poly-reference.json", import.meta.url), "utf8"));

test("a scramble undone gives the solved puzzle, and each turn three times changes nothing", () => {
  for (const e of POLY_EVENTS) {
    const p = poly(e), moves = p.randomScramble();
    const back = [...moves].reverse().map((m) => (m.endsWith("'") ? m.slice(0, -1) : `${m}'`));
    assert.deepEqual(p.apply(p.apply(p.solved, moves), back), p.solved, e);
    for (const t of p.turns) assert.deepEqual(p.apply(p.solved, [t, t, t]), p.solved, `${e} ${t}`);
    assert.throws(() => p.parse("R X"), /X isn't a/);
  }
});

test("the pose fit and the sticker places match Python's", () => {
  let checked = 0, worst = 0, fitGap = 0;
  for (const row of reference) {
    const p = poly(row.event);
    for (const g of [row.found, ...(row.jittered ?? [])]) {
      if (!g) continue;
      const label = `${row.image} (fit ${g.fit})`;
      assert.deepEqual(Array.from({ length: p.d.nFaces + 1 }, (_, f) => p.wellFormed(g.corners, f)), g.wellFormed, label);
      if (!p.d.quads) {
        const { err } = p.fit(g.corners, row.width, row.height);
        fitGap = Math.max(fitGap, Math.abs(err - g.fit));
        assert.ok(Math.abs(err - g.fit) < Math.max(2e-4, 0.02 * g.fit), `${label}: fit ${err} vs ${g.fit}`);
      }
      const centres = p.cellCentres(g.corners, row.width, row.height);
      if (g.fit < 0.05 || p.d.quads) {
        centres.forEach((c, i) => (worst = Math.max(worst, Math.hypot(c[0] - g.centres[i][0], c[1] - g.centres[i][1]))));
        centres.forEach((c, i) => assert.ok(Math.hypot(c[0] - g.centres[i][0], c[1] - g.centres[i][1]) < 0.5, `${label}: centre ${i}`));
      }
      assert.deepEqual(p.readable(g.corners, row.width, row.height), g.inside, label);
      checked++;
    }
  }
  assert.ok(checked >= 100, `checked ${checked}`);
  console.log(`# ${checked} corner sets: sticker centres within ${worst.toFixed(3)} px, fit error within ${fitGap.toExponential(1)}`);
});

test("verdicts match Python, field by field", () => {
  let checked = 0;
  for (const row of reference) {
    if (!row.verdicts) continue;
    const p = poly(row.event), probs = Float64Array.from(row.probs!.flat()), inside = row.found!.inside;
    for (const [kind, want] of Object.entries(row.verdicts)) {
      const alts = p.singleSlips(want.scramble), got = p.check(probs, inside, p.after(want.scramble), alts);
      const label = `${row.image}, ${kind} scramble`;
      assert.equal(alts.states.length, want.slips, `${label}: one-slip states`);
      assert.equal(got.status, want.status, label);
      assert.equal(got.reason, want.reason, label);
      if (want.observed) {
        assert.equal(got.observed, want.observed, label);
        assert.deepEqual(got.mismatches, want.mismatches, label);
        assert.equal(got.hypothesis, want.hypothesis, label);
      }
      assert.equal(got.slip || null, want.slip || null, label);
      checked++;
    }
  }
  assert.equal(checked, 2 * reference.filter((r) => r.verdicts).length);
});

test("follow-along gives Python's reports, look by look", async () => {
  const { PolyFollower } = await import("../src/lib/poly-follow.ts");
  const runs: { event: PolyEvent; scramble: string; looks: { probs: number[]; inside: boolean[]; report: Record<string, unknown> }[] }[] =
    JSON.parse(readFileSync(new URL("./fixtures/poly-follow.json", import.meta.url), "utf8"));
  let looks = 0;
  for (const run of runs) {
    const f = new PolyFollower(run.scramble, run.event);
    run.looks.forEach((lk, i) => {
      const got = f.look(Float64Array.from(lk.probs), lk.inside) as unknown as Record<string, unknown>;
      const label = `${run.event} "${run.scramble}", look ${i + 1}`;
      for (const [k, want] of Object.entries(lk.report)) assert.deepEqual(got[k] ?? null, want ?? null, `${label}: ${k}`);
      looks++;
    });
  }
  assert.ok(looks > 100);
});
