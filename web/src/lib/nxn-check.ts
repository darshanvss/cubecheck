// The check for any N×N cube: 7 corners, N×N stickers per face, scramble -> verdict.
// A port of python/cubecheck/nxn_check.py; tests/nxn.test.ts checks it gives Python's answers.
//
// The same steps as verifyProbs (check.ts), for every size: each visible face is a perspective-warped
// square, so its N×N sticker centres come from its 4 corners; all 24 ways the cube can face the camera
// are tried (an odd cube's middle stickers narrow it down; an even cube has none, so the photo decides
// alone); then a single slip that fits clearly better, or enough stickers the reader is sure are wrong,
// means mis-scramble. The cube rules that pick colours piece by piece are 3×3 only, so here each
// sticker's colour is the reader's own best guess.

import { FACES, FACE_QUADS, U_FLIP, V_FLIP, argmin, blank, hypotheses, squareToQuad, type Alternatives, type Point, type SlipWhere, type Verdict } from "./check.ts";
import { cube, type NState } from "./nxn.ts";

const cellsCache = new Map<number, Uint16Array[]>();

/** For each of the 24 orientations, the facelet behind each of the 3N² cells (face k of the image, row j, column i). */
export function cells(n: number): Uint16Array[] {
  if (cellsCache.has(n)) return cellsCache.get(n)!;
  const c = cube(n), h = n - 1;
  const out = hypotheses().map(([s, axes]) => {
    const idx: number[] = [];
    for (let k = 0; k < 3; k++) {
      const a = axes[k], au = axes[U_FLIP[k]], av = axes[V_FLIP[k]];
      const normal = [0, 0, 0];
      normal[a] = s[a];
      for (let j = 0; j < n; j++) {
        for (let i = 0; i < n; i++) {
          const p = [0, 0, 0];
          p[a] = s[a] * h;
          p[au] = s[au] * (h - 2 * i);
          p[av] = s[av] * (h - 2 * j);
          idx.push(c.index.get(`${p.map((v) => v + 0).join(",")}|${normal.map((v) => v + 0).join(",")}`)!);
        }
      }
    }
    return Uint16Array.from(idx);
  });
  cellsCache.set(n, out);
  return out;
}

/** The fixed middle sticker of each visible face (odd cubes only). */
export const middleCells = (n: number) => (n % 2 ? [0, 1, 2].map((k) => k * n * n + Math.floor(n / 2) * n + Math.floor(n / 2)) : []);

/** Image positions of the 3N² sticker centres: face k, row j, column i. */
export function cellCentres(corners: Point[], n: number): Point[] {
  return FACE_QUADS.flatMap((quad) => {
    const map = squareToQuad(quad.map((j) => corners[j]));
    return Array.from({ length: n * n }, (_, m) => map(((m % n) + 0.5) / n, (Math.floor(m / n) + 0.5) / n));
  });
}

/** Signed area of each face quad: faces seen from the front all wind one way, one seen from behind the other. */
export function faceWindings(corners: Point[]): number[] {
  return FACE_QUADS.map((quad) => {
    const p = quad.map((j) => corners[j]);
    let a = 0;
    for (let i = 0; i < 4; i++) a += p[i][0] * p[(i + 1) % 4][1] - p[(i + 1) % 4][0] * p[i][1];
    return a / 2;
  });
}

// How small a face may look (its area over the largest face's) and still be read, per size; sizes where
// two faces aren't enough (a 2×2 shows 8 of 24 stickers, and 40% of slips change none of them); the
// check's thresholds per size (a 2×2 shows 12 stickers, a 7×7 147); and the smallest readable sticker.
export const MIN_FACE_SHARE: Record<number, number> = { 2: 0.03, 3: 0.06, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1 };
export const NEEDS_THREE_FACES = new Set([2]);
export const THRESHOLDS: Record<number, { minEvidence: number; minSlipSeen: number; minMismatches: number }> = {
  2: { minEvidence: 6, minSlipSeen: 2, minMismatches: 3 },
  3: { minEvidence: 8, minSlipSeen: 3, minMismatches: 6 },
  4: { minEvidence: 8, minSlipSeen: 3, minMismatches: 8 },
  5: { minEvidence: 8, minSlipSeen: 3, minMismatches: 8 },
  6: { minEvidence: 8, minSlipSeen: 3, minMismatches: 12 },
  7: { minEvidence: 8, minSlipSeen: 3, minMismatches: 16 },
};
export const PX_PER_STICKER = 14;

/**
 * Which of the 3N² stickers can be read: inside the picture, on a face seen from the front and not nearly
 * edge-on (at least minFaceShare of the largest face's area). A cube held toward the scrambler often shows
 * two faces and a sliver of the third; the sliver is skipped, and so is a face seen from behind.
 */
export function readable(corners: Point[], n: number, width: number, height: number, minFaceShare = MIN_FACE_SHARE[n] ?? 0.1): boolean[] {
  const signed = faceWindings(corners), areas = signed.map(Math.abs), big = signed[argmin(areas.map((a) => -a))];
  const face = signed.map((s, k) => s * big > 0 && areas[k] >= minFaceShare * Math.max(...areas));
  return cellCentres(corners, n).map(([x, y], m) => face[Math.floor(m / (n * n))] && x >= 0 && x < width && y >= 0 && y < height);
}

/** out[i] = a[perm[i]]: a state after a move (a state, then a permutation), or two permutations combined. */
function permute<T extends Uint8Array | Uint16Array>(a: T, perm: Uint16Array): T {
  const out = (a instanceof Uint16Array ? new Uint16Array(perm.length) : new Uint8Array(perm.length)) as T;
  for (let i = 0; i < perm.length; i++) out[i] = a[perm[i]];
  return out;
}

/**
 * States one human slip away from the scramble (a move skipped, done wrong, an extra turn, two moves
 * swapped), each once, with this size's WCA moves. The states before each move (prefix) and the combined
 * effect of the moves after it (suffix) are worked out once, so a 7×7's ~11,000 slips stay quick.
 */
export function singleSlipsN(scramble: string, n: number): Alternatives {
  const c = cube(n), moves = c.parse(scramble), perms = moves.map((m) => c.move(m));
  const prefix: NState[] = [c.solved];
  for (const p of perms) prefix.push(permute(prefix[prefix.length - 1], p));
  const identity = Uint16Array.from({ length: c.size }, (_, i) => i);
  const suffix: Uint16Array[] = Array(moves.length + 1).fill(identity); // after all moves = after k moves, then [suffix[k]]
  for (let k = moves.length - 1; k >= 0; k--) suffix[k] = permute(perms[k], suffix[k + 1]);
  const seen = new Set([prefix[moves.length].join()]), out: Alternatives = { states: [], slips: [], where: [] };
  const add = (st: NState, text: string, move: number, kind: SlipWhere["kind"], done: string | null = null) => {
    const k = st.join();
    if (seen.has(k)) return;
    seen.add(k);
    out.states.push(st);
    out.slips.push(text);
    out.where.push({ move, kind, done });
  };
  const turns = c.eventTurns.map((t) => [t, c.move(t)] as const);
  moves.forEach((m, i) => {
    const before = prefix[i];
    add(permute(before, suffix[i + 1]), `move ${i + 1} (${m}) skipped`, i, "skipped");
    for (const [t, pt] of turns) if (t !== m) add(permute(permute(before, pt), suffix[i + 1]), `move ${i + 1} done as ${t} instead of ${m}`, i, "wrong", t);
    if (i + 1 < moves.length) add(permute(permute(permute(before, perms[i + 1]), perms[i]), suffix[i + 2]), `moves ${i + 1} and ${i + 2} swapped`, i, "swapped");
  });
  for (let i = 0; i <= moves.length; i++) {
    for (const [t, pt] of turns) add(permute(permute(prefix[i], pt), suffix[i]), i < moves.length ? `extra ${t} before move ${i + 1}` : `extra ${t} at the end`, i, "extra", t);
  }
  return out;
}

/** nxn_check.check: the reader's (3N² x 7) probabilities against the state a scramble should give. */
export function checkN(
  probs: ArrayLike<number>, // 3N² x 7, row-major: 6 colours (URFDLB), then covered
  inside: boolean[],
  n: number,
  state: NState,
  alternatives?: Alternatives,
  { covered = 0.5, cap = 5, sure = 0.05, maxUnknown = 0.48, unsure = 0.6, maxUnsure = 0.33, maxSlipLeft = 1, ...over }: Partial<Record<string, number>> = {},
): Verdict {
  const { minEvidence, minSlipSeen, minMismatches } = { ...(THRESHOLDS[n] ?? THRESHOLDS[3]), ...over };
  const m = 3 * n * n, P = (i: number, k: number) => probs[7 * i + k];
  if (NEEDS_THREE_FACES.has(n) && [0, 1, 2].filter((k) => inside.slice(k * n * n, (k + 1) * n * n).some(Boolean)).length < 3) {
    return blank("re-check", "turn a corner toward the camera: this cube needs three faces in view", inside.flatMap((v, i) => (v ? [] : [i])));
  }
  const unknown = Array.from({ length: m }, (_, i) => !inside[i] || P(i, 6) > covered);
  const unknownList = unknown.flatMap((u, i) => (u ? [i] : []));
  if (unknownList.length > maxUnknown * m) return blank("re-check", "too much of the cube is hidden or out of view", unknownList);
  const colour = Array.from({ length: m }, (_, i) => {
    let sum = 0;
    for (let k = 0; k < 6; k++) sum += P(i, k);
    return Array.from({ length: 6 }, (_, k) => P(i, k) / Math.max(sum, 1e-9));
  });
  const cost = colour.map((row) => row.map((p) => Math.min(-Math.log(Math.max(p, 1e-9)), cap)));
  if (colour.filter((row, i) => !unknown[i] && Math.max(...row) < unsure).length > maxUnsure * m) {
    return blank("re-check", "colours aren't clear enough (glare, too dark or washed out)", unknownList);
  }
  const rows = unknown.flatMap((u, i) => (u ? [] : [i])), all = cells(n);
  let fits = all.map((cl) => rows.reduce((sum, i) => sum + cost[i][state[cl[i]]], 0));
  const middles = middleCells(n).filter((i) => !unknown[i]);
  if (middles.length) {
    // Odd cubes: the middle stickers never move, so they say which way round the cube is.
    const centre = all.map((cl) => middles.reduce((sum, i) => sum + cost[i][state[cl[i]]], 0)), low = Math.min(...centre);
    fits = fits.map((f, h) => (centre[h] <= low + 1 ? f : Infinity));
  }
  const best = argmin(fits), cl = all[best], expected = Array.from(cl, (f) => state[f]);
  const nearest = cost.map(argmin);
  const mismatch = expected.map((e, i) => !unknown[i] && nearest[i] !== e && colour[i][e] < sure);
  const base: Verdict = {
    ...blank("correct", "matches the scramble", unknownList),
    mismatches: mismatch.flatMap((x, i) => (x ? [i] : [])),
    facelets: Array.from(cl),
    expected: expected.map((e) => FACES[e]).join(""),
    observed: nearest.map((c, i) => (unknown[i] ? "?" : FACES[c])).join(""),
    hypothesis: best,
    fit: fits[best],
  };
  if (alternatives?.states.length) {
    // Each slip in every orientation still possible (on an even cube, all 24).
    const rowCost = rows.map((i) => cost[i]);
    let k = 0, h = best, slipFit = Infinity;
    fits.forEach((f, o) => {
      if (!Number.isFinite(f)) return;
      const co = rows.map((i) => all[o][i]);
      alternatives.states.forEach((st, j) => {
        let s = 0;
        for (let r = 0; r < co.length && s < slipFit; r++) s += rowCost[r][st[co[r]]];
        if (s < slipFit) [k, h, slipFit] = [j, o, s];
      });
    });
    const slipped = Array.from(all[h], (f) => alternatives.states[k][f]);
    const changed = slipped.flatMap((s, i) => (s !== expected[i] && !unknown[i] ? [i] : []));
    base.evidence = fits[best] - slipFit;
    base.slipSeen = changed.filter((i) => colour[i][expected[i]] < 0.5).length;
    base.slipLeft = slipped.filter((s, i) => !unknown[i] && colour[i][s] < sure).length;
    if (base.evidence > minEvidence && base.slipSeen >= minSlipSeen) {
      if (base.slipLeft > maxSlipLeft) return { ...base, status: "mis-scramble", reason: "doesn't match the scramble" };
      return { ...base, status: "mis-scramble", reason: "doesn't match the scramble", mismatches: changed, slip: alternatives.slips[k], slipWhere: alternatives.where[k] };
    }
  }
  if (base.mismatches.length >= minMismatches) return { ...base, status: "mis-scramble", reason: "doesn't match the scramble" };
  return base;
}
