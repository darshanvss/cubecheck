// From the sticker reader's probabilities and a scramble to a verdict: correct, mis-scramble or re-check.
// A line-for-line port of python/cubecheck (cube.py, and verify.py's verify_probs); tests/check.test.ts
// checks it gives Python's verdicts.

export const FACES = "URFDLB";
type V3 = [number, number, number];
export type State = Uint8Array; // 54 colours, Kociemba order U R F D L B, 9 per face in reading order

// --- the cube: a facelet is (cubie position, outward normal); a face turn rotates its layer ---

const NORMALS: Record<string, V3> = { U: [0, 1, 0], R: [1, 0, 0], F: [0, 0, 1], D: [0, -1, 0], L: [-1, 0, 0], B: [0, 0, -1] };

function facePositions(face: string): V3[] {
  const pts: V3[] = [];
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      const a = r - 1, b = c - 1;
      pts.push(({ U: [b, 1, a], R: [1, -a, -b], F: [b, -a, 1], D: [b, -1, -a], L: [-1, -a, b], B: [-b, -a, -1] } as Record<string, V3>)[face]);
    }
  }
  return pts;
}

const FACELETS: [V3, V3][] = [...FACES].flatMap((f) => facePositions(f).map((p): [V3, V3] => [p, NORMALS[f]]));
const key = (p: number[], n: number[]) => `${p.map((v) => v + 0).join(",")}|${n.join(",")}`; // + 0 turns -0 into 0
const INDEX = new Map(FACELETS.map(([p, n], i) => [key(p, n), i]));

function rotate(axis: V3, quarterTurns: number, v: V3): V3 {
  const th = (-Math.PI / 2) * quarterTurns, s = Math.sin(th), c = 1 - Math.cos(th);
  const [x, y, z] = axis;
  const k = [[0, -z, y], [z, 0, -x], [-y, x, 0]];
  const kk = k.map((row) => [0, 1, 2].map((j) => row[0] * k[0][j] + row[1] * k[1][j] + row[2] * k[2][j]));
  const m = [0, 1, 2].map((i) => [0, 1, 2].map((j) => Math.round((i === j ? 1 : 0) + s * k[i][j] + c * kk[i][j])));
  return m.map((row) => row[0] * v[0] + row[1] * v[1] + row[2] * v[2]) as V3;
}

function movePermutation(face: string, quarterTurns: number): Uint8Array {
  const n = NORMALS[face], perm = Uint8Array.from({ length: 54 }, (_, i) => i); // new[dst] = old[src]
  FACELETS.forEach(([p, nv], src) => {
    if (p[0] * n[0] + p[1] * n[1] + p[2] * n[2] === 1) {
      perm[INDEX.get(key(rotate(n, quarterTurns, p), rotate(n, quarterTurns, nv)))!] = src;
    }
  });
  return perm;
}

export const MOVES: Record<string, Uint8Array> = {};
for (const f of FACES) {
  MOVES[f] = movePermutation(f, 1);
  MOVES[`${f}2`] = movePermutation(f, 2);
  MOVES[`${f}'`] = movePermutation(f, 3);
  MOVES[`${f}2'`] = MOVES[`${f}2`];
}

export const SOLVED: State = Uint8Array.from({ length: 54 }, (_, i) => Math.floor(i / 9));

export function apply(state: State, moves: string[]): State {
  let s = state;
  for (const m of moves) {
    const perm = MOVES[m], next = new Uint8Array(54);
    for (let i = 0; i < 54; i++) next[i] = s[perm[i]];
    s = next;
  }
  return s;
}

/** WCA face-turn notation (R U' F2 ...); throws on anything else. */
export function parseScramble(text: string): string[] {
  const moves = text.replace(/’/g, "'").split(/\s+/).filter(Boolean);
  const bad = moves.filter((m) => !(m in MOVES));
  if (bad.length) throw new Error(`not a 3x3 face turn: ${bad.join(" ")}`);
  return moves;
}

export const expectedState = (scramble: string) => apply(SOLVED, parseScramble(scramble));

export interface Alternatives {
  states: State[];
  slips: string[];
}

/** States from one human slip: a move skipped, changed, added, or two moves swapped. Each state once. */
export function singleSlips(scramble: string): Alternatives {
  const moves = parseScramble(scramble);
  const seen = new Set([apply(SOLVED, moves).join()]), out: Alternatives = { states: [], slips: [] };
  const add = (variant: string[], text: string) => {
    const st = apply(SOLVED, variant), k = st.join();
    if (!seen.has(k)) {
      seen.add(k);
      out.states.push(st);
      out.slips.push(text);
    }
  };
  const turns = [...FACES].flatMap((f) => [f, `${f}'`, `${f}2`]);
  moves.forEach((m, i) => {
    add([...moves.slice(0, i), ...moves.slice(i + 1)], `move ${i + 1} (${m}) skipped`);
    for (const t of turns) if (t !== m) add([...moves.slice(0, i), t, ...moves.slice(i + 1)], `move ${i + 1} done as ${t} instead of ${m}`);
    if (i + 1 < moves.length) add([...moves.slice(0, i), moves[i + 1], m, ...moves.slice(i + 2)], `moves ${i + 1} and ${i + 2} swapped`);
  });
  for (let i = 0; i <= moves.length; i++) {
    for (const t of turns) add([...moves.slice(0, i), t, ...moves.slice(i)], i < moves.length ? `extra ${t} before move ${i + 1}` : `extra ${t} at the end`);
  }
  return out;
}

// --- orientation: which cube corner faces the camera, and which visible face is image face 0 ---

const U_FLIP = [2, 0, 1], V_FLIP = [1, 2, 0];

function hypotheses(): [V3, number[]][] {
  const out: [V3, number[]][] = [];
  for (const s0 of [1, -1]) for (const s1 of [1, -1]) for (const s2 of [1, -1]) {
    const s: V3 = [s0, s1, s2], order = s0 * s1 * s2 > 0 ? [1, 0, 2] : [1, 2, 0];
    for (let r = 0; r < 3; r++) out.push([s, [0, 1, 2].map((k) => order[(k + r) % 3])]);
  }
  return out;
}

function cellFacelets(s: V3, axes: number[]): number[] {
  const idx: number[] = [];
  for (let k = 0; k < 3; k++) {
    const a = axes[k], au = axes[U_FLIP[k]], av = axes[V_FLIP[k]];
    const normal = [0, 0, 0];
    normal[a] = s[a];
    for (let j = 0; j < 3; j++) {
      for (let i = 0; i < 3; i++) {
        const p = [0, 0, 0];
        p[a] = s[a];
        p[au] = s[au] * (1 - i);
        p[av] = s[av] * (1 - j);
        idx.push(INDEX.get(key(p, normal))!);
      }
    }
  }
  return idx;
}

/** For each of the 24 orientations: the facelet (0..53) under each of the 27 cells (face k, row j, column i). */
export const CELLS = hypotheses().map(([s, axes]) => cellFacelets(s, axes));

// --- cube rules: every corner and edge is a real piece, each used once ---

type Options = number[][][]; // (piece, orientation, sticker) -> colour

function pieceOptions(): Map<string, { slot: number[]; options: Options }> {
  const byPos = new Map<string, number[]>();
  FACELETS.forEach(([pos], i) => {
    if (Math.abs(pos[0]) + Math.abs(pos[1]) + Math.abs(pos[2]) >= 2) {
      const k = pos.join();
      byPos.set(k, [...(byPos.get(k) ?? []), i]);
    }
  });
  const det = (m: number[][]) => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  // A corner's 3 facelets ordered so their normals form a right-handed set: one fixed turning direction.
  const clockwise = ([a, b, c]: number[]) => (det([a, b, c].map((i) => FACELETS[i][1])) > 0 ? [a, b, c] : [a, c, b]);
  const groups = [...byPos.values()];
  const corners = groups.filter((g) => g.length === 3).map(clockwise), edges = groups.filter((g) => g.length === 2);
  const cornerColours = corners.map((c) => c.map((i) => SOLVED[i])), edgeColours = edges.map((e) => e.map((i) => SOLVED[i]));
  const out = new Map<string, { slot: number[]; options: Options }>();
  for (const slot of corners) {
    const order = [...slot].sort((x, y) => x - y);
    const options = cornerColours.map((cols) => [0, 1, 2].map((t) => order.map((f) => cols[(slot.indexOf(f) + t) % 3])));
    out.set(order.join(), { slot: order, options });
  }
  for (const slot of edges) {
    const order = [...slot].sort((x, y) => x - y), first = slot.indexOf(order[0]);
    const options = edgeColours.map((cols) => {
      const f = [cols[first], cols[1 - first]];
      return [f, [f[1], f[0]]];
    });
    out.set(order.join(), { slot: order, options });
  }
  return out;
}

const OPTIONS = pieceOptions();
const SLOT_OF = new Map<number, string>();
for (const [k, { slot }] of OPTIONS) for (const f of slot) SLOT_OF.set(f, k);

const argmin = (row: ArrayLike<number>) => {
  let best = 0;
  for (let i = 1; i < row.length; i++) if (row[i] < row[best]) best = i;
  return best;
};

/** Give each slot (row) a different piece (column) with the lowest total cost, by dynamic programming over used sets. */
function distinct(cost: number[][]): number[] {
  const pieces = cost[0].length, size = 1 << pieces;
  let best = new Float64Array(size).fill(Infinity);
  best[0] = 0;
  const choices: Int16Array[] = [];
  for (const row of cost) {
    const next = new Float64Array(size).fill(Infinity), choice = new Int16Array(size).fill(-1);
    for (let p = 0; p < pieces; p++) {
      const bit = 1 << p;
      for (let mask = 0; mask < size; mask++) {
        if (!(mask & bit)) continue;
        const cand = best[mask ^ bit] + row[p];
        if (cand < next[mask]) {
          next[mask] = cand;
          choice[mask] = p;
        }
      }
    }
    best = next;
    choices.push(choice);
  }
  let mask = argmin(best);
  const out: number[] = [];
  for (let s = choices.length - 1; s >= 0; s--) {
    const p = choices[s][mask];
    out.push(p);
    mask ^= 1 << p;
  }
  return out.reverse();
}

/** Colour of each cell, reading every corner and edge as a whole, real piece (verify.decode_pieces). */
function decodePieces(cells: number[], dist6: number[][], known: boolean[], unique: boolean): number[] {
  const out = dist6.map(argmin);
  const groups = new Map<string, [number, number][]>();
  cells.forEach((f, n) => {
    const k = SLOT_OF.get(f);
    if (known[n] && k !== undefined) groups.set(k, [...(groups.get(k) ?? []), [n, OPTIONS.get(k)!.slot.indexOf(f)]]);
  });
  for (const size of [3, 2]) {
    const slots = [...groups.keys()].filter((k) => OPTIONS.get(k)!.slot.length === size);
    if (!slots.length) continue;
    const costs: number[][] = [], orients: number[][] = [];
    for (const k of slots) {
      const opts = OPTIONS.get(k)!.options;
      const c = opts.map((piece) => piece.map((orient) => groups.get(k)!.reduce((sum, [n, st]) => sum + dist6[n][orient[st]], 0)));
      costs.push(c.map((row) => Math.min(...row)));
      orients.push(c.map(argmin));
    }
    const chosen = unique ? distinct(costs) : costs.map(argmin);
    slots.forEach((k, i) => {
      const piece = chosen[i], orient = orients[i][piece];
      for (const [n, st] of groups.get(k)!) out[n] = OPTIONS.get(k)!.options[piece][orient][st];
    });
  }
  return out;
}

// --- the verdict ---

export interface Verdict {
  status: "correct" | "mis-scramble" | "re-check";
  reason: string;
  mismatches: number[]; // cells 0..26
  unknown: number[];
  facelets: number[]; // facelet index under each cell
  expected: string; // colour letter per cell
  observed: string; // colour letter per cell, "?" = unknown
  hypothesis: number;
  fit: number; // how well the scramble as written fits the photo (lower is better)
  slip: string; // the most likely slip, when one explains the photo
  evidence: number;
}

const blank = (status: Verdict["status"], reason: string, unknown: number[]): Verdict => ({
  status, reason, unknown, mismatches: [], facelets: [], expected: "", observed: "", hypothesis: -1, fit: Infinity, slip: "", evidence: 0,
});

/** Each cell's colour probabilities with "covered" left out, so the six colours add up to 1. */
export const colourShares = (probs: ArrayLike<number>) =>
  Array.from({ length: 27 }, (_, n) => {
    let sum = 0;
    for (let c = 0; c < 6; c++) sum += probs[7 * n + c];
    return Array.from({ length: 6 }, (_, c) => probs[7 * n + c] / Math.max(sum, 1e-9));
  });

/** How surprising each colour would be in each cell: -log p, capped so one bad reading can't decide alone. */
export const colourCosts = (colour: number[][], cap = 5) => colour.map((row) => row.map((p) => Math.min(-Math.log(Math.max(p, 1e-9)), cap)));

/** verify.verify_probs: the reader's (27 x 7) probabilities against the state a scramble should give. */
export function verifyProbs(
  probs: ArrayLike<number>, // 27 x 7, row-major: 6 colours (URFDLB), then covered
  inside: boolean[],
  state: State,
  alternatives?: Alternatives,
  { covered = 0.5, cap = 5, minEvidence = 8, sure = 0.05, minMismatches = 6, maxUnknown = 13, unsure = 0.6, maxUnsure = 9 } = {},
): Verdict {
  const P = (n: number, c: number) => probs[7 * n + c];
  const unknown = Array.from({ length: 27 }, (_, n) => !inside[n] || P(n, 6) > covered);
  const unknownList = unknown.flatMap((u, n) => (u ? [n] : []));
  if (unknownList.length > maxUnknown) return blank("re-check", "too much of the cube is hidden or out of view", unknownList);
  const colour = colourShares(probs), cost = colourCosts(colour, cap);
  if (colour.filter((row, n) => !unknown[n] && Math.max(...row) < unsure).length > maxUnsure) {
    return blank("re-check", "colours aren't clear enough (glare, too dark or washed out)", unknownList);
  }
  const rows = unknown.flatMap((u, n) => (u ? [] : [n]));
  const fits = CELLS.map((cells) => rows.reduce((sum, n) => sum + cost[n][state[cells[n]]], 0));
  // Face turns never move the centres, so they say which way round the cube is: of the orientations whose
  // visible centres fit, the one the whole photo fits best.
  const centres = [4, 13, 22].filter((n) => !unknown[n]);
  const centre = CELLS.map((cells) => centres.reduce((sum, n) => sum + cost[n][state[cells[n]]], 0));
  const best = argmin(fits.map((fit, h) => (centre[h] <= Math.min(...centre) + 1 ? fit : Infinity))), cells = CELLS[best];
  const expected = cells.map((f) => state[f]);
  const nearest = decodePieces(cells, cost, unknown.map((u) => !u), true);
  const mismatch = expected.map((e, n) => !unknown[n] && nearest[n] !== e && colour[n][e] < sure);
  const base: Verdict = {
    ...blank("correct", "matches the scramble", unknownList),
    mismatches: mismatch.flatMap((m, n) => (m ? [n] : [])),
    facelets: cells,
    expected: expected.map((e) => FACES[e]).join(""),
    observed: nearest.map((c, n) => (unknown[n] ? "?" : FACES[c])).join(""),
    hypothesis: best,
    fit: fits[best],
  };
  if (alternatives?.states.length) {
    const slips = alternatives.states.map((st) => rows.reduce((sum, n) => sum + cost[n][st[cells[n]]], 0));
    const k = argmin(slips), evidence = fits[best] - slips[k];
    base.evidence = evidence;
    if (evidence > minEvidence) {
      const changed = cells.flatMap((f, n) => (alternatives.states[k][f] !== expected[n] && !unknown[n] ? [n] : []));
      return { ...base, status: "mis-scramble", reason: "doesn't match the scramble", mismatches: changed, slip: alternatives.slips[k] };
    }
  }
  if (base.mismatches.length >= minMismatches) return { ...base, status: "mis-scramble", reason: "doesn't match the scramble" };
  return base;
}

// --- geometry: where the 27 stickers are, from the 7 corners ---

export const FACE_QUADS = [[6, 1, 0, 5], [6, 3, 2, 1], [6, 5, 4, 3]]; // C first, then the face's corners anticlockwise
export const SKELETON = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [6, 1], [6, 3], [6, 5]];
export type Point = [number, number];

/** Unit square -> quad (Heckbert): (0,0)->p0, (1,0)->p1, (1,1)->p2, (0,1)->p3. */
export function squareToQuad([[x0, y0], [x1, y1], [x2, y2], [x3, y3]]: Point[]) {
  const sx = x0 - x1 + x2 - x3, sy = y0 - y1 + y2 - y3;
  let a: number, b: number, c: number, d: number, e: number, f: number, g = 0, h = 0;
  if (Math.abs(sx) < 1e-9 && Math.abs(sy) < 1e-9) {
    [a, b, c, d, e, f] = [x1 - x0, x2 - x1, x0, y1 - y0, y2 - y1, y0];
  } else {
    const dx1 = x1 - x2, dx2 = x3 - x2, dy1 = y1 - y2, dy2 = y3 - y2, den = dx1 * dy2 - dx2 * dy1;
    g = (sx * dy2 - dx2 * sy) / den;
    h = (dx1 * sy - sx * dy1) / den;
    [a, b, c] = [x1 - x0 + g * x1, x3 - x0 + h * x3, x0];
    [d, e, f] = [y1 - y0 + g * y1, y3 - y0 + h * y3, y0];
  }
  return (u: number, v: number): Point => {
    const w = g * u + h * v + 1;
    return [(a * u + b * v + c) / w, (d * u + e * v + f) / w];
  };
}

/** Image positions of the 27 sticker centres: face k, row j, column i. */
export function cellCentres(corners: Point[]): Point[] {
  return FACE_QUADS.flatMap((quad) => {
    const map = squareToQuad(quad.map((j) => corners[j]));
    return Array.from({ length: 9 }, (_, n) => map(((n % 3) + 0.5) / 3, (Math.floor(n / 3) + 0.5) / 3));
  });
}

export const insideImage = (width: number, height: number, corners: Point[]) =>
  cellCentres(corners).map(([x, y]) => x >= 0 && x < width && y >= 0 && y < height);

/** Mean length of the 9 visible cube edges, in pixels. */
export const cubeSize = (corners: Point[]) =>
  SKELETON.reduce((sum, [a, b]) => sum + Math.hypot(corners[a][0] - corners[b][0], corners[a][1] - corners[b][1]), 0) / SKELETON.length;
