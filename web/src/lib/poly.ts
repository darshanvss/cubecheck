// Skewb, pyraminx and FTO (experimental): the puzzles, where their stickers are in a picture, and the check.
// A port of python/cubecheck/{puzzles,shapes,poly_check}.py; tests/poly.test.ts checks it gives Python's
// answers. The geometry Python works out once (each turn's reshuffle, the corners and every way to label
// them, the sticker places behind each face of a picture, sticker outlines) comes as numbers in
// poly-data.json (python/scripts/poly_web_data.py).
//
// A skewb is a cube: its 7 corners give 3 quads, each mapped from the unit square as for any cube. A
// pyraminx or FTO face is a triangle, and 3 corners don't say how it's foreshortened, so the whole shape is
// fitted to its corners as a camera would see it (`fit`: a pinhole camera through the middle of the
// picture, its zoom unknown), and each face is placed with that.
import type { Alternatives, Point, SlipWhere, Verdict } from "./check.ts";
import DATA from "./poly-data.json" with { type: "json" };

export const POLY_EVENTS = ["skewb", "pyram", "fto"] as const;
export type PolyEvent = (typeof POLY_EVENTS)[number];
export const isPoly = (p: unknown): p is PolyEvent => POLY_EVENTS.includes(p as PolyEvent);
export type PState = Uint8Array; // the colour (face number) on every sticker place

interface Thresholds {
  min_evidence: number;
  min_slip_seen: number;
  min_mismatches: number;
  min_known: number;
}

export interface PolyData {
  event: PolyEvent;
  name: string;
  faceKeys: string[];
  css: string[];
  places: number;
  solved: number[];
  turns: string[];
  perms: Record<string, number[]>;
  vertices: number[][];
  views: number[][];
  viewFaces: number[][];
  polys: number[][];
  quads: boolean;
  nFaces: number;
  perFace: number;
  keypoints: number;
  colours: number;
  spokes: number[];
  relabellings: number[][];
  frames: number[][];
  slots: number[][][];
  slotCentres: number[][];
  cells: number[][];
  facePx: number;
  mapMatrix: number[][];
  net: { polys: number[][][]; faces: number[][][]; labels: [string, number, number][]; width: number; height: number };
  minFaces: number;
  foundFaces: number;
  holdHint: string;
  thresholds: Thresholds;
  minSize: number;
  maxFit: number;
  scramble: { faces: string[]; length: number; tips: boolean };
  follow: { max_cost: number; min_evidence: number; min_faces: number };
}

type M3 = number[][]; // 3 x 3
type M34 = number[][]; // 3 x 4

// --- small linear algebra ---

const mul = (a: number[][], b: number[][]) => a.map((row) => b[0].map((_, j) => row.reduce((s, v, k) => s + v * b[k][j], 0)));
const cross = (a: number[], b: number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

/** Solve A x = b (n x n) by Gaussian elimination with pivoting; null when singular. */
function solve(A: number[][], b: number[]): number[] | null {
  const n = b.length, M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (Math.abs(M[p][c]) < 1e-14) return null;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((row, i) => row[n] / row[i]);
}

export function inv3(m: M3): M3 {
  const [[a, b, c], [d, e, f], [g, h, i]] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g, det = a * A + b * B + c * C;
  return [
    [A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
    [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
    [C / det, -(a * h - b * g) / det, (a * e - b * d) / det],
  ];
}

/** The 3 x 3 perspective map taking 4 points onto 4 others (OpenCV's getPerspectiveTransform). */
export function homography(src: number[][], dst: number[][]): M3 {
  const A: number[][] = [], b: number[] = [];
  src.forEach(([x, y], i) => {
    const [u, v] = dst[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    b.push(u);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    b.push(v);
  });
  const h = solve(A, b)!;
  return [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1]];
}

export const project = (m: M3, [x, y]: number[]): Point => {
  const w = m[2][0] * x + m[2][1] * y + m[2][2];
  return [(m[0][0] * x + m[0][1] * y + m[0][2]) / w, (m[1][0] * x + m[1][1] * y + m[1][2]) / w];
};

/** Rotation matrix for a rotation vector (Rodrigues). */
function rodrigues(w: number[]): M3 {
  const th = Math.hypot(w[0], w[1], w[2]);
  if (th < 1e-12) return [[1, -w[2], w[1]], [w[2], 1, -w[0]], [-w[1], w[0], 1]];
  const [x, y, z] = w.map((v) => v / th), c = Math.cos(th), s = Math.sin(th), C = 1 - c;
  return [
    [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
    [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
    [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
  ];
}

// --- the pose: the shape's corners as a camera sees them ---

interface Pose {
  R: M3;
  t: number[];
}

function reproject(obj: number[][], { R, t }: Pose, f: number, cx: number, cy: number): Point[] {
  return obj.map((X) => {
    const c = R.map((row, i) => row[0] * X[0] + row[1] * X[1] + row[2] * X[2] + t[i]);
    return [f * (c[0] / c[2]) + cx, f * (c[1] / c[2]) + cy];
  });
}

const sqCost = (a: Point[], b: Point[]) => a.reduce((s, p, i) => s + (p[0] - b[i][0]) ** 2 + (p[1] - b[i][1]) ** 2, 0);

/**
 * A first guess from a camera without perspective (scaled orthographic): the best 2 x 3 linear map from
 * the corners (around their middle) to the picture, turned into a rotation and a distance. Two guesses: as
 * found, and turned the other way in depth, which such a camera can't tell apart.
 */
function guesses(obj: number[][], img: Point[], f: number, cx: number, cy: number): Pose[] {
  const n = obj.length, mu = [0, 1, 2].map((k) => obj.reduce((s, X) => s + X[k], 0) / n);
  const xn = img.map(([x, y]) => [(x - cx) / f, (y - cy) / f]), mx = [0, 1].map((k) => xn.reduce((s, p) => s + p[k], 0) / n);
  const Xc = obj.map((X) => X.map((v, k) => v - mu[k])), xc = xn.map((p) => p.map((v, k) => v - mx[k]));
  const XtX = [0, 1, 2].map((i) => [0, 1, 2].map((j) => Xc.reduce((s, X) => s + X[i] * X[j], 0)));
  const XtXi = inv3(XtX);
  const A = [0, 1].map((r) => [0, 1, 2].map((j) => [0, 1, 2].reduce((s, k) => s + Xc.reduce((q, X, i) => q + xc[i][r] * X[k], 0) * XtXi[k][j], 0)));
  // R12 = (A Aᵀ)^(-1/2) A: the nearest two orthonormal rows; the scale is the mean singular value.
  const a = A[0][0] ** 2 + A[0][1] ** 2 + A[0][2] ** 2, b = A[0][0] * A[1][0] + A[0][1] * A[1][1] + A[0][2] * A[1][2];
  const d = A[1][0] ** 2 + A[1][1] ** 2 + A[1][2] ** 2, tr = a + d, disc = Math.sqrt(Math.max(0, ((a - d) / 2) ** 2 + b * b));
  const l1 = tr / 2 + disc, l2 = Math.max(tr / 2 - disc, 1e-18);
  let v1 = Math.abs(b) > 1e-15 ? [b, l1 - a] : a >= d ? [1, 0] : [0, 1];
  const nv = Math.hypot(v1[0], v1[1]);
  v1 = [v1[0] / nv, v1[1] / nv];
  const v2 = [-v1[1], v1[0]];
  const s1 = 1 / Math.sqrt(l1), s2 = 1 / Math.sqrt(l2);
  const Minv = [0, 1].map((i) => [0, 1].map((j) => s1 * v1[i] * v1[j] + s2 * v2[i] * v2[j]));
  const rows = [0, 1].map((i) => [0, 1, 2].map((k) => Minv[i][0] * A[0][k] + Minv[i][1] * A[1][k]));
  const scale = (Math.sqrt(l1) + Math.sqrt(l2)) / 2, tz = 1 / Math.max(scale, 1e-12);
  const unit = (v: number[]) => v.map((x) => x / Math.hypot(v[0], v[1], v[2]));
  return [1, -1].map((sz) => {
    // The second guess flips both rows' depth parts, then is made a rotation again (Gram-Schmidt).
    const r1 = unit([rows[0][0], rows[0][1], sz * rows[0][2]]), q = [rows[1][0], rows[1][1], sz * rows[1][2]];
    const dot = q[0] * r1[0] + q[1] * r1[1] + q[2] * r1[2], r2 = unit(q.map((v, i) => v - dot * r1[i]));
    const R = [r1, r2, cross(r1, r2)], T = [mx[0] * tz, mx[1] * tz, tz];
    return { R, t: T.map((v, i) => v - (R[i][0] * mu[0] + R[i][1] * mu[1] + R[i][2] * mu[2])) };
  });
}

/** Levenberg-Marquardt on the corners' distances in the picture (as OpenCV's solvePnPRefineLM). */
function refine(obj: number[][], img: Point[], pose: Pose, f: number, cx: number, cy: number): Pose {
  let cur = pose, cost = sqCost(reproject(obj, cur, f, cx, cy), img), lambda = 1e-3;
  const step = (p: Pose, d: number[]): Pose => ({ R: mul(p.R, rodrigues(d.slice(0, 3))), t: p.t.map((v, i) => v + d[3 + i]) });
  for (let it = 0; it < 200; it++) {
    const base = reproject(obj, cur, f, cx, cy), r = base.flatMap((p, i) => [p[0] - img[i][0], p[1] - img[i][1]]);
    const J = r.map(() => new Array(6).fill(0));
    for (let k = 0; k < 6; k++) {
      const e = k < 3 ? 1e-6 : 1e-6 * Math.max(1, Math.abs(cur.t[2])), d = new Array(6).fill(0);
      d[k] = e;
      const plus = reproject(obj, step(cur, d), f, cx, cy);
      d[k] = -e;
      const minus = reproject(obj, step(cur, d), f, cx, cy);
      plus.forEach((p, i) => {
        J[2 * i][k] = (p[0] - minus[i][0]) / (2 * e);
        J[2 * i + 1][k] = (p[1] - minus[i][1]) / (2 * e);
      });
    }
    const JtJ = [0, 1, 2, 3, 4, 5].map((i) => [0, 1, 2, 3, 4, 5].map((j) => J.reduce((s, row) => s + row[i] * row[j], 0)));
    const Jtr = [0, 1, 2, 3, 4, 5].map((i) => J.reduce((s, row, q) => s + row[i] * r[q], 0));
    let improved = false;
    for (let tries = 0; tries < 12 && !improved; tries++) {
      const A = JtJ.map((row, i) => row.map((v, j) => (i === j ? v * (1 + lambda) + 1e-12 : v)));
      const d = solve(A, Jtr.map((v) => -v));
      if (!d) break;
      const next = step(cur, d), c = sqCost(reproject(obj, next, f, cx, cy), img);
      if (Number.isFinite(c) && c < cost) {
        const gain = cost - c;
        cur = next;
        cost = c;
        lambda = Math.max(lambda / 10, 1e-12);
        improved = true;
        if (gain < 1e-12 * Math.max(cost, 1e-12) || cost < 1e-18) return cur;
      } else lambda *= 10;
    }
    if (!improved) break;
  }
  return cur;
}

// --- a puzzle ---

const keyOf = (s: PState) => s.join();
const permute = (a: PState, perm: Uint16Array): PState => {
  const out = new Uint8Array(perm.length);
  for (let i = 0; i < perm.length; i++) out[i] = a[perm[i]];
  return out;
};

export class Poly {
  readonly d: PolyData;
  readonly event: PolyEvent;
  readonly solved: PState;
  readonly cells: Uint16Array[];
  private readonly perms = new Map<string, Uint16Array>();

  constructor(d: PolyData) {
    this.d = d;
    this.event = d.event;
    this.solved = Uint8Array.from(d.solved);
    this.cells = d.cells.map((c) => Uint16Array.from(c));
    for (const t of d.turns) this.perms.set(t, Uint16Array.from(d.perms[t]));
  }

  get turns() {
    return this.d.turns;
  }

  move(m: string): Uint16Array {
    const p = this.perms.get(m);
    if (!p) throw new Error(`${m} isn't a ${this.d.name} move`);
    return p;
  }
  turn = (state: PState, m: string): PState => permute(state, this.move(m));
  apply = (state: PState, moves: string[]): PState => moves.reduce((s, m) => this.turn(s, m), state);

  /** The scramble's moves (shapes.parse_scramble); an Error naming the first that isn't a turn of this puzzle. */
  parse(text: string): string[] {
    const moves = text.replace(/[’′]/g, "'").split(/\s+/).filter(Boolean);
    for (const m of moves) {
      if (!this.perms.has(m)) {
        const faces = [...new Set(this.d.turns.map((t) => t.replace(/'$/, "")))].sort();
        throw new Error(`${m} isn't a ${this.d.name} move (${faces.join(" ")}, each also with ')`);
      }
    }
    return moves;
  }
  after = (text: string): PState => this.apply(this.solved, this.parse(text));

  // --- in the picture ---

  /** The picture's polygon for face k of the corners (C first, then anticlockwise). */
  facePoints = (kp: Point[], k: number): Point[] => this.d.polys[k].map((i) => kp[i]);

  /** The puzzle's size in the picture, in pixels: the mean length of the edges out of C. */
  size(kp: Point[]): number {
    const m = this.d.keypoints - 1, spokes = this.d.polys.map((p) => p[1]);
    return spokes.reduce((s, i) => s + Math.hypot(kp[i][0] - kp[m][0], kp[i][1] - kp[m][1]), 0) / spokes.length;
  }

  /** Signed area of each face in the picture: negative for a face seen from the front. */
  windings(kp: Point[]): number[] {
    return this.d.polys.map((poly) => {
      const p = poly.map((i) => kp[i]);
      let a = 0;
      for (let i = 0; i < p.length; i++) a += p[i][0] * p[(i + 1) % p.length][1] - p[(i + 1) % p.length][0] * p[i][1];
      return a / 2;
    });
  }

  /** At least `minFaces` faces seen from the front, each a proper convex shape (shapes.Shape.well_formed). */
  wellFormed(kp: Point[], minFaces: number): boolean {
    const front = this.windings(kp).map((w) => w < 0);
    let ok = 0;
    this.d.polys.forEach((poly, k) => {
      const p = poly.map((i) => kp[i]), n = p.length;
      const e = p.map((q, i) => [p[(i + 1) % n][0] - q[0], p[(i + 1) % n][1] - q[1]]);
      const turn = e.map((v, i) => v[0] * e[(i + 1) % n][1] - v[1] * e[(i + 1) % n][0]);
      if (front[k] && (turn.every((t) => t > 0) || turn.every((t) => t < 0))) ok++;
    });
    return ok >= minFaces;
  }

  /**
   * The 3 x 4 camera matrix that puts the shape's own corners on these, and how badly they fit: the mean
   * distance left, as a share of the puzzle's size (shapes.Shape.fit). Four zooms are tried, the best kept.
   */
  fit(kp: Point[], width: number, height: number): { pose: M34; err: number } {
    const obj = this.d.views[0].map((i) => this.d.vertices[i]);
    let best: M34 | null = null, bestErr = Infinity;
    for (const zoom of [0.8, 1.0, 1.3, 1.7]) {
      const f = zoom * height, cx = width / 2, cy = height / 2;
      for (const g of guesses(obj, kp, f, cx, cy)) {
        const p = refine(obj, kp, g, f, cx, cy);
        const pts = reproject(obj, p, f, cx, cy), err = pts.reduce((s, q, i) => s + Math.hypot(q[0] - kp[i][0], q[1] - kp[i][1]), 0) / pts.length;
        if (p.t[2] > 0 && err < bestErr) {
          const K = [[f, 0, cx], [0, f, cy], [0, 0, 1]];
          best = mul(K, p.R.map((row, i) => [...row, p.t[i]]));
          bestErr = err;
        }
      }
    }
    if (!best) {
      // No fit (corners in a line, say): no perspective at all, from the corners' x and y.
      const A = obj.map((X) => [X[0], X[1], 1]), AtA = [0, 1, 2].map((i) => [0, 1, 2].map((j) => A.reduce((s, r) => s + r[i] * r[j], 0)));
      const row = (c: 0 | 1) => solve(AtA, [0, 1, 2].map((i) => A.reduce((s, r, q) => s + r[i] * kp[q][c], 0))) ?? [0, 0, 0];
      const [a, b] = [row(0), row(1)];
      best = [[a[0], a[1], 0, a[2]], [b[0], b[1], 0, b[2]], [0, 0, 0, 1]];
    }
    return { pose: best, err: bestErr / Math.max(this.size(kp), 1e-6) };
  }

  /** 3 x 3 map from face k's (u, v) to the picture: its own 4 corners for a quad, the pose for a triangle. */
  faceMatrix(kp: Point[], k: number, pose: M34 | null): M3 {
    if (this.d.quads) return homography([[0, 0], [1, 0], [1, 1], [0, 1]], this.facePoints(kp, k));
    const [c, nxt, prev] = this.d.frames[k].map((i) => this.d.vertices[i]);
    const cols = [[...nxt.map((v, i) => v - c[i]), 0], [...prev.map((v, i) => v - c[i]), 0], [...c, 1]];
    return pose!.map((row) => cols.map((col) => row.reduce((s, v, i) => s + v * col[i], 0)));
  }

  faceMatrices(kp: Point[], width: number, height: number): M3[] {
    const pose = this.d.quads ? null : this.fit(kp, width, height).pose;
    return Array.from({ length: this.d.nFaces }, (_, k) => this.faceMatrix(kp, k, pose));
  }

  /** Where every (face, slot) sticker's middle falls in the picture. */
  cellCentres(kp: Point[], width: number, height: number, matrices = this.faceMatrices(kp, width, height)): Point[] {
    return matrices.flatMap((m) => this.d.slotCentres.map((uv) => project(m, uv)));
  }

  /** Every (face, slot) sticker's outline in the picture. */
  outlines(matrices: M3[]): Point[][] {
    return matrices.flatMap((m) => this.d.slots.map((poly) => poly.map((uv) => project(m, uv))));
  }

  /**
   * Which (face, slot) stickers can be read (poly_check.readable): inside the picture, on a face seen from
   * the front that isn't nearly edge-on. None at all when the corners don't fit the puzzle's shape.
   */
  readable(kp: Point[], width: number, height: number, minFaceShare = 0.1, maxFit = this.d.maxFit): boolean[] {
    const m = this.d.nFaces * this.d.perFace;
    if (maxFit && !this.d.quads && this.fit(kp, width, height).err > maxFit) return new Array(m).fill(false);
    const c = this.cellCentres(kp, width, height), signed = this.windings(kp), areas = signed.map(Math.abs);
    const front = signed.map((s) => s < 0), largest = Math.max(0, ...areas.filter((_, k) => front[k]));
    return c.map(([x, y], i) => {
      const k = Math.floor(i / this.d.perFace);
      return x >= 0 && x < width && y >= 0 && y < height && front[k] && areas[k] >= minFaceShare * largest;
    });
  }

  // --- the check ---

  /** States one human slip away from the scramble (poly_check.single_slips). */
  singleSlips(scramble: string): Alternatives {
    const moves = this.parse(scramble), perms = moves.map((m) => this.move(m)), size = this.d.places;
    const prefix: PState[] = [this.solved];
    for (const p of perms) prefix.push(permute(prefix[prefix.length - 1], p));
    const identity = Uint16Array.from({ length: size }, (_, i) => i);
    const compose = (a: Uint16Array, b: Uint16Array) => Uint16Array.from(b, (v) => a[v]);
    const suffix: Uint16Array[] = Array(moves.length + 1).fill(identity);
    for (let k = moves.length - 1; k >= 0; k--) suffix[k] = compose(perms[k], suffix[k + 1]);
    const seen = new Set([keyOf(prefix[moves.length])]), out: Alternatives = { states: [], slips: [], where: [] };
    const add = (st: PState, text: string, move: number, kind: SlipWhere["kind"], done: string | null = null) => {
      const k = keyOf(st);
      if (seen.has(k)) return;
      seen.add(k);
      out.states.push(st);
      out.slips.push(text);
      out.where.push({ move, kind, done });
    };
    const turns = this.d.turns.map((t) => [t, this.move(t)] as const);
    moves.forEach((m, i) => {
      add(permute(prefix[i], suffix[i + 1]), `move ${i + 1} (${m}) skipped`, i, "skipped");
      for (const [t, pt] of turns) if (t !== m) add(permute(permute(prefix[i], pt), suffix[i + 1]), `move ${i + 1} done as ${t} instead of ${m}`, i, "wrong", t);
      if (i + 1 < moves.length) add(permute(permute(permute(prefix[i], perms[i + 1]), perms[i]), suffix[i + 2]), `moves ${i + 1} and ${i + 2} swapped`, i, "swapped");
    });
    for (let i = 0; i <= moves.length; i++) {
      for (const [t, pt] of turns) add(permute(permute(prefix[i], pt), suffix[i]), i < moves.length ? `extra ${t} before move ${i + 1}` : `extra ${t} at the end`, i, "extra", t);
    }
    return out;
  }

  /**
   * poly_check.check: the reader's probabilities (faces x slots, then the puzzle's colours in its face
   * order and covered) against the state a scramble should give. `observed` and `expected` are the face
   * keys of each cell, joined by commas (an FTO has two-letter faces).
   */
  check(probs: ArrayLike<number>, inside: boolean[], state: PState, alternatives?: Alternatives,
    { covered = 0.5, cap = 5, sure = 0.05, unsure = 0.6, maxUnsure = 0.33, maxSlipLeft = 1 } = {}): Verdict {
    const d = this.d, t = d.thresholds, C = d.colours, m = inside.length, P = (i: number, k: number) => probs[(C + 1) * i + k];
    const facesRead = Array.from({ length: d.nFaces }, (_, k) => inside.slice(k * d.perFace, (k + 1) * d.perFace).some(Boolean)).filter(Boolean).length;
    const poorHold = facesRead < d.minFaces;
    const unknown = Array.from({ length: m }, (_, i) => !inside[i] || P(i, C) > covered);
    const unknownList = unknown.flatMap((u, i) => (u ? [i] : []));
    const blank = (reason: string): Verdict => ({
      status: "re-check", reason, unknown: unknownList, mismatches: [], facelets: [], expected: "", observed: "", hypothesis: -1, fit: Infinity,
      slip: "", slipWhere: null, evidence: 0, slipSeen: 0, slipLeft: 0,
    });
    if (facesRead === 0 || m - unknownList.length < t.min_known) return blank(poorHold ? d.holdHint : "too much of the puzzle is hidden or out of view");
    const colour = Array.from({ length: m }, (_, i) => {
      let sum = 0;
      for (let k = 0; k < C; k++) sum += P(i, k);
      return Array.from({ length: C }, (_, k) => P(i, k) / Math.max(sum, 1e-9));
    });
    const cost = colour.map((row) => row.map((p) => Math.min(-Math.log(Math.max(p, 1e-9)), cap)));
    const known = m - unknownList.length;
    if (colour.filter((row, i) => !unknown[i] && Math.max(...row) < unsure).length > maxUnsure * known) {
      return blank("colours aren't clear enough (glare, too dark or washed out)");
    }
    const rows = unknown.flatMap((u, i) => (u ? [] : [i]));
    const fits = this.cells.map((cl) => rows.reduce((s, i) => s + cost[i][state[cl[i]]], 0));
    let best = 0;
    fits.forEach((f, i) => f < fits[best] && (best = i));
    const cell = this.cells[best], expected = Array.from(cell, (f) => state[f]);
    const nearest = cost.map((row) => row.reduce((b, v, k) => (v < row[b] ? k : b), 0));
    const mismatch = expected.map((e, i) => !unknown[i] && nearest[i] !== e && colour[i][e] < sure);
    const keys = d.faceKeys;
    const base: Verdict = {
      status: "correct", reason: "matches the scramble",
      mismatches: mismatch.flatMap((x, i) => (x ? [i] : [])), unknown: unknownList, facelets: Array.from(cell),
      expected: expected.map((e) => keys[e]).join(","), observed: nearest.map((k, i) => (unknown[i] ? "?" : keys[k])).join(","),
      hypothesis: best, fit: fits[best], slip: "", slipWhere: null, evidence: 0, slipSeen: 0, slipLeft: 0,
    };
    if (alternatives?.states.length) {
      let k = 0, h = best, slipFit = Infinity;
      const alts = alternatives.states;
      for (let o = 0; o < this.cells.length; o++) {
        const at = rows.map((i) => this.cells[o][i]), cr = rows.map((i) => cost[i]);
        for (let j = 0; j < alts.length; j++) {
          const st = alts[j];
          let s = 0;
          for (let r = 0; r < at.length; r++) s += cr[r][st[at[r]]];
          // float32 sums in Python: the same order, so ties go the same way
          if (Math.fround(s) < slipFit) {
            k = j;
            h = o;
            slipFit = Math.fround(s);
          }
        }
      }
      const evidence = fits[best] - slipFit;
      const slipped = Array.from(this.cells[h], (f) => alts[k][f]);
      const changed = slipped.map((s, i) => s !== expected[i] && !unknown[i]);
      const seen = changed.filter((c, i) => c && colour[i][expected[i]] < 0.5).length;
      const left = slipped.filter((s, i) => !unknown[i] && colour[i][s] < sure).length;
      Object.assign(base, { evidence: Math.round(evidence * 100) / 100, slipSeen: seen, slipLeft: left });
      if (evidence > t.min_evidence && seen >= t.min_slip_seen) {
        if (left > maxSlipLeft) return { ...base, status: "mis-scramble", reason: "doesn't match the scramble" };
        return { ...base, status: "mis-scramble", reason: "doesn't match the scramble", mismatches: changed.flatMap((c, i) => (c ? [i] : [])), slip: alternatives.slips[k], slipWhere: alternatives.where[k] };
      }
    }
    if (base.mismatches.length >= t.min_mismatches) return { ...base, status: "mis-scramble", reason: "doesn't match the scramble" };
    if (poorHold) return { ...base, status: "re-check", reason: `fits so far: ${d.holdHint}` };
    return base;
  }

  /** A random-move scramble (shapes.random_scramble): never the same face twice in a row; a pyraminx's tips at the end. */
  randomScramble(random = Math.random, length?: number): string[] {
    const { faces, length: full, tips } = this.d.scramble, out: string[] = [];
    while (out.length < (length ?? full)) {
      const face = faces[Math.floor(random() * faces.length)];
      if (out.length && out[out.length - 1].replace(/'$/, "") === face) continue;
      out.push(face + (random() < 0.5 ? "" : "'"));
    }
    if (tips && !length) {
      for (const tip of "ulrb") {
        const turn = Math.floor(random() * 3);
        if (turn) out.push(tip + (turn === 2 ? "'" : ""));
      }
    }
    return out;
  }
}

const made = new Map<PolyEvent, Poly>();
export function poly(event: PolyEvent): Poly {
  if (!made.has(event)) made.set(event, new Poly((DATA as unknown as Record<PolyEvent, PolyData>)[event]));
  return made.get(event)!;
}
