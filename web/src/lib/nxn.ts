// Any N×N×N cube (2 to 7): moves in WCA notation and the sticker colours a scramble gives.
// A port of python/cubecheck/nxn.py. Facelets are in Kociemba face order U R F D L B, N×N per face
// in reading order; for N = 3 it gives exactly check.ts's numbers (tests/nxn.test.ts checks both).
//
// Positions use doubled coordinates so even cubes stay whole numbers: a 4×4's layers sit at
// -3, -1, 1, 3. A move turns every facelet whose position along the face normal is within its layers.
// Notation: R, R', R2 turn one layer; Rw turns two; 3Rw turns three (any depth below N); x, y, z the whole cube.

import { FACES } from "./check.ts";

type V3 = [number, number, number];
export type NState = Uint8Array; // 6N² colours, each the index (URFDLB) of the face whose colour it is

const NORMALS: Record<string, V3> = { U: [0, 1, 0], R: [1, 0, 0], F: [0, 0, 1], D: [0, -1, 0], L: [-1, 0, 0], B: [0, 0, -1] };
const ROTATIONS: Record<string, string> = { x: "R", y: "U", z: "F" }; // whole-cube turns go the way of these faces
const MOVE = /^(?:(\d+)?([URFDLB])(w)?|([xyz]))(2'?|')?$/;

export const SIZES = [2, 3, 4, 5, 6, 7] as const;

// The moves WCA scrambles (TNoodle) use for each size, and their usual length. Wide turns only come
// from the U, F and R sides on 4×4 and 6×6.
const WIDE = [...FACES].map((f) => `${f}w`);
export const EVENT_FAMILIES: Record<number, string[]> = {
  2: ["R", "U", "F"],
  3: [...FACES],
  4: [...FACES, "Rw", "Uw", "Fw"],
  5: [...FACES, ...WIDE],
  6: [...FACES, ...WIDE, "3Rw", "3Uw", "3Fw"],
  7: [...FACES, ...WIDE, ...[...FACES].map((f) => `3${f}w`)],
};
export const EVENT_LENGTH: Record<number, number> = { 2: 11, 3: 25, 4: 45, 5: 60, 6: 80, 7: 100 };
export const EVENT_NAME: Record<number, string> = { 2: "2×2", 3: "3×3", 4: "4×4", 5: "5×5", 6: "6×6", 7: "7×7" };

function rotation(axis: V3, quarterTurns: number): number[][] {
  const th = (-Math.PI / 2) * quarterTurns, s = Math.sin(th), c = 1 - Math.cos(th);
  const [x, y, z] = axis;
  const k = [[0, -z, y], [z, 0, -x], [-y, x, 0]];
  const kk = k.map((row) => [0, 1, 2].map((j) => row[0] * k[0][j] + row[1] * k[1][j] + row[2] * k[2][j]));
  return [0, 1, 2].map((i) => [0, 1, 2].map((j) => Math.round((i === j ? 1 : 0) + s * k[i][j] + c * kk[i][j])));
}

const times = (m: number[][], v: number[]): V3 => m.map((row) => row[0] * v[0] + row[1] * v[1] + row[2] * v[2]) as V3;
const key = (p: number[], n: number[]) => `${p.map((v) => v + 0).join(",")}|${n.join(",")}`; // + 0 turns -0 into 0

export class Cube {
  readonly size: number; // facelets, 6N²
  readonly facelets: [V3, V3][] = []; // (position, outward normal), doubled coordinates
  readonly index = new Map<string, number>();
  readonly solved: NState;
  private moves = new Map<string, Uint16Array>();

  readonly n: number;

  constructor(n: number) {
    this.n = n;
    if (!(n >= 2 && n <= 7)) throw new Error(`cube size ${n}: 2 to 7 supported`);
    this.size = 6 * n * n;
    const h = n - 1; // a face sits at ±h
    for (const f of FACES) {
      for (let r = 0; r < n; r++) {
        for (let c = 0; c < n; c++) {
          const a = 2 * r - h, b = 2 * c - h;
          const pos = ({ U: [b, h, a], R: [h, -a, -b], F: [b, -a, h], D: [b, -h, -a], L: [-h, -a, b], B: [-b, -a, -h] } as Record<string, V3>)[f];
          this.index.set(key(pos, NORMALS[f]), this.facelets.length);
          this.facelets.push([pos, NORMALS[f]]);
        }
      }
    }
    this.solved = Uint8Array.from({ length: this.size }, (_, i) => Math.floor(i / (n * n)));
  }

  /** new[dst] = old[src] for turning the outer `depth` layers of `face` (depth n = the whole cube). */
  private permutation(face: string, depth: number, quarterTurns: number): Uint16Array {
    const normal = NORMALS[face], rot = rotation(normal, quarterTurns), perm = Uint16Array.from({ length: this.size }, (_, i) => i);
    this.facelets.forEach(([p, nv], src) => {
      if (p[0] * normal[0] + p[1] * normal[1] + p[2] * normal[2] >= this.n - 1 - 2 * (depth - 1)) {
        perm[this.index.get(key(times(rot, p), times(rot, nv)))!] = src;
      }
    });
    return perm;
  }

  /** The permutation for one move in WCA notation; throws if it isn't one for this size. */
  move(name: string): Uint16Array {
    const known = this.moves.get(name);
    if (known) return known;
    const m = MOVE.exec(name);
    if (!m) throw new Error(`not a move: ${name}`);
    const [, digits, f, wide, rot, amount] = m;
    let face = f, depth: number;
    if (rot) {
      face = ROTATIONS[rot];
      depth = this.n;
    } else {
      if (digits && !wide) throw new Error(`not a WCA move: ${name}`);
      depth = digits ? Number(digits) : wide ? 2 : 1;
      if (depth < 1 || depth > Math.max(1, this.n - 1) || (wide && this.n < 3)) throw new Error(`not a ${this.n}x${this.n} move: ${name}`);
    }
    const perm = this.permutation(face, depth, amount === "'" ? 3 : amount ? 2 : 1);
    this.moves.set(name, perm);
    return perm;
  }

  isMove(name: string): boolean {
    try {
      this.move(name);
      return true;
    } catch {
      return false;
    }
  }

  /** Split a scramble into moves, checking each one; the error names the bad ones. */
  parse(text: string): string[] {
    const moves = splitMoves(text), bad = moves.filter((m) => !this.isMove(m));
    if (bad.length) throw new Error(`not ${this.n}×${this.n} moves: ${bad.join(" ")}`);
    return moves;
  }

  turn(state: NState, move: string): NState {
    const perm = this.move(move), next = new Uint8Array(this.size);
    for (let i = 0; i < this.size; i++) next[i] = state[perm[i]];
    return next;
  }

  apply(state: NState, moves: string[]): NState {
    return moves.reduce((s, m) => this.turn(s, m), state);
  }

  after(scramble: string | string[]): NState {
    return this.apply(this.solved, typeof scramble === "string" ? this.parse(scramble) : scramble);
  }

  /** Every move WCA scrambles of this size can contain (R, R', R2, Rw, ...). */
  get eventTurns(): string[] {
    return EVENT_FAMILIES[this.n].flatMap((f) => [f, `${f}'`, `${f}2`]);
  }
}

const cubes = new Map<number, Cube>();
/** One shared Cube per size (its moves are worked out once). */
export function cube(n: number): Cube {
  if (!cubes.has(n)) cubes.set(n, new Cube(n));
  return cubes.get(n)!;
}

export const splitMoves = (text: string) => text.replace(/[’‘]/g, "'").split(/\s+/).filter(Boolean);

/**
 * The cube size a scramble is for: the sizes whose moves it uses, then the one whose WCA scramble
 * length is nearest (a 3×3 and a 2×2 scramble use the same outer turns, but 25 moves against 11).
 * null if no size takes every move.
 */
export function guessSize(text: string): number | null {
  const moves = splitMoves(text);
  if (!moves.length) return null;
  const fits = SIZES.filter((n) => moves.every((m) => cube(n).isMove(m) && !/^[xyz]/.test(m)));
  if (!fits.length) return null;
  // Wide moves of a size's own family say more than length: Rw alone is 4×4 or bigger.
  const scored = fits.map((n) => {
    const fam = new Set(EVENT_FAMILIES[n]);
    const own = moves.every((m) => fam.has(m.replace(/2?'?$/, "")));
    return { n, own, off: Math.abs(moves.length - EVENT_LENGTH[n]) };
  });
  const pool = scored.some((s) => s.own) ? scored.filter((s) => s.own) : scored;
  return pool.reduce((a, b) => (b.off < a.off ? b : a)).n;
}

/**
 * A scramble shaped like TNoodle's (nxn.random_scramble): the size's WCA moves, no face turned twice in a
 * row, no three turns on one axis in a row. Random moves, not a random state: good for trying the page.
 */
export function randomScramble(n: number, length = EVENT_LENGTH[n], random = Math.random): string[] {
  const axis: Record<string, number> = { U: 0, D: 0, R: 1, L: 1, F: 2, B: 2 };
  const families = EVENT_FAMILIES[n], out: string[] = [];
  const faceOf = (fam: string) => fam.replace(/w$/, "").replace(/^\d+/, "");
  while (out.length < length) {
    const fam = families[Math.floor(random() * families.length)];
    if (out.length) {
      if (out[out.length - 1].replace(/2?'?$/, "") === fam) continue;
      const prev = out.slice(-2).map((m) => faceOf(m.replace(/2?'?$/, "")));
      if (prev.length === 2 && axis[prev[0]] === axis[prev[1]] && axis[prev[1]] === axis[faceOf(fam)]) continue;
    }
    out.push(fam + ["", "'", "2"][Math.floor(random() * 3)]);
  }
  return out;
}
