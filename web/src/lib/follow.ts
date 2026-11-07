// Follow-along: where in the scramble the cube is, look by look, and the first wrong move.
// A port of python/cubecheck/follow.py; tests/follow.test.ts checks it gives Python's answers.
//
// The scrambler turns the cube while the camera watches. Each still look gives the sticker reader's
// probabilities for the visible stickers; this keeps a score for every point of the scramble (0 = solved,
// up to every move done) and updates it with each look, the way a hidden Markov model does:
// 1. Before looking: probably the same point or a few moves on, but a jump anywhere is possible.
// 2. After looking: each point's score is multiplied by how well its stickers match the look, in
//    whichever of the 24 orientations fits it best.
// When no point fits, the likely mistakes near the best one are tried: a move done wrong, an extra turn,
// a skipped move or two moves swapped, followed by up to `after` (6) correct moves. A mistake must fit
// clearly better (by minEvidence, as the check) in two looks in a row before it's shown, with the way
// back. It clears after `confirm` looks back on track, except that a cube along the way back shown is
// taken at once. A half turn (R2) done as two quarter turns is halfway, not a mistake. Looks that show
// too little change nothing.

import { cells, middleCells } from "./nxn-check.ts";
import { cube, type Cube, type NState } from "./nxn.ts";

// How far on the scrambler probably is at the next still look: 0, 1, ... 6 moves.
const STEP = [0.4, 0.22, 0.12, 0.09, 0.07, 0.05, 0.05];

/** The quarter turn a half turn is made of: R2 -> R, Rw2' -> Rw; null for a quarter turn. */
export function quarter(move: string): string | null {
  const m = move.replace(/'$/, "");
  return m.endsWith("2") ? m.slice(0, -1) : null;
}

/** The turn that undoes a move: R -> R', Rw' -> Rw, R2 -> R2. */
export const inverse = (move: string) => (move.endsWith("2") ? move : move.endsWith("'") ? move.slice(0, -1) : `${move}'`);

const AXIS: Record<string, number> = { U: 0, D: 0, y: 0, R: 1, L: 1, x: 1, F: 2, B: 2, z: 2 };
const axisOf = (layers: string) => AXIS[layers.replace(/^\d+/, "")[0]];

/** A move as (layers, quarter turns clockwise): R2 -> ["R", 2], 3Rw' -> ["3Rw", 3]. */
export function turns(move: string): [string, number] {
  const m = move.replace(/'$/, "");
  const q = m.endsWith("2") ? 2 : move.endsWith("'") ? 3 : 1;
  return [m.replace(/2$/, ""), q];
}

/** Add up the turns of one axis that come together: D2 D -> D', D' U D -> U, R R' -> nothing. */
export function shorten(moves: string[]): string[] {
  let out = [...moves];
  for (;;) {
    const next: string[] = [], run = new Map<string, number>();
    let axis = -1;
    const flush = () => {
      for (const [k, q] of run) if (q % 4) next.push(k + ["", "", "2", "'"][q % 4]);
      run.clear();
    };
    for (const m of out) {
      const [k, q] = turns(m), a = axisOf(k);
      if (a !== axis) {
        flush();
        axis = a;
      }
      run.set(k, (run.get(k) ?? 0) + q);
    }
    flush();
    if (next.join(" ") === out.join(" ")) return out; // again: a run that cancels brings two runs of one axis together
    out = next;
  }
}

export const KINDS = ["skipped", "swapped", "wrong", "extra"] as const; // the order a tie between stories goes
export type Kind = (typeof KINDS)[number];

export interface Fix {
  undo: string[]; // newest first
  then: string[];
  resume: number; // carry on from this move (0-based)
  merged: boolean; // the undoing and the right move added up into one (D2 done for D: D')
}

export class Mistake {
  kind: Kind;
  move: number; // 0-based index of the scramble move where it went wrong
  done: string | null; // what was done instead (wrong), or the extra turn
  after: number; // correct moves done since
  state: NState;

  constructor(kind: Kind, move: number, done: string | null, after: number, state: NState) {
    this.kind = kind;
    this.move = move;
    this.done = done;
    this.after = after;
    this.state = state;
  }

  /** The way back: undo these (newest first), then do these, then carry on from `resume`. */
  fix(moves: string[]): Fix {
    const k = this.move, inv = (ms: string[]) => [...ms].reverse().map(inverse);
    let undo: string[], then: string[] = [], resume = k;
    if (this.kind === "wrong") {
      undo = [...inv(moves.slice(k + 1, k + 1 + this.after)), inverse(this.done!)];
      then = [moves[k]];
      resume = k + 1;
    } else if (this.kind === "swapped") {
      undo = inv([moves[k + 1], moves[k], ...moves.slice(k + 2, k + 2 + this.after)]);
    } else if (this.kind === "extra") {
      undo = [...inv(moves.slice(k, k + this.after)), inverse(this.done!)];
    } else {
      undo = inv(moves.slice(k + 1, k + 1 + this.after));
    }
    const way = shorten([...undo, ...then]), short = shorten(undo);
    if ([...short, ...then].join(" ") === way.join(" ")) return { undo: short, then, resume, merged: false };
    return { undo: way, then: [], resume, merged: true };
  }

  /** Smaller is the likelier story for the same cube (`at`: where it was last seen on track). */
  rank(moves: string[], at: number): number[] {
    const f = this.fix(moves);
    return [Number(this.move < at), f.undo.length + f.then.length, KINDS.indexOf(this.kind), -this.move, this.after];
  }
}

const less = (a: number[], b: number[]) => {
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] < b[i];
  return false;
};
const argmin = (a: ArrayLike<number>) => {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i] < a[best]) best = i;
  return best;
};
const keyOf = (s: NState) => s.join();

export interface Report {
  position: number; // moves done, as far as it can tell
  total: number;
  sure: number;
  clear: boolean; // this look was clear enough to judge
  half: boolean; // halfway through the next move's half turn
  status: "following" | "done" | "unclear" | "mistake";
  mistake?: { kind: Kind; move: number; done: string | null; after: number };
  undo?: string[];
  then?: string[];
  resume?: number;
  merged?: boolean;
}

export interface FollowOptions {
  after?: number;
  before?: number;
  jump?: number;
  temper?: number;
  minSeen?: number;
  maxCost?: number;
  minEvidence?: number;
  confirm?: number;
}

export class Follower {
  readonly n: number;
  readonly c: Cube;
  readonly moves: string[];
  readonly points: NState[]; // the cube after 0..L moves
  readonly halves = new Map<number, [NState, NState]>(); // halfway through move i's half turn, either way round
  readonly after: number;
  readonly before: number;
  private jump: number;
  private temper: number;
  private minSeen: number;
  private maxCost: number;
  private minEvidence: number;
  private confirm: number;
  belief: Float64Array;
  private cells: Uint16Array[];
  private middles: number[];
  private onScramble: Set<string>;
  private streak = 0;
  mistake: Mistake | null = null;
  private looks = 0;
  private back = 0;
  private half = false;
  private path = new Set<string>(); // the cubes along the way back shown (see show)

  constructor(scramble: string, n: number, { after = 6, before = 2, jump = 0.02, temper = 3, minSeen = 0.3, maxCost = 0.6, minEvidence = 8, confirm = 2 }: FollowOptions = {}) {
    this.n = n;
    this.c = cube(n);
    this.moves = this.c.parse(scramble);
    this.after = after;
    this.before = before;
    this.jump = jump;
    this.temper = temper;
    this.minSeen = minSeen;
    this.maxCost = maxCost;
    this.minEvidence = minEvidence;
    this.confirm = confirm;
    this.points = [this.c.solved];
    for (const m of this.moves) this.points.push(this.c.turn(this.points[this.points.length - 1], m));
    this.moves.forEach((m, i) => {
      const q = quarter(m);
      if (q) this.halves.set(i, [this.c.turn(this.points[i], q), this.c.turn(this.points[i], `${q}'`)]);
    });
    const L = this.moves.length;
    this.belief = new Float64Array(L + 1).fill(0.1 / (L + 1));
    this.belief[0] += 0.9; // a scramble starts from a solved cube
    this.cells = cells(n);
    this.middles = middleCells(n);
    this.onScramble = new Set([...this.points, ...[...this.halves.values()].flat()].map(keyOf));
  }

  // --- the two HMM steps ---

  /** Spread the scores the way scrambling goes: a few moves on, sometimes a jump anywhere. */
  predict(): Float64Array {
    const L1 = this.belief.length, out = new Float64Array(L1);
    STEP.forEach((w, d) => {
      for (let i = d; i < L1; i++) out[i] += w * this.belief[i - d];
      if (d) for (let i = Math.max(0, L1 - d); i < L1; i++) out[L1 - 1] += w * this.belief[i]; // can't go past the end
    });
    const sum = out.reduce((a, b) => a + b, 0);
    return out.map((v) => ((1 - this.jump) * v) / sum + this.jump / L1);
  }

  /** Total cost of the seen stickers for each state, in its best orientation. */
  fits(probs: ArrayLike<number>, unknown: boolean[], states: NState[], cap = 5): Float64Array {
    const m = unknown.length, rows: number[] = [];
    const full = Array.from({ length: m }, (_, i) => {
      let sum = 0;
      for (let k = 0; k < 6; k++) sum += probs[7 * i + k];
      return Array.from({ length: 6 }, (_, k) => Math.min(-Math.log(Math.max(probs[7 * i + k] / Math.max(sum, 1e-9), 1e-9)), cap));
    });
    unknown.forEach((u, i) => u || rows.push(i));
    let orients = this.cells.map((_, o) => o);
    const middles = this.middles.filter((k) => !unknown[k]);
    if (middles.length) {
      // Odd cubes: the middle stickers never move, so they say which way round the cube is.
      const centre = this.cells.map((cl) => middles.reduce((s, k) => s + full[k][this.c.solved[cl[k]]], 0)), low = Math.min(...centre);
      orients = orients.filter((o) => centre[o] <= low + 1);
    }
    const best = new Float64Array(states.length).fill(Infinity), cost = rows.map((i) => full[i]);
    for (const o of orients) {
      const at = rows.map((i) => this.cells[o][i]);
      states.forEach((st, j) => {
        let s = 0;
        for (let r = 0; r < at.length; r++) s += cost[r][st[at[r]]];
        if (s < best[j]) best[j] = s;
      });
    }
    return best;
  }

  // --- one look ---

  /** Update with one look: the reader's (3N² x 7) probabilities and which stickers are in view. */
  look(probs: ArrayLike<number>, inside: boolean[], covered = 0.5): Report {
    const unknown = inside.map((v, i) => !v || probs[7 * i + 6] > covered), seen = unknown.filter((u) => !u).length;
    if (seen < this.minSeen * inside.length) return this.report(false);
    let f = this.fits(probs, unknown, this.points);
    const half = new Float64Array(f.length).fill(Infinity); // halfway through move k's half turn, as point k
    if (this.halves.size) {
      const idx = [...this.halves.keys()], g = this.fits(probs, unknown, idx.flatMap((i) => this.halves.get(i)!));
      idx.forEach((i, j) => (half[i] = Math.min(g[2 * j], g[2 * j + 1])));
    }
    f = f.map((v, i) => Math.min(v, half[i]));
    const fmin = Math.min(...f);
    // A mistake first, judged from where the cube was before this look: near where it was believed to
    // be, and near the point this look fits best (a jump, or a tracker that started late).
    const at = argmin(this.belief.map((b) => -b));
    const ks = [...new Set([at, argmin(f), ...(this.mistake ? [this.mistake.move] : [])])].sort((a, b) => a - b);
    const cands = ks.flatMap((k) => this.mistakesNear(k, at));
    if (cands.length) {
      const g = this.fits(probs, unknown, cands.map((m) => m.state)), gmin = Math.min(...g);
      // Of the mistakes that fit about as well as the best, the likeliest story.
      let j = -1;
      g.forEach((v, i) => {
        if (v <= gmin + 1 && (j < 0 || less(cands[i].rank(this.moves, at), cands[j].rank(this.moves, at)))) j = i;
      });
      if (fmin - g[j] > this.minEvidence && g[j] / seen <= this.maxCost) {
        this.streak++;
        this.back = 0;
        if (this.streak >= this.confirm || this.path.has(keyOf(cands[j].state))) this.show(cands[j]);
        return this.report(true);
      }
    }
    this.streak = 0;
    if (fmin / seen > this.maxCost) return this.report(false); // nothing fits: mid-turn, glare, another puzzle
    if (this.mistake) {
      // Back on track (undone, or never wrong), once it's held.
      this.back++;
      const p = argmin(f);
      if (this.path.has(keyOf(this.points[p]))) {
        // Where the way back shown leads, or passes: done right.
        this.belief = new Float64Array(this.belief.length).fill(0.1 / this.belief.length);
        this.belief[p] += 0.9;
        this.show(null);
      } else if (this.back >= this.confirm) this.show(null);
    }
    this.looks++;
    const prior = this.predict(), like = f.map((v) => Math.exp(-(v - fmin) / this.temper));
    const post = prior.map((p, i) => p * like[i]), sum = post.reduce((a, b) => a + b, 0);
    this.belief = post.map((p) => p / sum);
    const k = argmin(this.belief.map((b) => -b));
    this.half = half[k] <= f[k] && this.halves.has(k);
    return this.report(true);
  }

  /** Show this mistake (null: back on track), and note the cubes along its way back: those are taken at the first look. */
  show(m: Mistake | null): void {
    this.mistake = m;
    this.back = 0;
    this.path = new Set();
    if (!m) return;
    let st = m.state;
    const f = m.fix(this.moves);
    for (const move of [...f.undo, ...f.then]) {
      const q = quarter(move);
      if (q) {
        this.path.add(keyOf(this.c.turn(st, q)));
        this.path.add(keyOf(this.c.turn(st, `${q}'`)));
      }
      st = this.c.turn(st, move);
      this.path.add(keyOf(st));
    }
  }

  /**
   * One slip at moves k-before .. k (0-based), then up to `after` correct moves; an extra turn after the
   * last move too. A cube two slips explain is kept as the likelier story (Mistake.rank).
   */
  mistakesNear(k: number, at = k): Mistake[] {
    const L = this.moves.length, out = new Map<string, Mistake>();
    for (let i = Math.max(0, k - this.before); i <= Math.min(L, k); i++) {
      const base = this.points[i];
      let tails: [Kind, string[]][] = [["wrong", this.moves.slice(i + 1)], ["extra", this.moves.slice(i)], ["skipped", this.moves.slice(i + 1)]];
      if (i + 1 < L) tails.push(["swapped", this.moves.slice(i + 2)]);
      if (i === L) tails = [["extra", []]]; // after the last move: only an extra turn
      for (const [kind, tail] of tails) {
        let firsts: (string | null)[];
        if (kind === "skipped" || kind === "swapped") firsts = [null];
        else if (kind === "wrong") firsts = this.c.eventTurns.filter((t) => t !== this.moves[i]);
        else firsts = this.c.eventTurns.filter((t) => !this.neighbours(i, t).has(turns(t)[0])); // R R2 is a wrong R, R' R a skip
        for (const t of firsts) {
          let st = kind === "swapped" ? this.c.turn(this.c.turn(base, this.moves[i + 1]), this.moves[i]) : t === null ? base : this.c.turn(base, t);
          for (let a = 0; a <= this.after; a++) {
            if (a) {
              if (a > tail.length) break;
              st = this.c.turn(st, tail[a - 1]);
            }
            if (kind === "skipped" && a === 0) continue;
            if (kind === "extra" && a === 0 && i < L) continue; // until move i is done too, it's move i done wrong
            const key = keyOf(st);
            if (this.onScramble.has(key)) continue;
            const m = new Mistake(kind, i, t, a, st), had = out.get(key);
            if (!had || less(m.rank(this.moves, at), had.rank(this.moves, at))) out.set(key, m);
          }
        }
      }
    }
    return [...out.values()];
  }

  /** The layers of the moves an extra turn t before move i meets: the moves either side, and on past those of t's axis. */
  neighbours(i: number, t: string): Set<string> {
    const axis = axisOf(turns(t)[0]), out = new Set<string>();
    for (const [step, start] of [[-1, i - 1], [1, i]]) {
      for (let j = start; j >= 0 && j < this.moves.length; j += step) {
        const k = turns(this.moves[j])[0];
        if (axisOf(k) !== axis) break;
        out.add(k);
      }
    }
    return out;
  }

  report(clear: boolean): Report {
    const k = argmin(this.belief.map((b) => -b));
    const out: Report = { position: k, total: this.moves.length, sure: Math.round(this.belief[k] * 1000) / 1000, clear, half: this.half && !this.mistake, status: "following" };
    if (this.mistake) {
      const m = this.mistake;
      return { ...out, status: "mistake", mistake: { kind: m.kind, move: m.move, done: m.done, after: m.after }, ...m.fix(this.moves) };
    }
    if (!clear && this.looks) return { ...out, status: "unclear" };
    return { ...out, status: k === this.moves.length && this.belief[k] > 0.8 ? "done" : "following" };
  }
}
