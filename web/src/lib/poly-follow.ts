// Follow-along for skewb, pyraminx and FTO (experimental): the cubes' tracker (follow.ts) on these puzzles.
// A port of python/cubecheck/poly_follow.py; tests/poly.test.ts checks it gives Python's answers.
//
// What differs from a cube:
// - Every turn is a third of a turn: no half turns, no "halfway".
// - No sticker stays put the way a 3×3's centres do, so every way round is tried at each look.
// - Some turns don't disturb each other: a pyraminx's tips and anything else, an FTO's opposite faces. Which
//   ones is worked out from the moves (`apart`), and used three ways: a later move done early isn't a
//   mistake (it counts as the point before it, reported as `ahead`); the way back keeps the moves it needn't
//   undo; and the way back is shortened across them (U D U is D U').
// - How much a look must show and how much better a slip must fit are per puzzle (poly-data.json's follow).
import { Follower, inverse, type Fix, type FollowOptions, type Mistake, type Report } from "./follow.ts";
import { poly, type Poly, type PolyEvent, type PState } from "./poly.ts";

const MAX_EARLY = 4; // later moves that may be done early at one point (a pyraminx has 4 tips)

/** What a move turns: R' -> R, BL -> BL, u' -> u. */
const layer = (move: string) => move.replace(/'$/, "");

function combinations<T>(items: T[]): T[][] {
  const out: T[][] = [];
  for (let r = 1; r <= items.length; r++) {
    const pick = (start: number, chosen: T[]) => {
      if (chosen.length === r) return void out.push(chosen);
      for (let i = start; i < items.length; i++) pick(i + 1, [...chosen, items[i]]);
    };
    pick(0, []);
  }
  return out;
}

export class PolyFollower extends Follower {
  readonly event: PolyEvent;
  readonly p: Poly;
  private apart: Set<string>;
  private minFaces: number;

  constructor(scramble: string, event: PolyEvent, options: FollowOptions & { minFaces?: number } = {}) {
    const p = poly(event), s = p.d.follow;
    super(scramble, { puzzle: p, moves: p.parse(scramble), turnList: p.turns, cells: p.cells, middles: [], colours: p.d.colours },
      { maxCost: s.max_cost, minEvidence: s.min_evidence, ...options });
    this.halfMargin = 1; // a move done early can be out of sight: then the look fits with and without it equally
    this.event = event;
    this.p = p;
    this.minFaces = options.minFaces ?? s.min_faces;
    const layers = [...new Set(p.turns.map(layer))].sort();
    const same = (a: Uint16Array, b: Uint16Array) => a.every((v, i) => v === b[i]);
    const compose = (a: Uint16Array, b: Uint16Array) => Uint16Array.from(b, (v) => a[v]); // a then b, as Python's a[b]
    this.apart = new Set();
    for (const a of layers) for (const b of layers) {
      if (a === b || same(compose(p.move(a), p.move(b)), compose(p.move(b), p.move(a)))) this.apart.add(`${a}|${b}`);
    }
    // Later moves done early: for point k, the moves after it that nothing in between disturbs, in every
    // combination. They count as point k (follow.ts's halves).
    const moves = this.moves, on = new Set(this.points.map((q) => q.join())), halves = new Map<number, PState[]>();
    for (let k = 0; k < moves.length; k++) {
      const early: number[] = [];
      for (let j = k + 1; j < moves.length && early.length < MAX_EARLY; j++) {
        let free = true;
        for (let i = k; i < j && free; i++) free = this.apart.has(`${layer(moves[j])}|${layer(moves[i])}`);
        if (free) early.push(j);
      }
      const states = combinations(early).map((js) => p.apply(this.points[k], js.map((j) => moves[j]))).filter((st) => !on.has(st.join()));
      if (states.length) halves.set(k, states);
    }
    this.setHalves(halves);
  }

  /** One look: the reader's probabilities and which stickers can be read; one face of a pyraminx says nothing. */
  look(probs: ArrayLike<number>, inside: boolean[], covered = 0.5): Report {
    const per = this.p.d.perFace;
    const faces = Array.from({ length: this.p.d.nFaces }, (_, k) => inside.slice(k * per, (k + 1) * per).some(Boolean)).filter(Boolean).length;
    if (faces < this.minFaces) {
      this.why = "few";
      return this.report(false);
    }
    return super.look(probs, inside, covered);
  }

  commutes = (a: string, b: string) => this.apart.has(`${layer(a)}|${layer(b)}`);

  /** Add up the turns of one layer that meet, through turns that don't disturb them: R R -> R', U D U -> D U'. */
  shorten(moves: string[]): string[] {
    const out: [string, number][] = []; // [layer, thirds of a turn]
    for (const m of moves) {
      const k = layer(m), q = m.endsWith("'") ? 2 : 1;
      let j = out.length - 1;
      while (j >= 0 && out[j][0] !== k && this.apart.has(`${out[j][0]}|${k}`)) j--;
      if (j >= 0 && out[j][0] === k) {
        out[j][1] = (out[j][1] + q) % 3;
        if (!out[j][1]) out.splice(j, 1);
      } else out.push([k, q]);
    }
    const next = out.map(([k, q]) => k + (q === 2 ? "'" : ""));
    return next.join(" ") === moves.join(" ") ? next : this.shorten(next);
  }

  /** The way back, keeping the moves done since the slip when putting it right doesn't disturb them. */
  fix(m: Mistake): Fix {
    const k = m.move, moves = this.moves, inv = (ms: string[]) => [...ms].reverse().map(inverse);
    if (m.kind === "swapped") return this.way(inv([moves[k + 1], moves[k], ...moves.slice(k + 2, k + 2 + m.after)]), [], k);
    const start = m.kind === "extra" ? k : k + 1; // the first right move done after the slip
    const since = moves.slice(start, start + m.after);
    let undo = m.done ? [inverse(m.done)] : [];
    const then = m.kind === "extra" ? [] : [moves[k]];
    let keep = 0;
    while (keep < since.length && [...undo, ...then].every((t) => this.commutes(since[keep], t))) keep++;
    undo = [...inv(since.slice(keep)), ...undo];
    if (m.kind === "skipped" && !keep) return this.way(undo, [], k); // everything since undone: carry on from the skipped move
    return this.way(undo, then, m.kind === "extra" ? start + keep : k + 1 + keep);
  }

  private way(undo: string[], then: string[], resume: number): Fix {
    const whole = this.shorten([...undo, ...then]);
    if (undo.length && [...this.shorten(undo), ...then].join(" ") === whole.join(" ")) return { undo: this.shorten(undo), then, resume, merged: false };
    return { undo: whole, then: [], resume, merged: true };
  }

  rank(m: Mistake, at: number): number[] {
    const f = this.fix(m);
    return [Number(m.move < at), f.undo.length + f.then.length, ["skipped", "swapped", "wrong", "extra"].indexOf(m.kind), -m.move, m.after];
  }

  /** An extra turn t before move i reaches a move of its own layer, either side, through moves that don't disturb it. */
  meets(i: number, t: string): boolean {
    const k = layer(t);
    for (const [step, start] of [[-1, i - 1], [1, i]]) {
      for (let j = start; j >= 0 && j < this.moves.length; j += step) {
        const other = layer(this.moves[j]);
        if (other === k) return true;
        if (!this.apart.has(`${other}|${k}`)) break;
      }
    }
    return false;
  }

  report(clear: boolean): Report {
    const out = super.report(clear);
    return { ...out, ahead: out.half, half: false };
  }
}
