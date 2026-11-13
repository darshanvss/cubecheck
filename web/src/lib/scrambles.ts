// A list of scrambles (one per line) for one puzzle, and which of them, if any, the puzzle in view matches.
import { FACES, singleSlips, verifyProbs, type Alternatives, type Verdict } from "./check.ts";
import { checkN, singleSlipsN } from "./nxn-check.ts";
import { EVENT_NAME, cube, guessSize, type NState } from "./nxn.ts";
import { POLY_EVENTS, isPoly, poly, type PolyEvent } from "./poly.ts";

/** What is being checked: an N×N cube's size (2..7), or one of the other puzzles (experimental). */
export type Puzzle = number | PolyEvent;
export const puzzleName = (p: Puzzle) => (isPoly(p) ? poly(p).d.name : EVENT_NAME[p]);

export interface Scramble {
  number: number; // 1-based, as listed
  label: string; // as the scramble sheet numbers it: "1", "2", ..., "E1" for an extra
  text: string;
  size: Puzzle;
  state?: NState;
  alternatives?: Alternatives; // one-slip versions: a cube that fits one of them better is a mis-scramble
  error?: string;
}

// A leading "1." or "2)" (as in TNoodle's printouts) is a label, not a move.
const LABEL = /^\s*(E?\d+)\s*[.)]\s*/;

export interface Line {
  label: string;
  text: string;
}

/** Pasted text, one scramble per line, with any "1." labels split off. */
export const parseLines = (text: string): Line[] =>
  text
    .split("\n")
    .map((line) => ({ label: LABEL.exec(line)?.[1] ?? "", text: line.replace(LABEL, "").trim() }))
    .filter((l) => l.text)
    .map((l, i) => ({ ...l, label: l.label || String(i + 1) }));

/**
 * The other puzzle a scramble is for, when its moves say so: a pyraminx's tips (u l r b), an FTO's BL and BR
 * (or 25 or more third turns), a skewb's 7 to 12 turns of R U L B. A 3×3 scramble always has half turns.
 */
export function guessPoly(text: string): PolyEvent | null {
  const moves = text.replace(/[’′]/g, "'").split(/\s+/).filter(Boolean);
  if (!moves.length || moves.some((m) => m.includes("2"))) return null;
  const fits = (e: PolyEvent) => moves.every((m) => poly(e).turns.includes(m));
  if (fits("pyram") && moves.some((m) => /^[ulrb]/.test(m))) return "pyram";
  if (fits("fto") && (moves.some((m) => /^B[LR]/.test(m)) || moves.length >= 25)) return "fto";
  if (fits("skewb") && moves.length >= 7 && moves.length <= 12) return "skewb";
  return null;
}

/** The puzzle most of the lines are for, or a 3×3 when nothing says. */
export function sizeOf(lines: Line[]): Puzzle {
  const votes = new Map<Puzzle, number>();
  for (const l of lines) {
    const n = guessPoly(l.text) ?? guessSize(l.text);
    if (n) votes.set(n, (votes.get(n) ?? 0) + 1);
  }
  return [...votes].sort((a, b) => b[1] - a[1])[0]?.[0] ?? 3;
}

export const PUZZLES: Puzzle[] = [2, 3, 4, 5, 6, 7, ...POLY_EVENTS];

/** Each line's expected cube and one-slip cubes, worked out once (a 7×7's take a moment). */
export function prepare(lines: Line[], size: Puzzle): Scramble[] {
  return lines.map((l, i) => {
    const base = { number: i + 1, label: l.label, text: l.text, size };
    try {
      if (isPoly(size)) {
        const p = poly(size), text = p.parse(l.text).join(" ");
        return { ...base, text, state: p.after(text), alternatives: p.singleSlips(text) };
      }
      const state = cube(size).after(l.text);
      return { ...base, state, alternatives: size === 3 ? singleSlips(l.text) : singleSlipsN(l.text, size) };
    } catch (err) {
      return { ...base, error: (err as Error).message };
    }
  });
}

export const netOf = (state: NState) => Array.from(state, (c) => FACES[c]).join("");

/** The check for one scramble: on a 3×3 with the cube rules (verifyProbs), on other sizes nxn_check's. */
export function judge(probs: ArrayLike<number>, inside: boolean[], s: Scramble): Verdict {
  if (isPoly(s.size)) return poly(s.size).check(probs, inside, s.state!, s.alternatives);
  return s.size === 3 ? verifyProbs(probs, inside, s.state!, s.alternatives) : checkN(probs, inside, s.size, s.state!, s.alternatives);
}

export interface Match {
  status: Verdict["status"];
  reason: string;
  scramble: Scramble | null; // the matching scramble, or the closest one
  verdict: Verdict | null;
}

/** Check the reading against every scramble: the one it matches, or else the closest. */
export function match(probs: ArrayLike<number>, inside: boolean[], scrambles: Scramble[]): Match {
  const usable = scrambles.filter((s) => s.state);
  const results = usable.map((s) => ({ scramble: s, verdict: judge(probs, inside, s) }));
  if (!results.length) return { status: "re-check", reason: "add a scramble to check against", scramble: null, verdict: null };
  const byFit = [...results].sort((a, b) => a.verdict.fit - b.verdict.fit);
  const correct = byFit.find((r) => r.verdict.status === "correct");
  if (correct) return { status: "correct", reason: "matches the scramble", ...correct };
  const judged = byFit.find((r) => r.verdict.status === "mis-scramble");
  if (!judged) return { status: "re-check", reason: results[0].verdict.reason, ...results[0] };
  return { status: "mis-scramble", reason: "doesn't match the scramble", ...judged };
}
