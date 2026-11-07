// A list of scrambles (one per line) for one cube size, and which of them, if any, the cube in view matches.
import { FACES, singleSlips, verifyProbs, type Alternatives, type Verdict } from "./check.ts";
import { checkN, singleSlipsN } from "./nxn-check.ts";
import { cube, guessSize, type NState } from "./nxn.ts";

export interface Scramble {
  number: number; // 1-based, as listed
  label: string; // as the scramble sheet numbers it: "1", "2", ..., "E1" for an extra
  text: string;
  size: number;
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

/** The size most of the lines are for (2..7), or 3 when nothing says. */
export function sizeOf(lines: Line[]): number {
  const votes = new Map<number, number>();
  for (const l of lines) {
    const n = guessSize(l.text);
    if (n) votes.set(n, (votes.get(n) ?? 0) + 1);
  }
  return [...votes].sort((a, b) => b[1] - a[1])[0]?.[0] ?? 3;
}

/** Each line's expected cube and one-slip cubes, worked out once (a 7×7's take a moment). */
export function prepare(lines: Line[], size: number): Scramble[] {
  return lines.map((l, i) => {
    const base = { number: i + 1, label: l.label, text: l.text, size };
    try {
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
