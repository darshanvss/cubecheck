// A list of scrambles (one per line), and which of them, if any, the cube in view matches.
import { FACES, expectedState, singleSlips, verifyProbs, type Alternatives, type State, type Verdict } from "./check";

export interface Scramble {
  number: number; // 1-based, as listed
  text: string;
  state?: State;
  alternatives?: Alternatives; // one-slip versions: a cube that fits one of them better is a mis-scramble
  error?: string;
}

// A leading "1." or "2)" (as in TNoodle's printouts) is a label, not a move.
const LABEL = /^\s*\d+\s*[.)]\s*/;

export function parseScrambles(text: string): Scramble[] {
  return text
    .split("\n")
    .map((line) => line.replace(LABEL, "").trim())
    .filter(Boolean)
    .map((line, i) => {
      try {
        return { number: i + 1, text: line, state: expectedState(line), alternatives: singleSlips(line) };
      } catch (err) {
        return { number: i + 1, text: line, error: (err as Error).message };
      }
    });
}

export const netOf = (state: State) => Array.from(state, (c) => FACES[c]).join("");

export interface Match {
  status: Verdict["status"];
  reason: string;
  scramble: Scramble | null; // the matching scramble, or the closest one
  verdict: Verdict | null;
}

/** Check the reading against every scramble: the one it matches, or else the closest. */
export function match(probs: ArrayLike<number>, inside: boolean[], scrambles: Scramble[]): Match {
  const usable = scrambles.filter((s) => s.state);
  const results = usable.map((s) => ({ scramble: s, verdict: verifyProbs(probs, inside, s.state!, s.alternatives) }));
  if (!results.length) return { status: "re-check", reason: "add a scramble to check against", scramble: null, verdict: null };
  const byFit = [...results].sort((a, b) => a.verdict.fit - b.verdict.fit);
  const correct = byFit.find((r) => r.verdict.status === "correct");
  if (correct) return { status: "correct", reason: "matches the scramble", ...correct };
  const judged = byFit.find((r) => r.verdict.status === "mis-scramble");
  if (!judged) return { status: "re-check", reason: results[0].verdict.reason, ...results[0] };
  return { status: "mis-scramble", reason: "doesn't match the scramble", ...judged };
}
