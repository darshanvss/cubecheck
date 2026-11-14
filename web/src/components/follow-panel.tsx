"use client";

import { Fragment, useEffect, useRef } from "react";
import { Select } from "@/components/scramble-source";
import type { SlipWhere } from "@/lib/check";
import type { Report } from "@/lib/follow";
import { isPoly, poly } from "@/lib/poly";
import { puzzleName, type Puzzle, type Scramble } from "@/lib/scrambles";
import { cn } from "@/lib/utils";

const TURN = (m: string) => (m.replace("'", "").endsWith("2") ? "⟳" : m.endsWith("'") ? "↺" : "↻"); // ⟳ = a half turn

/**
 * The scramble as rows of moves that wrap. `at` marks the next move (`half`: its half turn is halfway); all
 * done, the whole row turns green. On a mistake, the move that most likely slipped is marked: done wrong
 * (what was done, in a badge), skipped (struck out), swapped with the next (⇄), or an extra turn (a dashed
 * outline and "+D" on the move it came before). Marks never take room of their own: no move ever shifts.
 */
export function MoveRows({ moves, at, half = false, slip }: { moves: string[]; at: number | null; half?: boolean; slip?: SlipWhere | null }) {
  const finished = !slip && at === moves.length && moves.length > 0;
  const extraAt = slip?.kind === "extra" ? Math.min(slip.move, moves.length - 1) : -1; // after the last move: on the last
  return (
    <ol className="flex flex-wrap items-start gap-1" aria-label="The scramble, move by move">
      {moves.map((m, i) => {
        const hit = !!slip && slip.kind !== "extra" && (slip.move === i || (slip.kind === "swapped" && slip.move + 1 === i));
        const state = hit ? "bad" : at === null ? "" : finished || i < at ? "done" : i === at ? "next" : "";
        const badge = hit && slip?.kind === "wrong" ? slip.done : hit && slip?.kind === "swapped" && slip.move === i ? "⇄" : i === extraAt ? `+${slip!.done}` : null;
        return (
          <li
            key={i}
            className={cn(
              "relative grid min-w-9 justify-items-center rounded-md px-1.5 pb-0.5 pt-3 font-mono text-sm leading-5 transition-colors",
              state === "done" && (finished ? "bg-correct/15 text-foreground" : "text-muted-foreground/60"),
              state === "next" && (half ? "bg-amber-400/20 ring-2 ring-amber-500" : "bg-primary/15 ring-2 ring-primary"),
              state === "bad" && "bg-wrong/15 text-wrong ring-1 ring-wrong",
              i === extraAt && "outline-1 outline-dashed outline-wrong",
              hit && slip?.kind === "skipped" && "line-through",
            )}
            aria-current={state === "next" ? "step" : undefined}
          >
            <small className="absolute left-1 top-0.5 font-sans text-[9px] leading-none text-muted-foreground">{i + 1}</small>
            {m}
            {badge && <span className="absolute -right-1.5 -top-1.5 rounded bg-wrong px-1 font-mono text-[10px] font-semibold leading-[14px] text-white">{badge}</span>}
          </li>
        );
      })}
    </ol>
  );
}

function Tile({ move, kind, small = false }: { move: string; kind: "now" | "half" | "soon" | "undo" | "then"; small?: boolean }) {
  return (
    <span
      title={move}
      className={cn(
        "grid min-w-12 justify-items-center rounded-xl border px-2 py-1 font-mono font-bold",
        kind === "now" && "min-w-16 border-primary bg-primary/10 px-3 py-2 text-3xl text-primary",
        kind === "half" && "min-w-16 border-amber-500 bg-amber-400/15 px-3 py-2 text-3xl text-amber-600 dark:text-amber-400",
        kind === "soon" && "border-border text-lg text-muted-foreground",
        kind === "undo" && "border-wrong bg-wrong/10 text-wrong",
        kind === "then" && "border-correct bg-correct/10 text-correct",
        (kind === "undo" || kind === "then") && (small ? "min-w-10 px-1.5 text-lg" : "text-2xl"),
      )}
    >
      <i className="text-xs font-normal not-italic opacity-70">{TURN(move)}</i>
      {move}
    </span>
  );
}

const Group = ({ label, children }: { label: string; children: React.ReactNode }) => (
  <span className="grid gap-1">
    <span className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{label}</span>
    <span className="flex flex-wrap items-center gap-1.5">{children}</span>
  </span>
);

/** What to do now: the next move (and the three after it), or the way back from a mistake. */
export function Coach({ report, moves, offPath = false, puzzle = 3 }: { report: Report | null; moves: string[]; offPath?: boolean; puzzle?: Puzzle }) {
  const thing = isPoly(puzzle) ? "puzzle" : "cube";
  if (offPath) {
    return <p className="text-sm text-muted-foreground">This {thing} isn&rsquo;t at any point of this scramble. Follow-along has to watch it from solved: solve it (or pick the scramble you&rsquo;re doing) and start again.</p>;
  }
  if (!report) {
    const hold = isPoly(puzzle) ? poly(puzzle).d.holdHint.replace(/^turn /, "").split(",")[0] : "a corner toward the camera";
    return <p className="text-sm text-muted-foreground">Hold up a solved {isPoly(puzzle) ? puzzleName(puzzle) : "cube"}, {hold}, and start turning.</p>;
  }
  if (report.status === "mistake") {
    const undo = report.undo!, then = report.then!;
    // merged: the undoing and the right move add up to fewer turns (D2 done for D: just D').
    const label = report.merged ? "To fix it" : undo.length > 1 ? "Undo, in this order" : "Undo";
    const small = undo.length + then.length > 3; // a long way back stays on one line
    return (
      <div className="flex flex-wrap items-end gap-x-4 gap-y-3">
        <Group label={label}>
          {undo.map((m, i) => (
            <Fragment key={i}>
              {i > 0 && <span className="text-muted-foreground">→</span>}
              <Tile move={m} kind="undo" small={small} />
            </Fragment>
          ))}
        </Group>
        {then.length > 0 && <Group label="then">{then.map((m, i) => <Tile key={i} move={m} kind="then" small={small} />)}</Group>}
      </div>
    );
  }
  if (report.status === "done") return <p className="text-sm text-muted-foreground">All {report.total} moves done. Show a solved {thing} for the next scramble.</p>;
  const k = report.position, next = moves[k], soon = moves.slice(k + 1, k + 4);
  return (
    <div className="flex flex-wrap items-end gap-x-4 gap-y-3">
      {next && <Group label={report.half ? "Keep turning" : "Next"}><Tile move={next} kind={report.half ? "half" : "now"} /></Group>}
      {soon.length > 0 && <Group label="then">{soon.map((m, i) => <Tile key={i} move={m} kind="soon" />)}</Group>}
    </div>
  );
}

/**
 * Follow-along inside the scramble card: which scramble (a solved cube after the last one moves on to the
 * next by itself), its moves with the next one marked, and what to do now.
 */
export function FollowView({ scrambles, target, onPick, report, blurry, offPath }: {
  scrambles: Scramble[]; target: Scramble | undefined; onPick: (n: number) => void; report: Report | null; blurry: boolean; offPath: boolean;
}) {
  const rows = useRef<HTMLDivElement>(null);
  const moves = target?.text.split(/\s+/) ?? [];
  const at = report ? report.position : 0;
  // Keep the next move in view inside the box (never scrolling the page).
  useEffect(() => {
    const box = rows.current, cur = box?.querySelector<HTMLElement>("[aria-current=step]");
    if (!box || !cur) return;
    // Where the move sits in the scrolled list, measured on screen (offsetTop counts from a different box).
    const top = cur.getBoundingClientRect().top - box.getBoundingClientRect().top + box.scrollTop, bottom = top + cur.offsetHeight;
    if (top < box.scrollTop) box.scrollTop = top - 8;
    else if (bottom > box.scrollTop + box.clientHeight) box.scrollTop = bottom - box.clientHeight + 8;
  }, [at, target?.number]);
  if (!target) return <p className="m-auto text-sm text-muted-foreground">Add a scramble to follow.</p>;
  const slip = report?.status === "mistake" ? { move: report.mistake!.move, kind: report.mistake!.kind, done: report.mistake!.done } : null;
  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      <div className="flex items-center gap-2">
        <Select aria-label="The scramble to follow" value={target.number} onChange={(e) => onPick(Number(e.target.value))}>
          {scrambles.map((s) => <option key={s.number} value={s.number}>Scramble {s.label}</option>)}
        </Select>
        {/* Notes take the count's place, on the same line, so nothing below moves. */}
        {blurry && report && !offPath ? (
          <span className="ml-auto truncate text-xs font-medium text-amber-600 dark:text-amber-400">Can&rsquo;t see it clearly</span>
        ) : report?.status === "mistake" ? (
          <span className="ml-auto truncate text-xs font-medium text-wrong">{report.resume! >= report.total ? "then it's done" : `then on from move ${report.resume! + 1}`}</span>
        ) : (
          <span className="ml-auto shrink-0 text-xs tabular-nums text-muted-foreground">{report && report.position ? `${report.position} of ${report.total}` : `${moves.length} moves`}</span>
        )}
      </div>
      {/* The scrollbar's room is kept even without one, so the moves never rewrap when it appears. */}
      <div ref={rows} className="relative -mx-1.5 min-h-0 flex-1 overflow-y-auto px-1.5 pb-1.5 pt-1.5 [scrollbar-gutter:stable]">
        <MoveRows moves={moves} at={at} half={report?.half} slip={slip} />
      </div>
      {/* A fixed height, whatever it says: the moves above never move. */}
      <div className="h-[7.5rem] shrink-0 overflow-y-auto border-t border-border pt-3">
        <Coach report={report} moves={moves} offPath={offPath} puzzle={target.size} />
      </div>
    </div>
  );
}

/** The verdict banner's words for a follow-along report. */
export function followWords(r: Report, moves: string[]): { title: string; detail: string; tone: "wait" | "bad" } {
  const total = r.total;
  if (r.status === "mistake") {
    const m = r.mistake!;
    const what = m.kind === "wrong" ? `${m.done} instead of move ${m.move + 1}`
      : m.kind === "extra" ? `an extra ${m.done} ${m.move === total ? "after the last move" : `before move ${m.move + 1}`}`
      : m.kind === "swapped" ? `moves ${m.move + 1} and ${m.move + 2} swapped` : `move ${m.move + 1} skipped`;
    return { tone: "bad", title: "Wrong move", detail: `${what}${m.after ? `, then ${m.after} more` : ""}: go back as shown` };
  }
  const next = moves[r.position];
  if (!r.position && !r.half) return { tone: "wait", title: "Start from solved", detail: next ? `first move: ${next}` : "" };
  if (r.half) return { tone: "wait", title: `Move ${r.position + 1} of ${total}: halfway`, detail: `${next} is a half turn: one more quarter turn, the same way` };
  return { tone: "wait", title: `Move ${r.position} of ${total} done`, detail: next ? `next: ${next}` : "" };
}
