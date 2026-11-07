"use client";

import { Fragment } from "react";
import type { SlipWhere } from "@/lib/check";
import type { Report } from "@/lib/follow";
import { cn } from "@/lib/utils";

const TURN = (m: string) => (m.replace("'", "").endsWith("2") ? "⟳" : m.endsWith("'") ? "↺" : "↻"); // ⟳ = a half turn

/**
 * The scramble as rows of moves that wrap (never scrolled: only the highlight moves). `at` marks the next
 * move (`half`: its half turn is halfway); all done, the whole row turns green. On a mistake, the move that
 * most likely slipped is marked: done wrong (what was done under it), skipped (struck out), swapped with
 * the next (⇄), or an extra turn (a dashed move where it went in).
 */
export function MoveRows({ moves, at, half = false, slip }: { moves: string[]; at: number | null; half?: boolean; slip?: SlipWhere | null }) {
  const finished = !slip && at === moves.length && moves.length > 0;
  const extra = (i: number) =>
    slip?.kind === "extra" && slip.move === i && <span className="rounded-md border border-dashed border-wrong px-1.5 py-0.5 font-mono text-sm text-wrong">+{slip.done}</span>;
  return (
    <ol className="flex flex-wrap items-start gap-1" aria-label="The scramble, move by move">
      {moves.map((m, i) => {
        const hit = !!slip && slip.kind !== "extra" && (slip.move === i || (slip.kind === "swapped" && slip.move + 1 === i));
        const state = hit ? "bad" : at === null ? "" : finished || i < at ? "done" : i === at ? "next" : "";
        return (
          <Fragment key={i}>
            {extra(i)}
            <li
              className={cn(
                "relative grid min-w-9 justify-items-center rounded-md px-1.5 pb-0.5 pt-3 font-mono text-sm leading-5 transition-colors",
                state === "done" && (finished ? "bg-correct/15 text-foreground" : "text-muted-foreground/60"),
                state === "next" && (half ? "bg-amber-400/20 ring-2 ring-amber-500" : "bg-primary/15 ring-2 ring-primary"),
                state === "bad" && "bg-wrong/15 text-wrong ring-1 ring-wrong",
                hit && slip?.kind === "skipped" && "line-through",
              )}
              aria-current={state === "next" ? "step" : undefined}
            >
              <small className="absolute left-1 top-0.5 font-sans text-[9px] leading-none text-muted-foreground">{i + 1}</small>
              {m}
              {hit && slip?.kind === "wrong" && <span className="text-[11px] leading-4 no-underline">{slip.done}</span>}
            </li>
            {hit && slip?.kind === "swapped" && slip.move === i && <span className="self-center text-wrong">⇄</span>}
          </Fragment>
        );
      })}
      {extra(moves.length)}
    </ol>
  );
}

function Tile({ move, kind }: { move: string; kind: "now" | "half" | "soon" | "undo" | "then" }) {
  return (
    <span
      title={move}
      className={cn(
        "grid min-w-12 justify-items-center rounded-xl border px-2 py-1 font-mono font-bold",
        kind === "now" && "min-w-16 border-primary bg-primary/10 px-3 py-2 text-3xl text-primary",
        kind === "half" && "min-w-16 border-amber-500 bg-amber-400/15 px-3 py-2 text-3xl text-amber-600 dark:text-amber-400",
        kind === "soon" && "border-border text-lg text-muted-foreground",
        kind === "undo" && "border-wrong bg-wrong/10 text-2xl text-wrong",
        kind === "then" && "border-correct bg-correct/10 text-2xl text-correct",
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
export function Coach({ report, moves, blurry }: { report: Report | null; moves: string[]; blurry: boolean }) {
  if (!report) return <p className="text-sm text-muted-foreground">Hold up a solved cube, one corner toward the camera, then start turning.</p>;
  if (report.status === "mistake") {
    const undo = report.undo!, then = report.then!;
    // merged: the undoing and the right move add up to fewer turns (D2 done for D: just D').
    const label = report.merged ? "To fix it" : undo.length > 1 ? "Undo, in this order" : "Undo";
    return (
      <div className="flex flex-wrap items-end gap-x-4 gap-y-3">
        <Group label={label}>
          {undo.map((m, i) => (
            <Fragment key={i}>
              {i > 0 && <span className="text-muted-foreground">→</span>}
              <Tile move={m} kind="undo" />
            </Fragment>
          ))}
        </Group>
        {then.length > 0 && <Group label="then">{then.map((m, i) => <Tile key={i} move={m} kind="then" />)}</Group>}
        <span className="pb-1 text-sm text-muted-foreground">{report.resume! >= report.total ? "and the scramble is done" : `and carry on from move ${report.resume! + 1}`}</span>
      </div>
    );
  }
  if (report.status === "done") return <p className="text-sm text-muted-foreground">All {report.total} moves done: the verdict above checks the finished cube.</p>;
  const k = report.position, next = moves[k], soon = moves.slice(k + 1, k + 4);
  return (
    <div className="flex flex-wrap items-end gap-x-4 gap-y-3">
      {next && <Group label={report.half ? "Keep turning" : "Next"}><Tile move={next} kind={report.half ? "half" : "now"} /></Group>}
      {soon.length > 0 && <Group label="then">{soon.map((m, i) => <Tile key={i} move={m} kind="soon" />)}</Group>}
      <span className={cn("ml-auto pb-1 text-sm tabular-nums", blurry ? "font-medium text-amber-600 dark:text-amber-400" : "text-muted-foreground")}>
        {blurry ? "Can't see it clearly: hold still, corner to the camera" : `${k} of ${report.total} done`}
      </span>
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
