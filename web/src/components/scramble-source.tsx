"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { SIZES, randomScramble } from "@/lib/nxn";
import { POLY_EVENTS, isPoly, poly } from "@/lib/poly";
import { parseLines, puzzleName, sizeOf, type Line, type Puzzle } from "@/lib/scrambles";
import { NotTnoodleError, PasswordError, byEvent, eventName, loadPdfJs, readTnoodlePdf, setName, type ScrambleSet } from "@/lib/tnoodle";
import { cn } from "@/lib/utils";

export interface Source {
  lines: Line[];
  size: Puzzle; // a cube's N, or another puzzle (experimental)
  name: string; // what the list is: "3×3", "4×4 · Round 1 · Group A"
}

const SAMPLE = "R2 D' B2 U2 L2 F2 D R2 U' L2 B2 R' F U' L' D2 F' R' B D2 R'";
/** A native select (the phone's own picker) with its arrow close to the text. `className` sizes it. */
export function Select({ className, children, ...props }: React.ComponentProps<"select">) {
  return (
    <span className={cn("relative inline-flex min-w-0", className)}>
      <select
        {...props}
        className="h-9 w-full min-w-0 cursor-pointer appearance-none truncate rounded-lg border border-input bg-background py-0 pl-3 pr-8 text-sm font-medium shadow-xs outline-none focus-visible:ring-2 focus-visible:ring-ring/50 disabled:cursor-default disabled:opacity-50"
      >
        {children}
      </select>
      <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="pointer-events-none absolute right-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground">
        <path d="m6 9 6 6 6-6" />
      </svg>
    </span>
  );
}

/** A small icon button with a tooltip. */
export function IconButton({ label, onClick, children, disabled }: { label: string; onClick: () => void; children: React.ReactNode; disabled?: boolean }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button variant="ghost" size="icon" aria-label={label} onClick={onClick} disabled={disabled} className="text-muted-foreground hover:text-foreground">
          {children}
        </Button>
      </TooltipTrigger>
      <TooltipContent className="max-w-64 text-balance">{label}</TooltipContent>
    </Tooltip>
  );
}

export const Icon = ({ d, className }: { d: React.ReactNode; className?: string }) => (
  <svg aria-hidden="true" className={cn("size-[18px]", className)} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
    {d}
  </svg>
);
const DICE = <><rect x="3.5" y="3.5" width="17" height="17" rx="3.5" /><circle cx="8.5" cy="8.5" r="1" fill="currentColor" /><circle cx="15.5" cy="15.5" r="1" fill="currentColor" /><circle cx="12" cy="12" r="1" fill="currentColor" /></>;
const PDF = <><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" /><path d="M14 3v5h5M9 13h6M9 17h4" /></>;
const PENCIL = <><path d="M12 20h9" /><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z" /></>;
const CLOSE = <path d="M18 6 6 18M6 6l12 12" />;

/**
 * The scrambles: pasted (any size, guessed from the moves) or from a TNoodle PDF (event, round and group).
 * One card whose box keeps its size: the textarea while editing, a PDF's prompts, otherwise `children`
 * (the list to check against, or the scramble being followed).
 */
export function ScrambleSource({ onChange, children }: { onChange: (s: Source) => void; children: React.ReactNode }) {
  const [text, setText] = useState("");
  const [editing, setEditing] = useState(false);
  const [picked, setPicked] = useState<Puzzle | null>(null); // a puzzle chosen by hand; null: guessed from the moves
  const lines = useMemo(() => parseLines(text), [text]);
  const guessed = useMemo(() => sizeOf(lines), [lines]);
  const size = picked ?? guessed;
  const textBox = useRef<HTMLTextAreaElement>(null);

  const [pdf, setPdf] = useState<{ name: string; bytes: Uint8Array } | null>(null); // a PDF is open (kept in memory only)
  const [password, setPassword] = useState("");
  const [needsPassword, setNeedsPassword] = useState(false);
  const [pdfError, setPdfError] = useState("");
  const [busy, setBusy] = useState(false);
  const [sets, setSets] = useState<ScrambleSet[] | null>(null);
  const [event, setEvent] = useState("");
  const [setIndex, setSetIndex] = useState(0);
  const passwordBox = useRef<HTMLInputElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const events = useMemo(() => (sets ? byEvent(sets) : []), [sets]);
  const chosen = events.find((e) => e.event === event) ?? events.find((e) => e.size) ?? null;
  const chosenSet = chosen?.size ? chosen.sets[Math.min(setIndex, chosen.sets.length - 1)] : null;

  // Pasted scrambles stay in this browser between visits; they never leave the device.
  useEffect(() => {
    try {
      setText(localStorage.getItem("cubecheck-scrambles") ?? "");
    } catch {}
  }, []);
  const onText = (value: string) => {
    setText(value);
    try {
      localStorage.setItem("cubecheck-scrambles", value);
    } catch {}
  };

  useEffect(() => {
    if (!pdf) onChange({ lines, size, name: puzzleName(size) });
    else if (chosenSet) onChange({ lines: chosenSet.scrambles.map((s) => ({ label: s.label, text: s.text })), size: chosenSet.size!, name: `${eventName(chosen!.event)} · ${setName(chosenSet)}` });
    else onChange({ lines: [], size: 3, name: "" });
  }, [pdf, lines, size, chosen, chosenSet, onChange]);

  const open = async (file: { name: string; bytes: Uint8Array }, pass: string) => {
    setBusy(true);
    setPdfError("");
    try {
      const found = await readTnoodlePdf(await loadPdfJs(), file.bytes.slice(), pass); // pdf.js takes the bytes over
      setSets(found);
      setNeedsPassword(false);
      setPassword("");
      const first = byEvent(found).find((e) => e.size);
      setEvent(first?.event ?? "");
      setSetIndex(0);
      if (!first) setPdfError("This competition has no events this page can check yet (2×2 to 7×7, skewb, pyraminx).");
    } catch (err) {
      setSets(null);
      if (err instanceof PasswordError) {
        setNeedsPassword(true);
        setPdfError(err.message === "wrong password" ? "That didn't open it. Check it and try again." : "");
        setTimeout(() => passwordBox.current?.focus());
      } else if (err instanceof NotTnoodleError) {
        setNeedsPassword(false);
        setPdfError(err.message === "this file isn't a PDF" ? "That file isn't a PDF." : "This isn't a TNoodle scramble PDF.");
      } else setPdfError(`Couldn't read this PDF (${(err as Error).message}).`);
    } finally {
      setBusy(false);
    }
  };

  const choose = async (f: File | undefined) => {
    if (!f) return;
    const file = { name: f.name, bytes: new Uint8Array(await f.arrayBuffer()) };
    setPdf(file);
    setSets(null);
    setNeedsPassword(false);
    await open(file, "");
  };
  const closePdf = () => {
    setPdf(null);
    setSets(null);
    setNeedsPassword(false);
    setPdfError("");
  };

  const random = () => {
    const one = () => (isPoly(size) ? poly(size).randomScramble() : randomScramble(size)).join(" ");
    onText(Array.from({ length: 5 }, (_, i) => `${i + 1}. ${one()}`).join("\n"));
    setPicked(size);
    setEditing(false);
  };
  const edit = () => {
    setEditing(true);
    setTimeout(() => textBox.current?.focus());
  };

  const showText = !pdf && (editing || !lines.length);
  const count = pdf ? chosenSet?.scrambles.length ?? 0 : lines.length;

  let box: React.ReactNode;
  if (pdf && busy && !needsPassword) box = <p className="m-auto text-sm text-muted-foreground">Reading the PDF…</p>;
  else if (pdf && needsPassword) {
    box = (
      <form className="m-auto grid w-full max-w-72 gap-2.5 text-center" onSubmit={(e) => { e.preventDefault(); void open(pdf, password); }}>
        <p className="text-sm font-medium">This PDF is locked</p>
        <Input ref={passwordBox} type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Passcode or password" autoComplete="off" aria-label="The PDF's passcode or password" />
        <Button type="submit" disabled={busy || !password}>{busy ? "Opening…" : "Open"}</Button>
        <p className={cn("text-xs", pdfError ? "text-wrong" : "text-muted-foreground")}>{pdfError || "The group's passcode, or the competition's password"}</p>
      </form>
    );
  } else if (pdf && pdfError) box = <p className="m-auto max-w-72 text-center text-sm text-wrong">{pdfError}</p>;
  else if (showText) {
    box = (
      <Textarea
        ref={textBox}
        value={text}
        onChange={(e) => onText(e.target.value)}
        onBlur={() => lines.length && setEditing(false)}
        onPaste={() => setTimeout(() => setEditing(false))}
        placeholder={"Paste your scrambles, one per line\n\n" + SAMPLE}
        spellCheck={false}
        autoComplete="off"
        className="h-full min-h-0 resize-none font-mono leading-relaxed"
        aria-label="Scrambles, one per line"
      />
    );
  } else box = children;

  return (
    <Card className="gap-3 p-4 md:flex-1">
      <div className="flex items-center gap-2">
        {pdf ? (
          <>
            {/* The event says which cube: 3×3 for one-handed, blindfolded, fewest moves and multi-blind, 4×4 for 4×4 blindfolded, ... */}
            <Select className="max-w-[50%] shrink" aria-label="Event" value={chosen?.event ?? ""} disabled={!sets} onChange={(e) => { setEvent(e.target.value); setSetIndex(0); }}>
              {!sets && <option value="">Event</option>}
              {events.filter((e) => e.size).map((e) => <option key={e.event} value={e.event}>{eventName(e.event)}</option>)}
              {events.some((e) => !e.size) && (
                <optgroup label="Not checked yet">
                  {events.filter((e) => !e.size).map((e) => <option key={e.event} value={e.event} disabled>{eventName(e.event)}</option>)}
                </optgroup>
              )}
            </Select>
            <Select className="shrink" aria-label="Round and group" value={setIndex} disabled={!chosen?.size} onChange={(e) => setSetIndex(Number(e.target.value))}>
              {!chosen?.size && <option value={0}>Round and group</option>}
              {chosen?.size && chosen.sets.map((s, i) => <option key={s.title} value={i}>{setName(s)}</option>)}
            </Select>
          </>
        ) : (
          <Select aria-label="Puzzle" value={String(size)} onChange={(e) => { const v = e.target.value, n: Puzzle = isPoly(v) ? v : Number(v); setPicked(n === guessed ? null : n); }}>
            {SIZES.map((n) => <option key={n} value={n}>{puzzleName(n)}</option>)}
            <optgroup label="Experimental">
              {POLY_EVENTS.map((e) => <option key={e} value={e}>{puzzleName(e)}</option>)}
            </optgroup>
          </Select>
        )}
        <div className="ml-auto flex shrink-0 items-center">
          {!pdf && <IconButton label={`Five random ${puzzleName(size)} scrambles, to try it out`} onClick={random}><Icon d={DICE} /></IconButton>}
          {!pdf && lines.length > 0 && !showText && <IconButton label="Edit the scrambles" onClick={edit}><Icon d={PENCIL} /></IconButton>}
          <IconButton label="Open a TNoodle PDF. It's read in this browser, never uploaded." onClick={() => fileInput.current?.click()}><Icon d={PDF} /></IconButton>
          <input ref={fileInput} type="file" accept="application/pdf,.pdf" className="sr-only" tabIndex={-1} onChange={(e) => { void choose(e.target.files?.[0]); e.target.value = ""; }} />
        </div>
      </div>

      {/* On a phone a fixed height; beside the camera, as tall as the camera's side. */}
      <div className="flex h-80 min-h-0 flex-col md:h-auto md:min-h-80 md:flex-1 md:basis-0">{box}</div>

      <div className="flex h-6 items-center gap-2 text-xs text-muted-foreground">
        {pdf ? (
          <>
            <Icon d={PDF} className="size-3.5 shrink-0" />
            <span className="min-w-0 truncate">{pdf.name}</span>
            <button className="-m-1 shrink-0 rounded p-1 hover:text-foreground" onClick={closePdf} aria-label="Close the PDF"><Icon d={CLOSE} className="size-3.5" /></button>
            <span className="ml-auto shrink-0 tabular-nums">{count ? `${count} scrambles` : ""}</span>
          </>
        ) : (
          <span className="tabular-nums">{count ? `${count} scramble${count > 1 ? "s" : ""}` : "Or open a TNoodle PDF"}</span>
        )}
      </div>
    </Card>
  );
}
