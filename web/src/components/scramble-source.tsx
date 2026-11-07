"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { EVENT_LENGTH, EVENT_NAME, SIZES, randomScramble } from "@/lib/nxn";
import { parseLines, sizeOf, type Line } from "@/lib/scrambles";
import { NotTnoodleError, PasswordError, byEvent, eventName, loadPdfJs, readTnoodlePdf, setName, type ScrambleSet } from "@/lib/tnoodle";
import { cn } from "@/lib/utils";

export interface Source {
  lines: Line[];
  size: number;
  name: string; // what the list is: "3×3", "4×4 · Round 1 · Group A"
}

const SAMPLE = "R2 D' B2 U2 L2 F2 D R2 U' L2 B2 R' F U' L' D2 F' R' B D2 R'";
const select = "h-9 w-full min-w-0 rounded-md border border-input bg-background px-2 text-sm shadow-xs outline-none focus-visible:ring-2 focus-visible:ring-ring/50 disabled:opacity-50";

/** Where the scrambles come from: pasted (any size, guessed from the moves) or a TNoodle PDF (event, round, group). */
export function ScrambleSource({ onChange }: { onChange: (s: Source) => void }) {
  const [tab, setTab] = useState<"paste" | "pdf">("paste");
  const [text, setText] = useState("");
  const [picked, setPicked] = useState<number | null>(null); // a size chosen by hand; null: guessed from the moves
  const lines = useMemo(() => parseLines(text), [text]);
  const guessed = useMemo(() => sizeOf(lines), [lines]);
  const size = picked ?? guessed;

  const [pdf, setPdf] = useState<{ name: string; bytes: Uint8Array } | null>(null);
  const [password, setPassword] = useState("");
  const [needsPassword, setNeedsPassword] = useState(false);
  const [pdfError, setPdfError] = useState("");
  const [busy, setBusy] = useState(false);
  const [sets, setSets] = useState<ScrambleSet[] | null>(null);
  const [event, setEvent] = useState("");
  const [setIndex, setSetIndex] = useState(0);
  const passwordBox = useRef<HTMLInputElement>(null);
  const events = useMemo(() => (sets ? byEvent(sets) : []), [sets]);
  const chosen = events.find((e) => e.event === event) ?? events.find((e) => e.size) ?? null;
  const chosenSet = chosen?.sets[Math.min(setIndex, chosen.sets.length - 1)] ?? null;

  // Pasted scrambles stay in this browser between visits; they never leave the device. A PDF is kept in memory only.
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
    if (tab === "paste") onChange({ lines, size, name: EVENT_NAME[size] });
    else if (chosenSet?.size) {
      onChange({ lines: chosenSet.scrambles.map((s) => ({ label: s.label, text: s.text })), size: chosenSet.size, name: `${eventName(chosen!.event)} · ${setName(chosenSet)}` });
    } else onChange({ lines: [], size: 3, name: "" });
  }, [tab, lines, size, chosen, chosenSet, onChange]);

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
      if (!first) setPdfError("This competition has no cube events this page can check yet (2×2 to 7×7).");
    } catch (err) {
      setSets(null);
      if (err instanceof PasswordError) {
        setNeedsPassword(true);
        setPdfError(err.message === "wrong password" ? "That passcode didn't open it. Check it and try again." : "");
        setTimeout(() => passwordBox.current?.focus());
      } else if (err instanceof NotTnoodleError) {
        setNeedsPassword(false);
        setPdfError(err.message === "this file isn't a PDF" ? "That file isn't a PDF." : "This isn't a TNoodle scramble PDF: no page in it has a WCA scramble set (\"3x3x3 Round 1 Scramble Set A\").");
      } else setPdfError(`Couldn't read this PDF (${(err as Error).message})`);
    } finally {
      setBusy(false);
    }
  };

  const choose = async (f: File | undefined) => {
    if (!f) return;
    const file = { name: f.name, bytes: new Uint8Array(await f.arrayBuffer()) };
    setPdf(file);
    setSets(null);
    await open(file, "");
  };

  const random = () => {
    const n = size;
    onText(Array.from({ length: 5 }, (_, i) => `${i + 1}. ${randomScramble(n).join(" ")}`).join("\n"));
    setPicked(n);
  };

  return (
    <div className="grid gap-3">
      <div role="tablist" aria-label="Where the scrambles come from" className="grid grid-cols-2 gap-1 rounded-xl bg-muted p-1">
        {(["paste", "pdf"] as const).map((t) => (
          <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)}
            className={cn("rounded-lg px-3 py-1.5 text-sm font-medium transition-colors", tab === t ? "bg-background shadow-sm" : "text-muted-foreground hover:text-foreground")}>
            {t === "paste" ? "Paste" : "TNoodle PDF"}
          </button>
        ))}
      </div>

      {tab === "paste" ? (
        <>
          <div className="flex flex-wrap items-center gap-1.5" role="radiogroup" aria-label="Cube size">
            {SIZES.map((n) => (
              <button key={n} role="radio" aria-checked={size === n} onClick={() => setPicked(n === guessed ? null : n)}
                className={cn("rounded-full border px-2.5 py-0.5 text-xs font-semibold tabular-nums transition-colors", size === n ? "border-primary/50 bg-primary/10 text-primary" : "border-border text-muted-foreground hover:text-foreground")}>
                {EVENT_NAME[n]}
              </button>
            ))}
            <span className="ml-auto text-[11px] text-muted-foreground">{picked === null && lines.length ? "from the moves" : ""}</span>
          </div>
          <Textarea
            value={text}
            onChange={(e) => onText(e.target.value)}
            placeholder={"Paste the scrambles you're about to do, one per line.\n\n" + SAMPLE}
            spellCheck={false}
            autoComplete="off"
            className="max-h-48 min-h-32 overflow-y-auto font-mono leading-relaxed"
            aria-label="Scrambles, one per line"
          />
          <div className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
            <span>{lines.length ? `${lines.length} scramble${lines.length > 1 ? "s" : ""}` : "WCA notation, as TNoodle writes it"}</span>
            <Button variant="ghost" size="sm" onClick={random} title={`Five random ${EVENT_NAME[size]} scrambles of ${EVENT_LENGTH[size]} moves, for trying the page`}>
              Random {EVENT_NAME[size]}
            </Button>
          </div>
        </>
      ) : (
        <div className="grid gap-3">
          <div className="flex items-center gap-2">
            <label className="inline-flex h-9 shrink-0 cursor-pointer items-center rounded-md border border-input bg-background px-3 text-sm font-medium shadow-xs hover:bg-accent">
              {pdf ? "Another PDF…" : "Choose a PDF…"}
              <input type="file" accept="application/pdf,.pdf" className="sr-only" onChange={(e) => void choose(e.target.files?.[0])} />
            </label>
            <span className="min-w-0 truncate text-xs text-muted-foreground">{pdf?.name ?? "a group's Computer Display PDF, or a printing PDF"}</span>
          </div>
          {needsPassword && pdf && (
            <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); void open(pdf, password); }}>
              <Input ref={passwordBox} type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Passcode or PDF password" autoComplete="off" aria-label="The PDF's passcode or password" />
              <Button type="submit" disabled={busy || !password}>{busy ? "Opening…" : "Open"}</Button>
            </form>
          )}
          {needsPassword && !pdfError && <p className="text-xs text-muted-foreground">This PDF is locked. A Computer Display PDF has its own passcode (from the Delegate); a printing PDF has the competition&rsquo;s password.</p>}
          {pdfError && <p className="text-xs leading-relaxed text-wrong">{pdfError}</p>}
          {busy && !needsPassword && <p className="text-xs text-muted-foreground">Reading the PDF…</p>}
          {sets && events.length > 0 && (
            <div className="grid grid-cols-2 gap-2">
              <label className="grid gap-1 text-xs text-muted-foreground">
                Event
                <select className={select} value={chosen?.event ?? ""} onChange={(e) => { setEvent(e.target.value); setSetIndex(0); }}>
                  {events.map((e) => (
                    <option key={e.event} value={e.event} disabled={!e.size}>{eventName(e.event)}{e.size ? "" : " (not yet)"}</option>
                  ))}
                </select>
              </label>
              <label className="grid gap-1 text-xs text-muted-foreground">
                Round and group
                <select className={select} value={setIndex} onChange={(e) => setSetIndex(Number(e.target.value))} disabled={!chosen?.size}>
                  {chosen?.sets.map((s, i) => (
                    <option key={s.title} value={i}>{setName(s)}</option>
                  ))}
                </select>
              </label>
            </div>
          )}
          <p className="text-xs leading-relaxed text-muted-foreground">The PDF is read here, in this browser. Nothing is uploaded or kept, and it&rsquo;s gone when you leave the page.</p>
        </div>
      )}
    </div>
  );
}
