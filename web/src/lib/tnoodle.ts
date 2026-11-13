// Scrambles out of TNoodle PDFs (the WCA's official scramble program), read entirely on this device.
//
// TNoodle's PDFs (a group's Computer Display PDF, a group's printing PDF, or the whole competition's
// "All Scrambles" PDF) carry each scramble as text, so no pictures are needed: unlock the PDF with its
// passcode, read the text with its positions, and pick the scrambles out. Nothing is sent anywhere.
//
// Each page starts with a title like "3x3x3 Round 1 Scramble Set A", then the scrambles numbered 1, 2, ...
// and extras E1, E2, ..., long ones wrapped over several lines. A PDF with no such page isn't a TNoodle
// scramble sheet, and is refused. Fewest Moves sheets are laid out differently: one scramble per page,
// under a line that says "Scramble". Other puzzles' moves (Pyraminx tips, Megaminx R++, Square-1 (1,0) /)
// are read too, so every event shows; only the N×N cubes can be checked.

const TITLE = /^(.+?) Round (\d+) Scramble Set ([A-Z]+)(?: Attempt (\d+))?$/;
const ONE_GROUP = /^(.+?) Round (\d+)()(?: Attempt (\d+))?$/; // a round with one group: "2x2x2 Round 2", no "Scramble Set A"
const LABEL = /^(E?\d+)$/;
const DATE = /\b\d{4}-\d{2}-\d{2}\b/g;
const MOVE = /^(?:(?:\d*[URFDLB]w?|[xyz]|[urlb])(?:2'?|')?|[RD](?:\+\+|--)|\/|\(-?\d+,-?\d+\)\/?)$/;
const CUBE_EVENT = /^(\d)x\1x\1(?: |$)/; // "3x3x3", "4x4x4 Blindfolded", ...
// Only a WCA event's name, so a competition called "... Round 2" isn't taken for the title.
const CUBE_OR_PUZZLE = /^(\dx\dx\d( Blindfolded| One-Handed| Fewest Moves| Multiple Blindfolded)?|Fewest Moves|Megaminx|Pyraminx|Skewb|Square-1) Round/;
const FMC = /Fewest Moves$/; // a Fewest Moves sheet's title says just "Fewest Moves Round 1 ..."
const FMC_ATTEMPT = /^Scramble (\d+) of \d+$/;

export interface TnoodleScramble {
  label: string; // "1", "2", ... or "E1" for an extra
  extra: boolean;
  text: string;
}

export interface ScrambleSet {
  title: string; // "3x3x3 Round 1 Scramble Set A"
  event: string; // "3x3x3", "4x4x4 Blindfolded", "Pyraminx", ...
  round: number;
  group: string; // "A", "B", ...
  attempt: number | null; // Fewest Moves and Multi-Blind sheets: which attempt
  size: number | "skewb" | "pyram" | null; // the cube's N for N×N events (Fewest Moves is a 3×3), the puzzle for skewb and pyraminx (experimental), null for others
  scrambles: TnoodleScramble[];
}

export class PasswordError extends Error {}
export class NotTnoodleError extends Error {}

interface TextItem {
  str: string;
  transform: number[];
  width?: number;
}

interface Row {
  y: number;
  x: number;
  tokens: string[];
  near: string[];
}

/** Rows of text, top to bottom: y, x (leftmost), tokens, near (tokens before a wide gap). */
function rows(items: TextItem[]): Row[] {
  const out: { y: number; parts: { x: number; w: number; s: string }[] }[] = [];
  for (const it of items) {
    if (!it.str?.trim()) continue;
    const [x, y] = [it.transform[4], it.transform[5]];
    let row = out.find((r) => Math.abs(r.y - y) < 2);
    if (!row) out.push((row = { y, parts: [] }));
    row.parts.push({ x, w: it.width || 0, s: it.str.replace(/[’‘]/g, "'") }); // a curly prime is still a prime
  }
  return out
    .sort((a, b) => b.y - a.y)
    .map((r) => {
      r.parts.sort((a, b) => a.x - b.x);
      // For move lines, a wide gap ends the line: beyond it is the puzzle drawing (Megaminx's face labels).
      const cut = r.parts.findIndex((p, i) => i && p.x - (r.parts[i - 1].x + r.parts[i - 1].w) > 40);
      const words = (parts: typeof r.parts) => parts.map((p) => p.s).join(" ").split(/\s+/).filter(Boolean);
      return { y: r.y, x: r.parts[0].x, tokens: words(r.parts), near: words(cut > 0 ? r.parts.slice(0, cut) : r.parts) };
    });
}

/**
 * Give each number the run of move lines it sits beside. In the PDF the number is centred vertically on
 * its scramble, which may wrap over several lines, so the runs are chosen so that every number is as
 * close as possible to the middle of its own run (a small dynamic programme over where the runs split).
 */
function assign(labels: { y: number }[], lines: { y: number; moves: string[] }[]) {
  const n = labels.length, m = lines.length;
  const cost = (k: number, i: number, j: number) => (labels[k].y - (lines[i].y + lines[j].y) / 2) ** 2;
  const best = Array.from({ length: n + 1 }, () => new Array<number>(m + 1).fill(Infinity));
  const from = Array.from({ length: n + 1 }, () => new Array<number>(m + 1).fill(-1));
  best[0][0] = 0;
  for (let k = 1; k <= n; k++)
    for (let j = k; j <= m; j++)
      for (let i = k - 1; i < j; i++) {
        const c = best[k - 1][i] + cost(k - 1, i, j - 1);
        if (c < best[k][j]) [best[k][j], from[k][j]] = [c, i];
      }
  const runs: (typeof lines)[] = [];
  for (let k = n, j = m; k > 0; j = from[k][j], k--) runs.unshift(lines.slice(from[k][j], j));
  return runs;
}

/** One page's pdf.js text items -> a scramble set, or null when the page isn't one. */
export function parsePage(items: TextItem[]): ScrambleSet | null {
  const all = rows(items);
  const texts = all.map((row) => row.tokens.join(" ").replace(DATE, "").trim()); // the date sits on the title's line
  const text = texts.find((t) => TITLE.test(t)) || texts.find((t) => ONE_GROUP.test(t) && CUBE_OR_PUZZLE.test(t));
  const title = text ? TITLE.exec(text) || ONE_GROUP.exec(text) : null;
  if (!text || !title) return null;
  const [, event, round, letter, attempt] = title;
  const n = CUBE_EVENT.exec(event);
  const set: ScrambleSet = {
    title: text, event, round: Number(round), group: letter || "A", attempt: attempt ? Number(attempt) : null,
    size: n ? Number(n[1]) : FMC.test(event) ? 3 : event === "Skewb" ? "skewb" : event === "Pyraminx" ? "pyram" : null, scrambles: [],
  };
  if (FMC.test(set.event)) return fewestMoves(set, all);
  // Numbers (1, 2, ..., E1, E2) stand alone in the left column; move lines hold nothing but moves.
  const labels: { y: number; label: string }[] = [], lines: { y: number; moves: string[] }[] = [];
  for (const row of all) {
    for (const part of row.tokens) if (LABEL.test(part) && row.x < 60 && row.tokens[0] === part) labels.push({ y: row.y, label: part });
    const moves = row.near[0] && LABEL.test(row.near[0]) && row.x < 60 ? row.near.slice(1) : row.near;
    // Move lines start in the left column; letters further right are the puzzle drawing's face labels (Megaminx).
    if (moves.length && row.x < 120 && moves.every((t) => MOVE.test(t))) lines.push({ y: row.y, moves });
  }
  if (!labels.length || lines.length < labels.length) return null;
  const runs = assign(labels, lines);
  set.scrambles = labels.map((l, k) => ({ label: l.label, extra: l.label.startsWith("E"), text: runs[k].flatMap((line) => line.moves).join(" ") }));
  return set;
}

/** A Fewest Moves sheet: the one scramble under the "Scramble" line, labelled with its attempt. */
function fewestMoves(set: ScrambleSet, all: Row[]): ScrambleSet | null {
  const at = all.findIndex((row) => row.tokens.join(" ") === "Scramble");
  const line = at >= 0 ? all[at + 1] : null;
  if (!line || !line.tokens.length || !line.tokens.every((t) => MOVE.test(t))) return null;
  const n = all.map((row) => FMC_ATTEMPT.exec(row.tokens.join(" "))).find(Boolean);
  set.scrambles = [{ label: n ? n[1] : String(set.attempt || 1), extra: false, text: line.tokens.join(" ") }];
  return set;
}

// The part of pdf.js used here, so this file doesn't care where pdf.js came from (the page or Node's tests).
export interface PdfJs {
  getDocument(src: { data: Uint8Array; password?: string; isEvalSupported?: boolean; verbosity?: number }): {
    promise: Promise<{ numPages: number; getPage(n: number): Promise<{ getTextContent(): Promise<{ items: unknown[] }> }> }>;
    destroy(): Promise<void>;
  };
  PasswordResponses: { NEED_PASSWORD: number; INCORRECT_PASSWORD: number };
}

/**
 * Every scramble set in a TNoodle PDF, in the PDF's order. Throws PasswordError when it's locked and the
 * password is missing or wrong, NotTnoodleError when it isn't a TNoodle scramble PDF at all.
 */
export async function readTnoodlePdf(pdfjs: PdfJs, data: Uint8Array, password = ""): Promise<ScrambleSet[]> {
  const task = pdfjs.getDocument({ data, password, isEvalSupported: false, verbosity: 0 });
  let doc;
  try {
    doc = await task.promise;
  } catch (err) {
    const e = err as { name?: string; code?: number };
    if (e?.name === "PasswordException") {
      throw new PasswordError(e.code === pdfjs.PasswordResponses.INCORRECT_PASSWORD ? "wrong password" : "this PDF needs a password");
    }
    if (e?.name === "InvalidPDFException") throw new NotTnoodleError("this file isn't a PDF");
    throw err;
  }
  const sets = new Map<string, ScrambleSet>();
  for (let p = 1; p <= doc.numPages; p++) {
    const page = await doc.getPage(p);
    const set = parsePage((await page.getTextContent()).items as TextItem[]);
    if (!set) continue;
    const seen = sets.get(set.title); // a long set can run over two pages
    if (seen) seen.scrambles.push(...set.scrambles);
    else sets.set(set.title, set);
  }
  await task.destroy();
  if (!sets.size) throw new NotTnoodleError("no TNoodle scramble sets in this PDF");
  return [...sets.values()];
}

/** pdf.js, from this site (public/pdfjs, copied by scripts/copy-ort.mjs), loaded the first time a PDF is opened. */
let loading: Promise<PdfJs> | null = null;
export function loadPdfJs(): Promise<PdfJs> {
  loading ??= import(/* webpackIgnore: true */ /* turbopackIgnore: true */ "/pdfjs/pdf.min.mjs" as string).then((m) => {
    m.GlobalWorkerOptions.workerSrc = "/pdfjs/pdf.worker.min.mjs";
    return m as PdfJs;
  });
  return loading;
}

// --- a competition's sets, by event, round and group, for picking one ---

export interface EventSets {
  event: string;
  size: ScrambleSet["size"]; // null: not a puzzle this page can check yet
  sets: ScrambleSet[];
}

/** The sets grouped by event, in the order the WCA lists events (then any others, as found). */
export function byEvent(sets: ScrambleSet[]): EventSets[] {
  const order = ["3x3x3", "2x2x2", "4x4x4", "5x5x5", "6x6x6", "7x7x7", "3x3x3 Blindfolded", "Fewest Moves", "3x3x3 One-Handed", "Clock", "Megaminx", "Pyraminx", "Skewb", "Square-1", "4x4x4 Blindfolded", "5x5x5 Blindfolded", "3x3x3 Multiple Blindfolded"];
  const out = new Map<string, EventSets>();
  for (const s of sets) {
    if (!out.has(s.event)) out.set(s.event, { event: s.event, size: s.size, sets: [] });
    out.get(s.event)!.sets.push(s);
  }
  const rank = (e: string) => (order.includes(e) ? order.indexOf(e) : order.length);
  return [...out.values()].sort((a, b) => rank(a.event) - rank(b.event));
}

/** An event as people write it: "3x3x3" -> "3×3", "4x4x4 Blindfolded" -> "4×4 Blindfolded", "Fewest Moves" -> "3×3 Fewest Moves". */
export const eventName = (event: string) =>
  event.replace(/^Fewest Moves$/, "3x3x3 Fewest Moves").replace(/ Multiple Blindfolded$/, " Multi-Blind").replace(/^(\d)x\1x\1/, "$1×$1");

/** A short name for a set: "Round 1 · Group A", "Round 2" (a round of one group says no group), "Round 1 · Group A · Attempt 2". */
export const setName = (s: ScrambleSet) =>
  [`Round ${s.round}`, / Scramble Set /.test(s.title) ? `Group ${s.group}` : "", s.attempt ? `Attempt ${s.attempt}` : ""].filter(Boolean).join(" · ");
