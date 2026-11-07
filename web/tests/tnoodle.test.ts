// Reading TNoodle PDFs.
//
// Always run: pages built here in the layout TNoodle uses (a title, numbers in the left column, each
// scramble beside its number and wrapped over lines), and files that must be refused.
//
// Run against a real TNoodle competition when one is given: TNOODLE_FIXTURE=<folder> points at a
// competition's unzipped TNoodle output (Computer Display PDFs/, Printing/, Interchange/ and the
// "... Passcodes - SECRET.txt" file), and TNOODLE_PASSWORD is its PDF password. The Interchange JSON
// (not locked) is the answer key. No competition is kept in this repository: scrambles are secret.
//   TNOODLE_FIXTURE=~/tnoodle/my-test-comp TNOODLE_PASSWORD=... npm test
import assert from "node:assert/strict";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { NotTnoodleError, PasswordError, byEvent, readTnoodlePdf, type PdfJs } from "../src/lib/tnoodle.ts";
import { cube } from "../src/lib/nxn.ts";

const pdfjs = (await import("pdfjs-dist/legacy/build/pdf.mjs")) as unknown as PdfJs & { GlobalWorkerOptions: { workerSrc: string } };
pdfjs.GlobalWorkerOptions.workerSrc = new URL("../node_modules/pdfjs-dist/legacy/build/pdf.worker.mjs", import.meta.url).href;

// --- pages made here ---

/** A one-page PDF with lines of text at (x, y) points (xref offsets worked out, so pdf.js reads it without repairs). */
function makePdf(lines: [number, number, string][]): Uint8Array {
  const text = lines.map(([x, y, s]) => `BT /F1 12 Tf ${x} ${y} Td (${s}) Tj ET`).join("\n");
  const objects = [
    "<< /Type /Catalog /Pages 2 0 R >>",
    "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
    `<< /Length ${text.length} >>\nstream\n${text}\nendstream`,
    "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
  ];
  let out = "%PDF-1.4\n";
  const at = objects.map((o, i) => {
    const offset = out.length;
    out += `${i + 1} 0 obj\n${o}\nendobj\n`;
    return offset;
  });
  const xref = out.length;
  out += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n${at.map((o) => `${String(o).padStart(10, "0")} 00000 n \n`).join("")}`;
  out += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
  return new TextEncoder().encode(out);
}

test("a page laid out as TNoodle does gives its scrambles, a wrapped one joined up", async () => {
  const one = "R2 D' B2 U2 L2 F2 D R2 U' L2 B2 R' F U' L' D2 F' R' B D2 R'";
  const long = ["Rw2 U' Fw R2 Uw' B L' Fw2 D Rw' U2 F'", "Uw2 R B2 Fw' L2 D' Rw U"]; // wrapped over two lines
  const sets = await readTnoodlePdf(pdfjs, makePdf([
    [40, 740, "Practice 2026 2026-11-07"],
    [40, 720, "3x3x3 Round 1 Scramble Set A"],
    [40, 680, "1"], [70, 680, one],
    [40, 650, "2"], [70, 650, "U F2 R"],
    [40, 610, "E1"], [70, 610, "L' B D2"],
  ]));
  assert.deepEqual(sets.map((s) => [s.event, s.round, s.group, s.size]), [["3x3x3", 1, "A", 3]]);
  assert.deepEqual(sets[0].scrambles.map((s) => [s.label, s.extra, s.text]), [["1", false, one], ["2", false, "U F2 R"], ["E1", true, "L' B D2"]]);

  const four = await readTnoodlePdf(pdfjs, makePdf([[40, 720, "4x4x4 Round 2"], [40, 670, "1"], [70, 680, long[0]], [70, 660, long[1]]]));
  assert.deepEqual(four.map((s) => [s.event, s.round, s.group, s.size, s.title]), [["4x4x4", 2, "A", 4, "4x4x4 Round 2"]]);
  assert.equal(four[0].scrambles[0].text, long.join(" "));
  cube(4).parse(four[0].scrambles[0].text);
});

test("a PDF that isn't a TNoodle scramble sheet is refused, and so is a file that isn't a PDF", async () => {
  await assert.rejects(readTnoodlePdf(pdfjs, makePdf([[72, 700, "Minutes of the club meeting, Round 2"]])), NotTnoodleError);
  // A page that says "Round 1" but isn't a WCA event's scramble set.
  await assert.rejects(readTnoodlePdf(pdfjs, makePdf([[40, 720, "Quiz Round 1"], [40, 680, "1"], [70, 680, "R U"]])), NotTnoodleError);
  await assert.rejects(readTnoodlePdf(pdfjs, new TextEncoder().encode("just some text")), NotTnoodleError);
});

// --- a real TNoodle competition, when one is given ---

const dir = process.env.TNOODLE_FIXTURE ? path.resolve(process.env.TNOODLE_FIXTURE.replace(/^~/, process.env.HOME ?? "~")) : "";
const skip = dir && existsSync(dir) ? false : "set TNOODLE_FIXTURE to a TNoodle competition's unzipped folder";
const bytes = (...parts: string[]) => new Uint8Array(readFileSync(path.join(dir, ...parts)));
const EVENT_IDS: Record<string, string> = { "2x2x2": "222", "3x3x3": "333", "4x4x4": "444", "5x5x5": "555", "6x6x6": "666", "7x7x7": "777" };
const keyOf = (s: { event: string; round: number; group: string }) => `${EVENT_IDS[s.event]}-r${s.round}-${s.group}`;

/** "333-r1-A" -> TNoodle's scrambles and extras for that set (cube events). */
function answerKey(): Map<string, string[]> {
  const json = readdirSync(path.join(dir, "Interchange")).find((f) => f.endsWith(".json"))!;
  const wcif = JSON.parse(readFileSync(path.join(dir, "Interchange", json), "utf8")).wcif;
  const key = new Map<string, string[]>();
  for (const event of wcif.events)
    for (const round of event.rounds)
      round.scrambleSets.forEach((set: { scrambles: string[]; extraScrambles: string[] }, i: number) => {
        key.set(`${event.id}-r${round.id.split("-r")[1]}-${String.fromCharCode(65 + i)}`, [...set.scrambles, ...set.extraScrambles].map((s) => s.split(/\s+/).join(" ")));
      });
  return key;
}
const passcodes = () => {
  const secret = readdirSync(dir).find((f) => f.includes("Passcodes - SECRET") && !f.includes("ORDERED"))!;
  return [...readFileSync(path.join(dir, secret), "utf8").matchAll(/^(.+?): (\w{8})\s*$/gm)].map(([, title, code]) => ({ title, code }));
};

test("every cube Computer Display PDF, unlocked with its own passcode, gives exactly TNoodle's scrambles", { skip }, async () => {
  const key = answerKey();
  for (const { title, code } of passcodes().filter((p) => /^\dx\dx\d Round/.test(p.title))) {
    const sets = await readTnoodlePdf(pdfjs, bytes("Computer Display PDFs", `${title}.pdf`), code);
    assert.equal(sets.length, 1, title);
    assert.ok(title.startsWith(sets[0].title), title); // a round with one group says just "2x2x2 Round 2"
    assert.deepEqual(sets[0].scrambles.map((s) => s.text), key.get(keyOf(sets[0])), title);
    for (const s of sets[0].scrambles) cube(sets[0].size!).parse(s.text); // every scramble takes, at the title's size
  }
});

test("the All Scrambles printing PDF gives every cube set, grouped by event", { skip: skip || (!process.env.TNOODLE_PASSWORD && "set TNOODLE_PASSWORD") }, async () => {
  const key = answerKey();
  const all = readdirSync(path.join(dir, "Printing")).find((f) => f.endsWith("All Scrambles.pdf"))!;
  const sets = (await readTnoodlePdf(pdfjs, bytes("Printing", all), process.env.TNOODLE_PASSWORD)).filter((s) => s.size && s.event in EVENT_IDS);
  assert.ok(sets.length > 0);
  for (const set of sets) assert.deepEqual(set.scrambles.map((s) => s.text), key.get(keyOf(set)), set.title);
  const events = byEvent(sets).map((e) => e.event);
  assert.deepEqual(events, ["3x3x3", "2x2x2", "4x4x4", "5x5x5", "6x6x6", "7x7x7"].filter((e) => events.includes(e)));
});

test("a missing or wrong passcode is refused", { skip }, async () => {
  const { title } = passcodes()[0];
  const file = bytes("Computer Display PDFs", `${title}.pdf`);
  await assert.rejects(readTnoodlePdf(pdfjs, file.slice()), PasswordError);
  await assert.rejects(readTnoodlePdf(pdfjs, file.slice(), "wrongpass"), /wrong password/);
});
