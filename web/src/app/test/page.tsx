"use client";

// Runs the three networks in this browser on the test renders Python checked (python/scripts/web_reference.py)
// and compares every answer, plus the TypeScript check's verdicts. Not linked from the site.
import { useEffect, useState } from "react";
import { expectedState, insideImage, singleSlips, verifyProbs, type Point } from "@/lib/check";
import { findCube, framePixels, loadModels, readStickers } from "@/lib/vision";

interface Reference {
  image: string;
  finder: Point[];
  corners: Point[] | null;
  probs?: number[][];
  verdicts?: Record<string, { scramble: string; status: string; observed: string }>;
}

interface Row {
  image: string;
  finderOff: number;
  cornersOff: number;
  gap: number;
  same: number;
  verdicts: number;
  ms: number;
}

const off = (a: Point[], b: Point[]) => Math.max(...a.map((p, k) => Math.hypot(p[0] - b[k][0], p[1] - b[k][1])));
const argmax = (row: ArrayLike<number>) => Array.from(row).indexOf(Math.max(...Array.from(row)));
const median = (a: number[]) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];

let started = false; // React runs effects twice in development; run the comparison once

export default function Test() {
  const [rows, setRows] = useState<Row[]>([]);
  const [summary, setSummary] = useState("Loading models…");

  useEffect(() => {
    if (started) return;
    started = true;
    (async () => {
      const backend = (new URLSearchParams(location.search).get("backend") as "webgpu" | "wasm" | null) ?? undefined;
      const models = await loadModels("/models/", backend);
      const reference: Reference[] = await (await fetch("/check/reference.json", { cache: "no-cache" })).json();
      const out: Row[] = [];
      for (const ref of reference) {
        const img = await createImageBitmap(await (await fetch(`/check/${ref.image}`)).blob());
        const t0 = performance.now();
        const found = await findCube(models, img);
        const probs = found.corners && (await readStickers(models, framePixels(img), found.corners));
        const ms = performance.now() - t0;
        const row: Row = { image: ref.image, finderOff: off(found.steps[0].corners, ref.finder), cornersOff: NaN, gap: NaN, same: 0, verdicts: 0, ms };
        if (found.corners && ref.corners && probs && ref.probs) {
          row.cornersOff = off(found.corners, ref.corners);
          row.gap = 0;
          ref.probs.forEach((want, n) => {
            want.forEach((p, k) => (row.gap = Math.max(row.gap, Math.abs(p - probs[7 * n + k]))));
            row.same += Number(argmax(probs.slice(7 * n, 7 * n + 7)) === argmax(want));
          });
          // The whole check in the browser: these probabilities and corners, the TypeScript verdict.
          const inside = insideImage(img.width, img.height, found.corners);
          for (const want of Object.values(ref.verdicts ?? {})) {
            const got = verifyProbs(probs, inside, expectedState(want.scramble), singleSlips(want.scramble));
            row.verdicts += Number(got.status === want.status);
          }
        }
        out.push(row);
        setRows([...out]);
        setSummary(`Running on ${models.backend}… ${out.length}/${reference.length}`);
      }
      const done = out.filter((r) => !Number.isNaN(r.cornersOff));
      const same = done.reduce((s, r) => s + r.same, 0), verdicts = done.reduce((s, r) => s + r.verdicts, 0);
      setSummary(
        `${models.backend}: corners off by median ${median(done.map((r) => r.cornersOff)).toFixed(2)} px (at most ${Math.max(...done.map((r) => r.cornersOff)).toFixed(2)}) · ` +
          `stickers read the same ${same}/${27 * done.length} · verdicts the same ${verdicts}/${2 * done.length} · median ${median(out.slice(1).map((r) => r.ms)).toFixed(0)} ms per render`,
      );
    })().catch((err) => setSummary(`Failed: ${err?.message ?? err}`));
  }, []);

  return (
    <div className="mx-auto w-full max-w-4xl px-5 py-8">
      <h1 className="text-xl font-semibold">Browser model check</h1>
      <p className="mt-1 text-sm text-muted-foreground">The three networks and the check, in this browser, against Python&rsquo;s answers on the same test renders. Add ?backend=wasm to use the CPU.</p>
      <p id="summary" className="mt-4 font-medium">{summary}</p>
      <table className="mt-4 w-full text-right text-sm tabular-nums">
        <thead className="text-muted-foreground">
          <tr><th className="text-left font-normal">Render</th><th className="font-normal">Finder off (px)</th><th className="font-normal">Corners off (px)</th><th className="font-normal">Largest prob. gap</th><th className="font-normal">Same stickers</th><th className="font-normal">Same verdicts</th><th className="font-normal">ms</th></tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.image} className="border-t border-border">
              <td className="py-1 text-left">{r.image}</td><td>{r.finderOff.toFixed(2)}</td><td>{r.cornersOff.toFixed(2)}</td><td>{r.gap.toFixed(3)}</td>
              <td className={r.same === 27 ? "text-correct" : "text-wrong"}>{r.same}/27</td><td className={r.verdicts === 2 ? "text-correct" : "text-wrong"}>{r.verdicts}/2</td><td>{r.ms.toFixed(0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
