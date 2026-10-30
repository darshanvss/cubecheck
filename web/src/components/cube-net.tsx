"use client";

import { useEffect, useRef } from "react";

// Sticker colours in the WCA scheme (U white, R red, F green, D yellow, L orange, B blue); "?" = not seen.
export const STICKER: Record<string, string> = { U: "#f5f5f5", R: "#dc2626", F: "#16a34a", D: "#facc15", L: "#f97316", B: "#2563eb", "?": "#3f3f46" };
const AT: Record<string, [number, number]> = { U: [1, 0], L: [0, 1], F: [1, 1], R: [2, 1], B: [3, 1], D: [1, 2] };
const S = 24, GAP = 6, BLOCK = 3 * S + GAP;

/** The cube laid out flat (54 letters in U R F D L B order); flagged stickers get a red outline. */
export function CubeNet({ letters, flagged = new Set<number>(), label }: { letters: string; flagged?: Set<number>; label: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current!, dpr = devicePixelRatio || 1;
    canvas.width = (4 * BLOCK - GAP) * dpr;
    canvas.height = (3 * BLOCK - GAP) * dpr;
    const c = canvas.getContext("2d")!;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    [..."URFDLB"].forEach((face, fi) => {
      const [bx, by] = AT[face];
      for (let k = 0; k < 9; k++) {
        const idx = fi * 9 + k, x = bx * BLOCK + (k % 3) * S, y = by * BLOCK + Math.floor(k / 3) * S;
        c.fillStyle = STICKER[letters[idx]] ?? STICKER["?"];
        c.beginPath();
        c.roundRect(x + 1, y + 1, S - 2, S - 2, 3);
        c.fill();
        if (flagged.has(idx)) {
          c.strokeStyle = "#ef4444";
          c.lineWidth = 3;
          c.strokeRect(x + 2, y + 2, S - 4, S - 4);
        }
      }
    });
  }, [letters, flagged]);
  return <canvas ref={ref} role="img" aria-label={label} className="block h-auto w-full" />;
}
