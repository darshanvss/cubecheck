"use client";

import { poly, type PolyEvent } from "@/lib/poly";

const UNSEEN = "#3f3f46";

/**
 * A skewb, pyraminx or FTO laid out flat (python/cubecheck/poly_net.py): a skewb as a cube's cross, a
 * pyraminx as one big triangle, an FTO as its two halves seen straight on. `colours` is the face number on
 * each sticker place (-1: not seen); flagged places get a red outline.
 */
export function PolyNet({ event, colours, flagged = new Set<number>(), label }: { event: PolyEvent; colours: number[]; flagged?: Set<number>; label: string }) {
  const { net, css } = poly(event).d, pad = 0.06 * Math.max(net.width, net.height);
  const points = (q: number[][]) => q.map(([x, y]) => `${x.toFixed(3)},${y.toFixed(3)}`).join(" ");
  const stroke = 0.012 * Math.max(net.width, net.height);
  return (
    <svg role="img" aria-label={label} viewBox={`${-pad} ${-pad} ${net.width + 2 * pad} ${net.height + 2 * pad}`} className="block h-auto max-h-28 w-full">
      {net.polys.map((q, i) => (
        <polygon key={i} points={points(q)} fill={colours[i] >= 0 ? css[colours[i]] : UNSEEN} stroke="#000" strokeWidth={stroke} strokeLinejoin="round" />
      ))}
      {net.faces.map((q, i) => <polygon key={`f${i}`} points={points(q)} fill="none" stroke="#000" strokeWidth={2 * stroke} strokeLinejoin="round" />)}
      {net.polys.map((q, i) => flagged.has(i) && <polygon key={`x${i}`} points={points(q)} fill="none" stroke="#ef4444" strokeWidth={3 * stroke} strokeLinejoin="round" />)}
    </svg>
  );
}
