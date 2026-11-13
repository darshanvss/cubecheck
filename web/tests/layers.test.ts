// Thicker outer layers on big cubes: the TypeScript must put the layers, the straightened faces' pixels and
// the sticker centres where Python does (python/cubecheck/nxn_check.layer_edges, reader.face_maps).
//   node --test tests/
import assert from "node:assert/strict";
import { test } from "node:test";
import type { Point } from "../src/lib/check.ts";
import { LAYER_MM, cellCentres, layerEdges, layerMap, measuredRatio } from "../src/lib/nxn-check.ts";

const close = (got: ArrayLike<number>, want: number[], tol = 1e-5, label = "") =>
  want.forEach((w, i) => assert.ok(Math.abs(got[i] - w) <= tol, `${label}[${i}]: ${got[i]} vs ${w}`));

test("equal layers by default, and the measured 6×6 and 7×7", () => {
  close(layerEdges(4), [0, 0.25, 0.5, 0.75, 1]);
  close(layerEdges(6, measuredRatio(6)), [0, 13.5, 23, 32.5, 42, 51.5, 65].map((x) => x / 65));
  // python: layer_edges(7, LAYER_MM[7])
  close(layerEdges(7, LAYER_MM[7][0] / LAYER_MM[7][1]), [0, 0.19403, 0.316418, 0.438806, 0.561194, 0.683582, 0.80597, 1]);
  assert.equal(measuredRatio(3), 1);
});

test("each layer gets the same number of pixels when a face is straightened", () => {
  // python: np.interp(np.arange(168) / 168, np.linspace(0, 1, 8), layer_edges(7, LAYER_MM[7]))
  const along = layerMap(layerEdges(7, LAYER_MM[7][0] / LAYER_MM[7][1]), 168);
  close([0, 1, 23, 24, 25, 100, 167].map((x) => along[x]), [0, 0.008085, 0.185945, 0.19403, 0.199129, 0.581592, 0.991915], 1e-6);
  close(layerMap(layerEdges(5), 120), Array.from({ length: 120 }, (_, x) => x / 120), 1e-12, "even");
});

test("sticker centres move with the layers", () => {
  const kp: Point[] = [[100, 95], [60, 90], [100, 30], [160, 60], [160, 160], [100, 190], [100, 100]];
  const got = cellCentres(kp, 6, layerEdges(6, measuredRatio(6)));
  // python: cell_centres(kp, 6, layer_edges(6, LAYER_MM[6]))[[0, 7, 40, 107]]
  const want = [[100, 100.617], [100, 102.934], [131.336, 74.262], [153.156, 153.156]];
  [0, 7, 40, 107].forEach((m, i) => close(got[m], want[i], 2e-3, `centre ${m}`));
  assert.deepEqual(cellCentres(kp, 6, layerEdges(6)).map((p) => p.map((v) => v.toFixed(6))), cellCentres(kp, 6).map((p) => p.map((v) => v.toFixed(6))));
});
