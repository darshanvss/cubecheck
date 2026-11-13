// The corner model and sticker reader in the browser, with ONNX Runtime Web: on the GPU through
// WebGPU where the browser has it, otherwise on the CPU. Each step mirrors python/cubecheck
// (corners.py, reader.py); the /test page checks the answers against Python's. There is one sticker
// reader file per cube size (reader-N.onnx), loaded the first time that size is read.
import type * as Ort from "onnxruntime-web";
import { FACE_QUADS, squareToQuad, type Point } from "./check";
import { faceWindings, layerMap } from "./nxn-check";

type OrtModule = typeof Ort;
declare global {
  interface Window {
    ort?: OrtModule;
  }
}

export interface Meta {
  corners: string;
  reader: string;
  in_px: number;
  sticker_px: number; // each face is straightened to this many pixels per sticker
  sizes: number[]; // cube sizes with a reader
  refine_margin: number;
  passes: number;
  min_present: number;
  min_find: number; // the finder's own bar on the whole frame
  zoom: number; // when the finder is under its bar, look again this much closer (0: don't)
  two_faces: boolean; // the corner model was trained on views of exactly two faces
  classes: string;
}

export interface Models {
  ort: OrtModule;
  meta: Meta;
  finder: Ort.InferenceSession;
  refiner: Ort.InferenceSession;
  reader: (n: number) => Promise<Ort.InferenceSession>; // the sticker reader for an N×N cube
  backend: "webgpu" | "wasm";
}

// ONNX Runtime is served from this site (public/ort, copied by scripts/copy-ort.mjs) and loaded once.
let runtime: Promise<OrtModule> | null = null;
function loadRuntime(): Promise<OrtModule> {
  runtime ??= new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "/ort/ort.webgpu.min.js";
    script.onload = () => {
      const ort = window.ort!;
      ort.env.wasm.wasmPaths = "/ort/";
      ort.env.wasm.numThreads = 1;
      resolve(ort);
    };
    script.onerror = () => reject(new Error("couldn't load ONNX Runtime"));
    document.head.append(script);
  });
  return runtime;
}

// One set of sessions per page: two sets running on the same GPU device get mixed up ("Session mismatch").
const loaded = new Map<string, Promise<Models>>();

/** backend: the device's GPU through WebGPU where the browser has it (the default), else the CPU. */
export function loadModels(base = "/models/", backend?: Models["backend"]): Promise<Models> {
  const chosen = backend ?? (Reflect.has(navigator, "gpu") ? "webgpu" : "wasm");
  const key = `${base}|${chosen}`;
  if (!loaded.has(key)) loaded.set(key, openModels(base, chosen));
  return loaded.get(key)!;
}

async function openModels(base: string, backend: Models["backend"]): Promise<Models> {
  const ort = await loadRuntime();
  const meta: Meta = await (await fetch(`${base}meta.json`, { cache: "no-cache" })).json();
  const options = { executionProviders: backend === "webgpu" ? ["webgpu", "wasm"] : ["wasm"] };
  const open = async (name: string) => {
    // "no-cache" asks the server whether the file changed, so a new export is never hidden by an old copy.
    const bytes = await (await fetch(`${base}${name}.onnx`, { cache: "no-cache" })).arrayBuffer();
    return ort.InferenceSession.create(new Uint8Array(bytes), options);
  };
  // One at a time: creating them together while the WebGPU backend starts up fails.
  const finder = await open("finder"), refiner = await open("refiner");
  const readers = new Map<number, Promise<Ort.InferenceSession>>();
  const reader = (n: number) => {
    if (!meta.sizes.includes(n)) return Promise.reject(new Error(`no sticker reader for ${n}×${n}`));
    if (!readers.has(n)) readers.set(n, open(`reader-${n}`));
    return readers.get(n)!;
  };
  await reader(3);
  return { ort, meta, finder, refiner, reader, backend };
}

// --- crops: a square of `side` pixels around a centre, scaled onto the network's input ---

export interface View {
  cx: number;
  cy: number;
  side: number;
}

const fullView = (w: number, h: number): View => ({ cx: w / 2, cy: h / 2, side: Math.max(w, h) });
const mean = (pts: Point[]): Point => [pts.reduce((s, p) => s + p[0], 0) / pts.length, pts.reduce((s, p) => s + p[1], 0) / pts.length];

function refineView(corners: Point[], margin: number): View {
  const xs = corners.map((p) => p[0]), ys = corners.map((p) => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  return { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, side: Math.max(x1 - x0, y1 - y0) * margin };
}

type Source = CanvasImageSource & { width: number; height: number };

// Draw the view into a size x size canvas, grey (114) outside the frame, and return RGB in 0..1, channels first.
// OpenCV puts pixel centres on whole numbers and the canvas puts them on halves, hence the half-pixel shift.
function cut(source: CanvasImageSource, v: View, size: number, canvas: HTMLCanvasElement): Float32Array {
  const s = size / v.side, c = canvas.getContext("2d", { willReadFrequently: true })!;
  canvas.width = canvas.height = size;
  c.setTransform(1, 0, 0, 1, 0, 0);
  c.fillStyle = "rgb(114, 114, 114)";
  c.fillRect(0, 0, size, size);
  c.imageSmoothingEnabled = true;
  c.imageSmoothingQuality = "high";
  c.setTransform(s, 0, 0, s, size / 2 - v.cx * s + 0.5 * (1 - s), size / 2 - v.cy * s + 0.5 * (1 - s));
  c.drawImage(source, 0, 0);
  const rgba = c.getImageData(0, 0, size, size).data, n = size * size, out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    out[i] = rgba[4 * i] / 255;
    out[n + i] = rgba[4 * i + 1] / 255;
    out[2 * n + i] = rgba[4 * i + 2] / 255;
  }
  return out;
}

export interface Step {
  net: "finder" | "refiner";
  corners: Point[];
  present: number;
  view: View;
}

async function runCorners(models: Models, net: Ort.InferenceSession, source: CanvasImageSource, v: View, canvas: HTMLCanvasElement): Promise<Omit<Step, "net">> {
  const size = models.meta.in_px;
  const image = new models.ort.Tensor("float32", cut(source, v, size, canvas), [1, 3, size, size]);
  const out = await net.run({ image });
  const xy = out.corners.data as Float32Array, s = size / v.side;
  const corners: Point[] = [];
  for (let k = 0; k < 7; k++) corners.push([(xy[2 * k] * size - size / 2) / s + v.cx, (xy[2 * k + 1] * size - size / 2) / s + v.cy]);
  return { corners, present: (out.present.data as Float32Array)[0], view: v };
}

/**
 * The three faces are convex quads, all turning the same way, as any corner view of a real cube gives.
 * twoFaces also accepts a view of exactly two faces (looking at an edge): those two convex and turning
 * the same way, the third seen from behind (turning the other way) or edge-on (almost no area).
 */
export function cubeShaped(kp: Point[], twoFaces = false): boolean {
  const turns = FACE_QUADS.map((quad) => {
    const pts = quad.map((j) => kp[j]);
    return [0, 1, 2, 3].map((i) => {
      const [a, b, c] = [pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]];
      return (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
    });
  });
  const all = turns.flat();
  if (all.every((t) => t > 0) || all.every((t) => t < 0)) return true;
  if (!twoFaces) return false;
  const sign = turns.map((t) => (t.every((v) => v > 0) ? 1 : t.every((v) => v < 0) ? -1 : 0));
  const area = faceWindings(kp).map(Math.abs);
  return [0, 1, 2].some((third) => {
    const [a, b] = [0, 1, 2].filter((f) => f !== third);
    return sign[a] !== 0 && sign[a] === sign[b] && sign[third] !== sign[a] && Math.min(area[a], area[b]) > 0.2 * Math.max(...area);
  });
}

export interface Found {
  steps: Step[];
  corners: Point[] | null;
  present: number;
}

/**
 * Finder on the whole frame, then the refiner on close-ups (corners.find). When the finder is under its
 * bar, it looks again 2× closer around its rough guess: a cube small in the frame is only ~20 px across
 * in the whole-frame view. With `previous` (last frame's corners) it first tries one refiner pass around
 * them and skips the finder (corners.track).
 */
export async function findCube(models: Models, source: Source, previous: Point[] | null = null, canvas = document.createElement("canvas")): Promise<Found> {
  const { passes, refine_margin: margin, min_present: min, min_find: minFind, zoom, two_faces: twoFaces } = models.meta;
  const shaped = (kp: Point[]) => cubeShaped(kp, twoFaces);
  if (previous) {
    const step = await runCorners(models, models.refiner, source, refineView(previous, margin), canvas);
    if (step.present >= min && shaped(step.corners)) return { steps: [{ net: "refiner", ...step }], corners: step.corners, present: step.present };
  }
  let step = await runCorners(models, models.finder, source, fullView(source.width, source.height), canvas);
  if (zoom && step.present < minFind) {
    const [cx, cy] = mean(step.corners);
    const closer = await runCorners(models, models.finder, source, { cx, cy, side: Math.min(source.width, source.height) / zoom }, canvas);
    if (closer.present > step.present) step = closer;
  }
  const steps: Step[] = [{ net: "finder", ...step }];
  if (step.present < minFind) return { steps, corners: null, present: step.present };
  for (let n = 0; n < passes && step.present >= min; n++) {
    step = await runCorners(models, models.refiner, source, refineView(step.corners, margin), canvas);
    steps.push({ net: "refiner", ...step });
  }
  const ok = steps.length > passes && step.present >= min && shaped(step.corners);
  return { steps, corners: ok ? step.corners : null, present: step.present };
}

// Straighten each face to size x size like OpenCV's warpPerspective (bilinear, black outside the frame).
// edges (layerEdges) is for a cube whose outer layers are thicker: each layer is stretched or squeezed to
// the same number of pixels, so every sticker still lands in its own cell, where the reader expects it.
export function faceMaps(frame: ImageData, corners: Point[], size: number, edges?: number[]): Float32Array {
  const n = size * size, out = new Float32Array(3 * 3 * n), { data, width: W, height: H } = frame;
  const px = (x: number, y: number, ch: number) => (x < 0 || y < 0 || x >= W || y >= H ? 0 : data[4 * (y * W + x) + ch]);
  const along = edges ? layerMap(edges, size) : Float64Array.from({ length: size }, (_, x) => x / size);
  FACE_QUADS.forEach((quad, f) => {
    const map = squareToQuad(quad.map((j) => corners[j]));
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const [sx, sy] = map(along[x], along[y]);
        const x0 = Math.floor(sx), y0 = Math.floor(sy), fx = sx - x0, fy = sy - y0;
        for (let ch = 0; ch < 3; ch++) {
          const top = px(x0, y0, ch) * (1 - fx) + px(x0 + 1, y0, ch) * fx;
          const bottom = px(x0, y0 + 1, ch) * (1 - fx) + px(x0 + 1, y0 + 1, ch) * fx;
          out[f * 3 * n + ch * n + y * size + x] = (top * (1 - fy) + bottom * fy) / 255;
        }
      }
    }
  });
  return out;
}

/**
 * 3N² x 7 probabilities, row-major: face k, row j, column i; the 6 colours (URFDLB), then covered.
 * edges: layerEdges() for a cube whose layers aren't equally thick (none: equal layers).
 */
export async function readStickers(models: Models, frame: ImageData, corners: Point[], n = 3, edges?: number[]): Promise<Float32Array> {
  const size = models.meta.sticker_px * n, reader = await models.reader(n);
  const faces = new models.ort.Tensor("float32", faceMaps(frame, corners, size, edges), [1, 3, 3, size, size]);
  return (await reader.run({ faces })).probs.data as Float32Array;
}

/** The whole frame's pixels, for readStickers. */
export function framePixels(source: Source, canvas = document.createElement("canvas")): ImageData {
  canvas.width = source.width;
  canvas.height = source.height;
  const c = canvas.getContext("2d", { willReadFrequently: true })!;
  c.drawImage(source, 0, 0, source.width, source.height);
  return c.getImageData(0, 0, source.width, source.height);
}
