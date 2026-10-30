// The corner model and sticker reader in the browser, with ONNX Runtime Web: on the GPU through
// WebGPU where the browser has it, otherwise on the CPU. Each step mirrors python/cubecheck
// (corners.py, reader.py); the /test page checks the answers against Python's.
import type * as Ort from "onnxruntime-web";
import { FACE_QUADS, squareToQuad, type Point } from "./check";

type OrtModule = typeof Ort;
declare global {
  interface Window {
    ort?: OrtModule;
  }
}

export interface Meta {
  in_px: number;
  face_px: number;
  refine_margin: number;
  passes: number;
  min_present: number;
  classes: string;
}

export interface Models {
  ort: OrtModule;
  meta: Meta;
  finder: Ort.InferenceSession;
  refiner: Ort.InferenceSession;
  reader: Ort.InferenceSession;
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
  const finder = await open("finder"), refiner = await open("refiner"), reader = await open("reader");
  return { ort, meta, finder, refiner, reader, backend };
}

// --- crops: a square of `side` pixels around a centre, scaled onto the network's input ---

export interface View {
  cx: number;
  cy: number;
  side: number;
}

const fullView = (w: number, h: number): View => ({ cx: w / 2, cy: h / 2, side: Math.max(w, h) });

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

/** The three faces are convex quads, all turning the same way, as any real view of a cube gives. */
export function cubeShaped(kp: Point[]): boolean {
  const turns: number[] = [];
  for (const quad of FACE_QUADS) {
    const pts = quad.map((j) => kp[j]);
    for (let i = 0; i < 4; i++) {
      const [a, b, c] = [pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]];
      turns.push((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]));
    }
  }
  return turns.every((t) => t > 0) || turns.every((t) => t < 0);
}

export interface Found {
  steps: Step[];
  corners: Point[] | null;
  present: number;
}

/**
 * Finder on the whole frame, then the refiner on close-ups (corners.find). With `previous` (last
 * frame's corners) it first tries one refiner pass around them and skips the finder (corners.track).
 */
export async function findCube(models: Models, source: Source, previous: Point[] | null = null, canvas = document.createElement("canvas")): Promise<Found> {
  const { passes, refine_margin: margin, min_present: min } = models.meta;
  if (previous) {
    const step = await runCorners(models, models.refiner, source, refineView(previous, margin), canvas);
    if (step.present >= min && cubeShaped(step.corners)) return { steps: [{ net: "refiner", ...step }], corners: step.corners, present: step.present };
  }
  let step = await runCorners(models, models.finder, source, fullView(source.width, source.height), canvas);
  const steps: Step[] = [{ net: "finder", ...step }];
  for (let n = 0; n < passes && step.present >= min; n++) {
    step = await runCorners(models, models.refiner, source, refineView(step.corners, margin), canvas);
    steps.push({ net: "refiner", ...step });
  }
  const ok = steps.length > passes && step.present >= min && cubeShaped(step.corners);
  return { steps, corners: ok ? step.corners : null, present: step.present };
}

// Straighten each face to size x size like OpenCV's warpPerspective (bilinear, black outside the frame).
function faceMaps(frame: ImageData, corners: Point[], size: number): Float32Array {
  const n = size * size, out = new Float32Array(3 * 3 * n), { data, width: W, height: H } = frame;
  const px = (x: number, y: number, ch: number) => (x < 0 || y < 0 || x >= W || y >= H ? 0 : data[4 * (y * W + x) + ch]);
  FACE_QUADS.forEach((quad, f) => {
    const map = squareToQuad(quad.map((j) => corners[j]));
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const [sx, sy] = map(x / size, y / size);
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

/** 27 x 7 probabilities, row-major: face k, row j, column i; the 6 colours (URFDLB), then covered. */
export async function readStickers(models: Models, frame: ImageData, corners: Point[]): Promise<Float32Array> {
  const size = models.meta.face_px;
  const faces = new models.ort.Tensor("float32", faceMaps(frame, corners, size), [1, 3, 3, size, size]);
  return (await models.reader.run({ faces })).probs.data as Float32Array;
}

/** The whole frame's pixels, for readStickers. */
export function framePixels(source: Source, canvas = document.createElement("canvas")): ImageData {
  canvas.width = source.width;
  canvas.height = source.height;
  const c = canvas.getContext("2d", { willReadFrequently: true })!;
  c.drawImage(source, 0, 0, source.width, source.height);
  return c.getImageData(0, 0, source.width, source.height);
}
