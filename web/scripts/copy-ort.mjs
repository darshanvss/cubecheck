// Serve ONNX Runtime Web (public/ort/) and pdf.js (public/pdfjs/, for TNoodle PDFs) from this site instead
// of a CDN: the page works offline, and nothing about a visitor reaches another server. pdf.js is its legacy
// build: the modern one needs very new JavaScript (Map.getOrInsertComputed, Uint8Array.toHex) older phones lack.
import { cpSync, mkdirSync, readdirSync } from "node:fs";

const from = new URL("../node_modules/onnxruntime-web/dist/", import.meta.url);
const to = new URL("../public/ort/", import.meta.url);
mkdirSync(to, { recursive: true });
for (const file of readdirSync(from)) {
  if (/^ort\.webgpu\.min\.js$|^ort-wasm-simd-threaded\.jsep\.(mjs|wasm)$/.test(file)) cpSync(new URL(file, from), new URL(file, to));
}

const pdfFrom = new URL("../node_modules/pdfjs-dist/legacy/build/", import.meta.url);
const pdfTo = new URL("../public/pdfjs/", import.meta.url);
mkdirSync(pdfTo, { recursive: true });
for (const file of ["pdf.min.mjs", "pdf.worker.min.mjs"]) cpSync(new URL(file, pdfFrom), new URL(file, pdfTo));
cpSync(new URL("../node_modules/pdfjs-dist/LICENSE", import.meta.url), new URL("LICENSE", pdfTo));
