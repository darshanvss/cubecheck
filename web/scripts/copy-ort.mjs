// Serve ONNX Runtime Web from this site (public/ort/) instead of a CDN: the page works offline,
// and nothing about a visitor reaches another server.
import { cpSync, mkdirSync, readdirSync } from "node:fs";

const from = new URL("../node_modules/onnxruntime-web/dist/", import.meta.url);
const to = new URL("../public/ort/", import.meta.url);
mkdirSync(to, { recursive: true });
for (const file of readdirSync(from)) {
  if (/^ort\.webgpu\.min\.js$|^ort-wasm-simd-threaded\.jsep\.(mjs|wasm)$/.test(file)) cpSync(new URL(file, from), new URL(file, to));
}
