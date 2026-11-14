// Download the ONNX models from Hugging Face into public/models/ before a build (they aren't in git).
// Pinned to a release tag, so a model update is a deliberate change here. Files already present
// (for example freshly exported by python/scripts/export_onnx.py) are kept.
import { existsSync, mkdirSync, writeFileSync } from "node:fs";

const repo = "darshanvss/cubecheck";
const revision = "v0.3.0"; // c7 corners + r4 reader, one reader file per cube size; skewb, pyraminx, FTO in their own folders
const poly = ["meta.json", "finder.onnx", "refiner.onnx", "reader.onnx"];
const files = [
  "meta.json", "finder.onnx", "refiner.onnx", ...[2, 3, 4, 5, 6, 7].map((n) => `reader-${n}.onnx`),
  ...["skewb", "pyram", "fto"].flatMap((event) => poly.map((file) => `${event}/${file}`)),
];

const to = new URL("../public/models/", import.meta.url);
for (const file of files) {
  const target = new URL(file, to);
  mkdirSync(new URL(".", target), { recursive: true });
  if (existsSync(target)) continue;
  const url = `https://huggingface.co/${repo}/resolve/${revision}/onnx/${file}`;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status} ${response.statusText}`);
  writeFileSync(target, Buffer.from(await response.arrayBuffer()));
  console.log(`models: ${file}`);
}
