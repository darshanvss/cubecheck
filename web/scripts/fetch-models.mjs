// Download the ONNX models from Hugging Face into public/models/ before a build (they aren't in git).
// Pinned to a release tag, so a model update is a deliberate change here. Files already present
// (for example freshly exported by python/scripts/export_onnx.py) are kept.
import { existsSync, mkdirSync, writeFileSync } from "node:fs";

const repo = "darshanvss/cubecheck-3x3";
const revision = "v0.1.0";
const files = ["meta.json", "finder.onnx", "refiner.onnx", "reader.onnx"];

const to = new URL("../public/models/", import.meta.url);
mkdirSync(to, { recursive: true });
for (const file of files) {
  const target = new URL(file, to);
  if (existsSync(target)) continue;
  const url = `https://huggingface.co/${repo}/resolve/${revision}/onnx/${file}`;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status} ${response.statusText}`);
  writeFileSync(target, Buffer.from(await response.arrayBuffer()));
  console.log(`models: ${file}`);
}
