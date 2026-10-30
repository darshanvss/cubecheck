"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { CubeNet, STICKER } from "@/components/cube-net";
import { FACE_QUADS, cellCentres, cubeSize, insideImage, squareToQuad, type Point } from "@/lib/check";
import { match, netOf, parseScrambles, type Match } from "@/lib/scrambles";
import { CornerSmoother } from "@/lib/smooth";
import { findCube, loadModels, readStickers, type Models } from "@/lib/vision";
import { cn } from "@/lib/utils";

const MAX_WIDTH = 1280; // frames are checked at most this wide
const MIN_CUBE_PX = 60; // smaller than this, stickers are too few pixels to read
const HISTORY = 8; // a verdict settles over the last few frames

interface Frame {
  width: number;
  height: number;
  corners: Point[] | null;
  centres: Point[] | null;
  match: Match | null;
  tooSmall: boolean;
  ms: number;
}

type Shown = { tone: "wait" | "ok" | "bad"; title: string; detail: string };


const BLANK = "?".repeat(54); // a net with every sticker grey
const SAMPLE ="R2 D' B2 U2 L2 F2 D R2 U' L2 B2 R' F U' L' D2 F' R' B D2 R'";

export function Checker() {
  const [text, setText] = useState("");
  const scrambles = useMemo(() => parseScrambles(text), [text]);
  const [models, setModels] = useState<Models | null>(null);
  const [modelError, setModelError] = useState("");
  const [camera, setCamera] = useState<"off" | "starting" | "on" | "error">("off");
  const [photo, setPhoto] = useState<string | null>(null); // a photo checked instead of the camera
  const [cameraError, setCameraError] = useState("");
  const [mirror, setMirror] = useState(true);
  const [frame, setFrame] = useState<Frame | null>(null);
  const [settled, setSettled] = useState<string>("");

  const video = useRef<HTMLVideoElement>(null);
  const still = useRef<HTMLImageElement>(null);
  const overlay = useRef<HTMLCanvasElement>(null);
  const grab = useRef<HTMLCanvasElement | null>(null);
  const work = useRef<HTMLCanvasElement | null>(null);
  const smoother = useRef(new CornerSmoother());
  const history = useRef<string[]>([]);
  const latest = useRef({ models, scrambles, mirror });
  latest.current = { models, scrambles, mirror };

  // The scrambles stay in this browser between visits; they never leave the device.
  useEffect(() => {
    try {
      setText(localStorage.getItem("cubecheck-scrambles") ?? "");
    } catch {}
  }, []);
  const onText = (value: string) => {
    setText(value);
    history.current = [];
    try {
      localStorage.setItem("cubecheck-scrambles", value);
    } catch {}
  };

  useEffect(() => {
    loadModels().then(setModels, (err) => setModelError(String(err?.message ?? err)));
  }, []);

  const startCamera = useCallback(async () => {
    setCamera("starting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      const facing = stream.getVideoTracks()[0]?.getSettings().facingMode;
      setMirror(facing !== "environment"); // a selfie camera feels natural mirrored; a back camera doesn't
      video.current!.srcObject = stream;
      await video.current!.play();
      setCamera("on");
    } catch (err) {
      setCamera("error");
      setCameraError((err as Error).name === "NotAllowedError" ? "Camera access was blocked. Allow it in the browser's site settings and try again." : "No camera could be started.");
    }
  }, []);

  useEffect(() => {
    const v = video.current;
    return () => (v?.srcObject as MediaStream | null)?.getTracks().forEach((t) => t.stop());
  }, []);

  // Find the cube, read its stickers, check them against every scramble. Live video follows the cube from
  // the last frame, smooths the corners and lets the verdict settle over frames; a photo is judged alone.
  const analyse = useCallback(async (source: CanvasImageSource, width: number, height: number, live: boolean) => {
    grab.current ??= document.createElement("canvas");
    work.current ??= document.createElement("canvas");
    const t0 = performance.now(), scale = Math.min(1, MAX_WIDTH / width);
    const g = grab.current, w = Math.round(width * scale), h = Math.round(height * scale);
    if (g.width !== w || g.height !== h) Object.assign(g, { width: w, height: h });
    const gc = g.getContext("2d", { willReadFrequently: true })!;
    gc.drawImage(source, 0, 0, w, h);
    const now = t0 / 1000, s = smoother.current;
    const { models: m, scrambles: list } = latest.current;
    const found = await findCube(m!, g, live && s.fresh(now) ? s.kp : null, work.current!);
    const corners = !found.corners ? (s.reset(), null) : live ? s.update(found.corners, now) : found.corners;
    let result: Match | null = null, tooSmall = false;
    if (corners && list.some((x) => x.state)) {
      tooSmall = cubeSize(corners) < MIN_CUBE_PX;
      if (!tooSmall) {
        const probs = await readStickers(m!, gc.getImageData(0, 0, w, h), corners);
        result = match(probs, insideImage(w, h, corners), list);
      }
    }
    const key = result ? `${result.status}:${result.scramble?.number ?? 0}` : "";
    history.current = !result ? [] : live ? [...history.current, key].slice(-HISTORY) : [key];
    setSettled(live ? settle(history.current) : key);
    setFrame({ width: w, height: h, corners, centres: corners && cellCentres(corners), match: result, tooSmall, ms: performance.now() - t0 });
  }, []);

  useEffect(() => {
    if (camera !== "on" || !models || photo) return;
    let stop = false;
    const step = async () => {
      const v = video.current!;
      if (stop) return;
      if (v.readyState >= 2 && v.videoWidth) await analyse(v, v.videoWidth, v.videoHeight, true);
      if (!stop) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
    return () => {
      stop = true;
    };
  }, [camera, models, photo, analyse]);

  // A photo: checked once when it loads, again when the scrambles change.
  useEffect(() => {
    const img = still.current;
    if (!photo || !models || !img) return;
    const run = () => void analyse(img, img.naturalWidth, img.naturalHeight, false);
    if (img.complete && img.naturalWidth) run();
    else img.onload = run;
  }, [photo, models, scrambles, analyse]);

  const choosePhoto = (file: File | undefined) => {
    if (!file) return;
    if (photo) URL.revokeObjectURL(photo);
    smoother.current.reset();
    setFrame(null);
    setPhoto(URL.createObjectURL(file));
  };

  const showing = photo ? "photo" : camera === "on" ? "camera" : "none";

  const valid = scrambles.filter((s) => s.state);
  const shown = describe({ valid: valid.length, models, modelError, camera: photo ? "on" : camera, cameraError, frame, settled, live: !photo });
  const m = frame?.match;
  const observed = observedNet(m);
  const matched = settled.startsWith("correct") ? Number(settled.split(":")[1]) : 0;
  // The scramble the cube is compared with: the one it settled on matching, or the only one pasted once it's
  // settled as a mis-scramble. With several and none matching there's no telling which was meant, so the net
  // stays grey and no sticker is marked wrong.
  const target = valid.find((s) => s.number === matched) ?? (valid.length === 1 && settled.startsWith("mis-scramble") ? valid[0] : undefined);
  const marked = target && m?.scramble?.number === target.number ? m.verdict : undefined;
  const flagged = new Set(marked ? marked.mismatches.map((n) => marked.facelets[n]) : []);
  useEffect(() => drawOverlay(overlay.current, photo ? still.current : video.current, frame, !photo && mirror, !!marked), [frame, mirror, photo, marked]);

  return (
    <div className="mx-auto grid w-full max-w-[1536px] gap-6 px-5 py-6 md:grid-cols-[minmax(0,300px)_minmax(0,1fr)] lg:grid-cols-[minmax(0,380px)_minmax(0,1fr)] lg:px-8 xl:grid-cols-[minmax(0,440px)_minmax(0,1fr)]">
      <Card className="h-fit gap-4 p-5">
        <div className="flex items-center justify-between gap-3">
          <h2 className="text-sm font-semibold">Scrambles</h2>
          <Tooltip>
            <TooltipTrigger asChild>
              <span tabIndex={0} className="rounded-full border border-primary/40 bg-primary/10 px-2.5 py-0.5 text-xs font-semibold text-primary">3×3</span>
            </TooltipTrigger>
            <TooltipContent>More events coming soon</TooltipContent>
          </Tooltip>
        </div>
        <Textarea
          value={text}
          onChange={(e) => onText(e.target.value)}
          placeholder={"Paste the scrambles you're about to do, one per line.\n\n" + SAMPLE}
          spellCheck={false}
          autoComplete="off"
          className="max-h-48 min-h-36 overflow-y-auto font-mono leading-relaxed"
          aria-label="Scrambles, one per line"
        />
        {scrambles.length > 0 && (
          // Rows are as wide as the text box and their text starts where its text does (border + px-3).
          <ol className="space-y-1">
            {scrambles.map((s) => (
              <li key={s.number} className={cn("flex items-start gap-2.5 rounded-xl border px-3 py-1.5", s.number === matched ? "border-correct/60 bg-correct/10" : "border-transparent")}>
                <span className="min-w-5 shrink-0 font-mono text-sm leading-5 text-muted-foreground">{s.number}.</span>
                {s.error ? (
                  <span className="min-w-0 flex-1 text-sm leading-5 text-wrong">{s.error}</span>
                ) : (
                  <span className="min-w-0 flex-1 break-words font-mono text-sm leading-5">{s.text}</span>
                )}
                {s.number === matched && <span className="shrink-0 text-xs font-semibold leading-5 text-correct">match</span>}
              </li>
            ))}
          </ol>
        )}
        <p className="text-xs leading-relaxed text-muted-foreground">
          Hold the cube with one corner toward the camera, so three faces show. It&rsquo;s checked against every scramble in the list.
        </p>
      </Card>

      {/* No wider than a 16:9 picture that still fits the screen's height with the banner above it. */}
      <div className="mx-auto grid h-fit w-full min-w-0 max-w-[max(28rem,calc((100svh-16rem)*16/9))] gap-3">
        <div
          role="status"
          aria-live="polite"
          className={cn(
            "grid justify-items-center gap-0.5 rounded-2xl border-2 bg-card px-5 py-3 text-center",
            shown.tone === "ok" ? "border-correct" : shown.tone === "bad" ? "border-wrong" : "border-border",
          )}
        >
          {/* Fixed line heights, and a space when there's no detail: the picture below never moves. */}
          <span className={cn("block h-8 max-w-full truncate text-2xl font-bold leading-8 tracking-tight sm:h-9 sm:text-3xl sm:leading-9", shown.tone === "ok" ? "text-correct" : shown.tone === "bad" ? "text-wrong" : "text-foreground")}>{shown.title}</span>
          <span className="block h-5 max-w-full truncate text-sm leading-5 text-muted-foreground">{shown.detail || "\u00a0"}</span>
        </div>

        <div className={cn("relative overflow-hidden rounded-2xl border border-border", showing === "none" ? "bg-muted" : "bg-black")}>
          <video ref={video} playsInline muted className={cn("block aspect-video h-auto w-full object-contain", mirror && "-scale-x-100", showing !== "camera" && "opacity-0")} />
          {photo && <img ref={still} src={photo} alt="The photo being checked" className="absolute inset-0 h-full w-full object-contain" />}
          <canvas ref={overlay} className="pointer-events-none absolute inset-0 h-full w-full" />
          {showing === "none" && (
            <div className="absolute inset-0 grid place-items-center p-6 text-center">
              <div className="grid justify-items-center gap-3">
                <p className="max-w-sm text-sm text-muted-foreground">
                  {camera === "error" ? cameraError : "Everything runs on this device: no picture or scramble ever leaves it."}
                </p>
                <Button size="lg" onClick={startCamera} disabled={camera === "starting"}>
                  {camera === "starting" ? "Starting…" : camera === "error" ? "Try again" : "Start the camera"}
                </Button>
                <label className="cursor-pointer text-xs text-muted-foreground underline underline-offset-4 hover:text-foreground">
                  or check a photo
                  <input type="file" accept="image/*" className="sr-only" onChange={(e) => choosePhoto(e.target.files?.[0])} />
                </label>
              </div>
            </div>
          )}
          {showing !== "none" && valid.length > 0 && (
            <div className="pointer-events-none absolute inset-x-2.5 bottom-2.5 flex items-end justify-between gap-2.5">
              <figure className="w-[min(220px,30%)] rounded-xl bg-black/65 p-2">
                <figcaption className="mb-1.5 text-[11px] text-white/80">{target ? `Scramble ${target.number}` : "From the scramble"}</figcaption>
                <CubeNet letters={target ? netOf(target.state!) : BLANK} flagged={flagged} label="What the scramble should give" />
              </figure>
              <figure className="w-[min(220px,30%)] rounded-xl bg-black/65 p-2">
                <figcaption className="mb-1.5 text-[11px] text-white/80">What the camera reads</figcaption>
                <CubeNet letters={observed} flagged={flagged} label="What the camera reads" />
              </figure>
            </div>
          )}
        </div>

        <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
          <span className="tabular-nums">
            {models ? `Running on ${models.backend === "webgpu" ? "this device's GPU" : "this device's CPU"}` : modelError ? "" : "Loading the models…"}
            {frame && ` · ${frame.ms.toFixed(0)} ms per frame`}
          </span>
          <div className="flex gap-2">
            {photo && (
              <Button variant="outline" size="sm" onClick={() => { URL.revokeObjectURL(photo); setPhoto(null); setFrame(null); if (camera !== "on") void startCamera(); }}>
                Use the camera
              </Button>
            )}
            {showing === "camera" && (
              <Button variant="outline" size="sm" aria-pressed={mirror} onClick={() => setMirror(!mirror)}>
                Mirror {mirror ? "on" : "off"}
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/** Settled verdict over the last frames: "correct:N", "mis-scramble:N", "re-check:N", or "" while checking. */
function settle(history: string[]): string {
  const count = (pred: (h: string) => boolean) => history.filter(pred).length;
  const mis = count((h) => h.startsWith("mis-scramble"));
  if (mis >= 4) return history.findLast((h) => h.startsWith("mis-scramble"))!;
  for (const h of new Set(history)) if (h.startsWith("correct") && count((x) => x === h) >= 5 && mis <= 1) return h;
  if (count((h) => h.startsWith("re-check")) >= 4) return history.findLast((h) => h.startsWith("re-check"))!;
  return "";
}

function describe({ valid, models, modelError, camera, cameraError, frame, settled, live }: {
  valid: number; models: Models | null; modelError: string; camera: string; cameraError: string; frame: Frame | null; settled: string; live: boolean;
}): Shown {
  if (modelError) return { tone: "bad", title: "Couldn't load the models", detail: modelError };
  if (!valid) return { tone: "wait", title: "Add your scrambles", detail: "paste one or more, one per line" };
  if (camera === "error") return { tone: "wait", title: "No camera", detail: cameraError };
  if (camera !== "on") return { tone: "wait", title: "Start the camera", detail: "then show the cube with one corner toward it" };
  if (!models) return { tone: "wait", title: "Loading the models…", detail: "about 23 MB, once" };
  if (!frame) return { tone: "wait", title: live ? "Show the cube" : "Checking…", detail: "" };
  if (!frame.corners) return { tone: "wait", title: live ? "Show the cube" : "No cube found", detail: "one corner toward the camera, three faces visible" };
  if (frame.tooSmall) return { tone: "wait", title: "Move closer", detail: "the cube is too small to read" };
  const m = frame.match;
  if (!m) return { tone: "wait", title: "Checking…", detail: "" };
  const [status, n] = settled.split(":");
  if (status === "correct") return { tone: "ok", title: valid > 1 ? `Matches scramble ${n}` : "Correct", detail: "matches the scramble" };
  if (status === "mis-scramble") {
    return valid > 1 ? { tone: "bad", title: "Matches none", detail: "doesn't match any of the scrambles" } : { tone: "bad", title: "Mis-scramble", detail: "doesn't match the scramble" };
  }
  if (status === "re-check") return { tone: "wait", title: "Re-check", detail: recheckReason(m) };
  return { tone: "wait", title: "Checking…", detail: "hold the cube still for a moment" };
}

const recheckReason = (m: Match) => (m.status === "re-check" ? m.reason : "hold the cube still for a moment");

/** The 54-sticker net of what the camera read: the three visible faces, the rest grey. */
function observedNet(m: Match | null | undefined): string {
  const letters = Array(54).fill("?");
  m?.verdict?.facelets.forEach((f, n) => (letters[f] = m.verdict!.observed[n]));
  return letters.join("");
}

function drawOverlay(canvas: HTMLCanvasElement | null, video: HTMLElement | null, frame: Frame | null, mirror: boolean, marks: boolean) {
  if (!canvas || !video) return;
  const W = video.clientWidth, H = video.clientHeight, dpr = devicePixelRatio || 1;
  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  const c = canvas.getContext("2d")!;
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, W, H);
  if (!frame?.corners) return;
  // The video keeps its shape inside the box (object-contain): find where the picture sits.
  const s = Math.min(W / frame.width, H / frame.height), ox = (W - frame.width * s) / 2, oy = (H - frame.height * s) / 2;
  const P = ([x, y]: Point): Point => [mirror ? W - (ox + x * s) : ox + x * s, oy + y * s];
  const k = frame.corners.map(P);
  const line = (pts: Point[], close = false) => {
    c.beginPath();
    pts.forEach(([x, y], i) => (i ? c.lineTo(x, y) : c.moveTo(x, y)));
    if (close) c.closePath();
    c.stroke();
  };
  c.strokeStyle = "rgba(92, 200, 230, 0.55)";
  c.lineWidth = 1.2;
  for (const quad of FACE_QUADS) {
    const map = squareToQuad(quad.map((j) => k[j]));
    for (const t of [1 / 3, 2 / 3]) {
      line([map(t, 0), map(t, 1)]);
      line([map(0, t), map(1, t)]);
    }
  }
  c.strokeStyle = "#5cc8e6";
  c.lineWidth = 2.5;
  line(k.slice(0, 6), true);
  for (const j of [1, 3, 5]) line([k[6], k[j]]);
  const v = frame.match?.verdict;
  if (!v?.observed || !frame.centres) return;
  const bad = new Set(marks ? v.mismatches : []);
  frame.centres.forEach((pt, n) => {
    const [x, y] = P(pt);
    c.beginPath();
    c.arc(x, y, 6, 0, Math.PI * 2);
    c.fillStyle = STICKER[v.observed[n]] ?? STICKER["?"];
    c.fill();
    c.strokeStyle = "#000";
    c.lineWidth = 1.5;
    c.stroke();
    if (bad.has(n)) {
      c.beginPath();
      c.arc(x, y, 12, 0, Math.PI * 2);
      c.strokeStyle = "#ef4444";
      c.lineWidth = 3;
      c.stroke();
    }
  });
}
