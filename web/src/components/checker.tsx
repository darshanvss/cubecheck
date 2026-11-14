"use client";

import { useCallback, useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { CubeNet, STICKER } from "@/components/cube-net";
import { FollowView, followWords } from "@/components/follow-panel";
import { Icon, ScrambleSource, type Source } from "@/components/scramble-source";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { FACE_QUADS, cubeSize, squareToQuad, type Point, type Verdict } from "@/lib/check";
import { Follower, type Report } from "@/lib/follow";
import { PX_PER_STICKER, cellCentres, checkN, layerEdges, measuredRatio, readable } from "@/lib/nxn-check";
import { EVENT_NAME } from "@/lib/nxn";
import { match, netOf, prepare, type Match, type Scramble } from "@/lib/scrambles";
import { CornerSmoother } from "@/lib/smooth";
import { findCube, loadModels, readStickers, type Models } from "@/lib/vision";
import { cn } from "@/lib/utils";

const MAX_WIDTH = 1280; // frames are checked at most this wide
const HISTORY = 8; // a verdict settles over the last few frames
const HOLD_CLEAR = 1500; // follow-along says it can't see the cube only after this long without a clear look (ms)
// A cube seen well enough that fits no point of the scramble, for this long, isn't on it: scrambled before
// follow-along started watching, another scramble, or a slip followed by more moves than it looks back over.
// Longer than HOLD_CLEAR: a layer mid-turn fits nothing either.
const HOLD_OFF = 2500;
const MODE_KEY = "cubecheck-mode";
const GRID_KEY = "cubecheck-grid"; // "off": no grid or sticker dots drawn on the cube

// Smaller than this (the mean edge, in pixels), stickers are too few pixels to read.
const minCubePx = (n: number) => (n === 3 ? 60 : PX_PER_STICKER * n);

interface Frame {
  width: number;
  height: number;
  n: number;
  edges: number[] | undefined; // where the layers end on each face (uneven on a big cube), none: equal
  corners: Point[] | null;
  centres: Point[] | null;
  match: Match | null;
  tooSmall: boolean;
  ms: number;
  now: { point: number; verdict: Verdict } | null; // follow-along: the camera's read against where the scramble should be
}

type Shown = { tone: "wait" | "ok" | "bad"; title: string; detail: string };

/** Follow-along reports change what's shown only when two clear ones in a row agree; unclear looks keep the last. */
interface Steady {
  shown: Report | null;
  key: string;
  lastClear: number;
}
const NO_STEADY: Steady = { shown: null, key: "", lastClear: 0 };

export function Checker() {
  const [source, setSource] = useState<Source>({ lines: [], size: 3, name: "3×3" });
  const deferred = useDeferredValue(source); // a 7×7's one-slip cubes take a moment to work out
  const scrambles = useMemo(() => prepare(deferred.lines, deferred.size), [deferred]);
  const size = deferred.size;
  const [models, setModels] = useState<Models | null>(null);
  const [modelError, setModelError] = useState("");
  const [camera, setCamera] = useState<"off" | "starting" | "on" | "error">("off");
  const [photo, setPhoto] = useState<string | null>(null); // a photo checked instead of the camera
  const [cameraError, setCameraError] = useState("");
  // Always mirrored, like looking in a mirror: the cube moves the way your hands do. Only a phone's back
  // camera isn't (there the picture already matches what you see), so there's no switch for it.
  const [mirror, setMirror] = useState(true);
  const [grid, setGrid] = useState(true); // the cube's outline, its sticker grid and the colours read, drawn on the picture
  const [frame, setFrame] = useState<Frame | null>(null);
  const [settled, setSettled] = useState<string>("");

  // Follow-along (beta): one scramble, followed move by move.
  const [mode, setMode] = useState<"check" | "follow">("check");
  const [followed, setFollowed] = useState(0); // the scramble's number; 0: the first that parses
  const [report, setReport] = useState<Report | null>(null);
  const [blurry, setBlurry] = useState(false);
  const [offPath, setOffPath] = useState(false);
  const offSince = useRef<number | null>(null);
  const cubeSince = useRef<number | null>(null); // since when a cube has been in view (null: none now)
  const follower = useRef<Follower | null>(null);
  const steady = useRef<Steady>(NO_STEADY);
  const finished = useRef(false); // the followed scramble was seen done: a solved cube next moves on to the next one

  const video = useRef<HTMLVideoElement>(null);
  const still = useRef<HTMLImageElement>(null);
  const overlay = useRef<HTMLCanvasElement>(null);
  const grab = useRef<HTMLCanvasElement | null>(null);
  const work = useRef<HTMLCanvasElement | null>(null);
  const smoother = useRef(new CornerSmoother());
  const history = useRef<string[]>([]);

  const valid = scrambles.filter((s) => s.state);
  const following = mode === "follow";
  const target: Scramble | undefined = following ? valid.find((s) => s.number === followed) ?? valid[0] : undefined;
  // Big cubes have thicker outer layers: the grid follows the measured widths (6×6, 7×7), even elsewhere.
  const r = measuredRatio(size), edges = r === 1 ? undefined : layerEdges(size, r);
  const latest = useRef({ models, scrambles, following, target, size, valid, edges });
  latest.current = { models, scrambles, following, target, size, valid, edges };

  useEffect(() => {
    try {
      if (localStorage.getItem(MODE_KEY) === "follow") setMode("follow");
    } catch {}
  }, []);
  useEffect(() => {
    try {
      if (localStorage.getItem(GRID_KEY) === "off") setGrid(false);
    } catch {}
  }, []);
  const switchGrid = () => {
    setGrid(!grid);
    try {
      localStorage.setItem(GRID_KEY, grid ? "off" : "on");
    } catch {}
  };
  const switchMode = (m: "check" | "follow") => {
    setMode(m);
    try {
      localStorage.setItem(MODE_KEY, m);
    } catch {}
  };

  const onSource = useCallback((s: Source) => setSource((old) => (JSON.stringify(old) === JSON.stringify(s) ? old : s)), []);

  // A new scramble list or size starts the verdict over.
  useEffect(() => {
    history.current = [];
    setSettled("");
  }, [scrambles]);

  // A new scramble to follow (or the mode switched) starts the tracker from a solved cube.
  const restart = useCallback(() => {
    const t = latest.current.target;
    follower.current = t ? new Follower(t.text, t.size) : null;
    steady.current = NO_STEADY;
    finished.current = false;
    history.current = [];
    setReport(null);
    setBlurry(false);
    setOffPath(false);
    offSince.current = null;
    setSettled("");
  }, []);
  useEffect(restart, [following, target?.text, target?.size, restart]);

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

  // Find the cube, read its stickers, then check them against every scramble, and in follow-along tell the
  // tracker too. Live video follows the cube from the last frame, smooths the corners and lets the verdict
  // settle over frames; a photo is judged alone.
  const analyse = useCallback(async (source: CanvasImageSource, width: number, height: number, live: boolean) => {
    grab.current ??= document.createElement("canvas");
    work.current ??= document.createElement("canvas");
    const t0 = performance.now(), scale = Math.min(1, MAX_WIDTH / width);
    const g = grab.current, w = Math.round(width * scale), h = Math.round(height * scale);
    if (g.width !== w || g.height !== h) Object.assign(g, { width: w, height: h });
    const gc = g.getContext("2d", { willReadFrequently: true })!;
    gc.drawImage(source, 0, 0, w, h);
    const now = t0 / 1000, s = smoother.current;
    const { models: m, scrambles: list, following: fol, target: t, size: n, edges: e } = latest.current;
    const found = await findCube(m!, g, live && s.fresh(now) ? s.kp : null, work.current!);
    const corners = !found.corners ? (s.reset(), null) : live ? s.update(found.corners, now) : found.corners;
    let result: Match | null = null, tooSmall = false, where: Frame["now"] = null;
    const against = fol ? (t ? [t] : []) : list;
    if (corners && against.some((x) => x.state)) {
      tooSmall = cubeSize(corners) < minCubePx(n);
      if (!tooSmall) {
        const probs = await readStickers(m!, gc.getImageData(0, 0, w, h), corners, n, e), inside = readable(corners, n, w, h);
        if (fol && follower.current) {
          const r = follower.current.look(probs, inside), st = steady.current;
          if (r.clear) {
            const key = JSON.stringify([r.status, r.position, r.half, r.mistake ?? null]);
            steady.current = { lastClear: performance.now(), key, shown: !st.shown || key === st.key ? r : st.shown };
          } else if (!st.shown) steady.current = { ...st, shown: r };
          const shown = steady.current.shown!, f = follower.current;
          setReport(shown);
          // No "start over": a solved cube puts the tracker back at the start by itself, and after a scramble
          // was done it moves on to the next one.
          if (shown.status === "done") finished.current = true;
          else if (finished.current && r.clear && shown.status === "following" && shown.position === 0 && !shown.half) {
            finished.current = false;
            const { valid: v, target: cur } = latest.current, next = v[v.findIndex((x) => x === cur) + 1];
            if (next) setFollowed(next.number);
          }
          // Where the cube should be now (where the scrambler meant to be, after a slip), for the nets.
          const point = meantPoint(shown);
          where = { point, verdict: checkN(probs, inside, n, f.points[point]) };
        }
        result = match(probs, inside, against);
      }
    }
    if (fol) {
      const t1 = performance.now(), why = corners && !tooSmall ? follower.current?.why : undefined;
      // Off the scramble only while every look in that time is "off"; a clear look, or losing the cube, starts it over.
      offSince.current = why === "off" ? offSince.current ?? t1 : why === "few" ? offSince.current : null;
      // "Can't see it clearly" only for a cube in view: with none, there's nothing to see. A cube that just
      // came back gets the same 1.5 s as any other before the warning.
      cubeSince.current = corners ? cubeSince.current ?? t1 : null;
      setBlurry(cubeSince.current !== null && t1 - Math.max(steady.current.lastClear, cubeSince.current) > HOLD_CLEAR);
      setOffPath(offSince.current !== null && t1 - offSince.current > HOLD_OFF);
    }
    const key = result ? `${result.status}:${result.scramble?.number ?? 0}` : "";
    history.current = !result ? [] : live ? [...history.current, key].slice(-HISTORY) : [key];
    setSettled(live ? settle(history.current) : key);
    setFrame({ width: w, height: h, n, edges: e, corners, centres: corners && cellCentres(corners, n, e), match: result, tooSmall, ms: performance.now() - t0, now: where });
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
  const moves = useMemo(() => target?.text.split(/\s+/) ?? [], [target?.text]);
  const followDone = following && report?.status === "done";
  const shown = following && offPath && !photo
    ? { tone: "wait" as const, title: `Not on scramble ${target?.label ?? ""}`, detail: "follow-along watches from a solved cube: start from solved" }
    : following && report && !followDone && !photo
    ? followWords(report, moves)
    : describe({ valid: following ? Number(!!target) : valid.length, models, modelError, camera: photo ? "on" : camera, cameraError, frame, settled, live: !photo, size, done: followDone });
  const m = frame?.match;
  const observed = netFrom(m?.verdict, frame?.n ?? size);
  const matched = settled.startsWith("correct") ? Number(settled.split(":")[1]) : 0;
  // The scramble the cube is compared with: the one it settled on matching, or the only one there is once it's
  // settled as a mis-scramble. With several and none matching there's no telling which was meant, so the net
  // stays grey and no sticker is marked wrong.
  const against = following ? (target ? [target] : []) : valid;
  const compared = against.find((s) => s.number === matched) ?? (against.length === 1 && settled.startsWith("mis-scramble") ? against[0] : undefined);
  const marked = compared && m?.scramble?.number === compared.number ? m.verdict : undefined;
  // Following, the left net shows the whole scramble until a cube is being tracked (just picked, or done), and
  // in between where the cube should be by now, kept while the cube is out of view; after a slip, which stickers are off.
  const now = following && !followDone ? frame?.now ?? null : null;
  const tracking = following && !followDone && !!report;
  const point = now?.point ?? (report ? meantPoint(report) : 0);
  const flagSource = following && !followDone ? (report?.status === "mistake" ? now?.verdict : undefined) : marked;
  const flagged = useMemo(() => new Set(flagSource ? flagSource.mismatches.map((k) => flagSource.facelets[k]) : []), [flagSource]);
  const follower_ = follower.current, blankNet = "?".repeat(6 * size * size);
  const shouldBe = !following
    ? compared ? netOf(compared.state!) : blankNet
    : tracking && follower_ ? netOf(follower_.points[point]) : target?.state ? netOf(target.state) : blankNet;
  const shouldCaption = !following
    ? compared ? `Scramble ${compared.label}` : "From the scramble"
    : tracking ? (point === 0 ? "Should be: solved" : `Should be: after move ${point}`) : target ? `Scramble ${target.label}` : "From the scramble";
  const seen = now ? netFrom(now.verdict, frame!.n) : observed;
  useEffect(() => drawOverlay(overlay.current, photo ? still.current : video.current, grid ? frame : null, !photo && mirror, !!marked), [frame, mirror, photo, marked, grid]);

  const listView = (
    <ol className="min-h-0 flex-1 space-y-1 overflow-y-auto">
      {scrambles.map((s) => {
        const isMatch = s.number === matched;
        return (
          <li key={s.number} className={cn("flex items-start gap-2.5 rounded-xl border px-3 py-1.5", isMatch ? "border-correct/60 bg-correct/10" : "border-transparent")}>
            <span className="min-w-6 shrink-0 font-mono text-sm leading-5 text-muted-foreground">{s.label}.</span>
            {s.error ? <span className="min-w-0 flex-1 text-sm leading-5 text-wrong">{s.error}</span> : <span className="min-w-0 flex-1 break-words font-mono text-sm leading-5">{s.text}</span>}
            {isMatch && <span className="shrink-0 text-xs font-semibold leading-5 text-correct">match</span>}
          </li>
        );
      })}
    </ol>
  );

  return (
    // Wide screens: the camera stops growing (at 1280 px, or what fits the screen's height) and the scrambles get the rest.
    <div className="mx-auto grid w-full max-w-[2200px] gap-6 px-5 py-6 md:grid-cols-[minmax(0,340px)_minmax(0,1fr)] lg:grid-cols-[minmax(400px,1fr)_min(calc((100svh_-_14rem)*16/9),1280px,calc(100%_-_424px))] lg:px-8">
      <div className="flex min-w-0 flex-col gap-3">
        <div role="tablist" aria-label="Mode" className="grid grid-cols-2 gap-1 rounded-xl bg-muted p-1">
          {(["check", "follow"] as const).map((k) => {
            const tab = (
              <button key={k} role="tab" aria-selected={mode === k} onClick={() => switchMode(k)}
                className={cn("inline-flex h-8 items-center justify-center gap-2 rounded-lg px-3 text-sm font-medium transition-colors", mode === k ? "bg-card text-foreground shadow-sm ring-1 ring-border dark:bg-white/10" : "text-muted-foreground hover:text-foreground")}>
                {k === "check" ? "Check" : "Follow along"}
                {k === "follow" && <span className="rounded-md border border-current/35 px-1.5 text-[10px] font-semibold uppercase leading-4 tracking-wide opacity-80">Beta</span>}
              </button>
            );
            return k === "check" ? tab : (
              <Tooltip key={k}>
                <TooltipTrigger asChild>{tab}</TooltipTrigger>
                <TooltipContent className="max-w-64 text-balance">Follows one scramble move by move and shows the way back from a wrong move. Steady on 2×2 to 4×4, still learning on bigger cubes.</TooltipContent>
              </Tooltip>
            );
          })}
        </div>
        <ScrambleSource onChange={onSource}>
          {following ? <FollowView scrambles={valid} target={target} onPick={setFollowed} report={report} blurry={blurry} offPath={offPath} /> : listView}
        </ScrambleSource>
      </div>

      {/* No wider than a 16:9 picture that still fits the screen's height with the banner above it. */}
      <div className="mx-auto grid h-fit w-full min-w-0 max-w-[max(28rem,calc((100svh-14rem)*16/9))] gap-3">
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
          <span className="block h-5 max-w-full truncate text-sm leading-5 text-muted-foreground">{shown.detail || " "}</span>
        </div>

        <div className={cn("relative overflow-hidden rounded-2xl border border-border", showing === "none" ? "bg-muted" : "bg-black")}>
          <video ref={video} playsInline muted className={cn("block aspect-video h-auto w-full object-contain", mirror && "-scale-x-100", showing !== "camera" && "opacity-0")} />
          {photo && <img ref={still} src={photo} alt="The photo being checked" className="absolute inset-0 h-full w-full object-contain" />}
          <canvas ref={overlay} className="pointer-events-none absolute inset-0 h-full w-full" />
          {showing === "none" && (
            <div className="absolute inset-0 grid place-items-center p-6 text-center">
              <div className="grid justify-items-center gap-3">
                <p className="max-w-sm text-sm text-muted-foreground">
                  {camera === "error" ? cameraError : modelError ? "" : "Everything runs on this device: no picture, scramble or PDF ever leaves it."}
                </p>
                <Button size="lg" onClick={startCamera} disabled={camera === "starting"}>
                  {camera === "starting" ? "Starting…" : camera === "error" ? "Try again" : "Start the camera"}
                </Button>
                {/* Kept in follow-along too, so the box doesn't change; a photo is checked, never followed. */}
                <label className={cn("cursor-pointer text-xs text-muted-foreground underline underline-offset-4 hover:text-foreground", following && "invisible")}>
                  or check a photo
                  <input type="file" accept="image/*" className="sr-only" disabled={following} onChange={(e) => choosePhoto(e.target.files?.[0])} />
                </label>
              </div>
            </div>
          )}
          {showing !== "none" && (
            <div className="absolute left-2.5 right-2.5 top-2.5 flex items-start justify-between gap-2">
              <span className="rounded-md bg-black/55 px-1.5 py-0.5 text-[11px] tabular-nums text-white/75">
                {models ? `${models.backend === "webgpu" ? "GPU" : "CPU"}${frame ? ` · ${frame.ms.toFixed(0)} ms` : ""}` : "Loading…"}
              </span>
              <div className="flex gap-1.5">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button aria-label={grid ? "Hide the grid on the cube" : "Show the grid on the cube"} aria-pressed={grid} onClick={switchGrid} className="grid size-8 place-items-center rounded-lg bg-black/55 text-white/80 hover:text-white">
                      {/* a cube's corner with its grid; struck through while hidden */}
                      <Icon d={<><path d="M12 3 4 7.5v9L12 21l8-4.5v-9z" /><path d="M12 12v9M12 12 4 7.5M12 12l8-4.5" />{!grid && <path d="M3 3l18 18" />}</>} />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent>{grid ? "Hide the grid and colours drawn on the cube" : "Show the grid and colours on the cube"}</TooltipContent>
                </Tooltip>
                {photo && (
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <button aria-label="Close the photo" onClick={() => { URL.revokeObjectURL(photo); setPhoto(null); setFrame(null); if (camera !== "on") void startCamera(); }} className="grid size-8 place-items-center rounded-lg bg-black/55 text-white/80 hover:text-white">
                        <Icon d={<path d="M18 6 6 18M6 6l12 12" />} />
                      </button>
                    </TooltipTrigger>
                    <TooltipContent>Close the photo, back to the camera</TooltipContent>
                  </Tooltip>
                )}
              </div>
            </div>
          )}
          {showing !== "none" && (following ? !!target : against.length > 0) && (
            <div className="pointer-events-none absolute inset-x-2.5 bottom-2.5 flex items-end justify-between gap-2.5">
              <figure className="w-[min(220px,30%)] rounded-xl bg-black/65 p-2">
                <figcaption className="mb-1.5 truncate text-[11px] text-white/80">{shouldCaption}</figcaption>
                <CubeNet letters={shouldBe} flagged={flagged} label="What the cube should look like" />
              </figure>
              <figure className="w-[min(220px,30%)] rounded-xl bg-black/65 p-2">
                <figcaption className="mb-1.5 truncate text-[11px] text-white/80">What the camera reads</figcaption>
                <CubeNet letters={seen} flagged={flagged} label="What the camera reads" />
              </figure>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/** Where the scrambler meant to be: the tracked point, or after a slip, the point they thought they'd reached. */
function meantPoint(r: Report): number {
  const mk = r.mistake;
  return mk ? Math.min(r.total, mk.move + mk.after + (mk.kind === "extra" ? 0 : mk.kind === "swapped" ? 2 : 1)) : r.position;
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

function describe({ valid, models, modelError, camera, cameraError, frame, settled, live, size, done }: {
  valid: number; models: Models | null; modelError: string; camera: string; cameraError: string; frame: Frame | null; settled: string; live: boolean; size: number; done: boolean;
}): Shown {
  if (modelError) return { tone: "bad", title: "Couldn't load the models", detail: modelError };
  if (!valid) return { tone: "wait", title: "Add your scrambles", detail: "paste them, or open a TNoodle PDF" };
  if (camera === "error") return { tone: "wait", title: "No camera", detail: cameraError };
  if (camera !== "on") return { tone: "wait", title: "Start the camera", detail: "then show the cube, a corner toward it" };
  if (!models) return { tone: "wait", title: "Loading the models…", detail: "about 23 MB, once" };
  if (!frame) return { tone: "wait", title: live ? "Show the cube" : "Checking…", detail: "" };
  if (!frame.corners) return { tone: "wait", title: live ? "Show the cube" : "No cube found", detail: `a ${EVENT_NAME[size]}, one corner toward the camera` };
  if (frame.tooSmall) return { tone: "wait", title: "Move closer", detail: "the cube is too small to read" };
  const m = frame.match;
  if (!m) return { tone: "wait", title: "Checking…", detail: "" };
  const lead = done ? "All moves done · " : "";
  const [status] = settled.split(":");
  if (status === "correct") return { tone: "ok", title: valid > 1 ? `Matches scramble ${m.scramble?.label ?? ""}` : "Correct", detail: `${lead}matches the scramble` };
  if (status === "mis-scramble") {
    return valid > 1 ? { tone: "bad", title: "Matches none", detail: "doesn't match any of the scrambles" } : { tone: "bad", title: "Mis-scramble", detail: `${lead}doesn't match the scramble` };
  }
  if (status === "re-check") return { tone: "wait", title: "Re-check", detail: m.status === "re-check" ? m.reason : "hold the cube still for a moment" };
  return { tone: "wait", title: "Checking…", detail: "hold the cube still for a moment" };
}

/** The 6N²-sticker net of what the camera read: the visible faces, the rest grey. */
function netFrom(v: Verdict | null | undefined, n: number): string {
  const letters = Array(6 * n * n).fill("?");
  v?.facelets.forEach((f, k) => (letters[f] = v.observed[k] ?? "?"));
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
    for (let t = 1; t < frame.n; t++) {
      const at = frame.edges ? frame.edges[t] : t / frame.n;
      line([map(at, 0), map(at, 1)]);
      line([map(0, at), map(1, at)]);
    }
  }
  c.strokeStyle = "#5cc8e6";
  c.lineWidth = 2.5;
  line(k.slice(0, 6), true);
  for (const j of [1, 3, 5]) line([k[6], k[j]]);
  const v = frame.match?.verdict;
  if (!v?.observed || !frame.centres) return;
  const bad = new Set(marks ? v.mismatches : []), r = Math.max(2.5, 18 / frame.n);
  frame.centres.forEach((pt, n) => {
    if (v.observed[n] === "?" && !bad.has(n)) return;
    const [x, y] = P(pt);
    c.beginPath();
    c.arc(x, y, r, 0, Math.PI * 2);
    c.fillStyle = STICKER[v.observed[n]] ?? STICKER["?"];
    c.fill();
    c.strokeStyle = "#000";
    c.lineWidth = 1.5;
    c.stroke();
    if (bad.has(n)) {
      c.beginPath();
      c.arc(x, y, 2 * r, 0, Math.PI * 2);
      c.strokeStyle = "#ef4444";
      c.lineWidth = 3;
      c.stroke();
    }
  });
}
