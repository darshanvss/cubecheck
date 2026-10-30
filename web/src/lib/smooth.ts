// Steadier corners in live video (python/cubecheck/smooth.py): a One Euro filter (Casiez et al., 2012)
// averages over many frames while the corners barely move, and trusts the newest frame as they move faster.
// Speeds are in cube edges per second, so the same settings work close to the camera or far away.
import type { Point } from "./check";

// The 3 ways to start the corner order: shift P1..P6 by two places (one face), C stays.
const ROTATIONS = [0, 1, 2].map((r) => [...Array.from({ length: 6 }, (_, i) => (i + 2 * r) % 6), 6]);

const edge = (kp: Point[]) => [1, 3, 5].reduce((sum, i) => sum + Math.hypot(kp[i][0] - kp[6][0], kp[i][1] - kp[6][1]), 0) / 3;
const dist = (a: Point, b: Point) => Math.hypot(a[0] - b[0], a[1] - b[1]);

/** kp with its corner order turned to match ref (the model may start at any of the 3 faces). */
function align(kp: Point[], ref: Point[]): Point[] {
  const options = ROTATIONS.map((r) => r.map((i) => kp[i]));
  const cost = (k: Point[]) => k.reduce((sum, p, i) => sum + dist(p, ref[i]), 0);
  return options.reduce((best, k) => (cost(k) < cost(best) ? k : best));
}

const alpha = (cutoff: number, dt: number) => 1 / (1 + 1 / (2 * Math.PI * cutoff) / dt);

export class CornerSmoother {
  kp: Point[] | null = null;
  private speed = new Array(7).fill(0);
  private t = 0;

  constructor(private minCutoff = 1, private beta = 2, private dCutoff = 1, private maxGap = 0.5) {}

  reset() {
    this.kp = null;
  }

  /** There's a recent position to start from. */
  fresh(t: number) {
    return this.kp !== null && t - this.t <= this.maxGap;
  }

  update(raw: Point[], t: number): Point[] {
    if (!this.fresh(t)) return this.start(raw, t);
    const prev = this.kp!, kp = align(raw, prev), dt = Math.max(t - this.t, 1e-3), size = Math.max(edge(prev), 1);
    if (Math.max(...kp.map((p, i) => dist(p, prev[i]))) > 0.5 * size) return this.start(kp, t); // jumped: start over
    kp.forEach((p, i) => {
      this.speed[i] += alpha(this.dCutoff, dt) * (dist(p, prev[i]) / size / dt - this.speed[i]);
    });
    this.kp = prev.map((p, i) => {
      const a = alpha(this.minCutoff + this.beta * this.speed[i], dt);
      return [p[0] + a * (kp[i][0] - p[0]), p[1] + a * (kp[i][1] - p[1])];
    });
    this.t = t;
    return this.kp;
  }

  private start(kp: Point[], t: number) {
    this.kp = kp;
    this.speed.fill(0);
    this.t = t;
    return kp;
  }
}
