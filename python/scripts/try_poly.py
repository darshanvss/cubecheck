"""Try the experimental skewb, pyraminx or FTO check on a webcam or a picture.

    uv run python scripts/try_poly.py skewb "L' R' U L U' B' U L U' B' R"                 # webcam, live
    uv run python scripts/try_poly.py pyram "B L' U' L U' R' B' R U B' L u' l r b" --image photo.jpg --save out.jpg

Scramble the puzzle from solved in its usual scrambling orientation, then hold it up:
a corner toward the camera for a skewb or an FTO (three or four faces in view), an edge
toward it for a pyraminx (two faces). Without --image it prints an address to open in a
browser: the webcam picture with the corners found, the colour read on every sticker, and
the verdict: correct, mis-scramble (with the likely slip) or re-check.

Uses the puzzle's current models (shapes.MODELS) unless told otherwise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from cubecheck.poly_check import MIN_SIZE, check, readable, single_slips
from cubecheck.poly_corners import PolyCornerFinder
from cubecheck.poly_reader import load_poly_reader, read_stickers
from cubecheck.shapes import MODELS, shape

RUNS = Path(__file__).resolve().parents[2] / "runs"
STATUS = {"correct": (80, 200, 60), "mis-scramble": (40, 40, 230), "re-check": (0, 190, 240)}


def bgr(hex_colour: int) -> tuple[int, int, int]:
    return hex_colour & 255, (hex_colour >> 8) & 255, (hex_colour >> 16) & 255


def judge(img: np.ndarray, sh, finder, reader, state, slips, device: str, min_find: float):
    """Verdict text, its colour, and the picture with what was found drawn on it."""
    out = img.copy()
    found = finder.find(img, min_find=min_find, zoom=2.0)
    if found is None:
        return "no puzzle found", STATUS["re-check"], out
    kp = found[0].astype(np.float32)
    for poly in sh.polys:
        cv2.polylines(out, [kp[list(poly)].astype(np.int32)], True, (255, 255, 0), 2, cv2.LINE_AA)
    if sh.size(kp) < MIN_SIZE[sh.event]:
        return "re-check: too small, move closer", STATUS["re-check"], out
    probs = read_stickers(reader, img, kp, device)
    inside = readable(kp, sh, img.shape)
    v = check(probs, inside, sh, state, slips)
    colours = [bgr(sh.puzzle.colours[k]) for k in sh.puzzle.face_keys]
    r = max(3, int(sh.size(kp) / 22))
    for i, c in enumerate(sh.cell_centres(kp, img.shape)):
        if not inside[i]:
            continue
        p = (int(c[0]), int(c[1]))
        if probs[i, sh.colours] > 0.5:  # covered
            cv2.drawMarker(out, p, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 2 * r, 2)
            continue
        cv2.circle(out, p, r, colours[int(probs[i, : sh.colours].argmax())], -1, cv2.LINE_AA)
        cv2.circle(out, p, r, (0, 0, 255) if i in v.mismatches else (0, 0, 0), 3 if i in v.mismatches else 1, cv2.LINE_AA)
    text = v.status + (f": {v.slip}" if v.slip else f": {v.reason}" if v.status == "re-check" else "")
    return text, STATUS[v.status], out


def banner(img: np.ndarray, text: str, colour: tuple[int, int, int]) -> np.ndarray:
    cv2.rectangle(img, (0, 0), (img.shape[1], 44), (20, 20, 20), -1)
    cv2.putText(img, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("puzzle", choices=["skewb", "pyram", "fto"])
    ap.add_argument("scramble")
    ap.add_argument("--image", type=Path, help="a picture instead of the webcam")
    ap.add_argument("--save", type=Path, help="with --image: where to write the picture with the answer drawn on")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--port", type=int, default=8770, help="where the live page is served, on this computer only")
    ap.add_argument("--corners", type=Path)
    ap.add_argument("--reader", type=Path)
    ap.add_argument("--min-find", type=float, default=0.5)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    sh = shape(args.puzzle)
    finder = PolyCornerFinder.load(args.puzzle, args.corners or RUNS / "corners" / MODELS[args.puzzle][0], args.device)
    reader = load_poly_reader(str(args.reader or RUNS / "reader" / MODELS[args.puzzle][1] / "best.pt"), args.device)
    state = sh.puzzle.apply(sh.puzzle.solved, args.scramble.split())
    slips = single_slips(sh, args.scramble)

    if args.image:
        text, colour, out = judge(cv2.imread(str(args.image)), sh, finder, reader, state, slips, args.device, args.min_find)
        print(text)
        if args.save:
            cv2.imwrite(str(args.save), banner(out, text, colour))
        return
    serve(args, sh, finder, reader, state, slips)


def serve(args, sh, finder, reader, state, slips) -> None:
    """The webcam, judged frame by frame, as a page on this computer only (the project's OpenCV has no window of its own)."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    cam = cv2.VideoCapture(args.camera)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cam.read()[0]:
        raise SystemExit("no picture from the camera (allow camera access for this terminal, or try --camera 1)")

    class Page(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path != "/stream":
                body = f"<title>{args.puzzle} check</title><body style='margin:0;background:#111'><img src='/stream' style='width:100%'>".encode()
                self.send_response(200)
                self.send_header("content-type", "text/html")
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header("content-type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    ok, frame = cam.read()
                    if not ok:
                        break
                    text, colour, out = judge(frame, sh, finder, reader, state, slips, args.device, args.min_find)
                    jpg = cv2.imencode(".jpg", banner(out, text, colour), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                    self.wfile.write(b"--frame\r\ncontent-type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

    print(f"open http://127.0.0.1:{args.port} in a browser (Ctrl+C here to stop)")
    try:
        ThreadingHTTPServer(("127.0.0.1", args.port), Page).serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()


if __name__ == "__main__":
    main()
