# SPDX-License-Identifier: AGPL-3.0-or-later
"""A blank page, and the handwriting writing itself on it. No captions.

    uv run python examples/write_video.py out/me/library.json "Some text" -o writing.mp4

Needs ffmpeg on the PATH.
"""

from __future__ import annotations

import argparse
import subprocess

import cv2
import numpy as np

from inkhabit import library, render

W, H, FPS = 1920, 1080, 30
PAPER, INK = (247, 244, 237), (30, 28, 34)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("text")
    ap.add_argument("-o", "--out", default="writing.mp4")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--style", default="clean")
    ap.add_argument("--seconds", type=float, default=16.0, help="time spent writing")
    a = ap.parse_args()

    lib = library.load(a.library)
    human = render.style(lib, a.style).with_settings({"weight": 0.7})
    ink = render.Writer(lib, a.seed, human).write(a.text.replace("\\n", "\n"), x_height=34, max_width=W - 440)
    origin = np.array([(W - ink.size[0]) / 2, (H - ink.size[1]) / 2])

    pts = [np.round((s + origin) * 16).astype(np.int32) for s in ink.strokes]
    lengths = np.array([max(float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum()), 12.0) if len(s) > 1 else 12.0 for s in ink.strokes])
    ends = np.cumsum(lengths)

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", "-map_metadata", "-1", a.out],
        stdin=subprocess.PIPE)
    assert proc.stdin is not None
    page = np.full((H, W, 3), PAPER, np.uint8)
    blank = page.tobytes()
    for _ in range(int(0.8 * FPS)):  # a moment of blank page first
        proc.stdin.write(blank)

    done, n = 0, int(a.seconds * FPS)  # finished strokes stay on `page`; only the stroke in progress is redrawn
    for i in range(n):
        budget = ends[-1] * (i + 1) / n
        while done < len(pts) and ends[done] <= budget:
            _stroke(page, pts[done], ink.widths[done])
            done += 1
        frame = page
        if done < len(pts) and len(pts[done]) > 1:
            part = (budget - (ends[done] - lengths[done])) / lengths[done]
            k = max(2, int(len(pts[done]) * part))
            frame = page.copy()
            _stroke(frame, pts[done][:k], ink.widths[done])
        proc.stdin.write(frame.tobytes())
    last = page.tobytes()
    for _ in range(int(2.5 * FPS)):
        proc.stdin.write(last)
    proc.stdin.close()
    proc.wait()
    print(f"wrote {a.out}: {a.seconds + 3.3:.0f} seconds")


def _stroke(im: np.ndarray, q: np.ndarray, width: float) -> None:
    thick = max(1, int(round(width)))
    if len(q) == 1:
        cv2.circle(im, tuple(int(v) for v in q[0]), thick * 8, INK, -1, cv2.LINE_AA, 4)
    else:
        cv2.polylines(im, [q], False, INK, thick, cv2.LINE_AA, 4)


if __name__ == "__main__":
    main()
