# SPDX-License-Identifier: AGPL-3.0-or-later
"""Make a short demo video from a inkhabit library, styled as an agent session in a terminal.

    uv run python examples/demo_video.py out/me/library.json -o demo.mp4

Commands are typed out in a terminal; the handwriting they produce draws itself, stroke by stroke,
on paper cards. All text in the video is made up. Needs ffmpeg on the PATH.
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import replace

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from inkhabit import library, render

W, H, FPS = 1920, 1080, 30
BG, FG, DIM = (13, 15, 18), (215, 218, 224), (105, 113, 126)
GREEN, ORANGE, BLUE = (126, 231, 135), (255, 166, 87), (121, 192, 255)
PAPER, INK = (246, 243, 236), (32, 28, 30)
MONO = "/System/Library/Fonts/Menlo.ttc"
_fonts: dict[tuple[int, bool], ImageFont.FreeTypeFont] = {}


def mono(size: int, bold: bool = False):
    if (size, bold) not in _fonts:
        try:
            _fonts[(size, bold)] = ImageFont.truetype(MONO, size, index=1 if bold else 0)
        except OSError:
            _fonts[(size, bold)] = ImageFont.load_default(size=size)
    return _fonts[(size, bold)]


class Frame:
    """One video frame: terminal text and cards go on with PIL, ink with OpenCV."""

    def __init__(self):
        self.im = Image.new("RGB", (W, H), BG)
        self.d = ImageDraw.Draw(self.im)
        self.inks: list[tuple] = []

    def text(self, s: str, xy, size=34, color=FG, bold=False) -> float:
        self.d.text(xy, s, fill=color, font=mono(size, bold))
        return self.d.textlength(s, font=mono(size, bold))

    def typed(self, prefix: str, s: str, xy, t: float, size=34, color=FG, prefix_color=GREEN) -> None:
        """A command being typed, with a block cursor while it is in progress."""
        x = xy[0] + self.text(prefix, xy, size, prefix_color, True)
        n = int(round(len(s) * min(t, 1.0)))
        x += self.text(s[:n], (x, xy[1]), size, color)
        if 0 < t < 1:
            self.d.rectangle([x + 2, xy[1] + 4, x + size * 0.55, xy[1] + size * 1.15], fill=color)

    def card(self, box, alpha: float = 1.0) -> None:
        if alpha <= 0:
            return
        x0, y0, x1, y1 = box
        self.d.rounded_rectangle([x0 + 6, y0 + 10, x1 + 6, y1 + 10], 22, fill=(5, 6, 8))
        col = tuple(int(b + (p - b) * alpha) for b, p in zip(BG, PAPER))
        self.d.rounded_rectangle(box, 22, fill=col)

    def ink(self, ink, origin, progress=1.0, color=INK) -> None:
        self.inks.append((ink, origin, progress, color))

    def array(self) -> np.ndarray:
        a = np.asarray(self.im).copy()
        for ink, origin, progress, color in self.inks:
            draw(a, ink, origin, progress, color)
        return a


def draw(a: np.ndarray, ink: render.Ink, origin, progress: float, color) -> None:
    """Draw the first `progress` share of the ink, in the order it was written."""
    if progress <= 0:
        return
    lengths = [max(float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum()), 1.0) if len(s) > 1 else 1.0 for s in ink.strokes]
    budget = sum(lengths) * min(progress, 1.0)
    o = np.asarray(origin, float)
    for s, pw, ln in zip(ink.strokes, ink.widths, lengths):
        if budget <= 0:
            break
        p = s + o
        if budget < ln and len(p) > 1:  # this stroke is part-way through being written
            d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
            p = p[: max(2, int(np.searchsorted(d, budget)) + 1)]
        budget -= ln
        q = np.round(p * 16).astype(np.int32)
        thick = max(1, int(round(pw)))
        if len(q) == 1:
            cv2.circle(a, tuple(int(v) for v in q[0]), thick * 8, color, -1, cv2.LINE_AA, 4)
        else:
            cv2.polylines(a, [q], False, color, thick, cv2.LINE_AA, 4)


class Film:
    def __init__(self, path: str):
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
             "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart",
             "-map_metadata", "-1", path], stdin=subprocess.PIPE)
        self.frames = 0

    def scene(self, paint, seconds: float, hold: float = 1.0) -> None:
        """paint(frame, s) is called with s = seconds since the scene began; the last frame is then held."""
        assert self.proc.stdin is not None
        last = b""
        for i in range(int(seconds * FPS)):
            f = Frame()
            paint(f, (i + 1) / FPS)
            last = f.array().tobytes()
            self.proc.stdin.write(last)
            self.frames += 1
        for _ in range(int(hold * FPS)):
            self.proc.stdin.write(last)
            self.frames += 1

    def close(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        self.proc.wait()


def ramp(s: float, start: float, length: float) -> float:
    """0 before `start` seconds, rising to 1 over `length` seconds."""
    return float(np.clip((s - start) / length, 0, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("-o", "--out", default="inkhabit-demo.mp4")
    a = ap.parse_args()
    lib = library.load(a.library)
    film = Film(a.out)
    n_chars = len(render.with_fallbacks(lib)[0])

    def write(text, seed, xh=30.0, style="neat", width=None, **settings):
        human = render.style(lib, style).with_settings({"weight": 0.7, **settings})
        return render.Writer(lib, seed, human).write(text, x_height=xh, max_width=width)

    line = "Meet me by the library steps at four."

    # 1. the ask: a prompt, a tool call, and the result writing itself
    first = write(line, 7, xh=40)

    def s1(f, s):
        f.typed("> ", "write this note in my handwriting", (120, 150), ramp(s, 0.2, 1.3), 40)
        if s > 1.9:
            f.typed("● ", f'inkhabit write me.json "{line}" --seed 7', (120, 250), ramp(s, 1.9, 1.1), 30, DIM, ORANGE)
        f.card((120, 400, 1800, 760), ramp(s, 3.1, 0.3))
        f.ink(first, (220, 470), ramp(s, 3.3, 2.2))
        if s > 5.8:
            f.text("not a font. a pen your agent can hold.", (120, 860), 34, DIM)

    film.scene(s1, 6.2, 1.6)

    # 2. a font stamps; an agent writes
    word = "letter"
    stamp = render.Writer(lib, 1, replace(render.Human().scaled(0), whole_words=0, joins=0, weight=0.7)).write(word, x_height=44)
    hands = [write(word, sd, xh=44, style="natural", whole_words=0) for sd in (3, 8, 12, 21)]

    def s2(f, s):
        f.text("# a font stamps", (120, 90), 34, DIM)
        f.typed("$ ", 'for i in 1 2 3 4; do typeset "letter"; done', (120, 150), ramp(s, 0.1, 1.0), 32)
        f.text("# an agent writes", (120, 560), 34, DIM)
        if s > 3.0:
            f.typed("$ ", 'for i in 1 2 3 4; do inkhabit write "letter" --seed $i; done', (120, 620), ramp(s, 3.0, 1.1), 32)
        for k in range(4):
            x = 120 + k * 430
            f.card((x, 230, x + 390, 440), ramp(s, 1.2 + k * 0.12, 0.2))
            f.ink(stamp, (x + 60, 250), ramp(s, 1.4 + k * 0.3, 0.5))
            f.card((x, 700, x + 390, 910), ramp(s, 4.2 + k * 0.12, 0.2))
            f.ink(hands[k], (x + 60, 720), ramp(s, 4.4 + k * 0.45, 0.6))
        if s > 6.6:
            f.text("identical", (1560, 470), 28, DIM)
            f.text("never the same twice", (1330, 940), 28, GREEN)

    film.scene(s2, 7.0, 1.6)

    # 3. parameters: the same line rewritten under different flags
    flags = [("--style neat", write(line, 5, xh=38, style="neat")),
             ("--style rushed", write(line, 5, xh=38, style="rushed")),
             ("--set letter_height=1.25 slant=10 word_spacing=0.75",
              write(line, 5, xh=38, style="neat", letter_height=1.25, slant=10, word_spacing=0.75))]

    def s3(f, s):
        k = min(int(s // 2.4), 2)
        local = s - k * 2.4
        f.text("# handwriting, with flags", (120, 90), 34, DIM)
        f.typed("$ ", "inkhabit write me.json note.txt " + flags[k][0], (120, 170), ramp(local, 0.05, 0.6), 32)
        f.card((120, 330, 1800, 720))
        f.ink(flags[k][1], (220, 400), ramp(local, 0.7, 1.2))
        for i, name in enumerate(["letter_spacing", "word_spacing", "line_spacing", "letter_height", "letter_width", "slant",
                                  "weight", "tilt", "bounce", "drift", "wobble", "pressure", "joins", "whole_words"]):
            f.text(name, (120 + (i % 7) * 250, 820 + (i // 7) * 56), 26, DIM)

    film.scene(s3, 7.2, 1.2)

    # 4. where it comes from: the build log, and one letter's real variants
    log = [("$ inkhabit build ~/notes", FG), ("reading notebooks, scans and photos", DIM),
           ("pen strokes  ->  words  ->  letters   (forced alignment)", DIM),
           (f"{n_chars} characters, up to 8 real variants each", FG),
           (f"{len(lib.get('pairs', {}))} joined pairs, {len(lib.get('words', {}))} whole words lifted from the notes", FG),
           ("wrote library.json  handwriting.ttf  profile.md", GREEN)]
    variants = lib["glyphs"].get("a", {"variants": []})["variants"][:8]
    samples = [render.Ink([np.asarray(st, float) * 70 for st in v["strokes"]], [7.0] * len(v["strokes"])) for v in variants]

    def s4(f, s):
        for i, (txt, col) in enumerate(log):
            if s > 0.2 + i * 0.55:
                f.text(txt, (120, 110 + i * 62), 32, col)
        f.card((120, 560, 1800, 900), ramp(s, 3.4, 0.3))
        if s > 3.6:
            f.d.text((170, 590), "a", fill=(150, 90, 40), font=mono(30, True))
            f.d.text((215, 594), "eight samples of one letter, all real", fill=(120, 120, 120), font=mono(26))
        for k, ink in enumerate(samples):
            f.ink(ink, (230 + k * 195, 800), ramp(s, 3.7 + k * 0.22, 0.3))

    film.scene(s4, 6.0, 1.4)

    # 5. a form
    fields = [("name", "Alex Morgan"), ("date", "14 March 2027"), ("city", "Springfield, Illinois"), ("subject", "Astronomy and gardening")]
    answers = [write(v, 30 + i, xh=30, style="neat") for i, (_, v) in enumerate(fields)]

    def s5(f, s):
        f.typed("$ ", "inkhabit fill me.json form.pdf entries.json", (120, 90), ramp(s, 0.1, 0.9), 32)
        f.card((120, 190, 1800, 960), ramp(s, 1.0, 0.3))
        for k, ((label, _), ink) in enumerate(zip(fields, answers)):
            y = 330 + k * 160
            if s > 1.2:
                f.d.text((220, y - 6), label, fill=(110, 110, 110), font=mono(30))
                f.d.line([(470, y + 44), (1680, y + 44)], fill=(170, 170, 170), width=2)
            f.ink(ink, (490, y + 44 - 2.2 * 30 - 6), ramp(s, 1.4 + k * 0.9, 0.8), (28, 38, 110))
        if s > 5.0:
            f.d.text((220, 895), "made-up details", fill=(150, 150, 150), font=mono(24))

    film.scene(s5, 5.4, 1.4)

    # 6. characters that were never written
    groups = [("found in the notes", "= + - > × →"), ("built from those", "← ↑ ↓ ≥ ≤ ± ≠"), ("extrapolated to match", "* # & @ [ ] { } ✓")]
    sym = [(name, write(chars, 6, xh=36)) for name, chars in groups]

    def s6(f, s):
        f.text("# no sample of a character? work it out", (120, 90), 34, DIM)
        f.card((120, 190, 1800, 900), ramp(s, 0.2, 0.3))
        for k, (name, ink) in enumerate(sym):
            if s > 0.5 + k * 1.1:
                f.d.text((200, 300 + k * 210), name, fill=(110, 110, 110), font=mono(28))
            f.ink(ink, (760, 240 + k * 210), ramp(s, 0.6 + k * 1.1, 0.9))

    film.scene(s6, 4.2, 1.4)

    # 7. end: white ink on the dark terminal
    end = write("your notes in, your handwriting out", 14, xh=46, weight=0.8)

    def s7(f, s):
        f.ink(end, ((W - end.size[0]) / 2, 330), ramp(s, 0.2, 2.2), (236, 236, 240))
        if s > 2.6:
            label = "inkhabit   an agent skill"
            f.text(label, ((W - f.d.textlength(label, font=mono(32))) / 2, 660), 32, DIM)

    film.scene(s7, 3.0, 2.2)
    film.close()
    print(f"wrote {a.out}: {film.frames / FPS:.0f} seconds")


if __name__ == "__main__":
    main()
