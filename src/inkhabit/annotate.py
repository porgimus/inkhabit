# SPDX-License-Identifier: AGPL-3.0-or-later
"""Mark up a document by hand: underline, circle, box, bracket, strike through, highlight, arrows,
and notes written in the margin, all drawn with the same pen and the same unsteadiness as the writing.

Each mark is a pen path built from a few guide points, rounded and wobbled the way a hand would draw
it: an underline that is not quite straight, a circle that overshoots where it closes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pymupdf

from . import render

MARKS = ("underline", "double-underline", "strike", "circle", "box", "bracket", "highlight", "arrow", "tick", "star", "note")
COLOURS = {"blue": (0.08, 0.12, 0.45), "black": (0.1, 0.1, 0.12), "red": (0.72, 0.1, 0.1), "green": (0.1, 0.45, 0.2),
           "pencil": (0.35, 0.35, 0.37), "yellow": (1.0, 0.9, 0.2), "pink": (1.0, 0.45, 0.7), "orange": (1.0, 0.6, 0.15)}


class Pen:
    def __init__(self, lib: dict, seed: int = 0, size: float = 6.0):
        self.rng = np.random.default_rng(seed)
        self.unit = size  # the x-height of the accompanying writing, in points: marks are sized to match
        self.width = max(lib["meta"].get("font_weight", 0.16) * size, 0.95)
        self.wild = float(np.clip(lib.get("measured", {}).get("size", 0.1), 0.04, 0.15))

    def path(self, guide, closed: bool = False) -> np.ndarray:
        """A hand-drawn line through the guide points."""
        p = np.asarray(guide, float)
        for _ in range(2):  # round the corners
            q = [p[0]]
            for a, b in zip(p[:-1], p[1:]):
                q += [a * 0.75 + b * 0.25, a * 0.25 + b * 0.75]
            p = np.array(q + [p[-1]])
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        t = np.linspace(0, d[-1], max(8, int(d[-1] / (self.unit * 0.25)) + 2))
        p = np.c_[np.interp(t, d, p[:, 0]), np.interp(t, d, p[:, 1])]
        amp = self.unit * self.wild * 0.6
        f, ph = self.rng.uniform(0.6, 1.6, 2) / self.unit, self.rng.uniform(0, 6.28, 2)
        p[:, 0] += amp * np.sin(t * f[0] + ph[0])
        p[:, 1] += amp * np.sin(t * f[1] + ph[1])
        return p

    def j(self, scale: float = 1.0) -> float:
        return float(self.rng.normal(0, self.unit * 0.12 * scale))

    # -- marks; each returns a list of strokes in page points --------------------------------
    def underline(self, r, double: bool = False) -> list[np.ndarray]:
        x0, y0, x1, y1 = r
        y = y1 + self.unit * 0.22
        tilt = self.j(0.5)
        out = [self.path([[x0 - self.unit * 0.3 + self.j(), y + self.j(0.3)], [(x0 + x1) / 2, y + self.j(0.3)],
                          [x1 + self.unit * 0.3 + self.j(), y + tilt]])]
        if double:
            out.append(self.path([[x0 + self.j(), y + self.unit * 0.3], [x1 + self.j(), y + self.unit * 0.3 + tilt]]))
        return out

    def strike(self, r) -> list[np.ndarray]:
        x0, y0, x1, y1 = r
        y = (y0 + y1) / 2
        return [self.path([[x0 - self.unit * 0.3, y + self.j(0.5)], [x1 + self.unit * 0.3, y + self.j(0.5)]])]

    def circle(self, r) -> list[np.ndarray]:
        x0, y0, x1, y1 = r
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rx, ry = (x1 - x0) / 2 + self.unit * 0.4, (y1 - y0) / 2 + self.unit * 0.5  # snug: it must not swallow the next word
        start, turn = self.rng.uniform(200, 250), self.rng.uniform(385, 410)  # overshoots where it closes
        pts = []
        for i in range(31):
            a = math.radians(start + turn * i / 30)
            grow = 1 + 0.06 * i / 30  # a hand-drawn loop spirals out slightly
            pts.append([cx + rx * grow * math.cos(a) + self.j(0.25), cy + ry * grow * math.sin(a) + self.j(0.25)])
        return [self.path(pts)]

    def box(self, r) -> list[np.ndarray]:
        x0, y0, x1, y1 = r
        m = self.unit * 0.45
        a, b, c, d = [x0 - m, y0 - m], [x1 + m, y0 - m], [x1 + m, y1 + m], [x0 - m, y1 + m]
        return [self.path([[p[0] + self.j(0.6), p[1] + self.j(0.6)] for p in pair]) for pair in ((a, b), (b, c), (c, d), (d, a))]

    def bracket(self, r, side: str = "left") -> list[np.ndarray]:
        x0, y0, x1, y1 = r
        s = -1 if side == "left" else 1
        x = (x0 if side == "left" else x1) + s * self.unit * 0.9
        tick = -s * self.unit * 0.6
        return [self.path([[x + tick, y0 + self.j(0.4)], [x + self.j(0.3), y0 + self.unit * 0.3], [x + self.j(0.3), (y0 + y1) / 2],
                           [x + self.j(0.3), y1 - self.unit * 0.3], [x + tick, y1 + self.j(0.4)]])]

    def arrow(self, p0, p1, bend: float = 0.18) -> list[np.ndarray]:
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        v = p1 - p0
        n = np.array([-v[1], v[0]]) / (np.linalg.norm(v) + 1e-9)
        mid = (p0 + p1) / 2 + n * np.linalg.norm(v) * bend * self.rng.choice([-1, 1])
        shaft = self.path([p0, mid, p1])
        d = shaft[-1] - shaft[-4]
        d /= np.linalg.norm(d) + 1e-9
        size = self.unit * 1.1

        def barb(angle):
            c, s = math.cos(angle), math.sin(angle)
            return shaft[-1] - size * np.array([d[0] * c - d[1] * s, d[0] * s + d[1] * c])

        return [shaft, self.path([barb(0.5), shaft[-1], barb(-0.5)])]

    def tick(self, p) -> list[np.ndarray]:
        x, y = p
        u = self.unit
        return [self.path([[x, y - u * 0.5], [x + u * 0.5, y], [x + u * 1.7, y - u * 1.8]])]

    def star(self, p) -> list[np.ndarray]:
        x, y = p
        u = self.unit * 0.9
        return [self.path([[x + u * math.cos(a) + self.j(0.2), y + u * math.sin(a)], [x - u * math.cos(a), y - u * math.sin(a) + self.j(0.2)]])
                for a in (0.0, math.pi / 3, 2 * math.pi / 3)]


def _targets(page, item: dict) -> list[tuple[float, float, float, float]]:
    """Where a mark goes: an explicit rect, or every line of the printed text it names."""
    if "rect" in item:
        return [tuple(item["rect"])]
    if "find" in item:
        hits = page.search_for(item["find"])
        if not hits:
            from .forms import is_picture_only

            hits = _find_in_picture(page, item["find"]) if is_picture_only(page) else []
        if not hits:
            return []
        n = item.get("occurrence", 1) - 1
        first = hits[0]
        runs, cur = [], [first]  # one phrase can break across lines: search gives one box per line
        for h in hits[1:]:
            if abs(h.y0 - cur[-1].y0) < 2 or (h.y0 > cur[-1].y0 and h.x0 < cur[-1].x0 and h.y0 - cur[-1].y1 < cur[-1].height):
                cur.append(h)
            else:
                runs.append(cur)
                cur = [h]
        runs.append(cur)
        run = runs[min(n, len(runs) - 1)]
        return [(h.x0, h.y0, h.x1, h.y1) for h in run]
    return []


def _find_in_picture(page, phrase: str) -> list:
    """Find printed words on a scanned page by reading it. Tolerates small misreadings."""
    import re
    from difflib import SequenceMatcher

    from .forms import ocr_words

    norm = lambda t: re.sub(r"[^a-z0-9]", "", t.lower())  # noqa: E731
    words = ocr_words(page)
    want = [norm(w) for w in phrase.split() if norm(w)]
    if not want or not words:
        return []
    target = "".join(want)
    # the recogniser's order is not always reading order (columns): also try column by column, top to bottom
    lines: dict[int, list] = {}
    for w in words:
        lines.setdefault(w["line"], []).append(w)
    mid = page.rect.width * 0.47
    by_column = [w for _, ws in sorted(lines.items(), key=lambda kv: (min(x["rect"][0] for x in kv[1]) > mid, min(x["rect"][1] for x in kv[1])))
                 for w in sorted(ws, key=lambda x: x["rect"][0])]
    best, span, words_used = 0.0, None, words
    for order in (words, by_column):
        for i in range(len(order)):
            for n in (len(want) - 1, len(want), len(want) + 1):
                if n < 1 or i + n > len(order):
                    continue
                got = "".join(norm(w["text"]) for w in order[i : i + n])
                r = SequenceMatcher(None, got, target).ratio()
                if r > best:
                    best, span, words_used = r, (i, i + n), order
    if span is None or best < 0.8:
        return []
    words = words_used
    hits, by_line = [], {}
    for w in words[span[0] : span[1]]:
        by_line.setdefault(w["line"], []).append(w["rect"])
    for rects in by_line.values():
        hits.append(pymupdf.Rect(min(r[0] for r in rects), min(r[1] for r in rects), max(r[2] for r in rects), max(r[3] for r in rects)))
    return hits


def _text_blocks(page) -> list:
    """Blocks of printed text as (x0, y0, x1, y1, ...): from the PDF, or by reading a scanned page."""
    from .forms import is_picture_only, ocr_words

    if not is_picture_only(page):
        return [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    lines: dict[int, list] = {}
    for w in ocr_words(page):
        lines.setdefault(w["line"], []).append(w["rect"])
    rects = sorted(((min(r[0] for r in rs), min(r[1] for r in rs), max(r[2] for r in rs), max(r[3] for r in rs)) for rs in lines.values()),
                   key=lambda r: (round(r[0] / 40), r[1]))
    blocks: list[list[float]] = []
    for r in rects:  # lines stacked closely under one another, starting at about the same place, are one block
        if blocks and abs(r[0] - blocks[-1][0]) < 25 and 0 <= r[1] - blocks[-1][3] < 2.2 * (r[3] - r[1]):
            b = blocks[-1]
            blocks[-1] = [min(b[0], r[0]), b[1], max(b[2], r[2]), r[3]]
        else:
            blocks.append(list(r))
    return [tuple(b) + ("text", 0, 0) for b in blocks]


def annotate(pdf: str | Path, items: list[dict], lib: dict, out: str | Path, seed: int = 0, colour: str = "blue",
             x_height: float = 5.0) -> list[dict]:
    """Draw each item on the PDF. Items: {"type": one of MARKS, "page": 0, "find": "printed words" | "rect": [...],
    ["colour"], and for notes: "text", "side": "left"|"right"|"top"|"bottom", ["arrow": true], ["width"]}."""
    report = []
    from .forms import open_doc

    with open_doc(pdf) as doc:
        from .forms import is_picture_only

        for pg in doc:  # decide what kind of page each is before anything is drawn on it
            is_picture_only(pg)
        for k, item in enumerate(items):
            page = doc[item.get("page", 0)]
            kind = item["type"]
            rgb = COLOURS.get(item.get("colour", colour), COLOURS["blue"])
            pen = Pen(lib, seed + k, item.get("x_height", x_height))
            boxes = _targets(page, item)
            if kind != "note" and not boxes and "at" not in item and "from" not in item:
                report.append({**item, "status": "target not found"})
                continue
            strokes: list[np.ndarray] = []
            if kind in ("underline", "double-underline"):
                for b in boxes:
                    strokes += pen.underline(b, kind == "double-underline")
            elif kind == "strike":
                for b in boxes:
                    strokes += pen.strike(b)
            elif kind in ("circle", "box", "bracket"):
                x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
                x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
                area = (x0, y0, x1, y1)
                if kind == "bracket":
                    strokes += pen.bracket(area, item.get("side", "left"))
                else:
                    strokes += pen.circle(area) if kind == "circle" else pen.box(area)
            elif kind == "highlight":
                hl = COLOURS.get(item.get("colour", "yellow"), COLOURS["yellow"])
                for x0, y0, x1, y1 in boxes:
                    line = pen.path([[x0 - 1, (y0 + y1) / 2 + pen.j(0.3)], [x1 + 1, (y0 + y1) / 2 + pen.j(0.3)]])
                    shape = page.new_shape()
                    shape.draw_polyline([tuple(p) for p in line])
                    shape.finish(color=hl, width=(y1 - y0) * 0.85, lineCap=0, closePath=False, stroke_opacity=0.38)
                    shape.commit()
            elif kind == "arrow":
                strokes += pen.arrow(item["from"], item["to"])
            elif kind in ("tick", "star"):
                at = item.get("at") or [boxes[0][0] - pen.unit * 2.5, (boxes[0][1] + boxes[0][3]) / 2]
                strokes += getattr(pen, kind)(at)
            elif kind == "note":
                strokes, note_report = _note(page, item, lib, pen, boxes, seed + k, rgb)
                report.append({**item, **note_report})
                continue
            if strokes:
                render.onto_pdf_page(page, render.Ink(strokes, [pen.width] * len(strokes)), (0, 0), rgb)
            report.append({**item, "status": "ok"})
        doc.save(out)
    return report


def _note(page, item, lib, pen: Pen, boxes, seed: int, rgb):
    """A handwritten note in the margin beside its target, optionally with an arrow to it."""
    text_blocks = _text_blocks(page)
    wide = [b for b in text_blocks if b[2] - b[0] > 0.35 * page.rect.width]  # the text itself, not page numbers or headers
    text_blocks = wide or text_blocks
    body = (min(b[0] for b in text_blocks), min(b[1] for b in text_blocks), max(b[2] for b in text_blocks),
            max(b[3] for b in text_blocks)) if text_blocks else (72, 72, page.rect.width - 72, page.rect.height - 72)
    side = item.get("side", "right")
    pad = 8.0
    human, size = render.situation(lib, item.get("for", "label"))
    xh = item.get("x_height", pen.unit) * size / 0.8
    ty = (boxes[0][1] + boxes[0][3]) / 2 if boxes else item.get("y", body[1] + 20)  # the line the words start on
    moved = None
    if side in ("left", "right"):
        # the margin that matters is the one beside the column the words are in, not the edge of the page
        col = body
        if boxes:
            tx = (boxes[0][0] + boxes[0][2]) / 2
            inside = [b for b in text_blocks if b[0] - 2 <= tx <= b[2] + 2 and b[1] - 2 <= ty <= b[3] + 2]
            if inside:
                same = [b for b in text_blocks if abs(b[0] - inside[0][0]) < 20]  # this column, top to bottom
                col = (min(b[0] for b in same), min(b[1] for b in same), max(b[2] for b in same), max(b[3] for b in same))

        def room(which):
            if which == "right":
                nxt = [b[0] for b in text_blocks if b[0] > col[2] + 2 and b[1] < ty + 40 and b[3] > ty - 40]
                return col[2] + pad * 1.6, (min(nxt) if nxt else page.rect.width) - pad - (col[2] + pad * 1.6)
            prev = [b[2] for b in text_blocks if b[2] < col[0] - 2 and b[1] < ty + 40 and b[3] > ty - 40]
            start = (max(prev) if prev else 0) + pad
            return start, col[0] - pad * 2.6 - start

        x0, width = room(side)
        if width < xh * 6:  # no space on that side of this column: try the other side before giving up
            other_side = "left" if side == "right" else "right"
            x0b, wb = room(other_side)
            if wb >= xh * 6:
                side, x0, width, moved = other_side, x0b, wb, other_side
        body = col
    else:
        x0, width = body[0], body[2] - body[0]
    width = item.get("width", width)
    if width < xh * 6:
        return [], {"status": "no room in that margin"}
    writer = render.Writer(lib, seed, human)
    writer.min_pen = 0.95
    ink, used = writer.write_to_fit(item["text"], width, None, xh, min_x_height=2.6)
    if side in ("left", "right"):
        oy = float(np.clip(ty - ink.size[1] / 2, pad, page.rect.height - ink.size[1] - pad))
    else:
        oy = max(pad, body[1] - ink.size[1] - pad) if side == "top" else min(body[3] + pad, page.rect.height - ink.size[1] - pad)
    render.onto_pdf_page(page, ink, (x0, oy), rgb)
    strokes = []
    if item.get("arrow") and boxes:
        b = boxes[-1] if side == "right" else boxes[0]
        start = [x0 - 2 if side == "right" else x0 + ink.size[0] + 2, oy + min(ink.size[1] / 2, used * 2.2)]
        # point at the target if it is near that edge of the text; otherwise at its line, so the arrow
        # does not run across the print
        edge = body[2] if side == "right" else body[0]
        near = abs((b[2] if side == "right" else b[0]) - edge) < 10
        end = [(b[2] + 3 if side == "right" else b[0] - 3) if near else (edge + 4 if side == "right" else edge - 4), (b[1] + b[3]) / 2]
        if side in ("top", "bottom"):
            start = [x0 + min(ink.size[0], 40), oy + ink.size[1] if side == "top" else oy]
            end = [(b[0] + b[2]) / 2, b[1] - 2 if side == "top" else b[3] + 2]
        strokes = pen.arrow(start, end, bend=0.12)
        render.onto_pdf_page(page, render.Ink(strokes, [pen.width] * len(strokes)), (0, 0), rgb)
    small = used < 4.2  # too small to read comfortably: the note is too long for that margin
    return strokes, {"status": "ok" if not ink.missing and not small else "check", "x_height": round(used, 2),
                     **({"problem": "too small to read: shorten the note or use another side"} if small else {}),
                     "missing": "".join(sorted(ink.missing)),
                     **({"moved_to": moved} if moved else {})}


def load_items(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())
