# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build an installable .ttf from a handwriting library.

Each character keeps several real variants. The font's `calt` feature rotates
through them as you type, so the same letter is not repeated identically, and
the space glyph rotates through slightly different widths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fontTools.feaLib.builder import addOpenTypeFeaturesFromString
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from shapely.geometry import LineString, MultiPolygon, Point
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

XH = 400  # font units per x-height (1000 units per em)
N_ALT = 8
NAMES = {
    " ": "space", "!": "exclam", "%": "percent", "(": "parenleft", ")": "parenright",
    ",": "comma", ".": "period", ":": "colon", ";": "semicolon", "?": "question",
    "'": "quotesingle", "-": "hyphen", "$": "dollar", "&": "ampersand", "/": "slash", "_": "underscore", "=": "equal", "+": "plus", "@": "at", "*": "asterisk", "#": "numbersign", "<": "less", ">": "greater",
    "[": "bracketleft", "]": "bracketright", "{": "braceleft", "}": "braceright", "|": "bar", "\\": "backslash",
    "~": "asciitilde", "^": "asciicircum", "`": "grave", '"': "quotedbl",
    **{str(i): n for i, n in enumerate(
        ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"])},
}
# characters the writer never produced borrow a similar-looking one
FALLBACK = {**{c.upper(): (c, 1.45) for c in "cjkmnopsuvwxyz"}, **{c: (c.upper(), 0.7) for c in "cjkopsuvwxz"},
            }


def word_gap(lib: dict) -> float:
    """Space between words, in x-heights.

    Notes are written loosely: this writer's measured gap is lib["meta"]["word_gap"]. Set as running
    text that reads as too gappy, so by default we tighten it; "word_gap_use" overrides the choice.
    """
    meta = lib["meta"]
    if "word_gap_use" in meta:
        return float(meta["word_gap_use"])
    return float(min(max(meta.get("word_gap", 1.2) * 0.6, 0.8), 1.3))


def glyph_name(ch: str) -> str:
    if ch in NAMES:
        return NAMES[ch]
    if ch.isascii() and ch.isalpha():
        return ch if ch.islower() else f"{ch}.cap"
    return f"uni{ord(ch):04X}"  # accented letters and symbols


def _outline(strokes, pen_width: float, scale: float = 1.0, lean: float = 0.0):
    r = pen_width * XH / 2
    shapes = []
    for s in strokes:
        pts = [((x - y * lean) * XH * scale, -y * XH * scale) for x, y in s]  # lean: samples are stored upright
        shapes.append(Point(pts[0]).buffer(r, 6) if len(pts) == 1 or LineString(pts).length < 1
                      else LineString(pts).buffer(r, quad_segs=6, join_style="round"))
    geom = unary_union(shapes).simplify(2.0)
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    pen = TTGlyphPen(None)
    for poly in polys:
        poly = orient(poly, -1)  # TrueType: outer contours clockwise
        for ring in [poly.exterior, *poly.interiors]:
            pts = [(round(x), round(y)) for x, y in ring.coords[:-1]]
            if len(pts) < 3:
                continue
            pen.moveTo(pts[0])
            for p in pts[1:]:
                pen.lineTo(p)
            pen.closePath()
    return pen.glyph()


def with_fallbacks(lib: dict) -> tuple[dict, list[str]]:
    """Every character we can write: {char: (variants, scale)}, plus the ones that are not the writer's own samples."""
    from . import extend

    glyphs = {ch: (g["variants"], 1.0) for ch, g in lib["glyphs"].items()}
    added = []
    for ch, variants in extend.compose(lib).items():
        if ch not in glyphs:
            glyphs[ch] = (variants, 1.0)
            added.append(ch)
    for ch, (src, scale) in FALLBACK.items():
        if ch not in glyphs and src in lib["glyphs"]:
            glyphs[ch] = (lib["glyphs"][src]["variants"], scale)
            added.append(ch)
    # a letter or digit with no sample and no relative: a neutral shape fitted to the writer
    gone = "".join(c for c in extend.ALNUM if c not in glyphs)
    for ch, variants in (extend.neutral(lib, gone) if gone else {}).items():
        glyphs[ch] = (variants, 1.0)
        added.append(ch)
    for ch, variants in extend.extrapolate(lib).items():
        if ch not in glyphs:
            glyphs[ch] = (variants, 1.0)
            added.append(ch)
    for ch, src in (("¿", "?"), ("¡", "!")):  # upside-down marks, once the upright one exists from any source
        if ch not in glyphs and src in glyphs:
            vs = []
            for v in glyphs[src][0]:
                ys = [p[1] for s in v["strokes"] for p in s]
                lo, hi = min(ys), max(ys)
                vs.append({"strokes": [[[round(v["width"] - x, 3), round(lo + hi - y + 0.55, 3)] for x, y in s] for s in v["strokes"]],
                           "width": v["width"], "source": "composed"})
            glyphs[ch] = (vs, 1.0)
            added.append(ch)
    return glyphs, added


def build_font(lib: dict, path: str | Path, family: str = "My Handwriting", weight: float | None = None) -> dict:
    pen_width = weight or lib["meta"].get("font_weight", 0.13)
    gap = lib["spacing"]["letter_gap_mean"]
    gap_words = word_gap(lib)
    import math

    lean = math.tan(math.radians(lib["meta"].get("slant", 0.0)))
    table, borrowed = with_fallbacks(lib)
    order, cmap, glyphs, metrics = [".notdef"], {}, {}, {}
    empty = TTGlyphPen(None).glyph()
    glyphs[".notdef"], metrics[".notdef"] = empty, (XH, 0)
    classes: list[list[str]] = [[] for _ in range(N_ALT)]

    rng = np.random.default_rng(0)
    for i in range(N_ALT):
        name = "space" if i == 0 else f"space.alt{i}"
        order.append(name)
        glyphs[name] = empty
        metrics[name] = (round(gap_words * XH * (1 if i == 0 else rng.uniform(0.8, 1.25))), 0)
        classes[i].append(name)
    cmap[32] = "space"

    for ch, (variants, scale) in sorted(table.items()):
        base = glyph_name(ch)
        cmap[ord(ch)] = base
        names = []
        for i, var in enumerate(variants[:N_ALT]):
            name = base if i == 0 else f"{base}.alt{i}"
            g = _outline(var["strokes"], pen_width, scale, lean)
            lsb = round(gap * XH / 2)
            if g.numberOfContours:
                g.coordinates.translate((lsb + round(pen_width * XH / 2), 0))
            order.append(name)
            glyphs[name] = g
            metrics[name] = (round((var["width"] * scale + gap + pen_width) * XH), lsb)
            names.append(name)
        for i in range(N_ALT):
            classes[i].append(names[i % len(names)])

    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf(glyphs)
    for g in glyphs.values():
        g.recalcBounds(fb.font["glyf"])
    fb.setupHorizontalMetrics({n: (adv, glyphs[n].xMin if glyphs[n].numberOfContours else 0)
                               for n, (adv, _) in metrics.items()})
    fb.setupHorizontalHeader(ascent=950, descent=-450)
    fb.setupNameTable({"familyName": family, "styleName": "Regular"})
    fb.setupOS2(sTypoAscender=950, sTypoDescender=-450, usWinAscent=1000, usWinDescent=500, sxHeight=XH, sCapHeight=round(1.7 * XH))
    fb.setupPost()
    # pair spacing: each pair of characters gets its own measured gap (all variants of a letter alike)
    by_char: dict[str, list[str]] = {}
    for ch, (variants, _) in table.items():
        base = glyph_name(ch)
        by_char[ch] = [base] + [f"{base}.alt{i}" for i in range(1, min(len(variants), N_ALT))]
    kerns = []
    for pair, g in sorted(lib.get("pair_gap", {}).items()):
        delta = round((g - gap) * XH)
        if len(pair) == 2 and pair[0] in by_char and pair[1] in by_char and abs(delta) >= 12:
            kerns.append(f"  pos [{' '.join(by_char[pair[0]])}] [{' '.join(by_char[pair[1]])}] {max(-160, min(160, delta))};")
    fea = "\n".join(f"@v{i} = [{' '.join(c)}];" for i, c in enumerate(classes))
    rules = "\n".join(f"  sub @v{i} @v0' by @v{(i + 1) % N_ALT};" for i in range(N_ALT - 1))
    fea += f"\nfeature calt {{\n{rules}\n}} calt;\n"
    if kerns:
        fea += "feature kern {\n" + "\n".join(kerns) + "\n} kern;\n"
    addOpenTypeFeaturesFromString(fb.font, fea)
    fb.save(str(path))
    return {"glyphs": len(order), "characters": len(cmap), "borrowed": borrowed}


def typeset(ttf: str | Path, text: str, out: str | Path, size: float = 48, color=(40, 25, 20)) -> None:
    """Set text with the finished font file, as an app would: HarfBuzz shaping (alternates on), outlines filled."""
    import cv2
    import uharfbuzz as hb
    from fontTools.pens.recordingPen import DecomposingRecordingPen
    from fontTools.ttLib import TTFont

    glyphs = TTFont(str(ttf)).getGlyphSet()
    hbfont = hb.Font(hb.Face(hb.Blob.from_file_path(str(ttf))))
    k = size / 1000
    lines, runs, width = text.split("\n"), [], 0.0
    for line in lines:
        if not line:  # an empty line still takes up a line
            runs.append([])
            continue
        buf = hb.Buffer()
        buf.add_str(line)
        buf.guess_segment_properties()
        hb.shape(hbfont, buf, {"calt": True})
        runs.append([(hbfont.glyph_to_string(i.codepoint), p.x_advance) for i, p in zip(buf.glyph_infos, buf.glyph_positions)])
        width = max(width, sum(a for _, a in runs[-1]) * k)
    im = np.full((int(len(lines) * size * 1.5 + size), int(width + size), 3), 255, np.uint8)
    for li, run in enumerate(runs):
        x, base = size / 2, size * 1.3 + li * size * 1.5
        for name, adv in run:
            pen = DecomposingRecordingPen(glyphs)
            glyphs[name].draw(pen)
            polys, cur = [], []
            for op, args in pen.value:
                if op == "moveTo":
                    cur = [args[0]]
                elif op == "lineTo":
                    cur.append(args[0])
                elif op == "closePath" and len(cur) > 2:
                    polys.append((np.array([[x + px * k, base - py * k] for px, py in cur]) * 16).astype(np.int32))
            if polys:
                cv2.fillPoly(im, polys, color, cv2.LINE_AA, 4)
            x += adv * k
    if not cv2.imwrite(str(out), im):
        raise SystemExit(f"could not write {out}: use an existing folder and an image name such as .png")
