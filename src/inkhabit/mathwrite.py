# SPDX-License-Identifier: AGPL-3.0-or-later
"""Handwritten maths: fractions, powers, roots, integrals and worked steps lined up on the equals sign.

Accepts a small, forgiving subset of LaTeX, because that is what people and agents already write:

    x^2 + 3x - 4 = 0          \\frac{a}{b}      \\sqrt{a^2 + b^2}      \\int_0^1 x^2 dx
    \\sum_{i=1}^{n} i          x_1, \\theta, \\pi, \\Delta, \\infty      \\times \\div \\pm \\leq \\geq \\neq \\to
    \\text{so c is 5}    (a + b)^2     sin, cos, tan, log, ln, lim are kept as words

Everything is built from the writer's own digits, letters and signs; the fraction bars, root signs and
tall brackets are drawn with the same unsteady pen. Several lines given together are treated as worked
steps and aligned on their first equals sign, the way people set out a solution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .annotate import Pen
from .render import Ink, Writer

SYMBOLS = {"times": "×", "div": "÷", "pm": "±", "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "neq": "≠", "ne": "≠", "to": "→",
           "rightarrow": "→", "leftarrow": "←", "approx": "≈", "cdot": "·", "pi": "π", "theta": "θ", "Delta": "Δ", "infty": "∞",
           "alpha": "α", "mu": "μ", "lambda": "λ", "sigma": "σ", "prime": "′", "degree": "°", "uparrow": "↑", "downarrow": "↓",
           "quad": " ", ",": " ", ";": " ", "percent": "%"}
BINARY = set("=+-×÷→←<>≤≥≠±≈")
WORDS = ("sin", "cos", "tan", "log", "ln", "lim", "max", "min", "exp")


@dataclass
class Box:
    """Ink for one piece of an expression, in x-heights. Baseline at y = 0, y down."""

    strokes: list = field(default_factory=list)
    w: float = 0.0
    a: float = 0.0  # how far it rises above the baseline
    d: float = 0.0  # how far it drops below
    kind: str = "ord"  # ord | bin | big | space

    def moved(self, dx: float, dy: float = 0.0, k: float = 1.0) -> "Box":
        st = [np.asarray(s, float) * k + [dx, dy] for s in self.strokes]
        return Box(st, self.w * k, self.a * k - dy, self.d * k + dy, self.kind)


class Maths:
    def __init__(self, writer: Writer, lib: dict, seed: int = 0):
        self.w, self.lib = writer, lib
        self.pen = Pen(lib, seed, size=1.0)
        self.missing: set[str] = set()

    # ---- pieces -----------------------------------------------------------------------------
    def text(self, s: str, kind: str = "ord") -> Box:
        strokes, width, miss = self.w._word(s)
        self.missing |= miss
        if not strokes:
            return Box([], max(width, 0.3), 0.0, 0.0, kind)
        pts = np.vstack(strokes)
        x0 = pts[:, 0].min()
        return Box([s_ - [x0, 0] for s_ in strokes], float(pts[:, 0].max() - x0), float(max(-pts[:, 1].min(), 0)), float(max(pts[:, 1].max(), 0)), kind)

    def drawn(self, guide) -> np.ndarray:
        return self.pen.path(guide)

    def row(self, boxes: list[Box]) -> Box:
        out, x, a, d = [], 0.0, 0.0, 0.0
        prev = None
        for b in boxes:
            if b.kind == "space":
                x += b.w
                continue
            gap = 0.0 if prev is None else 0.6 if "bin" in (b.kind, prev.kind) else 0.3 if {"big", "stack"} & {prev.kind, b.kind} else 0.2
            x += gap
            out += [np.asarray(s, float) + [x, 0] for s in b.strokes]
            x += b.w
            a, d = max(a, b.a), max(d, b.d)
            prev = b
        return Box(out, x, a, d)

    def scripts(self, base: Box, sup: Box | None, sub: Box | None) -> Box:
        out, w = list(base.strokes), base.w
        a, d = base.a, base.d
        if base.kind == "stack":  # sums and limits: what is above and below sits centred over and under the sign
            parts = [(sup, True), (sub, False)]
            widest = max([base.w] + [p.w * 0.62 for p, _ in parts if p is not None])
            shift = (widest - base.w) / 2
            out = [np.asarray(s, float) + [shift, 0] for s in out]
            for p, above in parts:
                if p is None:
                    continue
                dy = -(base.a + 0.2 + p.d * 0.62) if above else base.d + 0.25 + p.a * 0.62
                s_ = p.moved((widest - p.w * 0.62) / 2, dy, 0.62)
                out += s_.strokes
                a, d = max(a, s_.a), max(d, s_.d)
            return Box(out, widest, a, d, "big")
        big = base.kind == "big"
        x = base.w + 0.05
        if sup is not None:
            s_ = sup.moved(x, -(max(base.a - 0.3, 0.8) if not big else base.a - 0.3), 0.62)
            out += s_.strokes
            w, a = max(w, x + s_.w), max(a, s_.a)
        if sub is not None:
            xs = x - (0.1 if big else 0.0)
            s_ = sub.moved(xs, (0.32 if not big else base.d + 0.05), 0.62)
            out += s_.strokes
            w, d = max(w, xs + s_.w), max(d, s_.d)
        return Box(out, w, a, d, "big" if big else "ord")

    def frac(self, num: Box, den: Box) -> Box:
        num, den = num.moved(0, 0, 0.92), den.moved(0, 0, 0.92)
        w = max(num.w, den.w) + 0.35
        axis = -0.52
        n = num.moved((w - num.w) / 2, axis - 0.24 - num.d)
        dn = den.moved((w - den.w) / 2, axis + 0.3 + den.a)
        bar = self.drawn([[0.0, axis + self.pen.j(0.2)], [w / 2, axis], [w, axis + self.pen.j(0.2)]])
        return Box(n.strokes + dn.strokes + [bar], w, n.a, dn.d)

    def sqrt(self, c: Box) -> Box:
        top = -(max(c.a, 1.0) + 0.4)
        sign = self.drawn([[0.0, -0.45], [0.12, -0.5], [0.3, c.d + 0.05], [0.3, c.d + 0.05], [0.5, top], [0.5, top],
                           [0.62 + c.w + 0.12, top + self.pen.j(0.2)]])
        inner = c.moved(0.62)
        return Box([sign] + inner.strokes, 0.62 + c.w + 0.15, -top, max(c.d, 0.05))

    def fence(self, c: Box) -> Box:
        """Brackets that grow with what is inside them."""
        if c.a <= 1.9 and c.d <= 0.7:
            return self.row([self.text("("), c, self.text(")")])
        top, bot = -(c.a + 0.15), c.d + 0.15
        mid = (top + bot) / 2
        left = self.drawn([[0.32, top], [0.05, mid - (bot - top) * 0.15], [0.05, mid + (bot - top) * 0.15], [0.32, bot]])
        right = self.drawn([[0.0, top], [0.27, mid - (bot - top) * 0.15], [0.27, mid + (bot - top) * 0.15], [0.0, bot]]) + [0.4 + c.w + 0.08, 0]
        return Box([left] + c.moved(0.4).strokes + [right], 0.4 + c.w + 0.4, -top, bot)

    # ---- reading the expression ---------------------------------------------------------------
    def parse(self, s: str, i: int = 0, close: str | None = None) -> tuple[list[Box], int]:
        out: list[Box] = []

        def arg(j: int) -> tuple[Box, int]:
            while j < len(s) and s[j] == " ":
                j += 1
            if j < len(s) and s[j] == "{":
                inner, j2 = self.parse(s, j + 1, "}")
                return self.row(inner), j2
            if j < len(s) and s[j] == "\\":
                boxes, j2 = self.parse_one(s, j)
                return self.row(boxes), j2
            m = re.match(r"\d+|.", s[j:]) if j < len(s) else None
            return (self.text(m.group()), j + len(m.group())) if m else (Box(), j)

        while i < len(s):
            ch = s[i]
            if close and ch == close:
                return out, i + 1
            if ch in "^_":
                first, i = arg(i + 1)
                sup, sub = (first, None) if ch == "^" else (None, first)
                if i < len(s) and s[i] in "^_" and s[i] != ch:  # both a power and a suffix
                    second, i = arg(i + 1)
                    sup, sub = (second, sub) if sup is None else (sup, second)
                base = out.pop() if out else Box()
                out.append(self.scripts(base, sup, sub))
                continue
            boxes, i = self.parse_one(s, i)
            out += boxes
        return out, i

    def parse_one(self, s: str, i: int) -> tuple[list[Box], int]:
        ch = s[i]
        if ch == " ":
            return [], i + 1
        if ch == "{":
            inner, j = self.parse(s, i + 1, "}")
            return [self.row(inner)], j
        if ch == "(":
            inner, j = self.parse(s, i + 1, ")")
            return [self.fence(self.row(inner))], j
        if ch == "\\":
            m = re.match(r"\\([A-Za-z]+|.)", s[i:])
            name, j = m.group(1), i + len(m.group())
            if name in WORDS:  # \lim, \sin ... are the same as the bare words
                return self.parse_one(name + s[j:], 0)[0], j
            if name == "frac":
                (n,), j = self._groups(s, j, 1)
                (d,), j = self._groups(s, j, 1)
                return [self.frac(n, d)], j
            if name == "sqrt":
                (c,), j = self._groups(s, j, 1)
                return [self.sqrt(c)], j
            if name == "text":
                k = s.find("}", j)
                k = len(s) if k < 0 else k
                words = s[j:k].lstrip(" {").split()
                boxes = []
                for n_, w in enumerate(words):
                    boxes += ([Box(w=0.75, kind="space")] if n_ else []) + [self.text(w)]
                return [Box(w=0.9, kind="space")] + boxes + [Box(w=0.9, kind="space")], k + 1
            if name in ("int", "sum"):
                if name == "int":  # a tall sign, drawn to the height of the line
                    sign = self.drawn([[0.62, -2.15], [0.48, -2.3], [0.36, -2.05], [0.3, -0.8], [0.24, 0.45], [0.12, 0.7], [0.0, 0.55]])
                    return [Box([sign], 0.62, 2.3, 0.7, "big")], j
                b = self.text("Σ").moved(0, 0.3, 1.45)
                b.kind = "stack"
                return [b], j
            if name in ("left", "right"):
                return [], j
            sym = SYMBOLS.get(name, name)
            if sym == " ":
                return [Box(w=0.6, kind="space")], j
            return [self.text(sym, "bin" if sym in BINARY else "ord")], j
        m = re.match(r"\d[\d.,]*\d|\d", s[i:])
        if m:
            return [self.text(m.group())], i + len(m.group())
        for word in WORDS:
            if s.startswith(word, i):
                b = self.text(word)
                if word in ("lim", "max", "min"):
                    b.kind = "stack"  # what follows as a suffix goes underneath
                    return [b], i + len(word)
                return [b, Box(w=0.12, kind="space")], i + len(word)
        if ch == "*":
            return [self.text("×", "bin")], i + 1
        if s.startswith("->", i):
            return [self.text("→", "bin")], i + 2
        if s.startswith(">=", i) or s.startswith("<=", i) or s.startswith("!=", i):
            return [self.text({">": "≥", "<": "≤", "!": "≠"}[ch], "bin")], i + 2
        if ch == "/" and False:
            return [], i + 1
        return [self.text(ch, "bin" if ch in BINARY else "ord")], i + 1

    def _groups(self, s: str, j: int, n: int) -> tuple[list[Box], int]:
        out = []
        for _ in range(n):
            while j < len(s) and s[j] == " ":
                j += 1
            if j < len(s) and s[j] == "{":
                inner, j = self.parse(s, j + 1, "}")
                out.append(self.row(inner))
            else:
                boxes, j = self.parse_one(s, j)
                out.append(self.row(boxes))
        return out, j

    def expression(self, s: str) -> Box:
        boxes, _ = self.parse(s.strip())
        return self.row(boxes)


def _split_equals(s: str) -> tuple[str, str] | None:
    depth = 0
    for i, ch in enumerate(s):
        depth += ch in "{(" 
        depth -= ch in "})"
        if ch == "=" and depth == 0 and i > 0:
            return s[:i], s[i:]
    return None


def write_maths(lib: dict, text: str, x_height: float = 8.0, seed: int = 0, human=None, line_gap: float = 3.4, align: bool = True,
                min_pen: float = 0.0, pen: float | None = None) -> Ink:
    """Write one expression, or several lines of working (separated by newlines or \\\\) aligned on '='."""
    from .render import style

    writer = Writer(lib, seed, human or style(lib, "careful"))
    m = Maths(writer, lib, seed)
    lines = [ln.strip() for ln in re.split(r"\\\\|\n", text) if ln.strip()]
    rows = []
    for ln in lines:
        parts = _split_equals(ln) if align and len(lines) > 1 else None
        if parts:
            left, right = m.expression(parts[0]), m.expression(parts[1])
            rows.append((left, right))
        else:
            rows.append((None, m.expression(ln)))
    lead = max((l.w for l, _ in rows if l is not None), default=0.0)
    ink = Ink()
    width = pen_w = 0.0
    pen_w = max((pen or writer.pen) * x_height * writer.h.weight, min_pen)
    y = 0.0
    for k, (left, right) in enumerate(rows):
        a = max(right.a, left.a if left else 0, 1.6)
        d = max(right.d, left.d if left else 0, 0.4)
        y += a + (0.6 if k else 0.3) if (a + d) > line_gap - 0.6 or k == 0 else max(line_gap - prev_d, a + 0.5)
        x = 0.0
        if left is not None:
            x = lead - left.w
            ink.strokes += [(np.asarray(s, float) + [x, y]) * x_height for s in left.strokes]
            x = lead + 0.42
        elif lead and len(rows) > 1:
            x = lead + 0.42 if _starts_with_relation(lines[k]) else 0.0
        ink.strokes += [(np.asarray(s, float) + [x, y]) * x_height for s in right.strokes]
        width = max(width, x + right.w)
        prev_d = d
        y += d
    ink.widths = [pen_w] * len(ink.strokes)
    ink.size = (width * x_height, (y + 0.3) * x_height)
    ink.missing = m.missing
    return ink


def _starts_with_relation(s: str) -> bool:
    return s.lstrip()[:1] in "=<>≤≥≈" or s.lstrip().startswith(("\\leq", "\\geq", "\\approx"))
