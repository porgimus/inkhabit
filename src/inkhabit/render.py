# SPDX-License-Identifier: AGPL-3.0-or-later
"""Write text the way the person writes it: never the same twice.

A font stamps identical shapes. This renderer works from the pen strokes in the
library and re-creates the small inconsistencies of real writing:

- a different real sample of each letter each time, never the same one twice in a row
- whole words lifted from the writer's notes when the library has them
- letters joined where this writer usually joins them
- drifting baseline, size and slant along a line; uneven spacing
- slight wobble in every stroke and in pen pressure

Everything is driven by one seed, so a result can be reproduced exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import ClassVar

import numpy as np

from .font import with_fallbacks, word_gap


@dataclass
class Human:
    """Everything that makes writing look hand-made. Two kinds of setting:

    Variation (how irregular; 0 = none, like a font):
    """

    morph: float = 0.5  # how often a letter is a blend of two real samples instead of one of them
    typical: float = 0.0  # 0..1: how often to stick to the writer's most usual shapes (more legible)
    line_size: float = 0.02  # each line comes out a little bigger or smaller
    squeeze: float = 0.12  # how much a word is squashed to fit the end of a line before wrapping
    whole_words: float = 0.2  # chance to reuse a whole written word when available
    joins: float = 1.0  # multiplier on the writer's own joining habits
    size: float = 0.035  # letter-to-letter size variation (fraction)
    height_var: float = 0.025  # extra variation in letter height only
    width_var: float = 0.03  # extra variation in letter width only
    word_size: float = 0.03  # whole words come out a bit bigger or smaller
    tilt: float = 1.5  # per-letter rotation, degrees
    bounce: float = 0.035  # per-letter vertical offset, x-heights
    spacing: float = 0.5  # letter-gap irregularity, as a fraction of the writer's measured spread
    word_spacing_var: float = 0.18  # word-gap irregularity (fraction)
    line_spacing_var: float = 0.04  # line-gap irregularity (fraction)
    margin: float = 0.15  # ragged left edge, x-heights
    drift: float = 0.08  # slow baseline wander along a line, x-heights
    slope: float = 0.5  # whole-line tilt varies by this much, degrees
    slope_bias: float = 0.0  # lines usually climb (+) or sink (-) by this much, degrees
    wobble: float = 0.012  # stroke tremor, x-heights
    pressure: float = 0.0  # pen width variation between strokes (fraction); 0 for a fixed-width pen

    # Dimensions (fixed adjustments; 1.0 / 0.0 = exactly as the writer writes):
    letter_spacing: float = 1.0  # gap between letters
    word_spacing: float = 1.0  # gap between words
    line_spacing: float = 1.0  # distance between lines
    letter_height: float = 1.0  # taller or squatter letters
    letter_width: float = 1.0  # wider or narrower letters
    slant: float = 0.0  # lean to the right, degrees
    weight: float = 1.0  # pen thickness

    VARIATION: ClassVar[tuple[str, ...]] = ("size", "height_var", "width_var", "word_size", "tilt", "bounce", "spacing", "word_spacing_var",
                 "line_spacing_var", "margin", "drift", "slope", "wobble", "pressure", "line_size")

    @classmethod
    def from_library(cls, lib: dict) -> "Human":
        """Defaults, with the amounts that were measured from this writer's own notes swapped in."""
        m = lib.get("measured", {})
        h = cls()
        # Letter gap, joining and line tilt are taken from the writer's notes. Size and baseline
        # irregularity are not: what can be measured of them in the notes is mostly measurement error
        # (the size of each word is itself an estimate), and copying it makes the writing look patchy.
        # pen thickness varies only as much as it does in the sources (not at all for a fixed-width pen)
        h.pressure = float(min(lib.get("meta", {}).get("pen_var", 0.0), 0.15))
        h.letter_spacing = m.get("letter_spacing_use", h.letter_spacing)
        h.joins = m.get("joins_use", h.joins)
        if "slope" in m:
            h.slope, h.slope_bias = m["slope"], m.get("slope_bias", 0.0)
        return h

    def scaled(self, k: float) -> "Human":
        """Turn all the irregularity up or down together; dimensions are left alone."""
        return replace(self, **{f: getattr(self, f) * k for f in self.VARIATION})

    def with_settings(self, settings: dict) -> "Human":
        """Override individual settings by name, e.g. {"word_spacing": 0.8, "slant": 6}."""
        names = {f.name for f in fields(self)}
        bad = sorted(set(settings) - names)
        if bad:
            raise ValueError(f"unknown setting(s) {bad}; choose from {sorted(names)}")
        return replace(self, **{k: float(v) for k, v in settings.items()})


STYLES = ("careful", "clean", "natural", "rushed")


def style(lib: dict, name: str = "clean") -> Human:
    """Named starting points.

    clean (the default; "neat" is the same): the writer at their best. Their most usual letter shapes,
    steady size and baseline, even spacing, few joins. Use this unless something else is asked for.
    natural: as the notes are really written, with their spacing and joining habits and more variation.
    rushed: in a hurry.
    """
    base = Human.from_library(lib)
    if name in ("clean", "neat"):
        return replace(base.scaled(0.6), typical=0.8, joins=0.5, whole_words=0.1, squeeze=0.05, morph=0.3,
                       letter_spacing=1.0)  # notes are spaced loosely; best-case writing is not
    if name == "careful":
        # The writer taking real care, as on a form or anything official: only their clearest shapes of each
        # letter, letters kept apart, a level line, even size and spacing. Still their hand, and still not
        # mechanical: a little variation is kept so that no two letters are identical.
        return replace(base.scaled(0.3), typical=1.0, joins=0.0, whole_words=0.0, morph=0.25, squeeze=0.0, slope_bias=0.0,
                       letter_spacing=1.08, word_spacing=1.05, word_size=0.0, drift=0.0)
    if name == "natural":
        return base
    if name == "rushed":  # more joins, leaning, tighter, wandering
        return replace(base.scaled(1.6), joins=1.3, whole_words=0.4, slant=5.0, letter_spacing=base.letter_spacing * 0.9,
                       word_spacing=0.9)
    raise ValueError(f"unknown style {name!r}; choose careful, clean, natural or rushed")


SITUATIONS = ("text", "form", "official", "heading", "label", "letter", "note", "scribble")


def situation(lib: dict, name: str) -> tuple[Human, float]:
    """Settings for what the writing is for, and how big to write relative to normal body text.
    All of them start from the clean style except the two that are meant to look casual.

    text: anything not otherwise specified; clean.          form: inside boxes and on lines; clean, no whole words.
    heading: larger, firmer, unjoined.                      label: small and plain, for captions and margin notes.
    letter: continuous prose with a little more life.       note: as the person's notes really look.
    scribble: hurried.
    """
    clean = style(lib, "clean")
    if name == "text":
        return clean, 1.0
    if name in ("form", "official"):  # forms, applications, envelopes, anything a stranger must read
        return form_preset(lib), 1.0
    if name == "heading":
        return replace(clean.scaled(0.8), joins=0.2, whole_words=0.0, weight=1.15, letter_spacing=1.05), 1.45
    if name == "label":
        return replace(clean.scaled(0.9), joins=0.2, whole_words=0.0, typical=0.9, squeeze=0.15), 0.8
    if name == "letter":
        return replace(clean.scaled(1.25), joins=0.7, typical=0.5, line_spacing_var=0.02), 1.0
    if name == "note":
        return style(lib, "natural"), 1.0
    if name == "scribble":
        return style(lib, "rushed"), 1.0
    raise ValueError(f"unknown situation {name!r}; choose from {', '.join(SITUATIONS)}")


def form_preset(lib: dict) -> Human:
    """How the writer behaves when filling in a form rather than taking notes.

    People write more carefully in boxes: steadier and more regular, mostly their usual letter shapes,
    fewer joined letters and fewer whole-word flourishes, and no squeezing at line ends. This is a
    starting point; change the numbers to match how you actually fill forms and compare a few seeds
    with:  inkhabit fill ... --seed 1
    """
    return style(lib, "careful")


def default_human(lib: dict) -> Human:
    return Human.from_library(lib)


@dataclass
class Ink:
    """Result of writing: polylines in output units, each with a pen width."""

    strokes: list[np.ndarray] = field(default_factory=list)
    widths: list[float] = field(default_factory=list)
    size: tuple[float, float] = (0.0, 0.0)
    missing: set[str] = field(default_factory=set)


class Writer:
    def __init__(self, lib: dict, seed: int | None = None, human: Human | None = None):
        self.lib = lib
        self.table, _ = with_fallbacks(lib)
        self.words = lib.get("words", {})
        self.join_rate = lib.get("join_rate", {})
        self.gap = lib["spacing"]["letter_gap_mean"]
        self.gap_sd = lib["spacing"]["letter_gap_sd"]
        self.word_gap = word_gap(lib)
        self.pen = lib["meta"].get("font_weight", 0.16)
        self.rng = np.random.default_rng(seed)
        self.h = human or style(lib, "clean")  # best-case writing unless told otherwise
        self.pairs = lib.get("pairs", {})
        self.pair_gap = lib.get("pair_gap", {})
        self.base_slant = float(lib.get("meta", {}).get("slant", 0.0))  # the writer's own lean; samples are stored upright
        self._last: dict[str, int] = {}
        self._word_scale = 1.0
        self.min_pen = 0.0  # thinnest line allowed, in output units. A real pen has a fixed tip: small writing does not get a finer line
        self.force: dict[str, int] = {}
        self.used_words = 0  # how many words so far were whole words lifted from the sources

    # -- choosing shapes -------------------------------------------------
    def _pick(self, key: str, n: int, prefer: list[int] | None = None) -> int:
        if key in self.force:  # testing one particular sample
            return min(self.force[key], n - 1)
        options = [i for i in range(n) if i != self._last.get(key)] or [0]
        if self.h.typical and self.rng.random() < self.h.typical:
            options = [i for i in options if i < 3] or options  # variants are stored most-usual first
        liked = [i for i in options if prefer and i in prefer]
        if liked and self.rng.random() < 0.75:  # usually a sample from the same place in a word
            options = liked
        i = int(self.rng.choice(options))
        self._last[key] = i
        return i

    def _wobble(self, p: np.ndarray) -> np.ndarray:
        if self.h.wobble <= 0 or len(p) < 3:
            return p
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        if seg[-1] < 0.3:  # tiny strokes (dots) get no tremor
            return p
        t = np.linspace(0, seg[-1], max(len(p), int(seg[-1] / 0.08) + 2))
        q = np.c_[np.interp(t, seg, p[:, 0]), np.interp(t, seg, p[:, 1])]
        phase, freq = self.rng.uniform(0, 6.28, 2), self.rng.uniform(1.5, 4.0, 2)
        q[:, 0] += self.h.wobble * np.sin(t * freq[0] + phase[0])
        q[:, 1] += self.h.wobble * np.sin(t * freq[1] + phase[1])
        return q

    def _place(self, strokes, x: float, scale: float = 1.0) -> list[np.ndarray]:
        """Apply per-letter size, tilt and bounce, then move to pen position x."""
        s = scale * self._word_scale * (1 + self.rng.normal(0, self.h.size))
        sx = s * self.h.letter_width * (1 + self.rng.normal(0, self.h.width_var))
        sy = s * self.h.letter_height * (1 + self.rng.normal(0, self.h.height_var))
        a = np.radians(self.rng.normal(0, self.h.tilt))
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        shear = np.array([[1.0, -np.tan(np.radians(self.h.slant + self.base_slant))], [0.0, 1.0]])  # y is down, so lean right
        m = shear @ rot @ np.diag([sx, sy])
        dy = self.rng.normal(0, self.h.bounce)
        return [self._wobble(np.asarray(st, float) @ m.T + [x, dy]) for st in strokes]

    def _word(self, word: str) -> tuple[list[np.ndarray], float, set[str]]:
        """Strokes for one word in x-height units, starting at x=0; returns (strokes, width, missing)."""
        self._word_scale = float(np.clip(1 + self.rng.normal(0, self.h.word_size), 0.75, 1.3))
        stretch = self._word_scale * self.h.letter_width
        if len(word) >= 3 and word in self.words and self.rng.random() < self.h.whole_words:
            v = self.words[word][self._pick("w:" + word, len(self.words[word]))]
            self.used_words += 1
            return self._place(v["strokes"], 0.0), v["width"] * stretch, set()
        out: list[np.ndarray] = []
        x, missing, prev, gap = 0.0, set(), None, 0.0
        k = -1
        while k + 1 < len(word):
            k += 1
            ch = word[k]
            pair = word[k : k + 2]
            if pair in self.pairs and self.rng.random() < self.join_rate.get(pair, 0) * self.h.joins * 0.9:
                v = self.pairs[pair][self._pick("p:" + pair, len(self.pairs[pair]))]
                out += self._place(v["strokes"], x)  # a real joined pair, lifted from the notes
                gap = self.gap * self.h.letter_spacing
                x += v["width"] * stretch + gap
                prev = None
                k += 1
                continue
            if ch not in self.table:
                missing.add(ch)
                x += 0.6
                prev = None
                continue
            variants, scale = self.table[ch]
            where = "s" if len(word) == 1 else "i" if k == 0 else "f" if k == len(word) - 1 else "m"
            var = variants[self._pick(ch, len(variants), [i for i, v in enumerate(variants) if v.get("pos") == where])]
            if ch.isupper() or ch.isdigit():  # a capital written small would read as its lowercase
                top = -min(p[1] for s in var["strokes"] for p in s)
                scale = scale * float(np.clip(1.4 / max(top * scale, 0.3), 1.0, 1.7))
            strokes, width = self._blend(var, variants)
            placed = self._place(strokes, x, scale)
            if prev and self._should_join(prev[0] + ch, out[-1], placed[0]):
                out[-1] = np.vstack([out[-1], self._bridge(out[-1], placed[0]), placed[0]])
                placed = placed[1:]
            out += placed
            nxt = word[k + 1] if k + 1 < len(word) else ""
            if nxt in ".,":  # a full stop or comma sits close to its word, whatever the notes' looser habit
                usual = min(self.pair_gap.get(ch + nxt, 1.0), self.gap * 0.75)
            elif nxt in ";:!?)" or ch in "(@" or nxt == "@":  # taller marks need the ordinary gap or they touch
                usual = self.gap * 1.1
            else:  # this pair's own spacing, tempered: a single measurement of a pair is noisier than the overall gap
                usual = 0.5 * self.pair_gap.get(ch + nxt, self.gap) + 0.5 * self.gap
            usual = float(np.clip(usual, 0.08, 0.6))  # measured 0 means "they touch when joined"; unjoined they need air
            gap = float(np.clip(usual + self.rng.normal(0, self.gap_sd * self.h.spacing), 0.03, max(usual, self.gap) + 2 * self.gap_sd))
            gap *= self.h.letter_spacing
            x += width * scale * stretch + gap
            prev = (ch,)
        return out, x - gap if word else 0.0, missing

    def _blend(self, var: dict, variants: list[dict]) -> tuple[list, float]:
        """A new shape between two real samples of the same letter, when they are drawn the same way.

        Two samples can be blended if they have the same number of strokes and each stroke runs the same
        way for about the same length. The result is neither sample, but stays inside what the writer does.
        """
        if self.h.morph <= 0 or self.rng.random() > self.h.morph or len(variants) < 2:
            return var["strokes"], var["width"]
        a = [np.asarray(s, float) for s in var["strokes"]]
        if self.h.typical >= 1.0:
            variants = variants[:3]  # being careful: blend only among the clearest shapes
        mates = []
        for other in variants:
            if other is var or len(other["strokes"]) != len(a):
                continue
            b = [np.asarray(s, float) for s in other["strokes"]]
            ok = True
            for p, q in zip(a, b):
                lp = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0
                lq = float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum()) if len(q) > 1 else 0.0
                if (len(p) == 1) != (len(q) == 1) or not 0.6 * lq <= lp <= 1.6 * lq + 1e-9 or np.dot(p[-1] - p[0], q[-1] - q[0]) < 0 \
                        or np.linalg.norm(p[0] - q[0]) > 0.5 or np.linalg.norm(p[-1] - q[-1]) > 0.5:
                    ok = False
                    break
            if ok:
                mates.append((other, b))
        if not mates:
            return var["strokes"], var["width"]
        other, b = mates[int(self.rng.integers(len(mates)))]
        t = float(self.rng.uniform(0.25, 0.75))

        def resample(p, n=28):
            if len(p) == 1:
                return np.repeat(p, n, axis=0)
            d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
            u = np.linspace(0, d[-1], n)
            return np.c_[np.interp(u, d, p[:, 0]), np.interp(u, d, p[:, 1])]

        out = []
        for p, q in zip(a, b):
            m = (1 - t) * resample(p) + t * resample(q)
            out.append(m[:1] if len(p) == 1 else m)
        return out, (1 - t) * var["width"] + t * other["width"]

    def _should_join(self, pair: str, a: np.ndarray, b: np.ndarray) -> bool:
        rate = self.join_rate.get(pair, 0.0) * self.h.joins
        if rate <= 0 or len(a) < 2 or len(b) < 2 or self.rng.random() > rate:
            return False
        if np.linalg.norm(np.diff(a, axis=0), axis=1).sum() < 0.5:
            return False  # the last stroke is a dot or a short bar, not where the pen leaves the letter
        d = b[0] - a[-1]  # only join when the pen would naturally travel right a short way
        return 0.0 < d[0] < 0.9 and abs(d[1]) < 0.7

    @staticmethod
    def _bridge(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Smooth connecting stroke that leaves a in its direction and arrives at b in b's."""
        p0, p3 = a[-1], b[0]
        d0 = a[-1] - a[max(len(a) - 3, 0)]
        d1 = b[min(2, len(b) - 1)] - b[0]
        L = np.linalg.norm(p3 - p0) / 3
        p1 = p0 + d0 / (np.linalg.norm(d0) + 1e-9) * L
        p2 = p3 - d1 / (np.linalg.norm(d1) + 1e-9) * L
        t = np.linspace(0, 1, 8)[1:-1, None]
        return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3

    # -- layout ----------------------------------------------------------
    def write_to_fit(self, text: str, width: float, height: float | None = None, x_height: float = 6.0,
                     min_x_height: float = 3.0, multiline: bool = True) -> tuple[Ink, float]:
        """Write inside a space the way a person does: first close up the gaps a little, then write smaller.

        Returns the ink and the x-height that was used. The same shapes are kept across attempts, so
        tightening does not also change which letters were chosen.
        """
        keep = self.h
        state = self.rng.bit_generator.state
        last = dict(self._last)
        xh, ink = x_height, None
        for squeeze in (1.0, 0.93, 0.86, None):
            while True:
                self.rng.bit_generator.state, self._last = state, dict(last)
                if squeeze is not None:
                    self.h = replace(keep, letter_spacing=keep.letter_spacing * squeeze, word_spacing=keep.word_spacing * squeeze)
                ink = self.write(text, xh, width if multiline else None)
                fits = ink.size[0] <= width and (height is None or ink.size[1] <= height)
                if fits or squeeze is not None or xh <= min_x_height:
                    break
                xh = max(min_x_height, xh * 0.93)
            if fits:
                break
        self.h = keep
        return ink, xh

    def write(self, text: str, x_height: float = 6.0, max_width: float | None = None,
              line_gap: float = 3.2, pen: float | None = None, justify: bool = False) -> Ink:
        """Write text. x_height and max_width in output units (points for PDF, px for images).

        The origin is the top-left of the block; the first baseline sits 2.2 x-heights down.
        """
        ink = Ink()
        pen_w = max((pen or self.pen) * x_height * self.h.weight, self.min_pen)
        word_gap = self.word_gap * self.h.word_spacing

        def space() -> float:
            return word_gap * max(0.55, 1 + self.rng.normal(0, self.h.word_spacing_var))

        limit = max_width / x_height if max_width else np.inf
        y, widest = 2.2, 0.0
        for para in text.split("\n"):
            lines: list[list[tuple[list[np.ndarray], float]]] = [[]]
            x = 0.0
            for word in para.split():
                strokes, w, miss = self._word(word)
                ink.missing |= miss
                over = x + w - limit
                if lines[-1] and 0 < over <= w * self.h.squeeze:  # nearly fits: squash it in, as people do
                    k = (limit - x) / w
                    strokes, w = [s * [k, 1.0] for s in strokes], w * k
                elif lines[-1] and over > 0:
                    lines.append([])
                    x = 0.0
                lines[-1].append((strokes, w))
                x += w + word_gap
            for li, line in enumerate(lines):
                x = abs(self.rng.normal(0, self.h.margin))
                extra = 0.0  # justified: every line but the last of a paragraph is spread to the full width
                if justify and max_width and li < len(lines) - 1 and len(line) > 1:
                    natural = sum(w for _, w in line) + word_gap * (len(line) - 1)
                    extra = float(np.clip((limit - natural) / (len(line) - 1), 0.0, word_gap * 0.8))
                slope = -np.tan(np.radians(self.rng.normal(self.h.slope_bias, self.h.slope)))  # y is down
                phase, period = self.rng.uniform(0, 6.28), self.rng.uniform(18, 40)
                grow = 1 + self.rng.normal(0, self.h.line_size)
                for strokes, w in line:
                    lift = slope * x + self.h.drift * np.sin(phase + x / period * 6.28)
                    for st in strokes:
                        ink.strokes.append((st * grow + [x, y + lift]) * x_height)
                        ink.widths.append(pen_w * max(0.6, 1 + self.rng.normal(0, self.h.pressure)) if self.h.pressure else pen_w)
                    widest = max(widest, x + w * grow)
                    x += w * grow + space() + extra
                y += line_gap * self.h.line_spacing * (1 + self.rng.normal(0, self.h.line_spacing_var if line_gap == 3.2 else 0.0))
        ink.size = (widest * x_height, (y - line_gap * self.h.line_spacing + 1.3) * x_height)
        return ink


# -- output ---------------------------------------------------------------
DOT = 0.72  # radius of a dot, in pen widths


def _is_dot(st: np.ndarray, pw: float) -> bool:
    return len(st) == 1 or float(np.linalg.norm(np.diff(st, axis=0), axis=1).sum()) < 1.3 * pw


def to_svg(ink: Ink, color: str = "#1a1a2e", margin: float = 8) -> str:
    w, h = ink.size[0] + 2 * margin, ink.size[1] + 2 * margin
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.1f} {h:.1f}" width="{w:.1f}" height="{h:.1f}" '
             f'fill="none" stroke="{color}" stroke-linecap="round" stroke-linejoin="round">']
    for st, pw in zip(ink.strokes, ink.widths):
        pts = st + margin
        if _is_dot(st, pw):
            c = pts.mean(0)
            parts.append(f'<circle cx="{c[0]:.2f}" cy="{c[1]:.2f}" r="{pw * DOT:.2f}" fill="{color}" stroke="none"/>')
            continue
        d = "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        parts.append(f'<path d="{d}" stroke-width="{pw:.2f}"/>')
    return "\n".join(parts) + "\n</svg>\n"


def to_image(ink: Ink, color=(26, 26, 46), background=(255, 255, 255), margin: int = 12, supersample: int = 3, scale: float = 1.0):
    """scale > 1 gives a larger picture of the same writing (for print), without changing its proportions."""
    import cv2
    from PIL import Image

    if scale != 1.0:
        ink = Ink([s * scale for s in ink.strokes], [w * scale for w in ink.widths], (ink.size[0] * scale, ink.size[1] * scale), ink.missing)
        margin = int(margin * scale)
    k = supersample
    w, h = int((ink.size[0] + 2 * margin) * k), int((ink.size[1] + 2 * margin) * k)
    im = np.full((max(h, 1), max(w, 1), 3), background, np.uint8)
    for st, pw in zip(ink.strokes, ink.widths):
        q = np.round((st + margin) * k * 16).astype(np.int32)
        thick = max(1, int(round(pw * k)))
        if _is_dot(st, pw):
            c = np.round((st.mean(0) + margin) * k * 16).astype(np.int32)
            cv2.circle(im, (int(c[0]), int(c[1])), int(round(pw * k * DOT * 16)), color, -1, cv2.LINE_AA, 4)
        else:
            cv2.polylines(im, [q], False, color, thick, cv2.LINE_AA, 4)
    return Image.fromarray(im).resize((w // k, h // k), Image.LANCZOS)


def onto_pdf_page(page, ink: Ink, origin: tuple[float, float], color=(0.1, 0.1, 0.18)) -> None:
    """Draw ink on a PyMuPDF page as vector strokes; origin = top-left of the text block, in points."""
    shape = page.new_shape()
    o = np.asarray(origin, float)
    for st, pw in zip(ink.strokes, ink.widths):
        pts = st + o
        if _is_dot(st, pw):  # a full stop or the dot of an i: a pen leaves a round mark, not a hairline speck
            shape.draw_circle(tuple(pts.mean(0)), pw * DOT)
            shape.finish(color=color, fill=color, width=0)
        else:
            shape.draw_polyline([tuple(p) for p in pts])
            shape.finish(color=color, width=pw, lineCap=1, lineJoin=1, closePath=False)
    shape.commit()
