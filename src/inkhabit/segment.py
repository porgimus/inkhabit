# SPDX-License-Identifier: AGPL-3.0-or-later
"""Split labelled words into letters.

Handwriting is often joined up, so we cannot assume one stroke per letter.
The method is a forced alignment, the same idea speech recognisers use:

1. Over-segment each word into small left-to-right pieces ("primitives"),
   cutting at gaps and at thin joining strokes.
2. We know the word's text, so find the way of grouping consecutive pieces
   into its letters that costs least. Round 0 only knows rough letter widths.
3. Learn what each letter looks like from the result, then re-align using
   shape. Repeat. Letters and alignments improve each other.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from .words import WordSample

ASC = set("bdfhklt") | set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") | set("0123456789") | set("()?!'")
DESC = set("gjpqy") | set("(),;¿¡")
NARROW = set("iljtfI.,:;'!()1r")
WIDE = set("mwMW")
LETTERS = r"[^\W\d_]+"  # letters of any alphabet, so accented ones count
TOKEN = re.compile(rf"[(¿¡]?({LETTERS})[.,:;?!)]?|\d+[.,%]?")
TOKEN_SHAPE = re.compile(rf"[(¿¡]?(?:{LETTERS}|\d+)[.,:;?!)%]?")  # for words whose text we already trust

FH, FW = 32, 24  # feature canvas
TOP, BOTTOM = 2.0, 1.1  # frame above / below baseline, in x-heights
MAX_SPAN = 5  # most primitives one letter may take


@dataclass
class Prepared:
    sample: WordSample
    text: str
    xh: float
    baseline: float
    prims: list[list[np.ndarray]]  # primitive -> polyline pieces
    bounds: np.ndarray  # (m, 2) x extent of each primitive
    feats: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)


@dataclass
class Aligned:
    prep: Prepared
    groups: list[tuple[int, int]]  # per letter: primitive range [i, j)
    costs: list[float]


def usable_text(text: str, vocab: set[str] | None) -> str | None:
    """The text if it is a word we can learn letters from. vocab=None accepts any word (no word list
    for the language); then only the alignment cost protects against misread words."""
    m = TOKEN.fullmatch(text)
    if not m:
        return None
    if vocab is None:
        return text
    if m.group(1) is not None and m.group(1).lower() not in vocab:
        return None
    return text


def _resample(p: np.ndarray, step: float) -> np.ndarray:
    if len(p) < 2 or not np.isfinite(step) or step <= 0:
        return p
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if d[-1] < step:
        return p[[0, -1]]
    t = np.linspace(0, d[-1], min(int(d[-1] / step) + 2, 4000))  # a wild size estimate must not ask for a huge array
    return np.c_[np.interp(t, d, p[:, 0]), np.interp(t, d, p[:, 1])]


XONLY = set("aceimnorsuvwxz")


def word_xheight(sample: WordSample) -> float | None:
    """x-height from one word: the band holding most of its ink, if enough letters are x-height ones."""
    t = [c for c in sample.text if c.isalpha()]
    if len(t) < 3 or sum(c in XONLY for c in t) / len(t) < 0.5:
        return None
    ys = np.sort(np.vstack([_resample(np.asarray(s, float), 0.3) for s in sample.strokes])[:, 1])
    n = int(len(ys) * 0.65)
    if n < 8:
        return None
    return float((ys[n:] - ys[: len(ys) - n]).min() / 0.68)  # 0.68: calibrated so all-x-height words give their full height


def size_map(samples: list[WordSample]) -> tuple[dict[int, float], float]:
    """x-height per word (by position in samples): own line's median, else page's, else global."""
    est = [word_xheight(s) for s in samples]
    line: dict[tuple, list[float]] = {}
    page: dict[tuple, list[float]] = {}
    for s, e in zip(samples, est):
        if e:
            line.setdefault((s.source, s.page, s.line), []).append(e)
            page.setdefault((s.source, s.page), []).append(e)
    glob = float(np.median([e for e in est if e])) if any(est) else 1.0
    out = {}
    for i, s in enumerate(samples):
        lv = line.get((s.source, s.page, s.line))
        pv = page.get((s.source, s.page))
        out[i] = float(np.median(lv)) if lv else float(np.median(pv)) if pv else glob
    return out, glob


def _baseline(pts: np.ndarray, text: str, xh: float) -> float:
    if not (set(text) & DESC):
        return float(np.percentile(pts[:, 1], 98))
    # descenders present: the baseline is the bottom of the densest x-height band
    ys = np.sort(pts[:, 1])
    lo = np.searchsorted(ys, ys - xh, "left")
    k = int(np.argmax(np.arange(len(ys)) - lo))
    return float(ys[k])


def _cuts(strokes: list[np.ndarray], xh: float) -> list[float]:
    """x positions to cut at: gaps, and thin single joining strokes."""
    sc = 12.0 / xh
    pts = np.vstack(strokes)
    x0, y0 = pts.min(0) - 1 / sc
    w = int((pts[:, 0].max() - x0) * sc) + 3
    h = int((pts[:, 1].max() - y0) * sc) + 3
    im = np.zeros((h, w), np.uint8)
    for s in strokes:
        q = np.round((s - [x0, y0]) * sc).astype(np.int32)
        if len(q) == 1:
            im[q[0, 1], q[0, 0]] = 1
        else:
            cv2.polylines(im, [q], False, 1, 1)
    edge = np.diff(np.pad(im, ((1, 1), (0, 0))).astype(np.int8), axis=0)
    runs = (edge == 1).sum(0)
    ink = im.sum(0)
    kind = np.where(runs == 0, 0, np.where((runs == 1) & (ink <= 3), 1, 2))
    cuts, i = [], 0
    while i < w:
        j = i
        while j < w and kind[j] == kind[i]:
            j += 1
        if kind[i] != 2 and i > 0 and j < w and (kind[i] == 0 or j - i >= 2):
            cuts.append((i + j - 1) / 2 / sc + x0)
        i = j
    return cuts


def _split_at(p: np.ndarray, cuts: list[float]) -> list[np.ndarray]:
    if len(p) < 2:
        return [p]
    pieces, cur = [], [p[0]]
    for a, b in zip(p[:-1], p[1:]):
        lo, hi = min(a[0], b[0]), max(a[0], b[0])
        hit = sorted((c for c in cuts if lo < c < hi), reverse=a[0] > b[0])
        for c in hit:
            t = (c - a[0]) / (b[0] - a[0])
            m = a + t * (b - a)
            cur.append(m)
            pieces.append(np.array(cur))
            cur = [m]
        cur.append(b)
    pieces.append(np.array(cur))
    return pieces


def prepare(sample: WordSample, text: str, xh: float) -> Prepared | None:
    step = xh / 14
    strokes = [_resample(np.asarray(s, float), step) for s in sample.strokes]
    pts = np.vstack(strokes)
    if np.ptp(pts[:, 0]) > 40 * xh or len(text) > 14:
        return None
    cuts = _cuts(strokes, xh)
    edges = np.array([-np.inf, *cuts, np.inf])
    buckets: dict[int, list[np.ndarray]] = {}
    for s in strokes:
        for piece in _split_at(s, cuts):
            cx = (piece[:, 0].min() + piece[:, 0].max()) / 2
            buckets.setdefault(int(np.searchsorted(edges, cx) - 1), []).append(piece)
    prims = [buckets[k] for k in sorted(buckets)]
    if len(prims) < len(text) or len(prims) > MAX_SPAN * len(text):
        return None
    bounds = np.array([[min(p[:, 0].min() for p in g), max(p[:, 0].max() for p in g)] for g in prims])
    return Prepared(sample, text, xh, _baseline(pts, text, xh), prims, bounds)


def feature(pieces: list[np.ndarray], baseline: float, xh: float) -> np.ndarray:
    """Small blurred picture of some ink, placed relative to the baseline."""
    sc = FH / ((TOP + BOTTOM) * xh)
    x0 = min(p[:, 0].min() for p in pieces)
    x1 = max(p[:, 0].max() for p in pieces)
    sx = sc * min(1.0, (FW - 4) / max((x1 - x0) * sc, 1e-6))
    ox = (FW - (x1 - x0) * sx) / 2
    im = np.zeros((FH, FW), np.uint8)
    for p in pieces:
        q = np.c_[(p[:, 0] - x0) * sx + ox, (p[:, 1] - baseline + TOP * xh) * sc]
        q = np.round(q * 16).astype(np.int32)
        if len(q) == 1:
            cv2.circle(im, (int(q[0, 0]), int(q[0, 1])), 16, 255, -1, cv2.LINE_AA, 4)
        else:
            cv2.polylines(im, [q], False, 255, 1, cv2.LINE_AA, 4)
    return cv2.GaussianBlur(im, (0, 0), 1.2).astype(np.float32).ravel() / 255.0


def _feat(prep: Prepared, i: int, j: int) -> np.ndarray:
    f = prep.feats.get((i, j))
    if f is None:
        f = feature([p for g in prep.prims[i:j] for p in g], prep.baseline, prep.xh)
        prep.feats[(i, j)] = f
    return f


def _width_prior(ch: str) -> float:
    return 0.45 if ch in NARROW else 1.5 if ch in WIDE else 1.15 if ch.isupper() else 1.0


def _width_bounds(ch: str) -> tuple[float, float]:
    """Plausible letter widths in x-heights; outside these the grouping is wrong."""
    if ch in WIDE:
        return 0.8, 3.0
    if ch in NARROW:
        return 0.0, 1.3
    return (0.35, 2.4) if ch.isupper() else (0.25, 1.8)


def align(prep: Prepared, models: dict[str, np.ndarray] | None, w_width: float = 0.3) -> Aligned | None:
    """Best grouping of consecutive primitives into the word's letters (DP)."""
    text, m, n = prep.text, len(prep.prims), len(prep.text)
    prior = np.array([_width_prior(c) for c in text])
    total = prep.bounds[-1, 1] - prep.bounds[0, 0]
    expect = prior / prior.sum() * max(total, prep.xh * 0.3)
    lo, hi = zip(*(_width_bounds(c) for c in text))
    INF = 1e9
    best = np.full((n + 1, m + 1), INF)
    back = np.zeros((n + 1, m + 1), int)
    best[0, 0] = 0
    for k, ch in enumerate(text):
        tmpl = models.get(ch) if models else None
        for j in range(k + 1, m - (n - k - 1) + 1):
            for i in range(max(k, j - MAX_SPAN), j):
                if best[k, i] >= INF:
                    continue
                width = prep.bounds[i:j, 1].max() - prep.bounds[i:j, 0].min()
                c = ((width - expect[k]) / prep.xh) ** 2
                if not lo[k] <= width / prep.xh <= hi[k]:
                    c += 40.0
                if tmpl is not None:
                    d = tmpl - _feat(prep, i, j)
                    c = w_width * c + float((d * d).sum(1).min())
                elif models:
                    c = w_width * c + 6.0  # letter with no model yet
                if best[k, i] + c < best[k + 1, j]:
                    best[k + 1, j] = best[k, i] + c
                    back[k + 1, j] = i
    if best[n, m] >= INF:
        return None
    groups, costs, j = [], [], m
    for k in range(n, 0, -1):
        i = back[k, j]
        groups.append((i, j))
        costs.append(best[k, j] - best[k - 1, i])
        j = i
    return Aligned(prep, groups[::-1], costs[::-1])


def _kmedoids(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    k = min(k, len(x))
    centres = x[rng.choice(len(x), k, replace=False)]
    for _ in range(8):
        d = ((x[:, None] - centres[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        centres = np.array([x[lab == c].mean(0) if (lab == c).any() else centres[c] for c in range(k)])
    return centres


def learn_models(aligned: list[Aligned], k: int = 6, keep: float = 0.6, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    by_char: dict[str, list[np.ndarray]] = {}
    for a in aligned:
        for ch, (i, j) in zip(a.prep.text, a.groups):
            by_char.setdefault(ch, []).append(_feat(a.prep, i, j))
    models = {}
    for ch, feats in by_char.items():
        x = np.array(feats)
        if len(x) > 1500:
            x = x[rng.choice(len(x), 1500, replace=False)]
        if len(x) >= 8:  # drop the samples least like the rest, then cluster
            d = ((x - np.median(x, 0)) ** 2).sum(1)
            x = x[d <= np.quantile(d, keep)]
        models[ch] = _kmedoids(x, max(1, min(k, len(x) // 4)), rng)
    return models


def estimate_slant(samples: list[WordSample], limit: int = 4000) -> float:
    """How far the writing leans, in degrees to the right of upright, from its near-vertical pen movement."""
    angles, weights = [], []
    for s in samples[:limit]:
        for p in s.strokes:
            p = np.asarray(p, float)
            if len(p) < 4:
                continue
            d = np.diff(p[:: max(1, len(p) // 16)], axis=0)
            ln = np.linalg.norm(d, axis=1)
            a = np.degrees(np.arctan2(d[:, 0] * -np.sign(d[:, 1]), np.abs(d[:, 1]) + 1e-9))
            keep = (np.abs(a) < 40) & (ln > 0)
            angles += a[keep].tolist()
            weights += ln[keep].tolist()
    if len(angles) < 200:
        return 0.0
    order = np.argsort(angles)  # weighted median: robust to curves and joining strokes
    cum = np.cumsum(np.asarray(weights)[order])
    return float(np.asarray(angles)[order][np.searchsorted(cum, cum[-1] / 2)])


def deslant(samples: list[WordSample], degrees: float) -> list[WordSample]:
    """Stand leaning writing upright. Letters are cut apart column by column, and in slanted writing the
    top of one letter hangs over the foot of the next, so the columns never clear. Cutting is done on
    the upright version; the lean is put back when writing."""
    from dataclasses import replace

    t = np.tan(np.radians(degrees))
    out = []
    for s in samples:
        pts = np.vstack([np.asarray(p, float) for p in s.strokes])
        y0 = float(np.median(pts[:, 1]))
        strokes = [np.c_[np.asarray(p, float)[:, 0] + (np.asarray(p, float)[:, 1] - y0) * t, np.asarray(p, float)[:, 1]] for p in s.strokes]
        out.append(replace(s, strokes=strokes))
    return out


def reference_models() -> dict[str, np.ndarray]:
    """What letters look like in a plain hand, as a starting point for cutting.

    Without any idea of letter shapes, the first pass can only guess from widths, and on cramped
    writing it settles on wrong splits (a "d" becomes its stem, with the bowl given to the letter
    before). Starting from generic shapes avoids that; later rounds replace them with the writer's own.
    """
    from .extend import ALNUM, _neutral_skeleton

    out = {}
    for ch in ALNUM:
        sk = _neutral_skeleton(ch)
        if sk:
            out[ch] = np.array([feature([np.asarray(s, float) for s in sk], 0.0, 1.0)])
    return out


def models_from_library(lib: dict) -> dict[str, np.ndarray]:
    """Letter models taken from a finished library's chosen samples, to start a new alignment from."""
    out = {}
    for ch, entry in lib["glyphs"].items():
        feats = [feature([np.asarray(s, float) for s in v["strokes"]], 0.0, 1.0) for v in entry["variants"]
                 if v.get("source") in (None, "harvested") and v["strokes"]]
        if feats and entry.get("count", 0) > 0:
            out[ch] = np.array(feats)
    return out


def run(samples: list[WordSample], vocab: set[str] | None, rounds: int = 4, log=print,
        seed_models: dict[str, np.ndarray] | None = None) -> tuple[list[Aligned], dict]:
    sizes, xh_global = size_map(samples)
    preps: list[Prepared] = []
    for i, s in enumerate(samples):
        text = s.text if s.trusted and TOKEN_SHAPE.fullmatch(s.text) else usable_text(s.text, vocab)
        if text and s.conf >= 0.5:
            p = prepare(s, text, sizes[i])
            if p:
                preps.append(p)
    log(f"x-height {xh_global:.2f} units; {len(preps)} words prepared of {len(samples)}")
    generic = None if seed_models or os.environ.get("INKHABIT_NO_REFERENCE") else reference_models()
    models: dict[str, np.ndarray] | None = seed_models or generic or None
    aligned: list[Aligned] = []
    for r in range(rounds):
        aligned = [a for a in (align(p, models) for p in preps) if a]
        if not aligned:  # nothing readable: the caller says so in plain words
            break
        if r == 0 and not models:  # no shapes to go on: start from the easy words, one piece per letter
            train = [a for a in aligned if len(a.prep.prims) == len(a.prep.text)]
        else:
            per = np.array([np.mean(a.costs) for a in aligned])
            train = [a for a, c in zip(aligned, per) if c <= np.quantile(per, 0.7)]
        models = learn_models(train)
        log(f"round {r}: {len(aligned)} aligned, {len(train)} used to learn {len(models)} characters")
    return aligned, {"xheight": xh_global}
