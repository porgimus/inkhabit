# SPDX-License-Identifier: AGPL-3.0-or-later
"""Turn pages of ink into labelled word samples (OCR text + the strokes under it)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .ink import Page, Stroke, render
from .ocr import recognize

OCR_SCALE_TARGET = 30.0  # render so a typical text line is about this many px tall
TILE_PX = 2800.0  # OCR tile height in pixels
MAX_PEN = 4.0  # wider strokes are highlighter, not writing


@dataclass
class WordSample:
    text: str
    conf: float
    strokes: list[np.ndarray]  # polylines in page units
    widths: list[float]
    box: tuple[float, float, float, float]
    source: str
    page: int
    line: int
    color: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0)
    trusted: bool = False  # text came from a corrected transcript, not straight from the recogniser


def _tiles(strokes: list[Stroke], TILE: float) -> list[tuple[float, float]]:
    """Cut a tall page into bands, cutting where there is no ink."""
    ys = np.array([[s.bbox[1], s.bbox[3]] for s in strokes])
    top, bottom = ys[:, 0].min() - 8, ys[:, 1].max() + 8
    cuts, y = [top], top
    while bottom - y > TILE * 1.3:
        cand = np.linspace(y + TILE * 0.7, y + TILE * 1.3, 60)
        crossing = [(np.sum((ys[:, 0] < c) & (ys[:, 1] > c)), abs(c - y - TILE), c) for c in cand]
        y = min(crossing)[2]
        cuts.append(y)
    cuts.append(bottom)
    return list(zip(cuts[:-1], cuts[1:]))


def page_words(page: Page, scale: float | None = None, custom_words: list[str] | None = None,
               lang: str | None = None) -> list[WordSample]:
    strokes = [s for s in page.strokes if s.width <= MAX_PEN and len(s.points) > 0]
    if len(strokes) < 5:
        return []
    boxes = np.array([s.bbox for s in strokes])
    if scale is None:  # zoom so tall letters are ~OCR_SCALE_TARGET px, whatever the source's units
        tall = np.percentile(boxes[:, 3] - boxes[:, 1], 80)
        scale = float(np.clip(OCR_SCALE_TARGET / max(tall, 1e-3), 1.0, 12.0))
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    x0 = boxes[:, 0].min() - 8
    width = boxes[:, 2].max() + 8 - x0
    out: list[WordSample] = []
    line_base = 0
    # A scan is read from the original picture: recognition does better on real ink than on our re-drawing.
    photo = page.image is not None
    if photo:
        x0, scale = 0.0, page.image_scale
        tiles = [(0.0, page.image.shape[0] / scale)]
    else:
        tiles = _tiles(strokes, TILE_PX / scale)
    for y0, y1 in tiles:
        sel = [s for s, b in zip(strokes, boxes) if b[3] > y0 and b[1] < y1]
        if not sel:
            continue
        if photo:
            from PIL import Image

            im = Image.fromarray(page.image)
        else:
            im = render(sel, (x0, y0), (width, y1 - y0), scale)
        words = recognize(im, custom_words, lang)
        taken = np.zeros(len(strokes), bool)
        for w in words:
            bx0, by0, bx1, by1 = (np.array(w.box) / scale + [x0, y0, x0, y0])
            pad = 0.15 * (by1 - by0)
            inside = (cx > bx0) & (cx < bx1) & (cy > by0 - pad) & (cy < by1 + pad) & ~taken
            idx = np.flatnonzero(inside)
            if len(idx) == 0:
                continue
            taken[idx] = True
            out.append(
                WordSample(w.text, w.conf, [strokes[i].points for i in idx],
                           [strokes[i].width for i in idx], (bx0, by0, bx1, by1),
                           page.source, page.index, line_base + w.line,
                           max(Counter(strokes[i].color for i in idx).items(), key=lambda kv: kv[1])[0])
            )
        line_base += 1 + max((w.line for w in words), default=0)
    return out


def apply_transcript(samples: list[WordSample], lines: list[str]) -> int:
    """Relabel one page's words from a corrected transcript (one string per written line, in reading order).

    The recogniser finds where the words are but often misreads them; a person or an agent reading the
    page gets the text right. We match each recognised line to a transcript line, then each word to a
    word, tolerating missed or extra words. Returns how many words were relabelled.
    """
    from difflib import SequenceMatcher

    def sim(a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def align(xs: list[str], ys: list[str], gap: float = 0.25) -> list[tuple[int, int]]:
        """Best in-order pairing of two word (or line) lists; unmatched items are skipped."""
        n, m = len(xs), len(ys)
        score = np.zeros((n + 1, m + 1))
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                score[i, j] = max(score[i - 1, j - 1] + sim(xs[i - 1], ys[j - 1]) - gap, score[i - 1, j], score[i, j - 1])
        pairs, i, j = [], n, m
        while i and j:
            if score[i, j] == score[i - 1, j]:
                i -= 1
            elif score[i, j] == score[i, j - 1]:
                j -= 1
            else:
                pairs.append((i - 1, j - 1))
                i, j = i - 1, j - 1
        return pairs[::-1]

    by_line: dict[int, list[WordSample]] = {}
    for s in samples:
        by_line.setdefault(s.line, []).append(s)
    ocr = sorted(by_line.values(), key=lambda ws: np.mean([w.box[1] for w in ws]))
    for ws in ocr:
        ws.sort(key=lambda w: w.box[0])
    done = 0
    left = [ln.split() for ln in lines]  # transcript words not yet used
    for li, lj in align([" ".join(w.text for w in ws) for ws in ocr], lines):
        want = lines[lj].split()
        for wi, wj in align([w.text for w in ocr[li]], want, gap=0.15):
            if sim(ocr[li][wi].text, want[wj]) >= 0.4:
                ocr[li][wi].text, ocr[li][wi].trusted, ocr[li][wi].conf = want[wj], True, 1.0
                left[lj][wj] = ""
                done += 1
    # tables and scattered notes are not read in line order: pair what is left by likeness alone
    spare = [w for ws in left for w in ws if w]
    loose = [s for s in samples if not s.trusted]
    scored = sorted(((sim(s.text, w), i, j) for i, s in enumerate(loose) for j, w in enumerate(spare)), reverse=True)
    used_i, used_j = set(), set()
    for score, i, j in scored:
        if score < 0.8:  # strict: a near-miss here would label the ink with the wrong letters
            break
        if i not in used_i and j not in used_j:
            loose[i].text, loose[i].trusted, loose[i].conf = spare[j], True, 1.0
            used_i.add(i)
            used_j.add(j)
            done += 1
    return done
