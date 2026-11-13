# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared ink types: a page is a list of pen strokes (centre-line polylines)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class Stroke:
    points: np.ndarray  # (n, 2) float, page units, y down
    width: float
    color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    order: int = 0  # writing order within the page (-1 when unknown, e.g. scans)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        mn, mx = self.points.min(0), self.points.max(0)
        return float(mn[0]), float(mn[1]), float(mx[0]), float(mx[1])


@dataclass
class Page:
    source: str
    index: int
    strokes: list[Stroke] = field(default_factory=list)
    image: np.ndarray | None = None  # the original picture, for scans (dropped once the words are read)
    image_scale: float = 1.0  # pixels per page unit


def render(strokes, origin, size, scale, min_px: int = 1) -> Image.Image:
    """Draw strokes black-on-white. origin/size in page units, scale = px per unit."""
    w, h = int(size[0] * scale), int(size[1] * scale)
    im = Image.new("L", (max(w, 1), max(h, 1)), 255)
    d = ImageDraw.Draw(im)
    o = np.asarray(origin, float)
    for s in strokes:
        pts = [tuple(p) for p in (s.points - o) * scale]
        px = max(min_px, int(round(s.width * scale)))
        if len(pts) == 1:
            x, y = pts[0]
            d.ellipse([x - px / 2, y - px / 2, x + px / 2, y + px / 2], fill=0)
        else:
            d.line(pts, fill=0, width=px, joint="curve")
    return im


def render_colour(page: Page, scale: float = 2.0, background=(255, 255, 255)) -> Image.Image:
    """Draw a page in its own ink colours, for looking at how the notes are laid out."""
    strokes = [s for s in page.strokes if len(s.points)]
    if not strokes:
        return Image.new("RGB", (10, 10), background)
    pts = np.vstack([s.points for s in strokes])
    o = pts.min(0) - 10
    w, h = ((pts.max(0) - o + 10) * scale).astype(int)
    im = Image.new("RGB", (int(w), int(h)), background)
    d = ImageDraw.Draw(im, "RGBA")
    for s in strokes:
        rgb = [int(c * 255) for c in s.color[:3]]
        if min(rgb) > 215 and min(background) > 200:  # white ink (dark-paper notes) shown dark on white
            rgb = [40, 40, 40]
        q = [tuple(p) for p in (s.points - o) * scale]
        alpha = 90 if s.width > 4 else 255  # highlighter is see-through
        if len(q) > 1:
            d.line(q, fill=(*rgb, alpha), width=max(1, int(round(s.width * scale * 0.7))), joint="curve")
    return im
