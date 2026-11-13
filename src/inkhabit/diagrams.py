# SPDX-License-Identifier: AGPL-3.0-or-later
"""Diagrams: find the sketches in someone's notes, keep them, and draw them again.

A page of notes is words plus everything else: graphs, arrows between ideas, boxes, little pictures.
`find` picks out each drawing (the pen strokes that are not part of a readable word and sit together),
with the words written in and around it as its labels. `catalogue` saves them all, with a numbered
picture sheet so a person or an agent can pick one. `redraw` draws one again: the same pen movements
with the small differences of a second attempt, at any size, in any colours, and with any label
rewritten in the person's handwriting. A label that is not rewritten keeps its original ink, because
that is their real writing and the machine's reading of a diagram label is often wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .ink import Page
from .words import MAX_PEN, WordSample

BIG = 2.5  # a stroke longer than this many letter-heights is drawing, not writing
REACH = 1.2  # strokes this many letter-heights apart belong to the same drawing
MIN_SIDE = 3.5  # a drawing is at least this many letter-heights across and down


def _len(p: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def find(page: Page, samples: list[WordSample]) -> list[dict]:
    """The drawings on one page. Each: {box, strokes: [{points, width, color, label}], labels: [{text, box}]}.
    `label` on a stroke is the index of the label it belongs to, or None for the drawing itself."""
    strokes = [s for s in page.strokes if len(s.points) and s.width <= MAX_PEN]
    if not strokes or not samples:
        return []
    owner: dict[int, int] = {}
    for k, w in enumerate(samples):
        for p in w.strokes:
            owner[id(p)] = k
    unit = float(np.median([w.box[3] - w.box[1] for w in samples]))  # a written line's height
    boxes = np.array([s.bbox for s in strokes])
    size = np.hypot(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    free = np.array([id(s.points) not in owner for s in strokes])
    flat = (boxes[:, 3] - boxes[:, 1]) < 0.35 * unit  # underlines and dashes are not drawings by themselves
    seeds = np.flatnonzero(free & (size > BIG * unit) & ~(flat & (size < 12 * unit)))
    if len(seeds) == 0:
        return []
    # join seeds, and the free strokes near them, into groups
    members = np.flatnonzero(free)
    parent = {int(i): int(i) for i in members}

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pad = REACH * unit
    grown = boxes + [-pad / 2, -pad / 2, pad / 2, pad / 2]
    for a in seeds:
        near = members[(grown[members, 0] < grown[a, 2]) & (grown[members, 2] > grown[a, 0])
                       & (grown[members, 1] < grown[a, 3]) & (grown[members, 3] > grown[a, 1])]
        for b in near:
            parent[root(int(b))] = root(int(a))
    groups: dict[int, list[int]] = {}
    seed_set = set(int(s) for s in seeds)
    for i in members:
        groups.setdefault(root(int(i)), []).append(int(i))
    out = []
    for idx in groups.values():
        if not seed_set & set(idx):
            continue
        b = boxes[idx]
        x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
        ink = sum(_len(strokes[i].points) for i in idx)
        if min(x1 - x0, y1 - y0) < MIN_SIDE * unit or ink < 14 * unit or len(idx) < 2:
            continue
        # words inside the drawing, or just beside it, are its labels
        m = 0.8 * unit
        labels, label_of = [], {}
        for k, w in enumerate(samples):
            cx, cy = (w.box[0] + w.box[2]) / 2, (w.box[1] + w.box[3]) / 2
            if x0 - m < cx < x1 + m and y0 - m < cy < y1 + m:
                label_of[k] = len(labels)
                labels.append({"text": w.text, "box": [round(float(v), 2) for v in w.box]})
                x0, y0, x1, y1 = min(x0, w.box[0]), min(y0, w.box[1]), max(x1, w.box[2]), max(y1, w.box[3])
        inside = [i for i, s in enumerate(strokes)
                  if (i in idx) or owner.get(id(s.points)) in label_of
                  or (free[i] and x0 < (boxes[i, 0] + boxes[i, 2]) / 2 < x1 and y0 < (boxes[i, 1] + boxes[i, 3]) / 2 < y1)]
        out.append({
            "source": page.source, "page": page.index, "unit": round(unit, 3),
            "box": [round(float(v), 2) for v in (x0, y0, x1, y1)],
            "strokes": [{"points": np.round(strokes[i].points, 2).tolist(), "width": round(float(strokes[i].width), 3),
                         "color": [round(float(c), 3) for c in strokes[i].color[:3]],
                         "label": label_of.get(owner.get(id(strokes[i].points)))} for i in inside],
            "labels": labels,
        })
    out.sort(key=lambda d: d["box"][1])
    return out


def _rgb(c, dark_on_white: bool = True) -> tuple[int, int, int]:
    rgb = [int(v * 255) for v in c[:3]]
    return (40, 40, 48) if dark_on_white and min(rgb) > 215 else tuple(rgb)  # white ink is shown dark on white


def picture(d: dict, scale: float = 3.0, number_labels: bool = False):
    """A drawing as an image. With number_labels, each label gets its index in red beside it."""
    from PIL import Image, ImageDraw

    x0, y0, x1, y1 = d["box"]
    m = 10
    im = Image.new("RGB", (int((x1 - x0) * scale) + 2 * m, int((y1 - y0) * scale) + 2 * m), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    for s in d["strokes"]:
        pts = [((x - x0) * scale + m, (y - y0) * scale + m) for x, y in s["points"]]
        px = max(1, int(round(s["width"] * scale)))
        if len(pts) == 1:
            dr.ellipse([pts[0][0] - px, pts[0][1] - px, pts[0][0] + px, pts[0][1] + px], fill=_rgb(s["color"]))
        else:
            dr.line(pts, fill=_rgb(s["color"]), width=px, joint="curve")
    if number_labels:
        for k, lab in enumerate(d["labels"]):
            dr.text(((lab["box"][0] - x0) * scale + m, (lab["box"][1] - y0) * scale + m - 9), str(k), fill=(220, 0, 0))
    return im


def catalogue(found: list[dict], out_dir: str | Path) -> dict:
    """Save drawings as <out>/diagrams.json and one numbered picture each in <out>/diagrams/."""
    out = Path(out_dir)
    (out / "diagrams").mkdir(parents=True, exist_ok=True)
    for k, d in enumerate(found):
        d["id"] = k
        picture(d, number_labels=True).save(out / "diagrams" / f"{k:03d}.png")
    (out / "diagrams.json").write_text(json.dumps(found, ensure_ascii=False))
    return {"diagrams": len(found), "labels": sum(len(d["labels"]) for d in found)}


def _again(p: np.ndarray, rng: np.random.Generator, unit: float, amount: float) -> np.ndarray:
    """The same stroke as a second attempt: a slow drift along its length, never a jitter."""
    if len(p) < 3 or amount <= 0:
        return p
    t = np.linspace(0, 1, len(p))
    drift = np.zeros_like(p)
    for f in (1, 2, 3):
        drift += rng.normal(0, 1, 2) * np.sin(np.pi * f * t[:, None] + rng.uniform(0, 6.3)) / f
    small = min(1.0, _len(p) / (5 * unit))  # a short tick has less room to wander than a long curve
    return p + drift * unit * 0.07 * amount * small


def redraw(d: dict, lib: dict | None = None, labels: dict | None = None, seed: int = 0, vary: float = 1.0,
           width: float | None = None, colours: dict | None = None) -> dict:
    """Draw a catalogued drawing again. Returns {size, strokes: [{points, width, color}], missing, tight}
    (tight: new labels that had to be written smaller to fit before the next thing on their line).

    labels: {label index or its old text: new text}; rewritten in the person's hand (needs lib), in the
    old label's place and size. width: the new overall width (same units as the source; default as drawn).
    vary: 0 = an exact copy, 1 = the natural difference between two attempts, 2 = looser.
    colours: {"old r,g,b" or "all": [r, g, b]} to change ink colours."""
    from . import render

    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = d["box"]
    k = (width / (x1 - x0)) if width else 1.0
    unit = d["unit"]
    wanted = {}
    for key, text in (labels or {}).items():
        hits = [i for i, lab in enumerate(d["labels"]) if str(key) == str(i) or str(key) == lab["text"]]
        if not hits:
            raise SystemExit(f"no label {key!r} in this diagram; it has: " + ", ".join(f"{i}={lab['text']!r}" for i, lab in enumerate(d["labels"])))
        for i in hits:
            wanted[i] = text
    if wanted and lib is None:
        raise SystemExit("rewriting labels needs the handwriting library")

    def colour(c):
        key = ",".join(str(v) for v in c)
        return (colours or {}).get(key, (colours or {}).get("all", c))

    # the whole drawing leans and stretches a little, as a second attempt does
    lean, sx, sy = rng.normal(0, 0.012) * vary, 1 + rng.normal(0, 0.02) * vary, 1 + rng.normal(0, 0.02) * vary
    centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])

    def place(p: np.ndarray) -> np.ndarray:
        q = (p - centre) * [sx, sy]
        q = np.c_[q[:, 0] - q[:, 1] * lean, q[:, 1] + q[:, 0] * lean]
        return (q + centre - [x0, y0]) * k

    out, missing = [], set()
    for s in d["strokes"]:
        if s["label"] in wanted:
            continue
        p = _again(np.asarray(s["points"], float), rng, unit, vary * (0.4 if s["label"] is not None else 1.0))
        out.append({"points": place(p), "width": s["width"] * k, "color": colour(s["color"])})
    tight = []
    if wanted:
        writer = render.Writer(lib, seed, render.situation(lib, "label")[0])
        ink_of = {i: np.vstack([np.asarray(s["points"], float) for s in d["strokes"] if s["label"] == i] or [np.zeros((0, 2))])
                  for i in range(len(d["labels"]))}
        heights = [np.ptp(p[:, 1]) for p in ink_of.values() if len(p) > 3]
        # one writing size for the whole drawing, from its own labels (most have a tall letter: 1.6 small-letter heights)
        xh = float(np.median(heights)) / 1.6 if heights else unit * 0.5
        for i, text in wanted.items():
            lab, old = d["labels"][i], ink_of[i]
            left, base = (old[:, 0].min(), old[:, 1].max()) if len(old) else (lab["box"][0], lab["box"][3])
            ink = writer.write(text, xh)
            missing |= set(ink.missing)
            new = [np.asarray(st, float) for st in ink.strokes]
            if not new:  # nothing to write: the old label is simply left out
                continue
            pts = np.vstack(new)
            origin = np.array([pts[:, 0].min(), 2.2 * xh])  # the writer puts the baseline 2.2 small-letter heights down
            # room before the next thing on the same line that is not this label
            room = np.inf
            for s in d["strokes"]:
                if s["label"] == i:
                    continue
                q = np.asarray(s["points"], float)
                if q[:, 1].max() > base - 1.4 * xh and q[:, 1].min() < base + 0.3 * xh and q[:, 0].min() > left + xh:
                    room = min(room, q[:, 0].min() - left - 0.5 * xh)
            squeeze = float(np.clip(room / max(np.ptp(pts[:, 0]), 1e-6), 0.6, 1.0))
            if squeeze < 1.0:
                tight.append(text)
            mine = [s for s in d["strokes"] if s["label"] == i]
            pen = float(np.median([s["width"] for s in mine])) if mine else unit * 0.08
            col = mine[0]["color"] if mine else [0.1, 0.1, 0.18]
            for st in new:
                out.append({"points": place((st - origin) * squeeze + [left, base]), "width": pen * k, "color": colour(col)})
    return {"size": [(x1 - x0) * k, (y1 - y0) * k], "strokes": out, "missing": sorted(missing), "tight": tight}


def save(drawing: dict, path: str | Path, scale: float = 4.0) -> None:
    """Write a redrawn diagram as .png, .svg or .pdf."""
    path = Path(path)
    w, h = drawing["size"]
    m = 8.0
    if path.suffix.lower() == ".png":
        d = {"box": [0, 0, w, h], "labels": [], "strokes": [{**s, "points": np.asarray(s["points"]).tolist()} for s in drawing["strokes"]]}
        picture(d, scale).save(path)
    elif path.suffix.lower() == ".svg":
        body = []
        for s in drawing["strokes"]:
            pts = " ".join(f"{x + m:.2f},{y + m:.2f}" for x, y in np.asarray(s["points"]))
            r, g, b = _rgb(s["color"])
            body.append(f'<polyline points="{pts}" fill="none" stroke="rgb({r},{g},{b})" stroke-width="{s["width"]:.2f}" stroke-linecap="round" stroke-linejoin="round"/>')
        path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w + 2 * m:.1f} {h + 2 * m:.1f}">' + "".join(body) + "</svg>")
    elif path.suffix.lower() == ".pdf":
        import pymupdf

        from . import render

        doc = pymupdf.open()
        page = doc.new_page(width=w + 2 * m, height=h + 2 * m)
        for s in drawing["strokes"]:
            r, g, b = _rgb(s["color"])
            render.onto_pdf_page(page, render.Ink([np.asarray(s["points"], float)], [s["width"]]), (m, m), (r / 255, g / 255, b / 255))
        doc.save(path)
    else:
        raise SystemExit("write a .png, .svg or .pdf")
