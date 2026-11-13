# SPDX-License-Identifier: AGPL-3.0-or-later
"""Find places to write on a PDF, and write there in handwriting."""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf

from .render import Human, Writer, form_preset, onto_pdf_page


MIN_PEN_PT = 0.95  # about a 0.35 mm pen: the thinnest line a filled-in document should have


def open_doc(path: str | Path):
    """Open the document to write on. A picture (a photographed or scanned form) becomes a one-page PDF with
    the picture as its page, so everything else works the same way."""
    from .sources import IMAGE_EXT, kind

    path = Path(path)
    k = kind(path)
    if k in IMAGE_EXT:
        from PIL import Image

        from .scan import _to_rgb, _upright

        import io

        if k in (".heic", ".heif"):
            from .scan import _images

            rgb = _images(path, k)[0][0]
        else:
            rgb = _upright(_to_rgb(Image.open(path)))
        h, w = rgb.shape[:2]
        scale = 595.0 / w if w <= h else 842.0 / w  # about A4 across, so point sizes are familiar
        doc = pymupdf.open()
        page = doc.new_page(width=w * scale, height=h * scale)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, "JPEG", quality=92)
        page.insert_image(page.rect, stream=buf.getvalue())
        return doc
    doc = pymupdf.open(path)
    if doc.needs_pass:
        raise ValueError(f"{path.name} is password-protected: export an unlocked copy and use that")
    _picture_pages.clear()  # what was remembered about pages belongs to the document before this one
    _ocr_cache.clear()
    return doc


_picture_pages: dict[tuple[int, int], bool] = {}


def is_picture_only(page) -> bool:
    """A scan or photo: no real text and no drawn shapes, only an image. Remembered per page, because
    once we start drawing on it the page has shapes of its own."""
    key = (id(page.parent), page.number)
    if key not in _picture_pages:
        _picture_pages[key] = len(page.get_text("words")) < 3 and len(page.get_drawings()) < 3 and bool(page.get_images())
    return _picture_pages[key]


_ocr_cache: dict[tuple[int, int], list] = {}


def ocr_words(page) -> list[dict]:
    """Printed words on a picture-only page, read by the recogniser: [{text, rect (points), line}]."""
    key = (id(page.parent), page.number)
    if key not in _ocr_cache:
        import numpy as np
        from PIL import Image

        from .ocr import recognize

        k = 200 / 72
        px = page.get_pixmap(dpi=200, colorspace=pymupdf.csRGB)
        im = Image.fromarray(np.frombuffer(px.samples, np.uint8).reshape(px.height, px.width, 3))
        _ocr_cache[key] = [{"text": w.text, "rect": [v / k for v in w.box], "line": w.line} for w in recognize(im)]
    return _ocr_cache[key]


def _classify(rects: list, pno: int) -> list[dict]:
    """Boxes into tick boxes, rows of character cells, and plain boxes."""
    out, used = [], set()
    rects = sorted(rects, key=lambda r: (round(r.y0 / 6), r.x0))  # rows first (allowing a little tilt), then left to right
    for i, r in enumerate(rects):  # a run of equal cells side by side is one field written a character per cell
        if i in used:
            continue
        run = [i]
        for j in range(i + 1, len(rects)):
            q, last = rects[j], rects[run[-1]]  # compared with its neighbour, so a slightly tilted scan still chains
            if j not in used and abs(q.y0 - last.y0) < 3 and abs(q.height - r.height) < 3 and abs(q.width - r.width) < 3 and abs(q.x0 - last.x1) < 4:
                run.append(j)
        if len(run) >= 3 and r.width < 40:
            used.update(run)
            out.append({"kind": "cells", "page": pno, "cells": len(run),
                        "rect": [round(v, 1) for v in (r.x0, r.y0, rects[run[-1]].x1, r.y1)], "use": {"comb": len(run)}})
    for i, r in enumerate(rects):
        if i in used:
            continue
        if r.width <= 22 and 0 < r.height <= 22 and 0.6 < r.width / r.height < 1.6:
            out.append({"kind": "tickbox", "page": pno, "rect": [round(v, 1) for v in r], "use": {"check": True}})
        else:
            out.append({"kind": "box", "page": pno, "rect": [round(v, 1) for v in r],
                        **({"use": {"valign": "middle"}} if r.height < 40 else {})})
    return out


def _picture_fields(page, pno: int, read_labels: bool = True) -> list[dict]:
    """Lines, boxes and labels found by looking at a scanned or photographed form."""
    import cv2
    import numpy as np

    k = 150 / 72
    px = page.get_pixmap(dpi=150, colorspace=pymupdf.csGRAY)
    g = np.frombuffer(px.samples, np.uint8).reshape(px.height, px.width)
    # thin printed lines are faint after scanning: pick out anything darker than its surroundings
    back = cv2.medianBlur(g, 31)
    dark = ((back.astype(np.int16) - g.astype(np.int16)) > 22).astype(np.uint8) * 255
    horiz = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(14 * k), 1)))
    # for boxes, shorter runs count too (a tick box is only a few millimetres across)
    h2 = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(7 * k), 1)))
    v2 = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(7 * k))))
    grid = cv2.dilate(h2 | v2, np.ones((3, 3), np.uint8))
    rects = []
    contours, hier = cv2.findContours(grid, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for c, h in zip(contours, hier[0] if hier is not None else []):
        if h[3] == -1:  # we want the holes: the empty insides of drawn boxes
            continue
        x, y, w, hh = cv2.boundingRect(c)
        if w >= 5 * k and hh >= 5 * k and w < g.shape[1] * 0.95 and hh < g.shape[0] * 0.6 and cv2.contourArea(c) > 0.75 * w * hh:
            rects.append(pymupdf.Rect((x - 1) / k, (y - 1) / k, (x + w + 1) / k, (y + hh + 1) / k))
    out = _classify(rects, pno)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(horiz, connectivity=8)
    for i in range(1, n):
        x, y, w, hh = stats[i, :4]
        r = (x / k, y / k, (x + w) / k, (y + hh) / k)
        if w < 40 * k or hh > 4 * k + 0.02 * w:  # a tilted scan makes a long line a shallow staircase
            continue
        # the top or bottom edge of a box, or of a row of cells: together the boxes on that level cover it
        y_line = (r[1] + r[3]) / 2
        covered = sum(max(0.0, min(r[2], b.x1) - max(r[0], b.x0)) for b in rects if abs(y_line - b.y0) < 6 or abs(y_line - b.y1) < 6)
        if covered > 0.5 * (r[2] - r[0]):
            continue  # the top or bottom edge of a box, not a line to write on
        out.append({"kind": "line", "page": pno, "rect": [round(v, 1) for v in (r[0], r[1] - 16, r[2], r[1])]})
    lines: dict[int, list] = {}
    for w in ocr_words(page) if read_labels else []:
        lines.setdefault(w["line"], []).append(w)
    for ws in lines.values():
        ws.sort(key=lambda w: w["rect"][0])
        text = " ".join(w["text"] for w in ws)
        if any(ch.isalnum() for ch in text):
            out.append({"kind": "label", "page": pno, "text": text[:80],
                        "rect": [round(v, 1) for v in (ws[0]["rect"][0], min(w["rect"][1] for w in ws), ws[-1]["rect"][2], max(w["rect"][3] for w in ws))]})
    labels = [e["rect"] for e in out if e["kind"] == "label"]

    def through_text(e) -> bool:  # a "line" that is really the feet of a row of printed letters
        x0, _, x1, y = e["rect"]
        return any(l[1] - 1 <= y <= l[3] + 2 and min(x1, l[2]) - max(x0, l[0]) > 0.5 * (x1 - x0) for l in labels)

    out = [e for e in out if not (e["kind"] == "line" and through_text(e))]
    for e in out:
        e["rect"] = [round(float(v), 1) for v in e["rect"]]
    return out


def fields(pdf: str | Path) -> list[dict]:
    """Everything an agent needs to decide where to write: form fields, blank lines, boxes, tick boxes,
    rows of character cells, and the printed labels (one per printed line), all with positions in points.
    Works on real PDFs and, by looking at the picture, on scanned or photographed forms."""
    out = []
    with open_doc(pdf) as doc:
        for pno, page in enumerate(doc):
            if is_picture_only(page):
                out += _picture_fields(page, pno)
                continue
            for w in page.widgets() or []:
                out.append({"kind": "field", "page": pno, "name": w.field_name, "type": w.field_type_string,
                            "rect": [round(v, 1) for v in w.rect], "value": w.field_value})
            rects = []
            for d in page.get_drawings():
                r = d["rect"]
                if r.width > 40 and r.height < 2.5:
                    out.append({"kind": "line", "page": pno, "rect": [round(v, 1) for v in (r.x0, r.y0 - 16, r.x1, r.y0)]})
                elif 6 <= r.width < page.rect.width * 0.95 and 6 <= r.height < page.rect.height * 0.6 and d.get("fill") is None:
                    rects.append(r)
            found = _classify(rects, pno)
            # tables are usually drawn as separate lines, not boxes: look at the page as a picture as well and
            # add any cells that way, unless a drawn box already covers them
            def overlap(a, b):
                ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
                return ix / max(1e-6, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))

            for e in _picture_fields(page, pno, read_labels=False):
                if e["kind"] in ("box", "tickbox", "cells") and not any(overlap(e["rect"], f["rect"]) > 0.5 for f in found):
                    found.append(e)
            out += found

            def is_edge(e) -> bool:  # a ruled line that is really the edge of boxes or table cells
                x0, _, x1, y = e["rect"]
                covered = sum(max(0.0, min(x1, f["rect"][2]) - max(x0, f["rect"][0])) for f in found
                              if abs(y - f["rect"][1]) < 4 or abs(y - f["rect"][3]) < 4)
                return covered > 0.5 * (x1 - x0)

            out[:] = [e for e in out if not (e["kind"] == "line" and e["page"] == pno and is_edge(e))]
            for blk in page.get_text("dict")["blocks"]:
                for ln in blk.get("lines", []):
                    text = "".join(sp["text"] for sp in ln["spans"]).strip()
                    if text:
                        out.append({"kind": "label", "page": pno, "rect": [round(v, 1) for v in ln["bbox"]], "text": text[:80]})
            for inst in page.search_for("____"):  # blanks typed as underscores
                out.append({"kind": "line", "page": pno, "rect": [round(v, 1) for v in (inst.x0, inst.y0 - 4, inst.x1, inst.y1)]})
    return out


def fill(pdf: str | Path, entries: list[dict], lib: dict, out: str | Path, seed: int = 0,
         color=(0.08, 0.1, 0.35), human: Human | None = None) -> list[dict]:
    """Write each entry inside its rect.

    entry = {page, rect:[x0,y0,x1,y1], text} or {field, text}, with optional:
      "check": true        tick the box at rect instead of writing ("mark": "cross" for an X)
      "comb": n            the rect is n equal cells; write one character in each (dates, postcodes, IDs)
      "valign": "middle"   a single line inside a closed box: centre it instead of sitting it on the bottom edge
      "align": "center" | "right"
      "line_pitch": pts    multi-line text on ruled lines this far apart; the first baseline sits on y0 + pitch
      "for": "heading" | "label" | "letter" | "note" | "scribble"   what the entry is (default: a form field)
      "math": true         the text is maths (LaTeX-like); lines separated by \\\\ are worked steps
      "x_height", "style": {...}   size and per-entry setting overrides
    """
    import numpy as np

    from .render import Ink

    base = human or form_preset(lib)
    writer = Writer(lib, seed=seed, human=base)
    writer.min_pen = MIN_PEN_PT
    report = []
    with open_doc(pdf) as doc:
        by_name = {w.field_name: (pno, list(w.rect)) for pno, page in enumerate(doc) for w in (page.widgets() or [])}
        for e in entries:
            if "field" in e:
                if e["field"] not in by_name:
                    report.append({**e, "status": "no such field"})
                    continue
                pno, rect = by_name[e["field"]]
            else:
                pno, rect = e["page"], e["rect"]
            x0, y0, x1, y1 = rect
            size = 1.0
            if e.get("for"):  # what this entry is: a heading, a label, a letter... (see render.situation)
                from .render import situation

                h, size = situation(lib, e["for"])
                writer.h = h.with_settings(e.get("style", {}))
            else:
                writer.h = base.with_settings(e.get("style", {}))  # per-entry overrides, e.g. {"slant": 4}
            pen = max(lib["meta"].get("font_weight", 0.16) * 6.0 * writer.h.weight, MIN_PEN_PT)
            if e.get("check"):
                from .annotate import Pen

                p = Pen(lib, int(writer.rng.integers(1 << 30)), size=max(3.0, (y1 - y0) * 0.42))
                cx, cy, r = (x0 + x1) / 2, (y0 + y1) / 2, min(x1 - x0, y1 - y0) / 2
                if e.get("mark") == "cross":
                    strokes = [p.path([[cx - r * 0.7, cy - r * 0.7], [cx + r * 0.75, cy + r * 0.7]]), p.path([[cx + r * 0.7, cy - r * 0.75], [cx - r * 0.7, cy + r * 0.7]])]
                else:  # a tick that starts inside the box and flicks out of its top right corner, as people do
                    strokes = [p.path([[cx - r * 0.7, cy], [cx - r * 0.15, cy + r * 0.65], [cx + r * 1.25, cy - r * 1.2]])]
                onto_pdf_page(doc[pno], Ink(strokes, [pen] * len(strokes)), (0, 0), color)
                report.append({"check": True, "page": pno, "rect": rect, "status": "ok"})
                continue
            text = str(e["text"])
            if e.get("math"):  # an answer in maths: fractions, powers, worked steps (see mathwrite)
                from .mathwrite import write_maths

                xh = e.get("x_height") or 6.5
                while True:
                    ink = write_maths(lib, text, xh, int(writer.rng.integers(1 << 30)), writer.h, min_pen=MIN_PEN_PT)
                    if (ink.size[0] <= x1 - x0 - 2 and ink.size[1] <= y1 - y0) or xh <= 3.2:
                        break
                    xh *= 0.92
                fits = ink.size[0] <= x1 - x0 - 2 and ink.size[1] <= y1 - y0
                onto_pdf_page(doc[pno], ink, (x0 + 3, y0 + 2), color)
                report.append({"text": text, "page": pno, "x_height": round(xh, 2), "fits": bool(fits), "missing": "".join(sorted(ink.missing)),
                               "status": "ok" if fits and not ink.missing else "check"})
                continue
            if e.get("comb"):
                n = int(e["comb"])
                cell = (x1 - x0) / n
                xh = e.get("x_height") or min((y1 - y0) * 0.36, cell * 0.5)
                missing = set()
                for k, ch in enumerate(text.replace(" ", "")[:n]):
                    ink = writer.write(ch, xh)
                    missing |= ink.missing
                    if not ink.strokes:
                        continue
                    pts = np.vstack(ink.strokes)
                    w = pts[:, 0].max() - pts[:, 0].min()
                    ox = x0 + k * cell + (cell - w) / 2 - pts[:, 0].min()
                    oy = y1 - 2.2 * xh - 0.16 * (y1 - y0)
                    onto_pdf_page(doc[pno], ink, (ox, oy), color)
                fits = len(text.replace(" ", "")) <= n
                report.append({"text": text, "page": pno, "x_height": round(xh, 2), "fits": fits, "missing": "".join(sorted(missing)),
                               "status": "ok" if fits and not missing else "check"})
                continue
            pitch = e.get("line_pitch")
            multiline = e.get("multiline", bool(pitch) or (y1 - y0) > 40)
            # start at a comfortable size and let fitting shrink it: people fill a big box with normal-sized writing,
            # not tiny writing. Only headings are scaled up; labels keep the size the space suggests.
            xh = e.get("x_height") or max(size, 1.0) * (min(pitch * 0.34, 7.0) if pitch else 7.0 if multiline
                                                        else min(7.5, max(3.5, (y1 - y0) * 0.4)))
            if pitch:  # on ruled lines: the line spacing is fixed by the paper, so only the size can give
                while True:
                    ink = writer.write(text, xh, x1 - x0 - 4, line_gap=pitch / xh)
                    if (ink.size[0] <= x1 - x0 - 2 and ink.size[1] <= y1 - y0 + xh) or xh <= 3.0:
                        break
                    xh *= 0.93
            else:
                ink, xh = writer.write_to_fit(text, x1 - x0 - 4, (y1 - y0) if multiline else None, xh, multiline=multiline)
            fits = ink.size[0] <= x1 - x0 - 2 and (not multiline or ink.size[1] <= y1 - y0 + (xh if pitch else 0))
            # single line: sit the baseline just above the bottom of the box
            if pitch:
                oy = y0 + pitch - 2.2 * xh - 0.6
            elif e.get("valign") == "middle" and not multiline:  # inside a closed box: centred, clear of the edges
                oy = y0 + (y1 - y0) / 2 + 0.55 * xh - 2.2 * xh
            else:
                oy = y0 + 1 if multiline else y1 - 2.2 * xh - 0.1 * (y1 - y0)
            slack = max(0.0, x1 - x0 - 4 - ink.size[0])
            ox = x0 + 2 + (slack / 2 if e.get("align") == "center" else slack if e.get("align") == "right" else writer.rng.uniform(0, 3))
            onto_pdf_page(doc[pno], ink, (ox, oy), color)
            report.append({"text": text, "page": pno, "x_height": round(xh, 2), "fits": bool(fits),
                           "missing": "".join(sorted(ink.missing)), "status": "ok" if fits and not ink.missing else "check"})
        doc.save(out)
    return report


def load_entries(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())
