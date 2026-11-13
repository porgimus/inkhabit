# SPDX-License-Identifier: AGPL-3.0-or-later
"""A practice sheet for characters the library is short of, and reading it back in.

The sheet is a grid: one row per character, several boxes to write it in, each with a baseline and a
small-letter guide line. Because we know which character every box is for and where its baseline is,
what is written there needs no recognition and no size estimate: it is the cleanest source there is.
Write on it in a note app (import the PDF) or print it, then give the result back.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pymupdf

from . import library, sources

GUIDE = (0.74, 0.80, 0.93)  # the sheet's own lines, in a colour no pen is likely to match
PAGE = (612.0, 792.0)
MARGIN, LABEL, BOX_W, ROW_H, XH = 40.0, 44.0, 78.0, 62.0, 16.0


def layout(chars: str, reps: int = 6) -> list[dict]:
    """Where every box is: [{page, ch, rect, baseline}]"""
    per_page = int((PAGE[1] - 2 * MARGIN - 20) // ROW_H)
    boxes = []
    for i, ch in enumerate(chars):
        page, row = divmod(i, per_page)
        y0 = MARGIN + 20 + row * ROW_H
        for r in range(reps):
            x0 = MARGIN + LABEL + r * (BOX_W + 6)
            boxes.append({"page": page, "ch": ch, "rect": [x0, y0, x0 + BOX_W, y0 + ROW_H - 8], "baseline": y0 + ROW_H - 8 - 16})
    return boxes


def make(chars: str, path: str | Path, reps: int = 6) -> int:
    """Write the practice sheet PDF (and <path>.json describing it). Returns the number of pages."""
    chars = "".join(dict.fromkeys(c for c in chars if not c.isspace()))
    if not chars:
        raise SystemExit("no characters to put on a sheet: every character already has enough samples")
    reps = max(1, min(reps, int((PAGE[0] - MARGIN - LABEL - 10 + 6) // (BOX_W + 6))))  # as many boxes as fit across the page
    boxes = layout(chars, reps)
    doc = pymupdf.open()
    for pno in range(max(b["page"] for b in boxes) + 1):
        page = doc.new_page(width=PAGE[0], height=PAGE[1])
        page.insert_text((MARGIN, MARGIN + 4), "Write each character in every box, the way you normally write it. "
                         "Sit it on the lower line; small letters reach the upper line.", fontsize=9, color=(0.4, 0.4, 0.45))
        seen = set()
        for b in (b for b in boxes if b["page"] == pno):
            x0, y0, x1, y1 = b["rect"]
            page.draw_rect(pymupdf.Rect(x0, y0, x1, y1), color=GUIDE, width=0.6)
            page.draw_line((x0, b["baseline"]), (x1, b["baseline"]), color=GUIDE, width=0.6)
            page.draw_line((x0, b["baseline"] - XH), (x1, b["baseline"] - XH), color=GUIDE, width=0.3, dashes="[2 2]")
            if b["ch"] not in seen:
                seen.add(b["ch"])
                page.insert_text((MARGIN + 6, b["baseline"]), b["ch"], fontsize=22, color=(0.25, 0.25, 0.3))
    doc.save(str(path))
    Path(str(path) + ".json").write_text(json.dumps({"chars": chars, "reps": reps}))
    return len(doc)


def read_back(filled: str | Path, sheet_json: str | Path, lib: dict, log=print) -> dict[str, int]:
    """Add what was written on a filled-in sheet to the library. Returns {char: samples added}."""
    spec = json.loads(Path(sheet_json).read_text())
    boxes = layout(spec["chars"], spec["reps"])
    added: dict[str, int] = {}
    for page in sources.load(filled):
        mine = [b for b in boxes if b["page"] == page.index]
        ink = [s for s in page.strokes if not np.allclose(s.color[:3], GUIDE, atol=0.04) and s.width <= 6]
        for b in mine:
            x0, y0, x1, y1 = b["rect"]
            inside = [s.points for s in ink if x0 - 2 <= (s.bbox[0] + s.bbox[2]) / 2 <= x1 + 2 and y0 - 4 <= (s.bbox[1] + s.bbox[3]) / 2 <= y1 + 6]
            if not inside:
                continue
            pts = np.vstack(inside)
            if np.ptp(pts[:, 0]) > BOX_W * 0.97 and np.ptp(pts[:, 1]) < 2:
                continue  # a guide line that slipped through, not writing
            left = pts[:, 0].min()
            strokes = [np.round((p - [left, b["baseline"]]) / XH, 3).tolist() for p in inside]
            entry = lib["glyphs"].setdefault(b["ch"], {"count": 0, "variants": []})
            if entry["variants"] and entry["variants"][0].get("source") in ("neutral", "derived", "extrapolated", "composed"):
                entry["variants"] = []  # real samples replace stand-ins
            entry["variants"].insert(0, {"strokes": strokes, "width": round(float(np.ptp(pts[:, 0]) / XH), 3), "source": "sheet"})
            entry["variants"] = entry["variants"][: max(library.VARIANTS, 8)]
            entry["count"] = entry.get("count", 0) + 1
            added[b["ch"]] = added.get(b["ch"], 0) + 1
    if "rejected" in lib:
        lib["rejected"] = [c for c in lib["rejected"] if c not in added]
    lib["glyphs"] = dict(sorted(lib["glyphs"].items()))
    log(f"added {sum(added.values())} samples for {len(added)} characters: {''.join(sorted(added))}")
    return added


def wanted(lib: dict, thin: int = 15) -> str:
    """Characters worth practising: missing, stand-ins, or with few samples."""
    from .font import with_fallbacks

    table, borrowed = with_fallbacks(lib)
    want = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:;?!'\"()-&@#*/+=%$"
    out = []
    for c in want:
        e = lib["glyphs"].get(c)
        if c in borrowed or e is None or e.get("count", 0) < thin or len(e["variants"]) < 4:
            out.append(c)
    return "".join(out)
