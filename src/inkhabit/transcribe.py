# SPDX-License-Identifier: AGPL-3.0-or-later
"""Draft transcripts for `build --transcripts`.

Reads each page and writes the text out line by line, keeping the indentation
of the notes. Recognition is Apple Vision; it can be primed with the writer's
own vocabulary (from a inkhabit library). The draft is often wrong on fast
handwriting: it is meant to be corrected against the page by eye and then
given back to `build` as the labels.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import sources
from .words import page_words

UNSURE = 0.5


def page_lines(page, custom_words: list[str] | None = None, lang: str | None = None) -> list[dict]:
    """[{text, conf, indent, y}] in reading order (top to bottom)."""
    words = page_words(page, custom_words=custom_words, lang=lang)
    if not words:
        return []
    lines: dict[int, list] = {}
    for w in words:
        lines.setdefault(w.line, []).append(w)
    heights = [w.box[3] - w.box[1] for w in words]
    unit = max(float(np.median(heights)) * 1.5, 1e-6)  # one indent step is about this wide
    rows = []
    for ws in lines.values():
        ws.sort(key=lambda w: w.box[0])
        rows.append({"text": " ".join(w.text for w in ws), "conf": float(np.mean([w.conf for w in ws])),
                     "left": ws[0].box[0], "y": float(np.mean([w.box[1] for w in ws])),
                     "bottom": float(np.mean([w.box[3] for w in ws]))})
    rows.sort(key=lambda r: r["y"])
    margin = float(np.percentile([r["left"] for r in rows], 5))
    for r in rows:
        r["indent"] = int(max(0, round((r["left"] - margin) / unit)))
    gap = np.median(np.diff([r["y"] for r in rows])) if len(rows) > 2 else 0
    for a, b in zip(rows, rows[1:]):
        b["new_block"] = bool(gap and b["y"] - a["y"] > 1.6 * gap)
    return rows


def transcribe(path: str | Path, library: dict | None = None, mark_unsure: bool = True,
               lang: str | None = None) -> tuple[str, dict]:
    """Markdown transcript of one source file, and a small report."""
    custom = sorted((library or {}).get("words", {})) or None
    out, n_lines, n_unsure = [f"# {Path(path).stem}", ""], 0, 0
    for page in sources.load(path):
        rows = page_lines(page, custom, lang)
        page.image = None
        if not rows:
            continue
        out += [f"## Page {page.index + 1}", ""]
        for r in rows:
            if r.get("new_block"):
                out.append("")
            unsure = r["conf"] < UNSURE
            n_lines += 1
            n_unsure += unsure
            out.append("  " * min(r["indent"], 8) + r["text"] + ("  [?]" if unsure and mark_unsure else ""))
        out.append("")
    return "\n".join(out), {"lines": n_lines, "unsure_lines": int(n_unsure)}


def parse_transcript(text: str) -> dict[int, list[str]]:
    """Read back a (corrected) transcript written by `transcribe`: {page index: [line, ...]}."""
    pages: dict[int, list[str]] = {}
    page = 0
    for raw in text.splitlines():
        line = raw.strip()
        m = re.match(r"## Page (\d+)", line)
        if m:
            page = int(m.group(1)) - 1
        elif line and not line.startswith("#"):
            line = line.replace("[?]", "").replace("[illegible]", "").strip()
            if line:
                pages.setdefault(page, []).append(line)
    return pages
