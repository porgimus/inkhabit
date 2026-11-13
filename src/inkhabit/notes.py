# SPDX-License-Identifier: AGPL-3.0-or-later
r"""Write a page of notes the way the person takes notes, not just in their letters.

Input is a plain outline:

    # Heading
    A line of text                     one idea per line
      - an indented point              two spaces per level; "-" gives a dash bullet
        a consequence -> a result      "->" becomes a drawn arrow
    > a side remark                    written to the right of the line above, with an arrow to it
    ! a key line                       gets a box drawn round it
    $ T = 2\pi \sqrt{\frac{L}{g}}     maths; several "$" lines in a row are worked steps, aligned on "="
    @ axes Amount Month              a small sketch; follow with "@ curve up Daylight", "@ curve down Frost",
    @ flow cause -> effect -> result   "@ point Planting"; or a chain of boxes joined by arrows
    @ theirs 34 3=EUR                  drawing 34 from their own notes, drawn again, label 3 rewritten
                                       a blank line starts a new block (and a new colour)

Layout habits come from the writer's profile when there is one (`profile.json` from build): which ink
colours they use and how often, how far they indent, how loosely they space lines. Colour changes by
block, not by importance, because that is how most people who colour their notes do it; indented points
are led in by a hooked arrow or a dash.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pymupdf

from . import render
from .annotate import Pen

ON_DARK = {"blue": (0.0, 0.48, 1.0), "pink": (1.0, 0.25, 0.7), "cyan": (0.2, 0.9, 1.0), "orange": (0.98, 0.62, 0.15), "white": (0.97, 0.97, 0.97),
           "yellow": (1.0, 0.95, 0.2), "red": (1.0, 0.3, 0.3), "green": (0.3, 0.85, 0.45), "purple": (0.75, 0.5, 1.0), "black": (0.97, 0.97, 0.97),
           "grey": (0.8, 0.8, 0.8)}
ON_LIGHT = {"blue": (0.05, 0.25, 0.75), "pink": (0.8, 0.1, 0.5), "cyan": (0.0, 0.5, 0.65), "orange": (0.85, 0.42, 0.0), "white": (0.1, 0.1, 0.12),
            "yellow": (0.7, 0.55, 0.0), "red": (0.78, 0.1, 0.1), "green": (0.1, 0.5, 0.2), "purple": (0.45, 0.2, 0.7), "black": (0.1, 0.1, 0.12),
            "grey": (0.35, 0.35, 0.37)}
PAPERS = {"dark": ((0.19, 0.19, 0.19), (0.34, 0.35, 0.36)), "white": ((1, 1, 1), None), "ruled": ((0.99, 0.985, 0.96), (0.72, 0.8, 0.9))}


def parse(text: str) -> list[dict]:
    """Outline -> lines: {kind: heading|line|aside, level, text, bullet, key, new_block}."""
    out, new_block = [], True
    for raw in text.replace("\t", "  ").splitlines():
        if not raw.strip():
            new_block = True
            continue
        level = (len(raw) - len(raw.lstrip(" "))) // 2
        body = raw.strip()
        item = {"kind": "line", "level": level, "bullet": False, "key": False, "new_block": new_block}
        if body.startswith("@"):
            item.update(kind="diagram", text=body[1:].strip())
            out.append(item)
            new_block = False
            continue
        if body.startswith("$"):
            item.update(kind="math")
            item["text"] = body.lstrip("$ ").rstrip("$ ")
            out.append(item)
            new_block = False
            continue
        if body.startswith("#"):
            item.update(kind="heading", level=0, new_block=True)
            body = body.lstrip("# ")
        elif body.startswith(">"):
            item["kind"] = "aside"
            body = body.lstrip("> ")
        else:
            if body.startswith("!"):
                item["key"] = True
                body = body.lstrip("! ")
            if re.match(r"[-*•]\s", body):
                item["bullet"] = True
                body = body[2:].strip()
        item["text"] = re.sub(r"\s*->\s*", " → ", body)
        out.append(item)
        new_block = False
    return out


def _habits(profile: dict | None, paper: str) -> dict:
    p = profile or {}
    h = p.get("habits", {})
    names = [n for n in p.get("ink_colours", {}) if n in ON_DARK] or ["blue", "black"]
    shares = np.array([p.get("ink_colours", {}).get(n, 1.0) for n in names], float)
    if paper != "dark":  # white ink is the neutral ink on dark paper; on light paper that role is black
        names = ["black" if n == "white" else n for n in names]
    return {"colours": names, "shares": shares / (shares.sum() or 1.0), "indent": float(np.clip(h.get("indent_step_x_heights") or 4.5, 2.5, 7.0)),
            "lead_in": float(np.clip(h.get("indented_lines_with_lead_in_mark", 0.15) * 2.5, 0.2, 0.8)),
            "underline_headings": 0.7}


def write_notes(outline: str, lib: dict, out: str | Path, profile: dict | None = None, paper: str = "dark", seed: int = 0,
                page_size: tuple[float, float] | None = None, x_height: float | None = None, line_pitch: float | None = None,
                colours: list[str] | None = None, pen: float | None = None, diagrams: list[dict] | None = None) -> dict:
    """Lay the outline out as handwritten notes on one or more pages. Returns a short report."""
    lines = parse(outline)
    hab = _habits(profile, paper)
    # size the page, the writing and the line spacing as in the person's own notes, when the profile says
    p_ = profile or {}
    if x_height is None:
        x_height = float(np.clip(p_.get("x_height_units") or 5.2, 4.2, 9.0))
    if line_pitch is None:
        line_pitch = float(np.clip((p_.get("line_gap_x_heights") or 4.3) * x_height, 3.6 * x_height, 6.5 * x_height))
    if page_size is None:
        w = float(np.clip(p_.get("page_width_units") or 455.0, 380.0, 900.0))
        page_size = (w, round(w * 1.294, 1))
    if colours:
        hab["colours"], hab["shares"] = colours, np.full(len(colours), 1 / len(colours))
    table = ON_DARK if paper == "dark" else ON_LIGHT
    unknown = [c for c in hab["colours"] if c not in table]
    if unknown:
        raise SystemExit(f"unknown ink colour {', '.join(unknown)}: use {', '.join(table)}")
    bg, rule = PAPERS[paper]
    rng = np.random.default_rng(seed)
    human, _ = render.situation(lib, "note")
    # in their own notes people space words as loosely as they really do (typed text is tightened; notes are not)
    from .font import word_gap

    real_gap = lib["meta"].get("word_gap")
    if real_gap:
        human = human.with_settings({"word_spacing": float(np.clip(real_gap / word_gap(lib), 1.0, 1.7))})
    writer = render.Writer(lib, seed, human)
    writer.min_pen = 0.95
    # their pen, but by default never so thick that loops fill; pass pen= (a share of the x-height) to match it exactly
    pen_ratio = pen or float(np.clip(lib["meta"].get("pen_ratio", 0.2), 0.14, 0.24))
    W, H = page_size
    left, right, top, bottom = 26.0, 22.0, 46.0, 34.0
    doc = pymupdf.open()
    missing: set[str] = set()

    def new_page():
        page = doc.new_page(width=W, height=H)
        page.draw_rect(page.rect, fill=bg, color=None)
        if rule:
            y = top
            while y < H - 10:
                page.draw_line((0, y), (W, y), color=rule, width=0.4)
                y += line_pitch
        return page

    page, y = new_page(), top
    colour_name, last_end, last_colour = None, None, None

    def pick_colour():
        opts = [(n, s) for n, s in zip(hab["colours"], hab["shares"]) if n != colour_name] or list(zip(hab["colours"], hab["shares"]))
        ps = np.array([s for _, s in opts])
        return opts[int(rng.choice(len(opts), p=ps / ps.sum()))][0]

    # consecutive maths lines are one piece of working, aligned on the equals sign
    merged = []
    for item in lines:
        if item["kind"] in ("math", "diagram") and merged and merged[-1]["kind"] == item["kind"] and not item["new_block"]:
            merged[-1]["text"] += "\n" + item["text"]
        else:
            merged.append(dict(item))
    lines = merged
    prev_level = 0
    for item in lines:
        if colour_name is None or (item["new_block"] and item["kind"] != "aside"):
            colour_name = pick_colour()
            if y > top:
                y += line_pitch * (1.0 if item["kind"] == "heading" else 0.85)  # a gap between blocks of thought
                y = top + round((y - top) / line_pitch) * line_pitch if rule else y
        rgb = table[colour_name]
        if item["kind"] == "aside" and last_end is not None:
            # a remark beside the line above, joined to it by an arrow; if there is no room it goes underneath
            ax = last_end[0] + 30
            room = W - right - ax
            xh = x_height * 0.9
            other = table[pick_colour()]
            if room > 90:
                ink = writer.write(item["text"], xh, room, line_gap=line_pitch / xh, pen=pen_ratio)
                oy = last_end[1] - 2.2 * xh
                render.onto_pdf_page(page, ink, (ax, oy), other)
                p = Pen(lib, int(rng.integers(1 << 30)), xh)
                strokes = p.arrow([last_end[0] + 5, last_end[1] - xh * 0.5], [ax - 4, last_end[1] - xh * 0.5], bend=0.06)
                render.onto_pdf_page(page, render.Ink(strokes, [max(pen_ratio * xh, 0.95)] * len(strokes)), (0, 0), other)
                missing |= ink.missing
                continue
            item = {**item, "kind": "line", "level": prev_level + 1, "bullet": False, "key": False}
            rgb = other
        if item["kind"] == "diagram":
            x = left + max(item["level"], 1) * hab["indent"] * x_height
            own = _their_diagram(item["text"], diagrams, lib, int(rng.integers(1 << 30)), x_height, page.rect.width - x - 24)
            height = own["size"][1] if own else _diagram_height(item["text"])
            if y + height + line_pitch > H - bottom:
                page, y = new_page(), top
            others = [table[n] for n in hab["colours"] if table[n] != rgb] or [rgb]
            if own:  # one of their own drawings, drawn again
                missing |= set(own["missing"])
                for st in own["strokes"]:
                    c = tuple(st["color"])
                    c = rgb if (min(c) > 0.84 and paper != "dark") or (max(c) < 0.2 and paper == "dark") else c
                    render.onto_pdf_page(page, render.Ink([np.asarray(st["points"], float)], [max(st["width"], 0.6)]), (x, y + line_pitch * 0.6), c)
            else:
                missing |= _diagram(page, item["text"], x, y + line_pitch * 0.6, lib, writer, rng, rgb, others, x_height, pen_ratio)
            n = int(np.ceil((height + line_pitch * 0.8) / line_pitch))
            last_end = None
            y += n * line_pitch
            prev_level = item["level"]
            continue
        if item["kind"] == "math":
            from .mathwrite import write_maths

            x = left + item["level"] * hab["indent"] * x_height
            ink = write_maths(lib, item["text"], x_height * 1.1, int(rng.integers(1 << 30)), human, line_gap=line_pitch / (x_height * 1.1),
                              min_pen=0.95, pen=pen_ratio)
            n = max(1, int(np.ceil(ink.size[1] / line_pitch)))
            if y + n * line_pitch > H - bottom:
                page, y = new_page(), top
            render.onto_pdf_page(page, ink, (x, y + line_pitch * 0.45), rgb)
            missing |= ink.missing
            n = max(1, int(np.ceil((ink.size[1] + line_pitch * 0.45) / line_pitch)))
            last_end = (x + ink.size[0], y + n * line_pitch)
            y += n * line_pitch
            prev_level = item["level"]
            continue
        heading = item["kind"] == "heading"
        xh = x_height * (1.3 if heading else 1.0)
        x = left + item["level"] * hab["indent"] * x_height + rng.normal(0, 1.2)
        if "$" in item["text"] and item["text"].count("$") % 2 == 0:
            ink = _inline(lib, writer, item["text"], xh, pen_ratio, int(rng.integers(1 << 30)))  # words with maths in the line
        else:
            ink = writer.write(item["text"], xh, W - right - x, line_gap=line_pitch / xh, pen=pen_ratio * (1.1 if heading else 1.0))
        n = max(1, int(round((ink.size[1] / xh - 3.5) / (line_pitch / xh))) + 1)
        if y + n * line_pitch > H - bottom:
            page, y = new_page(), top
        base = y + line_pitch  # the first baseline of this item
        render.onto_pdf_page(page, ink, (x, base - 2.2 * xh), rgb)
        missing |= ink.missing
        p = Pen(lib, int(rng.integers(1 << 30)), xh)
        marks = []
        if item["level"] > 0 and item["level"] > prev_level and rng.random() < hab["lead_in"]:
            # a hooked arrow dropping from under the parent line into this one
            px = x - hab["indent"] * x_height * 0.55
            marks += p.arrow([px, base - line_pitch * 0.75], [x - 4, base - xh * 0.45], bend=0.28)
        elif item["bullet"]:
            marks += [p.path([[x - 9, base - xh * 0.5 + p.j(0.3)], [x - 3.5, base - xh * 0.52 + p.j(0.3)]])]
        if heading and rng.random() < hab["underline_headings"]:
            marks += p.underline((x, base - xh, x + ink.size[0], base))
        if item["key"]:
            marks += p.box((x - 1, base - xh * 1.9, x + ink.size[0] + 1, base + (n - 1) * line_pitch + xh * 0.55))
        if marks:
            render.onto_pdf_page(page, render.Ink(marks, [max(pen_ratio * x_height, 0.95)] * len(marks)), (0, 0), rgb)
        last_line_w = ink.size[0] if n == 1 else max((s[:, 0].max() for s in ink.strokes if s[:, 1].mean() > (n - 1) * line_pitch), default=0)
        last_end = (x + last_line_w, base + (n - 1) * line_pitch)
        y += n * line_pitch
        prev_level = item["level"]
    doc.save(str(out))
    return {"pages": len(doc), "lines": len(lines), "colours": hab["colours"], "missing": "".join(sorted(missing)),
            "page_size": page_size, "x_height": round(x_height, 2), "line_pitch": round(line_pitch, 2), "from_profile": bool(profile)}


def _their_diagram(text: str, catalogue: list[dict] | None, lib: dict, seed: int, x_height: float, room: float) -> dict | None:
    """`theirs 34 3=EUR "7=Euro area"`: drawing 34 from the person's own notes (see diagrams.py), redrawn at
    the size of this page's writing, with any numbered labels rewritten."""
    import shlex

    from . import diagrams as dg

    parts = shlex.split(text)
    if not parts or parts[0] != "theirs":
        return None
    if not catalogue:
        raise SystemExit("`@ theirs` needs diagrams.json beside the library: run `inkhabit diagrams <their notes> -o <out>` first")
    if len(parts) < 2 or not parts[1].isdigit() or int(parts[1]) >= len(catalogue):
        raise SystemExit(f"`@ theirs N`: N is a drawing number from 0 to {len(catalogue) - 1}")
    d = catalogue[int(parts[1])]
    labels = dict(p.split("=", 1) for p in parts[2:] if "=" in p)
    # their small letters are about 0.35 of a written line's height; make them this page's writing size
    natural = (d["box"][2] - d["box"][0]) * x_height / (0.35 * d["unit"])
    return dg.redraw(d, lib, labels, seed, 1.0, min(natural, room))


def _diagram_height(text: str) -> float:
    return 125.0 if any(ln.split()[:1] == ["axes"] for ln in text.splitlines()) else 34.0


def _diagram(page, text: str, x: float, y: float, lib: dict, writer, rng, rgb, others, xh: float, pen_ratio: float) -> set:
    """A small sketch in the same pen. Commands, one per line:
         axes <up label> <across label>     then any of:  curve up|down|hump|flat <label>,  point <label>
         flow first thing -> second thing -> third thing
    """
    missing: set[str] = set()
    pen_w = max(pen_ratio * xh, 0.95)
    p = Pen(lib, int(rng.integers(1 << 30)), xh)

    def label(s: str, at, colour, anchor="left"):
        ink = writer.write(s, xh * 0.9, pen=pen_ratio)
        missing.update(ink.missing)
        ox = at[0] - (ink.size[0] if anchor == "right" else ink.size[0] / 2 if anchor == "center" else 0)
        render.onto_pdf_page(page, ink, (ox, at[1] - 2.2 * xh * 0.9), colour)
        return ink.size[0]

    def strokes(st, colour):
        render.onto_pdf_page(page, render.Ink(st, [pen_w] * len(st)), (0, 0), colour)

    cmds = [ln.strip() for ln in text.splitlines() if ln.strip()]
    w, h = 150.0, 105.0
    ox, oy = x + 26, y + h  # the origin of the axes
    k = 0
    for cmd in cmds:
        name, _, rest = cmd.partition(" ")
        if name == "axes":
            up, _, across = rest.partition(" ")
            strokes(p.arrow([ox, oy], [ox + p.j(0.3), y - 4], bend=0.015) + p.arrow([ox, oy], [ox + w + 8, oy + p.j(0.3)], bend=0.015), rgb)
            label(up, (ox - 5, y + 6), rgb, "right")
            label(across, (ox + w + 12, oy + xh * 0.4), rgb)
        elif name == "curve":
            kind, _, lab = rest.partition(" ")
            t = np.linspace(0.06, 0.94, 12)
            shapes = {"up": 0.92 - 0.84 * t ** 1.15, "down": 0.08 + 0.84 * t ** 1.15, "flat": np.full_like(t, 0.5),
                      "hump": 1.0 - 3.6 * t * (1 - t), "valley": 0.08 + 3.6 * t * (1 - t) * 0 + 3.4 * (t - 0.5) ** 2}
            if kind not in shapes:
                raise SystemExit(f"unknown curve {kind!r}: use one of {', '.join(shapes)}")
            yy = shapes[kind]
            colour = others[k % len(others)]
            k += 1
            pts = np.c_[ox + t * w, y + yy * h]
            strokes([p.path(pts)], colour)
            if lab:
                label(lab, (pts[-1][0] + 4, pts[-1][1] + xh * 0.35), colour)
        elif name == "point":
            cx, cy = ox + 0.5 * w, y + 0.5 * h
            dash = lambda a, b: [p.path([a + (b - a) * u, a + (b - a) * (u + 0.09)]) for u in np.arange(0.0, 0.95, 0.18)]  # noqa: E731
            strokes(dash(np.array([cx, cy]), np.array([cx, oy])) + dash(np.array([cx, cy]), np.array([ox, cy])), rgb)
            strokes([p.path([[cx - 1.2, cy], [cx + 1.2, cy + 0.4]])], rgb)
            if rest:
                label(rest, (cx + 7, cy - xh * 0.9), rgb)
        elif name not in ("flow",):
            raise SystemExit(f"unknown sketch command {name!r}: use axes, curve, point, flow or theirs")
        elif name == "flow":
            cx = x
            parts = [s.strip() for s in rest.replace("→", "->").split("->") if s.strip()]
            for i, part in enumerate(parts):
                wpx = label(part, (cx + 5, y + 20), rgb)
                strokes(p.box((cx + 5, y + 20 - xh * 1.9, cx + 5 + wpx, y + 20 + xh * 0.6)), rgb)
                cx += wpx + 10
                if i < len(parts) - 1:
                    strokes(p.arrow([cx + 4, y + 20 - xh * 0.5], [cx + 30, y + 20 - xh * 0.5], bend=0.05), rgb)
                    cx += 34
    return missing


def _inline(lib: dict, writer, text: str, xh: float, pen_ratio: float, seed: int):
    """One line that mixes words and maths: "Standard form: $ax^2 + bx + c = 0$". Not wrapped."""
    from .mathwrite import Maths

    m = Maths(writer, lib, seed)
    ink = render.Ink()
    x = 0.0
    for k, piece in enumerate(text.split("$")):
        if not piece.strip():
            continue
        if k % 2:  # maths
            box = m.expression(piece)
            ink.strokes += [(np.asarray(s, float) + [x, 2.2]) * xh for s in box.strokes]
            x += box.w + 0.9
        else:
            part = writer.write(piece.strip(), xh)
            ink.strokes += [s + [x * xh, 0] for s in part.strokes]
            ink.missing |= part.missing
            x += part.size[0] / xh + 0.9
    ink.missing |= m.missing
    ink.widths = [max(pen_ratio * xh * writer.h.weight, writer.min_pen)] * len(ink.strokes)
    ink.size = ((x - 0.9) * xh, 3.5 * xh)
    return ink


def load_profile(path: str | Path | None) -> dict | None:
    return json.loads(Path(path).read_text()) if path and Path(path).exists() else None
