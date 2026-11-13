# SPDX-License-Identifier: AGPL-3.0-or-later
"""Use-case benchmark: made-up documents where handwriting is useful, filled or marked up, then scored.

    uv run python bench/usecases.py <library.json> [--out out/usecases]

For every case it reports:
  fits      share of entries the tool itself says fitted
  inside    share of the new ink that lies inside the places it was meant for
  on print  share of the new ink that landed on top of printed text or lines (lower is better)
  read      share of the intended words a recogniser reads back from the result
All names, addresses and text are invented.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

from inkhabit import annotate, forms, library
from inkhabit.ocr import recognize

DPI = 200
K = DPI / 72


def norm(t):
    return re.sub(r"[^a-z0-9]", "", t.lower())


# ---- the documents ---------------------------------------------------------------------------------
def application(path):
    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text((60, 70), "Community Garden: Plot Application", fontsize=17, fontname="hebo")
    E = []
    y = 110
    for label, text in (("Full name", "Alex Morgan"), ("Street address", "42 Example Street, Flat 3"), ("Town", "Springfield"),
                        ("Phone", "555 010 2026"), ("Email", "alex.morgan@example.com")):
        p.insert_text((60, y + 12), label, fontsize=10, fontname="helv")
        p.draw_line((170, y + 15), (540, y + 15), width=0.6)
        E.append({"page": 0, "rect": [170, y - 3, 540, y + 15], "text": text})
        y += 34
    p.insert_text((60, y + 13), "Date of birth", fontsize=10, fontname="helv")
    for k in range(8):
        p.draw_rect(pymupdf.Rect(170 + k * 20, y, 190 + k * 20, y + 20), width=0.6)
    E.append({"page": 0, "rect": [170, y, 330, y + 20], "text": "14032005", "comb": 8})
    p.insert_text((360, y + 13), "Postcode", fontsize=10, fontname="helv")
    for k in range(5):
        p.draw_rect(pymupdf.Rect(420 + k * 20, y, 440 + k * 20, y + 20), width=0.6)
    E.append({"page": 0, "rect": [420, y, 520, y + 20], "text": "60637", "comb": 5})
    y += 44
    p.insert_text((60, y + 10), "Plot size wanted (tick one)", fontsize=10, fontname="helv")
    for k, name in enumerate(("Small", "Medium", "Large")):
        p.draw_rect(pymupdf.Rect(230 + k * 100, y, 242 + k * 100, y + 12), width=0.7)
        p.insert_text((248 + k * 100, y + 10), name, fontsize=10, fontname="helv")
    E.append({"page": 0, "rect": [330, y, 342, y + 12], "check": True})
    y += 34
    p.insert_text((60, y + 10), "Have you gardened before?", fontsize=10, fontname="helv")
    for k, name in enumerate(("Yes", "No")):
        p.draw_rect(pymupdf.Rect(230 + k * 70, y, 242 + k * 70, y + 12), width=0.7)
        p.insert_text((248 + k * 70, y + 10), name, fontsize=10, fontname="helv")
    E.append({"page": 0, "rect": [230, y, 242, y + 12], "check": True, "mark": "cross"})
    y += 36
    p.insert_text((60, y + 10), "What would you like to grow, and why?", fontsize=10, fontname="helv")
    p.draw_rect(pymupdf.Rect(60, y + 18, 540, y + 130), width=0.6)
    E.append({"page": 0, "rect": [60, y + 18, 540, y + 130],
              "text": "Mostly tomatoes, beans and a few herbs. I would like somewhere quiet to work outside at the weekend, "
                      "and to share what I grow with the neighbours on my street."})
    y += 150
    p.insert_text((60, y + 10), "Initials", fontsize=10, fontname="helv")
    p.draw_rect(pymupdf.Rect(110, y - 2, 150, y + 16), width=0.6)
    E.append({"page": 0, "rect": [110, y - 2, 150, y + 16], "text": "AM"})
    p.insert_text((180, y + 10), "Emergency contact and relationship", fontsize=10, fontname="helv")
    p.draw_rect(pymupdf.Rect(360, y - 2, 540, y + 16), width=0.6)
    E.append({"page": 0, "rect": [360, y - 2, 540, y + 16], "text": "Jordan Morgan, brother, 555 010 7788"})
    doc.save(path)
    return ("fill", E)


def worksheet(path):
    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text((60, 70), "Science 8B   Worksheet 12: Separating mixtures", fontsize=14, fontname="hebo")
    qa = [("1. Name two ways of separating a mixture.", "Filtering and evaporating. A magnet also works for iron."),
          ("2. What is left when salt water is heated until dry?", "White crystals of salt are left in the dish."),
          ("3. Why can sand be filtered out but salt cannot?", "Sand does not dissolve, so the paper catches it. Salt dissolves and passes through."),
          ("4. Which change of state turns a liquid into a gas?", "Boiling or evaporating.")]
    E, y = [], 110
    for q, ans in qa:
        p.insert_text((60, y), q, fontsize=11, fontname="helv")
        for k in range(3):
            p.draw_line((60, y + 26 + k * 24), (540, y + 26 + k * 24), width=0.5, color=(0.55, 0.55, 0.55))
        E.append({"page": 0, "rect": [60, y + 2, 540, y + 74], "text": ans, "line_pitch": 24})
        y += 120
    doc.save(path)
    return ("fill", E)


def card(path):
    doc = pymupdf.open()
    p = doc.new_page(width=420, height=595)
    p.draw_rect(pymupdf.Rect(18, 18, 402, 577), color=(0.75, 0.6, 0.3), width=2)
    p.insert_text((118, 90), "Happy Birthday", fontsize=26, fontname="tibi", color=(0.5, 0.25, 0.1))
    E = [{"page": 0, "rect": [60, 150, 360, 190], "text": "Dear Sam,", "for": "letter", "x_height": 9},
         {"page": 0, "rect": [60, 200, 360, 430], "for": "letter", "x_height": 8,
          "text": "Hope you have a brilliant day and that the cake is as good as last year. Thanks for all the lifts, "
                  "the late night revision and the terrible jokes. See you on Saturday!"},
         {"page": 0, "rect": [200, 450, 360, 490], "text": "From all of us", "for": "letter", "x_height": 9, "align": "right"}]
    doc.save(path)
    return ("fill", E)


def envelope(path):
    doc = pymupdf.open()
    p = doc.new_page(width=650, height=320)
    p.draw_rect(pymupdf.Rect(560, 24, 620, 96), width=0.8, dashes="[3 3]")
    p.insert_text((572, 64), "STAMP", fontsize=9, fontname="helv", color=(0.5, 0.5, 0.5))
    E = [{"page": 0, "rect": [220, 120, 560, 280], "text": "Sam Rivera\n18 Orchard Lane\nSpringfield\nIL 62704", "for": "official", "x_height": 10},
         {"page": 0, "rect": [24, 20, 260, 80], "text": "A Morgan, 42 Example Street\nSpringfield IL 60637", "for": "official", "x_height": 6}]
    doc.save(path)
    return ("fill", E)


def diagram(path):
    doc = pymupdf.open()
    p = doc.new_page(width=595, height=420)
    p.insert_text((60, 50), "Figure 3. The water cycle (label the diagram)", fontsize=12, fontname="hebo")
    p.draw_oval(pymupdf.Rect(80, 250, 300, 330), color=(0.2, 0.4, 0.8), width=1.2)
    p.draw_circle((470, 110), 30, color=(0.9, 0.6, 0.1), width=1.2)
    p.draw_bezier((330, 140), (350, 100), (410, 100), (430, 140), color=(0.4, 0.4, 0.4), width=1.2)
    p.draw_line((330, 140), (430, 140), color=(0.4, 0.4, 0.4), width=1.2)
    spots = [([110, 345, 270, 367], "sea"), ([420, 150, 540, 172], "sun"), ([300, 60, 460, 82], "cloud forms"),
             ([180, 170, 320, 192], "evaporation"), ([440, 250, 570, 272], "rain falls")]
    E = []
    for r, text in spots:
        p.draw_line((r[0], r[3]), (r[2], r[3]), width=0.5, color=(0.5, 0.5, 0.5))
        E.append({"page": 0, "rect": r, "text": text, "for": "label", "align": "center"})
    doc.save(path)
    return ("fill", E)


def notebook(path):
    doc = pymupdf.open()
    p = doc.new_page()
    for k in range(27):
        p.draw_line((0, 100 + k * 24), (595, 100 + k * 24), width=0.5, color=(0.6, 0.72, 0.85))
    p.draw_line((70, 0), (70, 842), width=0.6, color=(0.9, 0.5, 0.5))
    E = [{"page": 0, "rect": [80, 76, 560, 100], "text": "Mixtures and solutions", "for": "heading", "x_height": 8.5},
         {"page": 0, "rect": [80, 124, 560, 400], "line_pitch": 24, "for": "note",
          "text": "A mixture has two or more substances that are not joined chemically. A solution forms when a solute dissolves "
                  "in a solvent. Salt in water is clear and cannot be filtered; sand in water is cloudy and can be filtered out. "
                  "To separate them we can filter, evaporate or distil."}]
    doc.save(path)
    return ("fill", E)


def todo(path):
    doc = pymupdf.open()
    p = doc.new_page(width=300, height=300)
    p.draw_rect(p.rect, fill=(1.0, 0.95, 0.55), color=None)
    items = ["call the dentist", "buy milk and bread", "return library books", "fix bike chain"]
    E = [{"page": 0, "rect": [24, 18, 276, 52], "text": "To do today", "for": "heading", "x_height": 8}]
    for k, it in enumerate(items):
        y = 70 + k * 48
        p.draw_rect(pymupdf.Rect(26, y + 8, 40, y + 22), width=0.8, color=(0.35, 0.35, 0.35))
        E.append({"page": 0, "rect": [50, y, 280, y + 30], "text": it, "for": "scribble", "x_height": 7})
        if k < 2:
            E.append({"page": 0, "rect": [26, y + 8, 40, y + 22], "check": True})
    doc.save(path)
    return ("fill", E)


def handout(path):
    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text((60, 60), "Reading 4: Supply, demand and price", fontsize=14, fontname="hebo")
    a = ("A market is not a place so much as a habit. People return to the same stall because it was there yesterday, and the "
         "seller sets a price that expects them to return. When the price rises a little, most buyers grumble and pay.\n\n"
         "This is what economists mean when they say that demand is elastic. A rubber band stretches easily at first and "
         "then, quite suddenly, refuses.")
    b = ("Supply behaves differently. A baker cannot conjure a second oven overnight, so in the short run the quantity on offer "
         "barely moves.\n\nGive the baker a year, however, and ovens appear. Time is the hidden ingredient in every price. "
         "Neither side acts alone: a price is an agreement.")
    p.insert_textbox(pymupdf.Rect(60, 85, 285, 500), a, fontsize=10, fontname="tiro", lineheight=1.4)
    p.insert_textbox(pymupdf.Rect(310, 85, 535, 500), b, fontsize=10, fontname="tiro", lineheight=1.4)
    items = [{"type": "underline", "find": "A market is not a place so much as a habit"}, {"type": "circle", "find": "elastic"},
             {"type": "highlight", "find": "Time is the hidden ingredient in every price"},
             {"type": "bracket", "find": "A baker cannot conjure a second oven overnight", "side": "left"},
             {"type": "note", "find": "elastic", "side": "left", "text": "key term", "arrow": True},
             {"type": "note", "find": "A baker cannot conjure", "side": "left", "text": "short run"},
             {"type": "note", "find": "Time is the hidden ingredient", "side": "right", "text": "use in essay", "arrow": True},
             {"type": "star", "find": "Neither side acts alone"},
             {"type": "note", "side": "bottom", "text": "Compare with the Velmar harbour case from last week", "for": "note"}]
    doc.save(path)
    return ("annotate", items)


def exam(path):
    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text((60, 70), "Mathematics: Quiz 3   (show your working)", fontsize=14, fontname="hebo")
    qs = [("1. Solve  x squared + 3x - 4 = 0.", r"x = \frac{-3 \pm \sqrt{9 + 16}}{2} \\ = \frac{-3 \pm 5}{2} \\ x = 1 \text{or} x = -4"),
          ("2. Differentiate  y = 3x squared + 2x.", r"\frac{dy}{dx} = 6x + 2"),
          ("3. Evaluate the integral of x squared from 0 to 3.", r"\int_0^3 x^2 dx = \frac{27}{3} \\ = 9")]
    E, y = [], 110
    for q, ans in qs:
        p.insert_text((60, y), q, fontsize=11, fontname="helv")
        p.draw_rect(pymupdf.Rect(60, y + 10, 540, y + 150), width=0.6)
        E.append({"page": 0, "rect": [66, y + 14, 536, y + 148], "text": ans, "math": True, "x_height": 8})
        y += 190
    doc.save(path)
    return ("fill", E)


def table(path):
    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text((60, 70), "Club sign-up sheet", fontsize=15, fontname="hebo")
    cols, rows, x0, y0, rh = [60, 220, 330, 440, 540], 5, 60, 100, 30
    for r in range(rows + 2):  # drawn as separate lines, the way tables usually are
        p.draw_line((x0, y0 + r * rh), (cols[-1], y0 + r * rh), width=0.6)
    for c in cols:
        p.draw_line((c, y0), (c, y0 + (rows + 1) * rh), width=0.6)
    for c, head in zip(cols, ("Name", "Year", "Phone", "Day")):
        p.insert_text((c + 6, y0 + 20), head, fontsize=11, fontname="hebo")
    data = [("Alex Morgan", "Second", "555 0102", "Tuesday"), ("Sam Rivera", "First", "555 0144", "Friday"),
            ("Jo Castell", "Third", "555 0170", "Monday"), ("Robin Vale", "First", "555 0199", "Tuesday")]
    E = []
    for r, row in enumerate(data):
        for c, text in enumerate(row):
            E.append({"page": 0, "rect": [cols[c] + 2, y0 + (r + 1) * rh + 2, cols[c + 1] - 2, y0 + (r + 2) * rh - 2], "text": text, "valign": "middle"})
    doc.save(path)
    return ("fill", E)


def planner(path):
    doc = pymupdf.open()
    p = doc.new_page(width=842, height=420)
    p.insert_text((40, 50), "Week plan", fontsize=15, fontname="hebo")
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    plans = ["Lab report due\nlibrary 4 to 6", "Gym\ncall Grandma", "Essay draft\nbuy printer ink", "Group meeting 2:30", "Hand in essay\nfilm night"]
    E = []
    for k, (d, plan) in enumerate(zip(days, plans)):
        x = 40 + k * 154
        p.draw_rect(pymupdf.Rect(x, 80, x + 146, 360), width=0.7)
        p.insert_text((x + 8, 100), d, fontsize=12, fontname="hebo")
        E.append({"page": 0, "rect": [x + 6, 112, x + 142, 350], "text": plan, "for": "note", "x_height": 6.5})
    doc.save(path)
    return ("fill", E)


def _photo(make_blank):
    """The same document as a tilted, blurred photograph (a picture, no text layer)."""
    def make(path):
        import cv2

        from inkhabit.forms import open_doc

        tmp = Path(str(path) + ".src.pdf")
        kind, entries = make_blank(tmp)
        with pymupdf.open(tmp) as d:
            px = d[0].get_pixmap(dpi=140)
            im = np.frombuffer(px.samples, np.uint8).reshape(px.height, px.width, 3).astype(np.float32)
        h, w = im.shape[:2]
        im = cv2.warpAffine(im, cv2.getRotationMatrix2D((w / 2, h / 2), 0.6, 1.0), (w, h), borderValue=(250, 250, 250))
        im = cv2.GaussianBlur(im, (0, 0), 0.7) + np.random.default_rng(1).normal(0, 4, im.shape) + np.linspace(-8, 6, w)[None, :, None]
        jpg = Path(str(path) + ".jpg")
        cv2.imwrite(str(jpg), np.clip(im, 0, 255).astype(np.uint8)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 70])
        with open_doc(jpg) as d:
            d.save(path)
        tmp.unlink()
        return kind, entries
    return make


CASES = {"application form": application, "worksheet answers": worksheet, "greeting card": card, "envelope": envelope,
         "diagram labels": diagram, "ruled notebook page": notebook, "to-do note": todo, "annotated handout": handout,
         "maths quiz answers": exam, "table (sign-up sheet)": table, "week planner": planner,
         "photographed form": _photo(application), "photographed handout": _photo(handout)}


# ---- scoring ---------------------------------------------------------------------------------------
def pix(pdf):
    with pymupdf.open(pdf) as d:
        px = d[0].get_pixmap(dpi=DPI, colorspace=pymupdf.csRGB)
        return np.frombuffer(px.samples, np.uint8).reshape(px.height, px.width, 3).astype(np.int16)


def score(before, after, entries, kind):
    a, b = pix(before), pix(after)
    new = np.abs(a - b).max(2) > 40                         # the ink that was added
    paper = np.median(a.reshape(-1, 3), 0)
    printed = np.abs(a - paper).max(2) > 60                 # what was already printed there
    import cv2

    printed = cv2.dilate(printed.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    row = {"entries": len(entries), "ink_px": int(new.sum())}
    row["on_print"] = float((new & printed).sum() / max(new.sum(), 1))
    if kind != "fill":
        return row
    allowed = np.zeros_like(new)
    found = total = 0
    for e in entries:
        x0, y0, x1, y1 = e["rect"]
        h = y1 - y0
        pad_y = 0.35 * h if not e.get("check") else h * 1.3
        X0, Y0, X1, Y1 = int((x0 - 3) * K), int((y0 - pad_y) * K), int((x1 + (3 if not e.get("check") else h * 1.4)) * K), int((y1 + 0.3 * h) * K)
        allowed[max(Y0, 0):Y1, max(X0, 0):X1] = True
        if e.get("check"):
            continue
        crop = np.where(new[max(Y0, 0):Y1, max(X0, 0):X1], 0, 255).astype(np.uint8)
        if crop.shape[0] < 60:                                 # the recogniser needs some height
            crop = np.asarray(Image.fromarray(crop).resize((crop.shape[1] * 2, crop.shape[0] * 2)))
        crop = np.pad(crop, 30, constant_values=255)
        got = " ".join(w.text for w in recognize(Image.fromarray(crop).convert("RGB")))
        if e.get("math"):
            continue  # maths is not something the recogniser can read back; judged by eye
        if e.get("comb"):
            total += 1
            found += SequenceMatcher(None, norm(got), norm(e["text"])).ratio()
        else:
            want = [norm(w) for w in e["text"].split() if norm(w)]
            have = norm(got)
            got_words = {norm(w) for w in got.split()}
            total += len(want)
            found += sum(1 for w in want if w in got_words or (len(w) > 3 and w in have))
    row["inside"] = float((new & allowed).sum() / max(new.sum(), 1))
    if total:
        row["read"] = found / total
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("library")
    ap.add_argument("--out", default="out/usecases")
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lib = library.load(a.library)
    rows = {}
    print(f"{'use case':22s} {'entries':>7s} {'fits':>6s} {'inside':>7s} {'on print':>9s} {'read':>6s}")
    for name, make in CASES.items():
        stem = name.replace(" ", "-")
        blank, done = out / f"{stem}.blank.pdf", out / f"{stem}.pdf"
        kind, entries = make(blank)
        if kind == "fill":
            rep = forms.fill(blank, entries, lib, done, seed=a.seed)
            fits = np.mean([r["status"] == "ok" for r in rep])
        else:
            rep = annotate.annotate(blank, entries, lib, done, seed=a.seed)
            fits = np.mean([r["status"] == "ok" for r in rep])
            # score only the pen: highlighting, underlining and circling are meant to touch the print
            pen_only = out / f"{stem}.notes-only.pdf"
            annotate.annotate(blank, [e for e in entries if e["type"] in ("note", "star", "tick", "bracket")], lib, pen_only, seed=a.seed)
            row = {"fits": float(fits), **score(blank, pen_only, entries, kind)}
            rows[name] = row
        if kind == "fill":
            row = {"fits": float(fits), **score(blank, done, entries, kind)}
        rows[name] = row
        with pymupdf.open(done) as d:
            d[0].get_pixmap(dpi=110).save(out / f"{stem}.png")
        (out / f"{stem}.entries.json").write_text(json.dumps(entries, indent=1))
        print(f"{name:22s} {row['entries']:7d} {row['fits']:6.0%} " + (f"{row['inside']:7.0%}" if "inside" in row else "      -") +
              f" {row['on_print']:9.1%} " + (f"{row['read']:6.0%}" if "read" in row else "     -"), flush=True)
    (out / "results.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
