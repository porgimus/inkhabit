# SPDX-License-Identifier: AGPL-3.0-or-later
"""Invented writers and pages of their notes, for the benchmark (see bench/run.py).

Nothing here comes from any real person's writing: letter shapes start from a system font's centre
line and are bent by a fixed, person-specific distortion, so the "person" has a consistent hand that
the pipeline has never seen. Outputs go to data/fake/ (the bundle an agent is given) and
data/fake_truth/ (the answers, kept away from the agent).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize

from inkhabit.scan import _smooth, _trace

FONTS = {"mara": ("/System/Library/Fonts/Noteworthy.ttc", 0), "other": ("/System/Library/Fonts/Supplemental/Chalkboard.ttc", 0),
         "comic": ("/System/Library/Fonts/Supplemental/Comic Sans MS.ttf", 0), "marker": ("/System/Library/Fonts/MarkerFelt.ttc", 0)}
_cache: dict = {}


def skeleton(ch: str, who: str):
    """Centre-line strokes of a character in x-height units (baseline 0, y down)."""
    if (ch, who) in _cache:
        return _cache[(ch, who)]
    path, idx = FONTS[who]
    font = ImageFont.truetype(path, 300, index=idx)

    def mask(c):
        im = Image.new("L", (700, 700), 0)
        ImageDraw.Draw(im).text((200, 150), c, fill=255, font=font)
        return np.asarray(im) > 128

    x = mask("x")
    ys = np.flatnonzero(x.any(1))
    base, xh = ys.max(), ys.max() - ys.min()
    m = mask(ch)
    if not m.any():
        _cache[(ch, who)] = None
        return None
    x0 = np.flatnonzero(m.any(0)).min()
    out = []
    for line in _trace(skeletonize(m)):
        if len(line) >= 2 and np.linalg.norm(np.diff(line, axis=0), axis=1).sum() < 0.12 * xh and out:
            continue
        out.append((_smooth(line) - [x0, base]) / xh)
    _cache[(ch, who)] = out or None
    return _cache[(ch, who)]


class Person:
    """A consistent hand: global habits plus, for every character, two fixed ways of forming it."""

    def __init__(self, who: str, seed: int, slant: float, squeeze: float, tall: float, tail: float, wild: float, join: float):
        self.who, self.seed, self.slant, self.squeeze, self.tall, self.tail, self.wild, self.join = who, seed, slant, squeeze, tall, tail, wild, join

    def form(self, ch: str, which: int):
        """The person's own shape for a character: the neutral skeleton bent by a distortion fixed for (person, char, form)."""
        sk = skeleton(ch, self.who)
        if sk is None:
            return None
        r = np.random.default_rng(abs(hash((self.seed, ch, which))) % (2**32))
        ax, ay = r.uniform(0.05, 0.13, 2) * (1 if which == 0 else 1.5)
        fx, fy, px, py = r.uniform(1.2, 3.0), r.uniform(1.2, 3.0), r.uniform(0, 6.28), r.uniform(0, 6.28)
        shear_extra = r.normal(0, 0.06)
        out = []
        for s in sk:
            p = s.copy()
            p[:, 1] = np.where(p[:, 1] < -1.05, p[:, 1] * self.tall, p[:, 1])
            p[:, 1] = np.where(p[:, 1] > 0.08, p[:, 1] * self.tail, p[:, 1])
            dx = ax * np.sin(p[:, 1] * fx + px) + 0.5 * ax * np.sin(p[:, 0] * fy + py)
            dy = ay * np.sin(p[:, 0] * fy + py)
            p[:, 0] = (p[:, 0] + dx) * self.squeeze
            p[:, 1] = p[:, 1] + dy
            p[:, 0] -= p[:, 1] * (math.tan(math.radians(self.slant)) + shear_extra)
            out.append(p)
        x0 = min(q[:, 0].min() for q in out)
        return [q - [x0, 0] for q in out]

    def word(self, text: str, rng) -> tuple[list[np.ndarray], float]:
        strokes, x, prev_end = [], 0.0, None
        for ch in text:
            f = self.form(ch, int(rng.random() < 0.35))
            if f is None:
                x += 0.5
                prev_end = None
                continue
            s = 1 + rng.normal(0, self.wild)
            a = math.radians(rng.normal(0, 3.5 * self.wild / 0.08))
            rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            dy = rng.normal(0, self.wild * 0.9)
            placed = []
            for q in f:
                t = np.linspace(0, 1, len(q))
                wob = np.c_[0.02 * np.sin(t * rng.uniform(4, 9) + rng.uniform(0, 6)), 0.02 * np.sin(t * rng.uniform(4, 9) + rng.uniform(0, 6))]
                placed.append((q * s + wob) @ rot.T + [x, dy])
            if prev_end is not None and ch.isalpha() and rng.random() < self.join and len(placed[0]) > 1:
                d = placed[0][0] - prev_end
                if 0 < d[0] < 0.8 and abs(d[1]) < 0.6:  # a joining stroke from the last letter into this one
                    mid = (prev_end + placed[0][0]) / 2 + [0, 0.12]
                    t = np.linspace(0, 1, 7)[:, None]
                    strokes.append((1 - t) ** 2 * prev_end + 2 * (1 - t) * t * mid + t**2 * placed[0][0])
            strokes += placed
            w = max(q[:, 0].max() for q in placed) - x
            prev_end = placed[-1][-1] if ch.isalpha() and len(placed[-1]) > 1 else None
            x += w + max(0.04, rng.normal(0.16, 0.07))
        return strokes, x


NOTES = {
    "history": [
        "# The Harbour Town of Velmar", "Founded around 1340 by salt traders from the north.",
        "- first quay built in 1362, rebuilt twice after storms", "- population grew from 800 to about 4,500 by 1500",
        "The guild of weavers controlled most trade until the 1600s.", "Key export: dyed wool cloth, also dried fish and barrels.",
        "> why did the guild lose power?", "1. cheaper cloth arrived by sea", "2. the plague year of 1631 killed a third of members",
        "3. a new charter gave equal rights to outside merchants", "# Life in the Old Quarter",
        "Houses were narrow, four floors high, with a workshop below.", "Water came from six public wells; the largest was by the market.",
        "Children worked from age ten, usually as apprentices.", "Bread, onions, and salted herring made up the daily diet.",
        "- quiz next Friday on chapters 3 and 4", "- bring the yellow workbook and a ruler",
        "Remember: the fire of 1702 destroyed just over half the town.", "Rebuilding took twenty years and used brick, not timber.",
        "Exam tip: explain causes before effects, and give exact dates.",
    ],
    "science": [
        "# Mixtures and Solutions", "A mixture has two or more substances that are not joined chemically.",
        "A solution forms when a solute dissolves in a solvent.", "- salt in water: clear, cannot be filtered",
        "- sand in water: cloudy, can be filtered out", "Ways to separate: filtering, evaporating, distilling, magnets.",
        "Experiment 5: heat 50 ml of salt water until dry.", "Result: 3.2 g of white crystals left in the dish.",
        "> check the mass again, maybe 3.4 g?", "Zinc and copper do not dissolve; sugar does very quickly.",
        "# Changes of State", "Melting, freezing, boiling and condensing are physical changes.",
        "Water boils at 100 degrees and freezes at zero.", "Particles move faster when heated and spread further apart.",
        "- solid: fixed shape, particles vibrate", "- liquid: takes the shape of the jar",
        "- gas: fills the whole box, squeezes easily", "Question 7: why does a puddle vanish on a warm day?",
        "Because the water evaporates; it joins the air as vapour.", "Homework: draw the water cycle and label six parts.",
    ],
    "reading": [
        "# Book Report: The Glass Orchard", "A quiet novel about two sisters who inherit a failing farm.",
        "Main idea: looking after something is harder than owning it.", "- Junia, the elder, wants to sell and move to the city",
        "- Brix, the younger, plants forty apple trees in secret", "The first winter kills half of them; she replants in spring.",
        "> best chapter is eleven, the frost night", "Style: short sentences, very little dialogue, lots of weather.",
        "I liked how the ending does not explain everything.", "Favourite line: we kept what we could carry and called it enough.",
        "# Vocabulary To Learn", "exquisite, quarrel, vivid, juxtapose, zealous, flux", "buoyant, wharf, jinx, pivot, quench, hazard",
        "Use each word in a sentence before Monday.", "Mixed up affect and effect again; check the rule.",
        "- affect is usually the verb, effect the noun", "- practise twenty spellings every evening",
        "Zoe quizzed me and I got just six wrong.", "Extra reading: pages 140 to 195 of the grey anthology.",
        "Ask about joining the debate club; it meets at 3:30.",
    ],
    "maths": [
        "# Fractions and Percentages", "To find 15% of 240, work out 10% then add half of it.", "So 24 plus 12 gives 36.",
        "- a quarter is 25%, three quarters is 75%", "- one fifth is 20%, one eighth is 12.5%",
        "Converting: divide the top by the bottom, then multiply by 100.", "Example: 7 over 8 equals 0.875, which is 87.5%.",
        "> always check by estimating first", "Mistake last week: forgot to simplify 18 over 24 to 3 over 4.",
        "Zero point six is the same as six tenths, or 60%.",
    ],
    "plans": [
        "# Trip to the Coast", "Leave Saturday at 7:45, back by Sunday evening.", "Jack is driving; Zoe and Quinn bring the tent.",
        "- pack: jacket, boots, torch, extra socks, map", "- food: bread, cheese, six apples, a box of tea",
        "Budget is about 60 each, including the ferry.", "Call the campsite before Thursday to fix the booking.",
        "> ask Vera if we can borrow her stove", "Weather looks mixed: sun early, wind and rain by five.",
        "Walk the cliff path if dry, otherwise the museum.", "# Things To Do This Week", "1. finish the history essay by Wednesday",
        "2. email Mr Okafor about the lab report", "3. buy a birthday card for Grandpa", "4. fix the bike chain and pump both tyres",
        "5. return library books (two are overdue)", "Kept forgetting the keys, so hang them by the door.",
        "Next month: save 25 a week for the new headphones.", "Quote I liked: every expert was once a beginner.",
        "Note to self: just start, the first page is the hardest.",
    ],
}


def page_image(person: Person, lines: list[str], rng, size=(1654, 2339), ruled=True, second: Person | None = None, xh: float = 30.0,
               boxes: list | None = None, vec: list | None = None):
    """Draw one page of notes. Returns (BGR image, list of what was actually written per line)."""
    W, H = size
    paper = np.full((H, W, 3), (236, 240, 243), np.uint8)
    if ruled:
        for y in range(260, H - 120, 84):
            cv2.line(paper, (0, y), (W, y), (214, 196, 170), 2)
        cv2.line(paper, (170, 0), (170, H), (150, 150, 222), 2)
    ink_layer = np.zeros((H, W), np.uint8)
    y = 260.0
    truth = []
    colour = (92, 44, 30)
    for raw in lines:
        kind = raw[:1]
        text = raw.lstrip("#->0123456789. ") if kind in "#->" else raw
        heading, bullet, aside = kind == "#", kind == "-", kind == ">"
        scale = 1.3 if heading else 0.85 if aside else 1.0
        x = 205.0 + (70 if bullet else 0) + (420 if aside else 0) + rng.normal(0, 9)
        if heading:
            y += 40
        slope = math.tan(math.radians(rng.normal(0.4, 0.5)))
        x_start, written = x, []
        if bullet:
            cv2.line(ink_layer, (int(x - 34), int(y - 12)), (int(x - 14), int(y - 11)), 255, 3, cv2.LINE_AA)
        if raw[:2] in ("1.", "2.", "3.", "4.", "5."):
            text = raw
        words = text.split()
        k = 0
        while k < len(words):
            wd = words[k]
            strokes, wlen = person.word(wd, rng)
            wpx = wlen * xh * scale
            if x + wpx > W - 110:  # wrap, carrying on underneath
                y += 84
                x = x_start + rng.normal(0, 8)
            base = y - 6 + slope * (x - x_start) + rng.normal(0, 2.0)
            crossed = rng.random() < 0.035 and len(wd) > 3  # a slip: written, struck out, written again
            for s in strokes:
                q = np.round((s * xh * scale + [x, base]) * 16).astype(np.int32)
                thick = 3 if rng.random() > 0.25 else 4
                if vec is not None and not crossed:
                    vec.append((s * xh * scale + [x, base], thick))
                if len(q) > 1:
                    cv2.polylines(ink_layer, [q], False, 255, thick, cv2.LINE_AA, 4)
                else:
                    cv2.circle(ink_layer, (int(q[0][0]), int(q[0][1])), 3 * 16, 255, -1, cv2.LINE_AA, 4)
            if crossed:
                cv2.line(ink_layer, (int(x - 4), int(base - xh * 0.45)), (int(x + wpx + 4), int(base - xh * 0.55)), 255, 3, cv2.LINE_AA)
                cv2.line(ink_layer, (int(x - 2), int(base - xh * 0.25)), (int(x + wpx + 6), int(base - xh * 0.3)), 255, 3, cv2.LINE_AA)
                x += wpx + xh * rng.uniform(0.5, 0.9)
                continue  # the same word is written again next
            written.append(wd)
            if boxes is not None:
                boxes.append((wd, x, base - 2.0 * xh * scale, x + wpx, base + 0.9 * xh * scale))
            x += wpx + xh * scale * max(0.35, rng.normal(0.75, 0.2))
            k += 1
        if heading:  # a wavy underline
            xs = np.linspace(x_start - 6, x + 4, 30)
            pts = np.c_[xs, y + 14 + 3 * np.sin(xs / 37.0) + rng.normal(0, 0.8, 30)]
            cv2.polylines(ink_layer, [np.round(pts * 16).astype(np.int32)], False, 255, 3, cv2.LINE_AA, 4)
        if aside:  # an arrow pointing back at the main column
            cv2.arrowedLine(ink_layer, (int(x_start - 20), int(y - 14)), (int(x_start - 150), int(y - 30)), 255, 3, cv2.LINE_AA, tipLength=0.2)
        truth.append(" ".join(written))
        y += 84 * (1.0 if not heading else 1.15)
        if y > H - 190:
            break
    # a doodle in the corner, as people do
    cx, cy = W - 230, H - 260
    t = np.linspace(0, 5 * math.pi, 160)
    spiral = np.c_[cx + (8 + 9 * t) * np.cos(t), cy + (8 + 9 * t) * np.sin(t) * 0.8]
    cv2.polylines(ink_layer, [np.round(spiral * 16).astype(np.int32)], False, 255, 3, cv2.LINE_AA, 4)
    if second is not None:  # a few words in someone else's hand, in the margin
        s2, _ = second.word("see", rng)
        for word_, off in (("see", 0), ("page", 130), ("12", 290)):
            st, _ = second.word(word_, rng)
            for s in st:
                q = np.round((s * 34 + [W - 560 + off, 190]) * 16).astype(np.int32)
                if len(q) > 1:
                    cv2.polylines(ink_layer, [q], False, 255, 4, cv2.LINE_AA, 4)
    a = (ink_layer.astype(np.float32) / 255)[..., None] * 0.92
    img = (paper * (1 - a) + np.array(colour, np.float32) * a).astype(np.uint8)
    return img, truth


def degrade(img, rng, kind="scan"):
    """What a scanner or a phone does to a page."""
    H, W = img.shape[:2]
    out = img.astype(np.float32)
    if kind == "scan":
        M = cv2.getRotationMatrix2D((W / 2, H / 2), rng.normal(0, 1.1), 1.0)
        out = cv2.warpAffine(out, M, (W, H), borderValue=(250, 250, 250))
        out += np.linspace(-10, 8, W)[None, :, None] + rng.normal(0, 4.5, out.shape)
        out = cv2.GaussianBlur(out, (0, 0), 0.9)
    else:  # a phone photo: perspective, a shadow across it, softer focus, a table behind the page
        m = 90
        src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        dst = np.float32([[m + rng.uniform(0, 70), m + rng.uniform(0, 50)], [W - m - rng.uniform(0, 90), m + rng.uniform(20, 90)],
                          [W - m + rng.uniform(-30, 20), H - m - rng.uniform(0, 60)], [m - rng.uniform(0, 40), H - m + rng.uniform(-40, 10)]])
        out = cv2.warpPerspective(out, cv2.getPerspectiveTransform(src, dst), (W, H), borderValue=(70, 88, 110))
        yy, xx = np.mgrid[0:H, 0:W]
        shadow = 1 - 0.33 / (1 + np.exp(-((xx * 0.7 + yy * 0.5) - (W * 0.55 + rng.uniform(-200, 200))) / 130.0))
        out *= shadow[..., None]
        out = cv2.GaussianBlur(out, (0, 0), 1.3) + rng.normal(0, 6, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def jpeg(img, q):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes()


def main(root="data"):
    rng = np.random.default_rng(2027)
    bundle, truth_dir = Path(root) / "fake", Path(root) / "fake_truth"
    for d in (bundle, truth_dir):
        d.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            f.unlink()
    mara = Person("mara", 11, slant=15, squeeze=0.8, tall=1.22, tail=1.4, wild=0.075, join=0.45)
    other = Person("other", 5, slant=-4, squeeze=1.1, tall=0.95, tail=0.9, wild=0.04, join=0.05)
    truth = {}
    lines_all = [(topic, ln) for topic, lns in NOTES.items() for ln in lns]
    pages, cur = [], []
    for topic, ln in lines_all:
        cur.append(ln)
        if len(cur) >= 10:
            pages.append(cur)
            cur = []
    if cur:
        pages.append(cur)
    # 1. a scanned PDF of the first four pages (image only, no text layer)
    doc = pymupdf.open()
    for i, lines in enumerate(pages[:4]):
        img, tr = page_image(mara, lines, rng, second=other if i == 1 else None)
        truth[f"school notes scan.pdf#page{i + 1}"] = tr
        pg = doc.new_page(width=595, height=842)
        pg.insert_image(pg.rect, stream=jpeg(degrade(img, rng, "scan"), 62))
    doc.save(bundle / "school notes scan.pdf")
    # 2. phone photos of two more pages; one is stored sideways with only a metadata flag saying so
    for i, lines in enumerate(pages[4:6]):
        img, tr = page_image(mara, lines, rng)
        name = f"IMG_{4031 + i}.jpg"
        truth[name] = tr
        photo = degrade(img, rng, "photo")
        pil = Image.fromarray(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB))
        if i == 1:
            pil = pil.transpose(Image.ROTATE_90)
            exif = Image.Exif()
            exif[0x0112] = 6  # "rotate 90 degrees clockwise to view"
            pil.save(bundle / name, "JPEG", quality=70, exif=exif)
        else:
            pil.save(bundle / name, "JPEG", quality=68)
    # 3. a printed worksheet with handwritten answers, scanned
    W, H = 1654, 2339
    sheet = Image.new("RGB", (W, H), (244, 244, 240))
    d = ImageDraw.Draw(sheet)
    f_print = ImageFont.truetype("/System/Library/Fonts/Supplemental/Times New Roman.ttf", 40)
    d.text((150, 130), "Science 8B   Worksheet 12: Separating mixtures", fill=(20, 20, 20), font=f_print)
    qa = [("1. Name two ways of separating a mixture.", "filtering and evaporating the water"),
          ("2. What is left when salt water is heated until dry?", "white crystals of salt in the dish"),
          ("3. Why can sand be filtered out but salt cannot?", "sand does not dissolve but salt does"),
          ("4. Which change of state turns a liquid into a gas?", "boiling or evaporating the liquid"),
          ("5. Give one example of a solution used at home.", "sugar in tea or juice mixed with water")]
    base = np.asarray(sheet)[:, :, ::-1].copy()
    ink = np.zeros((H, W), np.uint8)
    tr = []
    for j, (q_, ans) in enumerate(qa):
        y = 300 + j * 330
        cv2.putText(base, "", (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        pil_q = Image.fromarray(base[:, :, ::-1])
        ImageDraw.Draw(pil_q).text((150, y), q_, fill=(20, 20, 20), font=f_print)
        base = np.asarray(pil_q)[:, :, ::-1].copy()
        cv2.line(base, (150, y + 170), (W - 150, y + 170), (120, 120, 120), 2)
        x = 170.0
        for wd in ans.split():
            st, wl = mara.word(wd, rng)
            for s in st:
                qq = np.round((s * 32 + [x, y + 160 + rng.normal(0, 2)]) * 16).astype(np.int32)
                if len(qq) > 1:
                    cv2.polylines(ink, [qq], False, 255, 3, cv2.LINE_AA, 4)
            x += wl * 32 + 26
        tr.append(ans)
    a = (ink.astype(np.float32) / 255)[..., None] * 0.9
    img = (base * (1 - a) + np.array((120, 60, 40), np.float32) * a).astype(np.uint8)
    truth["worksheet 12.pdf#page1"] = tr
    doc = pymupdf.open()
    pg = doc.new_page(width=595, height=842)
    pg.insert_image(pg.rect, stream=jpeg(degrade(img, rng, "scan"), 66))
    doc.save(bundle / "worksheet 12.pdf")
    # 4. a page in somebody else's hand, with a name that hints at it
    img, tr = page_image(other, ["# Notes from Dana", "Borrowed these for the test on mixtures.", "- filtering uses paper and a funnel",
                                 "- distilling needs heat and a cold tube", "Thanks for lending me your book last week."], rng)
    cv2.imwrite(str(bundle / "from Dana.jpg"), degrade(img, rng, "scan"), [cv2.IMWRITE_JPEG_QUALITY, 70])
    truth["from Dana.jpg"] = {"writer": "someone else", "lines": tr}
    # 5. a typed document that has no handwriting at all
    doc = pymupdf.open()
    pg = doc.new_page()
    pg.insert_textbox(pymupdf.Rect(72, 72, 520, 700), "Term timetable\n\nMonday: maths, science, history\nTuesday: art, English, sport\n"
                      "Wednesday: science, music, geography\n\nThis page is typed and contains no handwriting.", fontsize=13)
    doc.save(bundle / "timetable.pdf")
    # 6. the remaining pages as a second scanned PDF, lower quality
    doc = pymupdf.open()
    for i, lines in enumerate(pages[6:] or pages[:1]):
        img, tr = page_image(mara, lines, rng, ruled=(i % 2 == 0))
        truth[f"scan0003.pdf#page{i + 1}"] = tr
        pg = doc.new_page(width=595, height=842)
        small = cv2.resize(degrade(img, rng, "scan"), None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
        pg.insert_image(pg.rect, stream=jpeg(small, 52))
    doc.save(bundle / "scan0003.pdf")
    (truth_dir / "text.json").write_text(json.dumps(truth, indent=1))
    forms = {ch: [[np.round(s, 3).tolist() for s in mara.form(ch, w)] for w in (0, 1)] for ch in
             "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" if mara.form(ch, 0)}
    (truth_dir / "forms.json").write_text(json.dumps(forms))
    used = sorted(set("".join(" ".join(v) if isinstance(v, list) else " ".join(v["lines"]) for v in truth.values())))
    print("bundle:", sorted(p.name for p in bundle.iterdir()))
    print("pages by the writer:", sum(1 for k in truth if "Dana" not in k), "| words:", sum(len(" ".join(v).split()) for v in truth.values() if isinstance(v, list)))
    print("characters that occur:", "".join(c for c in used if c != " "))


if __name__ == "__main__":
    main(*sys.argv[1:])
