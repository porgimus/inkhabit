# SPDX-License-Identifier: AGPL-3.0-or-later
"""Benchmark: build a font from invented writers' notes and score it against the known truth.

    uv run python bench/run.py [--quick] [--out out/bench]

Each writer is delivered three ways (pen strokes in a PDF, a scan, a phone photo) and built twice
(labels from the true text, and from the recogniser alone). For every case it reports where accuracy
is lost, stage by stage:

  words    share of the written words that ended up with the right text attached
  ink      of those, the share that also got the right ink (not a neighbour's, not another line's)
  letters  share of the samples the library kept that are the right letter
  hers     share that are closer to this writer's true shape than to a generic font or another writer
  read     share of words written with the result that the recogniser reads back correctly

Needs macOS (system fonts for the invented hands, Apple Vision for recognition). No real handwriting is used.

A caution learnt the hard way: the invented hands are bent versions of system fonts, and the pipeline's
reference letter shapes come from a system font too. Anything that compares samples with reference
shapes is therefore flattered here. A review step that looked excellent on this benchmark threw away a
third of the good samples in a real person's font. Confirm every gain on real handwriting before keeping it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pymupdf

sys.path.insert(0, str(Path(__file__).parent))
import writers as fw  # noqa: E402

from inkhabit import library, pipeline, render, segment, sources, tune  # noqa: E402
from inkhabit.segment import feature  # noqa: E402

WRITERS = {  # name: (font, seed, slant, squeeze, tall, tail, wild, join, x-height in px)
    "upright print": ("other", 3, 0, 1.0, 1.0, 1.0, 0.04, 0.05, 30),
    "heavy slant": ("mara", 11, 15, 0.8, 1.22, 1.4, 0.075, 0.45, 30),
    "mostly joined": ("mara", 21, 7, 0.9, 1.1, 1.2, 0.06, 0.9, 30),
    "tiny": ("comic", 8, 2, 0.95, 1.0, 1.0, 0.05, 0.1, 21),
    "large and loopy": ("marker", 14, 5, 1.05, 1.3, 1.5, 0.09, 0.3, 40),
    "left leaning": ("comic", 30, -12, 0.9, 1.1, 1.2, 0.06, 0.25, 30),
}
FORMATS = ("strokes", "scan", "photo")
W, H = 1654, 2339
PT = 595 / W


def norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())


def degrade(img, rng, kind):
    """Returns the degraded image and a function mapping page pixels to where they ended up."""
    out = img.astype(np.float32)
    if kind == "scan":
        M = cv2.getRotationMatrix2D((W / 2, H / 2), rng.normal(0, 1.1), 1.0)
        out = cv2.warpAffine(out, M, (W, H), borderValue=(250, 250, 250))
        out += np.linspace(-10, 8, W)[None, :, None] + rng.normal(0, 4.5, out.shape)
        out = cv2.GaussianBlur(out, (0, 0), 0.9)
        fn = lambda p: np.c_[p, np.ones(len(p))] @ M.T  # noqa: E731
    else:
        m = 90
        src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        dst = np.float32([[m + rng.uniform(0, 70), m + rng.uniform(0, 50)], [W - m - rng.uniform(0, 90), m + rng.uniform(20, 90)],
                          [W - m + rng.uniform(-30, 20), H - m - rng.uniform(0, 60)], [m - rng.uniform(0, 40), H - m + rng.uniform(-40, 10)]])
        Hm = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(out, Hm, (W, H), borderValue=(70, 88, 110))
        yy, xx = np.mgrid[0:H, 0:W]
        out *= (1 - 0.33 / (1 + np.exp(-((xx * 0.7 + yy * 0.5) - (W * 0.55 + rng.uniform(-200, 200))) / 130.0)))[..., None]
        out = cv2.GaussianBlur(out, (0, 0), 1.3) + rng.normal(0, 6, out.shape)
        fn = lambda p: cv2.perspectiveTransform(np.float32(p)[None], Hm)[0]  # noqa: E731
    return np.clip(out, 0, 255).astype(np.uint8), fn


def make_case(person, xh, fmt, pages, root: Path, rng):
    """Write the source files for one writer in one format. Returns {source stem: [(page, truth lines, word boxes in page units)]}."""
    root.mkdir(parents=True, exist_ok=True)
    truth = {}
    doc = pymupdf.open() if fmt != "photo" else None
    for i, lines in enumerate(pages):
        boxes, vec = [], []
        img, tr = fw.page_image(person, lines, rng, ruled=fmt != "strokes" and not os.environ.get("BENCH_PLAIN"), xh=xh, boxes=boxes, vec=vec)
        if fmt == "strokes":
            pg = doc.new_page(width=595, height=842)
            for pts, thick in vec:
                if len(pts) > 1:
                    pg.draw_polyline([tuple(p * PT) for p in pts], color=(0.1, 0.15, 0.4), width=thick * PT, lineCap=1, lineJoin=1)
            unit, fn = PT, (lambda p: p)
            stem, idx = "notes", i
        else:
            bad, fn = degrade(img, rng, fmt)
            ok, buf = cv2.imencode(".jpg", bad, [cv2.IMWRITE_JPEG_QUALITY, 64])
            if fmt == "scan":
                pg = doc.new_page(width=595, height=842)
                pg.insert_image(pg.rect, stream=buf.tobytes())
                unit, stem, idx = PT, "notes", i
            else:
                (root / f"photo{i + 1}.jpg").write_bytes(buf.tobytes())
                unit, stem, idx = 72 / 300, f"photo{i + 1}", 0
        placed = []
        for wd, x0, y0, x1, y1 in boxes:
            c = np.asarray(fn(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float))) * unit
            placed.append((wd, c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()))
        truth.setdefault(stem, []).append((idx, tr, placed))
    if doc is not None:
        doc.save(root / "notes.pdf")
    return truth


def stage_scores(samples, truth) -> tuple[float, float]:
    """(share of written words with the right text attached, share of those with the right ink)."""
    total = found = good = 0
    for stem, pgs in truth.items():
        for idx, _, placed in pgs:
            mine = [(norm(s.text), np.vstack(s.strokes)) for s in samples if Path(s.source).stem == stem and s.page == idx]
            mine = [(t, p.min(0), p.max(0)) for t, p in mine]
            for wd, x0, y0, x1, y1 in placed:
                if not norm(wd):
                    continue
                total += 1
                w = x1 - x0
                best = None
                for t, lo, hi in mine:
                    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
                    if t == norm(wd) and x0 - 0.5 * w - 6 < cx < x1 + 0.5 * w + 6 and y0 - (y1 - y0) < cy < y1 + (y1 - y0):
                        inter = min(hi[0], x1) - max(lo[0], x0)
                        iou = inter / max(max(hi[0], x1) - min(lo[0], x0), 1e-6)
                        tall_ok = hi[1] - lo[1] < 1.6 * (y1 - y0) and y0 - 0.25 * (y1 - y0) < cy < y1 + 0.25 * (y1 - y0)
                        score = iou if tall_ok else 0.0
                        best = score if best is None else max(best, score)
                if best is not None:
                    found += 1
                    good += best >= 0.6
    return found / max(total, 1), good / max(found, 1)


def letter_scores(lib, person, rival) -> tuple[float, float, int]:
    lean = np.tan(np.radians(lib["meta"].get("slant", 0.0)))
    chars = [c for c in fw_alnum() if person.form(c, 0)]
    T = {c: [feat(person.form(c, w)) for w in (0, 1)] for c in chars}
    N = {c: feat(fw.skeleton(c, person.who)) for c in chars}
    O = {c: feat(rival.form(c, 0)) for c in chars if rival.form(c, 0)}
    real = {c: e for c, e in lib["glyphs"].items() if c in T and e.get("count", 0) > 0 and e["variants"]
            and e["variants"][0].get("source") in (None, "sheet")}
    ratios = []
    for c in "acemnorsuvwxz":
        if c in real:
            h_lib = np.median([np.ptp([p[1] for s in v["strokes"] for p in s]) for v in real[c]["variants"]])
            h_true = np.ptp([p[1] for s in person.form(c, 0) for p in s])
            ratios.append(h_true / max(h_lib, 1e-6))
    k = float(np.median(ratios)) if ratios else 1.0
    right = hers = total = 0
    for c, e in real.items():
        for v in e["variants"]:
            st = [np.asarray(s, float) * k for s in v["strokes"]]
            st = [np.c_[p[:, 0] - p[:, 1] * lean, p[:, 1]] for p in st]
            x0 = min(q[:, 0].min() for q in st)
            f = feat([q - [x0, 0] for q in st])
            best = min(T, key=lambda ch: min(((f - t) ** 2).sum() for t in T[ch]))
            total += 1
            right += best.lower() == c.lower()
            d_true = min(((f - t) ** 2).sum() for t in T[c])
            hers += d_true < min(((f - N[c]) ** 2).sum(), ((f - O[c]) ** 2).sum() if c in O else 1e9)
    return right / max(total, 1), hers / max(total, 1), len(real)


def feat(strokes):
    return feature([np.asarray(s, float) for s in strokes], 0.0, 1.0)


def fw_alnum():
    return "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def build(files, labels: Path | None, vocab):
    pipeline.TRANSCRIPTS, pipeline.LANG, pipeline.INK = labels, None, None
    pages, samples = [], []
    for f in files:
        pg, ws = pipeline._read(f)
        pages += pg
        samples += ws
    raw = samples
    slant = segment.estimate_slant(samples)
    if abs(slant) >= 4:
        samples = segment.deslant(samples, slant)
    else:
        slant = 0.0
    aligned, meta = segment.run(samples, vocab, log=lambda *a: None)
    if len([a for a in aligned if len(a.prep.text) >= 2]) < 30:
        return raw, None
    meta["slant"] = round(slant, 1)
    pens = [s.width for p in pages for s in p.strokes if s.width <= 4]
    meta["font_weight"] = float(np.clip(np.median(pens) / meta["xheight"], 0.09, 0.15))
    labelled = sum(s.trusted for s in samples)
    lib = library.build(aligned, meta, word_cost=6.0 if labelled > 0.5 * max(1, len(aligned)) else 2.5)
    lib["pair_gap"] = library.pair_gaps(aligned)
    library.unify(lib, log=lambda *a: None)
    return raw, lib


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/bench")
    ap.add_argument("--quick", action="store_true", help="two writers, four pages")
    ap.add_argument("--only", help="run one writer")
    ap.add_argument("--formats", default=",".join(FORMATS), help="comma-separated: strokes,scan,photo")
    ap.add_argument("--labels", default="truth,ocr", help="comma-separated: truth,ocr")
    a = ap.parse_args()
    out = Path(a.out)
    vocab = pipeline.load_vocab()
    lines = [ln for lns in fw.NOTES.values() for ln in lns]
    pages = [lines[i : i + 10] for i in range(0, len(lines), 10)][: 4 if a.quick else 8]
    names = [a.only] if a.only else list(WRITERS)[: 2 if a.quick else None]
    rival = fw.Person("other", 5, slant=-4, squeeze=1.1, tall=0.95, tail=0.9, wild=0.04, join=0.05)
    rows = []
    print(f"{'writer':16s} {'format':8s} {'labels':7s} {'words':>6s} {'ink':>6s} {'letters':>8s} {'hers':>6s} {'read':>6s} {'chars':>6s}")
    for name in names:
        font, seed, slant, squeeze, tall, tail, wild, join, xh = WRITERS[name]
        person = fw.Person(font, seed, slant=slant, squeeze=squeeze, tall=tall, tail=tail, wild=wild, join=join)
        for fmt in a.formats.split(","):
            root = out / name.replace(" ", "-") / fmt
            for f in root.glob("*") if root.exists() else []:
                if f.is_file():
                    f.unlink()
            truth = make_case(person, xh, fmt, pages, root, np.random.default_rng(seed * 7 + len(fmt)))
            labels = root / "labels"
            labels.mkdir(exist_ok=True)
            for stem, pgs in truth.items():
                body = [f"# {stem}", ""]
                for idx, tr, _ in pgs:
                    body += [f"## Page {idx + 1}", ""] + tr + [""]
                (labels / f"{stem}.md").write_text("\n".join(body))
            files = [f for f in sources.discover(root) if f.parent == root]
            for mode in a.labels.split(","):
                t = time.time()
                samples, lib = build(files, labels if mode == "truth" else None, vocab)
                used = [s for s in samples if s.trusted] if mode == "truth" else [s for s in samples if segment.usable_text(s.text, vocab)]
                words, ink = stage_scores(used, truth)
                if lib is None:
                    row = {"writer": name, "format": fmt, "labels": mode, "words": words, "ink": ink, "letters": 0, "hers": 0, "read": 0, "chars": 0}
                else:
                    letters, hers, n = letter_scores(lib, person, rival)
                    read = tune.legibility(lib, seeds=(1, 2, 3), style="clean")
                    row = {"writer": name, "format": fmt, "labels": mode, "words": words, "ink": ink, "letters": letters, "hers": hers,
                           "read": read, "chars": n}
                    library.save(lib, root / f"library-{mode}.json")
                row["seconds"] = round(time.time() - t)
                rows.append(row)
                print(f"{name:16s} {fmt:8s} {mode:7s} {row['words']:6.0%} {row['ink']:6.0%} {row['letters']:8.0%} {row['hers']:6.0%} "
                      f"{row['read']:6.0%} {row['chars']:6d}", flush=True)
    (out / "results.json").write_text(json.dumps(rows, indent=1))
    print("\naverages by format and labels:")
    for fmt in a.formats.split(","):
        for mode in a.labels.split(","):
            sel = [r for r in rows if r["format"] == fmt and r["labels"] == mode]
            if sel:
                print(f"  {fmt:8s} {mode:6s} " + "  ".join(f"{k} {np.mean([r[k] for r in sel]):.0%}" for k in ("words", "ink", "letters", "hers", "read")))


if __name__ == "__main__":
    main()
