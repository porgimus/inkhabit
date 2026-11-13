# SPDX-License-Identifier: AGPL-3.0-or-later
"""Format check: the same page of an invented writer's notes, delivered every way someone might hand it
over. Reports, for each, whether it loads, how many strokes and words come out, and how many of the words
that were written are read. Run: uv run python bench/formats.py [--out out/formats]
"""

from __future__ import annotations

import argparse
import io
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import writers as fw  # noqa: E402

from inkhabit import sources  # noqa: E402
from inkhabit.words import page_words  # noqa: E402

W, H = 1654, 2339
PT = 595 / W


def norm(t):
    return re.sub(r"[^a-z0-9]", "", t.lower())


def make(root: Path) -> tuple[dict[str, Path], list[str]]:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    person = fw.Person("other", 3, slant=0, squeeze=1.0, tall=1.0, tail=1.0, wild=0.04, join=0.05)
    lines = [ln for lns in fw.NOTES.values() for ln in lns][:10]
    vec: list = []
    img, truth = fw.page_image(person, lines, rng, ruled=False, xh=30, vec=vec)
    words = [norm(w) for ln in truth for w in ln.split() if norm(w)]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    out: dict[str, Path] = {}

    def save(name: str, data: bytes | None = None) -> Path:
        p = root / name
        if data is not None:
            p.write_bytes(data)
        out[name] = p
        return p

    # pictures
    pil.save(save("page.png"))
    pil.save(save("page.jpg"), quality=75)
    pil.save(save("page.tif"))
    pil.save(save("page.bmp"))
    pil.save(save("page.webp"), quality=80)
    pil.convert("L").save(save("grey.png"))
    pil.convert("CMYK").save(save("cmyk.jpg"), quality=80)
    Image.fromarray((np.asarray(pil.convert("L")).astype(np.uint16) * 257)).save(save("16bit.png"))
    Image.fromarray(255 - rgb).save(save("white-on-black.png"))
    # ink on a transparent background, as note apps export it
    alpha = 255 - np.asarray(pil.convert("L"))
    alpha = np.where(alpha > 20, 255, 0).astype(np.uint8)
    rgba = np.dstack([np.full_like(rgb, 30), alpha])
    Image.fromarray(rgba, "RGBA").save(save("transparent.png"))
    pil.rotate(90, expand=True).save(save("sideways-no-flag.jpg"), quality=75)
    ex = Image.Exif()
    ex[0x0112] = 6
    pil.transpose(Image.ROTATE_90).save(save("sideways-with-flag.jpg"), quality=75, exif=ex)
    pil.resize((W // 3, H // 3)).save(save("small-600px.jpg"), quality=70)
    pil.resize((W * 3, H * 3)).save(save("huge-5000px.jpg"), quality=70)
    pil.save(save("two-pages.tif"), save_all=True, append_images=[pil])
    if shutil.which("sips"):
        subprocess.run(["sips", "-s", "format", "heic", str(root / "page.png"), "--out", str(save("page.heic"))], capture_output=True)
    pil.save(save("UPPERCASE.JPG"), quality=75)
    shutil.copy(root / "page.jpg", save("no-extension"))
    # PDFs
    def pdf(name, build):
        d = pymupdf.open()
        build(d)
        d.save(save(name))

    def image_page(d, rotate=0):
        pg = d.new_page(width=595, height=842)
        buf = io.BytesIO()
        pil.save(buf, "JPEG", quality=75)
        pg.insert_image(pg.rect, stream=buf.getvalue())
        if rotate:
            pg.set_rotation(rotate)

    def stroke_page(d, outline=False):
        pg = d.new_page(width=595, height=842)
        for pts, thick in vec:
            if len(pts) > 1:
                pg.draw_polyline([tuple(p * PT) for p in pts], color=(0.1, 0.15, 0.4), width=thick * PT, lineCap=1, lineJoin=1)

    pdf("scan.pdf", image_page)
    pdf("scan-rotated-90.pdf", lambda d: image_page(d, 90))
    pdf("strokes.pdf", stroke_page)
    pdf("strokes-and-typed-text.pdf", lambda d: (stroke_page(d), d[0].insert_text((60, 40), "Lecture 4 handout: printed title", fontsize=11)))
    pdf("mixed-scan-then-strokes.pdf", lambda d: (image_page(d), stroke_page(d)))
    pdf("typed-only.pdf", lambda d: d.new_page().insert_text((72, 100), "This page is typed. No handwriting here.", fontsize=12))
    pdf("empty-page.pdf", lambda d: d.new_page())
    enc = pymupdf.open()
    image_page(enc)
    enc.save(save("password.pdf"), encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="secret")
    # containers and odd ones
    with zipfile.ZipFile(save("bundle.zip"), "w") as z:
        z.write(root / "page.jpg", "notes/page one.jpg")
        z.write(root / "strokes.pdf", "notes/more/strokes.pdf")
    (root / "folder" / "deep").mkdir(parents=True, exist_ok=True)
    shutil.copy(root / "page.png", root / "folder" / "deep" / "inner.png")
    out["folder/"] = root / "folder"
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" fill="none" stroke="#1a2a66" stroke-width="3">']
    for pts, _ in vec:
        if len(pts) > 1:
            svg.append('<path d="M' + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + '"/>')
    save("ink.svg", ("\n".join(svg) + "\n</svg>").encode())
    save("broken.jpg", b"\xff\xd8\xff\xe0 this is not really a picture")
    save("notes.txt", b"These are typed notes, not handwriting.")
    return out, words


def check(path: Path, words: list[str]) -> dict:
    try:
        files = sources.discover(path)
        if not files:
            return {"status": "not picked up", "detail": "discover() found nothing"}
        pages = [p for f in files for p in sources.load(f)]
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "detail": repr(e)[:110]}
    strokes = sum(len(p.strokes) for p in pages)
    got = []
    try:
        for p in pages[:2]:
            got += [norm(w.text) for w in page_words(p)]
    except Exception as e:  # noqa: BLE001
        return {"status": "error reading", "detail": repr(e)[:110], "pages": len(pages), "strokes": strokes}
    want = set(words)
    read = len(want & set(got)) / max(len(want), 1)
    return {"status": "ok" if read > 0.5 else "empty" if strokes < 30 else "poor", "pages": len(pages), "strokes": strokes, "read": read}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/formats")
    a = ap.parse_args()
    files, words = make(Path(a.out))
    print(f"{'input':30s} {'result':14s} {'pages':>5s} {'strokes':>8s} {'words read':>11s}")
    for name, path in files.items():
        r = check(path, words)
        print(f"{name:30s} {r['status']:14s} {r.get('pages', ''):>5} {r.get('strokes', ''):>8} "
              + (f"{r['read']:>10.0%}" if "read" in r else f"   {r.get('detail', '')}"), flush=True)


if __name__ == "__main__":
    main()
