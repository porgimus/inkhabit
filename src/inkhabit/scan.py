# SPDX-License-Identifier: AGPL-3.0-or-later
"""Scanned pages (PDF or image) -> pen strokes.

A scan is only pixels, so we recover the pen's path: clean the page to ink
only, thin every line to one pixel wide (skeleton), then walk the skeleton
into polylines. Pen width comes from how thick the ink was before thinning.

Units are PDF points (1/72 inch) so scans and digital ink share one scale.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from .ink import Page, Stroke

DPI = 300  # for images, whose real size is unknown
TARGET_WIDTH_PX = 3600  # PDFs are rendered at whatever resolution gives about this width


def _to_rgb(im) -> np.ndarray:
    """Any picture as 8-bit RGB on a plain background. Handles 16-bit, palettes, CMYK and transparency."""
    from PIL import Image, ImageOps

    im = ImageOps.exif_transpose(im)  # phone photos store their rotation in metadata; apply it
    if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        a = np.asarray(im).astype(np.float64)
        lo, hi = float(a.min()), float(a.max())
        im = Image.fromarray(((a - lo) / max(hi - lo, 1e-9) * 255).astype(np.uint8))
    if im.mode == "P" and "transparency" in im.info:
        im = im.convert("RGBA")
    if im.mode in ("RGBA", "LA"):
        rgba = np.asarray(im.convert("RGBA")).astype(np.float64)
        alpha = rgba[..., 3:] / 255
        solid = rgba[..., :3][alpha[..., 0] > 0.5]
        # ink exported on a transparent background: put it on paper that contrasts with the ink
        paper = 255.0 if (solid.size == 0 or solid.mean() < 140) else 30.0
        return (rgba[..., :3] * alpha + paper * (1 - alpha)).astype(np.uint8)
    return np.asarray(im.convert("RGB"))


def _upright(rgb: np.ndarray) -> np.ndarray:
    """Turn a page that was photographed or scanned sideways or upside down the right way up.
    Judged by reading it: the right way up is the one where the recogniser finds the most words."""
    from PIL import Image

    from .ocr import recognize

    h, w = rgb.shape[:2]
    k = 1400 / max(h, w)
    small = cv2.resize(rgb, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else rgb

    def score(a):
        words = recognize(Image.fromarray(a))
        good = [x for x in words if len(x.text) >= 3 and x.text.isalpha()]
        wide = [x for x in words if (x.box[2] - x.box[0]) >= (x.box[3] - x.box[1])]
        return len(good) + 0.5 * len(wide), len(words), len(wide)

    base, n, wide = score(small)
    if n >= 8 and wide >= 0.8 * n:
        return rgb  # reads well as it is, in lines that run across
    best, turn = base, 0
    for t in (1, 3, 2):  # quarter turn either way, then upside down
        s_, _, _ = score(np.ascontiguousarray(np.rot90(small, t)))
        if s_ > best * 1.25 + 2:
            best, turn = s_, t
    return np.ascontiguousarray(np.rot90(rgb, turn)) if turn else rgb


def _images(path: Path, kind: str | None = None) -> list[tuple[np.ndarray, float]]:
    """(RGB image, pixels per PDF point) for every page."""
    kind = kind or path.suffix.lower()
    if kind == ".pdf":
        import pymupdf as fitz

        out = []
        with fitz.open(path) as doc:
            if doc.needs_pass:
                raise ValueError(f"{path.name} is password-protected: export an unlocked copy and use that")
            for page in doc:
                dpi = float(np.clip(TARGET_WIDTH_PX / (page.rect.width / 72), 200, 600))
                pix = page.get_pixmap(dpi=int(dpi), colorspace=fitz.csRGB, alpha=False)
                im = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()
                # printed text (slides, worksheets, typed forms) is not handwriting: paint it out
                paper = np.median(im.reshape(-1, 3)[::97], axis=0).astype(np.uint8)
                k = pix.width / page.rect.width
                printed = 0
                for span in page.get_texttrace():
                    # type 3 = invisible text: the searchable layer that note apps put over handwriting.
                    # Painting that out would erase the handwriting itself.
                    if span.get("type") == 3 or span.get("opacity", 1) == 0:
                        continue
                    printed += 1
                    q = fitz.Rect(span["bbox"]) * page.rotation_matrix  # where it lands on the rendered page
                    im[max(int(q.y0 * k) - 3, 0) : int(q.y1 * k) + 3, max(int(q.x0 * k) - 3, 0) : int(q.x1 * k) + 3] = paper
                out.append((im if printed else _upright(im), k))
        return out
    from PIL import Image, ImageSequence

    if kind in (".heic", ".heif"):  # iPhone photos: Pillow cannot read these, macOS can convert them
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            jpg = Path(tmp) / "page.jpg"
            subprocess.run(["sips", "-s", "format", "jpeg", str(path), "--out", str(jpg)], check=True, capture_output=True)
            return [(_upright(_to_rgb(Image.open(jpg))), DPI / 72)]
    try:
        im = Image.open(path)
        frames = [_to_rgb(f.copy()) for f in ImageSequence.Iterator(im)]  # a TIFF can hold several pages
    except Exception as e:
        raise ValueError(f"{path.name} could not be opened as a picture ({type(e).__name__}): the file may be damaged") from None
    return [(_upright(f), DPI / 72) for f in frames[:200]]


def ink_mask(rgb: np.ndarray, px_per_pt: float = DPI / 72) -> np.ndarray:
    """True where there is handwriting.

    Works on any paper and ink colour (white ink on dark paper, coloured pens)
    by measuring how far each pixel is from the local paper colour, and copes
    with uneven light and ruled or printed lines.
    """
    blur = cv2.GaussianBlur(rgb, (3, 3), 0).astype(np.int16)
    k = int(px_per_pt * 18) | 1  # paper colour = local median, wide enough to ignore the writing
    small = cv2.resize(rgb, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
    paper = cv2.resize(cv2.medianBlur(small, min(max(k // 8, 3) | 1, 255)), rgb.shape[1::-1], interpolation=cv2.INTER_LINEAR)
    diff = np.abs(blur - paper.astype(np.int16)).max(axis=2).astype(np.uint8)
    level, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Scanning and photographing blur ink outwards: a threshold that takes in the soft edge makes strokes fat,
    # which closes the loops of letters and fuses neighbours. Cut at a fixed share of the full ink strength.
    inked = diff[diff > max(level * 0.8, 28)]
    strong = float(np.percentile(inked, 80)) if inked.size else 255.0
    share = float(os.environ.get("INKHABIT_INK_SHARE", 0.6))
    mask = (diff > max(level * 0.8, 28, share * strong)).astype(np.uint8) * 255
    mask = remove_rules(mask, px_per_pt)
    mask = _drop_faint_lines(mask, diff, px_per_pt)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    small_cc = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] < max(6, (px_per_pt * 0.35) ** 2))
    mask[np.isin(lab, small_cc)] = 0
    return mask > 0


def _drop_faint_lines(mask: np.ndarray, diff: np.ndarray, px_per_pt: float) -> np.ndarray:
    """Pale ruled lines on a skewed or crumpled scan break into fragments too short to be caught as lines.
    They are still much paler than the pen. If the page has a pale kind of ink, and most of it lies along
    long flat runs, it is ruling, and all of it goes. Pale ink that is not line-like (pencil) is kept."""
    on = mask > 0
    if on.sum() < 500:
        return mask
    strong = float(np.percentile(diff[on], 90))
    faint = on & (diff < 0.45 * strong)
    if faint.sum() < 0.05 * on.sum():
        return mask
    run = max(int(px_per_pt * 9), 24)
    flat = cv2.morphologyEx(faint.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((1, 9), np.uint8))
    lines = cv2.morphologyEx(flat, cv2.MORPH_OPEN, np.ones((1, run), np.uint8)) | cv2.morphologyEx(flat, cv2.MORPH_OPEN, np.ones((run, 1), np.uint8))
    if (lines.astype(bool) & faint).sum() < 0.5 * faint.sum():
        return mask
    keep = on & ~faint
    # the pen's own soft edges are pale too: keep pale pixels that touch strong ink
    keep |= faint & (cv2.dilate(keep.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~(lines.astype(bool) & ~cv2.dilate(keep.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool))
    return (keep.astype(np.uint8) * 255)


def remove_rules(mask: np.ndarray, px_per_pt: float) -> np.ndarray:
    """Erase ruled and printed lines, including ones tilted or bowed by a camera.

    A rule is a long thin run. We look for runs along directions a few degrees
    either side of horizontal and vertical, and erase them except where a pen
    stroke crosses (there the ink is much taller than the rule is thick).
    """
    h, w = mask.shape
    run = max(int(px_per_pt * 30), w // 16)
    rules = np.zeros_like(mask)
    for base in (0.0, 90.0):
        for ang in np.arange(-5.0, 5.01, 0.5):
            a = np.radians(base + ang)
            dx, dy = np.cos(a) * run / 2, np.sin(a) * run / 2
            size = (int(abs(dy)) * 2 + 3, int(abs(dx)) * 2 + 3)
            kern = np.zeros(size, np.uint8)
            c = (size[1] // 2, size[0] // 2)
            cv2.line(kern, (int(round(c[0] - dx)), int(round(c[1] - dy))), (int(round(c[0] + dx)), int(round(c[1] + dy))), 1, 1)
            rules |= cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)
    if not rules.any():
        return mask
    thick = max(3, int(round(px_per_pt * 0.9)))
    rules = cv2.dilate(rules, np.ones((3, 3), np.uint8))
    # keep ink that crosses a rule: it extends well beyond the rule's thickness on the perpendicular
    tall = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((thick * 2 + 1, 1), np.uint8))
    wide = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, thick * 2 + 1), np.uint8))
    horiz = cv2.morphologyEx(rules, cv2.MORPH_OPEN, np.ones((1, 9), np.uint8))
    crossing = np.where(horiz > 0, tall, wide)
    return mask & ~(rules & ~crossing)


def _trace(skel: np.ndarray) -> list[np.ndarray]:
    """Walk a 1-pixel skeleton into polylines, breaking at junctions."""
    sk = np.pad(skel, 1).astype(np.uint8)
    nb = cv2.filter2D(sk, -1, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.float32), borderType=cv2.BORDER_CONSTANT)
    junction = (sk > 0) & (nb >= 3)
    body = (sk > 0) & ~junction
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    seen = np.zeros_like(body)
    lines = []

    def walk(y, x):
        path = [(y, x)]
        seen[y, x] = True
        while True:
            for dy, dx in offs:
                ny, nx = path[-1][0] + dy, path[-1][1] + dx
                if body[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    path.append((ny, nx))
                    break
            else:
                return path

    def cap(path, end):
        y, x = path[end]
        for dy, dx in offs:  # reach into the junction so pieces still touch
            if junction[y + dy, x + dx]:
                return [(y + dy, x + dx)]
        return []

    body_nb = cv2.filter2D(body.astype(np.uint8), -1, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.float32),
                           borderType=cv2.BORDER_CONSTANT)
    for pass_ends in (True, False):  # open paths first, then closed loops
        ys, xs = np.nonzero(body & ~seen & ((body_nb <= 1) if pass_ends else True))
        for y, x in zip(ys, xs):
            if seen[y, x]:
                continue
            p = walk(y, x)
            p = cap(p, 0) + p + cap(p, -1)
            lines.append(np.array(p, float)[:, ::-1] - 1)
    ys, xs = np.nonzero(junction & (cv2.filter2D(body.astype(np.uint8), -1, np.ones((3, 3), np.float32)) == 0))
    lines += [np.array([[x - 1, y - 1]], float) for y, x in zip(ys, xs)]
    return lines


def _smooth(p: np.ndarray) -> np.ndarray:
    if len(p) < 5:
        return p
    k = np.array([1, 2, 3, 2, 1]) / 9
    q = np.c_[np.convolve(np.pad(p[:, 0], 2, "edge"), k, "valid"), np.convolve(np.pad(p[:, 1], 2, "edge"), k, "valid")]
    q[0], q[-1] = p[0], p[-1]
    return q


def _path_points(items) -> list[np.ndarray]:
    """PDF path items -> polylines (one per connected run), sampling the curves."""
    runs: list[list] = []
    last = None
    for it in items:
        op = it[0]
        if op == "l":
            pts = [(it[1].x, it[1].y), (it[2].x, it[2].y)]
        elif op == "c":
            p0, p1, p2, p3 = (np.array([q.x, q.y]) for q in it[1:5])
            t = np.linspace(0, 1, 7)[:, None]
            pts = [tuple(v) for v in (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3]
        elif op == "qu":
            q = it[1]
            pts = [(q.ul.x, q.ul.y), (q.ur.x, q.ur.y), (q.lr.x, q.lr.y), (q.ll.x, q.ll.y), (q.ul.x, q.ul.y)]
        else:
            last = None
            continue
        if last is not None and abs(pts[0][0] - last[0]) < 1e-3 and abs(pts[0][1] - last[1]) < 1e-3:
            runs[-1] += pts[1:]
        else:
            runs.append(list(pts))
        last = pts[-1]
    return [np.array(r, float) for r in runs if len(r) >= 1]


def vector_strokes(page) -> list[Stroke] | None:
    """Pen strokes read straight from a PDF that still holds them as shapes (exports from note apps).

    Two kinds exist. Some apps write each stroke as a line with a width: that is the pen path itself.
    Others write the outline of each stroke as a filled shape: then we thin each shape on its own, so
    neighbouring strokes never merge the way they do when a whole page of pixels is thinned.
    Returns None when the page is not made of ink shapes (a scan, a typed document).
    """
    from skimage.morphology import skeletonize

    pw, ph = page.rect.width, page.rect.height
    out: list[Stroke] = []
    for d in page.get_drawings():
        r = d["rect"]
        if r.width > 0.4 * pw and r.height < 3 or r.height > 0.4 * ph and r.width < 3:
            continue  # ruled lines and margins
        if r.width > 0.6 * pw and r.height > 0.6 * ph:
            continue  # the paper itself
        if any(it[0] == "re" for it in d["items"]):
            continue  # boxes and backgrounds, not pen
        runs = _path_points(d["items"])
        if not runs:
            continue
        if "s" in d["type"] and d.get("color") is not None:  # a stroked line: the pen path as written
            colour = (*[float(c) for c in d["color"][:3]], 1.0)
            for p in runs:
                out.append(Stroke(p, float(d.get("width") or 1.0), colour, len(out)))
        elif d["type"] == "f" and d.get("fill") is not None:  # the outline of one stroke
            colour = (*[float(c) for c in d["fill"][:3]], 1.0)
            pts = np.vstack(runs)
            lo = pts.min(0)
            size = pts.max(0) - lo
            area = sum(abs(float(np.dot(p[:-1, 0], p[1:, 1]) - np.dot(p[1:, 0], p[:-1, 1]))) / 2 for p in runs if len(p) > 2)
            length = sum(float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) for p in runs)
            thick = max(2 * area / max(length, 1e-6), 0.15)  # a thin shape: area = length x thickness
            if max(size) < 1.5 * thick + 0.3:  # a dot
                out.append(Stroke(pts.mean(0)[None, :], float(thick), colour, len(out)))
                continue
            k = float(np.clip(9 / thick, 2, 40))
            k = min(k, 1400 / max(size.max(), 1e-6))
            canvas = np.zeros((int(size[1] * k) + 9, int(size[0] * k) + 9), np.uint8)
            for p in runs:  # one at a time: the pieces of a stroke overlap, and filling them together would cancel the overlaps
                if len(p) > 2:
                    cv2.fillPoly(canvas, [np.round((p - lo) * k + 4).astype(np.int32)], 1)
            skel = skeletonize(canvas > 0)
            pieces = [ln for ln in _trace(skel) if len(ln) >= 2]
            if not pieces:
                out.append(Stroke(pts.mean(0)[None, :], float(thick), colour, len(out)))
                continue
            keep = [ln for ln in pieces if np.linalg.norm(np.diff(ln, axis=0), axis=1).sum() > 0.8 * thick * k] or pieces[:1]
            for ln in keep:  # drop the little whiskers thinning leaves at stroke ends
                out.append(Stroke((_smooth(ln) - 4) / k + lo, float(thick), colour, len(out)))
    return out if len(out) >= 20 else None


def _tidy(mask: np.ndarray) -> np.ndarray:
    """Clean the ink shape before thinning it. Scanned ink has pinholes and ragged edges; thinning turns
    every pinhole into a little ring and every bump into a whisker, which shatters letters into bits."""
    m = cv2.medianBlur(mask.astype(np.uint8) * 255, 5) > 0
    dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 3)
    w = float(np.percentile(dist[m], 75) * 2) if m.any() else 2.0  # roughly the pen's thickness in pixels
    holes = (~m).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=4)
    small = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] < 0.5 * w * w)  # far smaller than the loop of any letter
    m[np.isin(lab, small)] = True
    return m


def _prune(lines: list[np.ndarray], width: float) -> list[np.ndarray]:
    """Drop the whiskers thinning leaves on the sides of strokes: very short pieces hanging off a junction."""
    if len(lines) < 2:
        return lines
    ends: dict[tuple[int, int], int] = {}
    for ln in lines:
        for p in (ln[0], ln[-1]):
            key = (int(round(p[0])), int(round(p[1])))
            ends[key] = ends.get(key, 0) + 1
    out = []
    for ln in lines:
        length = float(np.linalg.norm(np.diff(ln, axis=0), axis=1).sum()) if len(ln) > 1 else 0.0
        a = ends[(int(round(ln[0][0])), int(round(ln[0][1])))] > 1
        b = ends[(int(round(ln[-1][0])), int(round(ln[-1][1])))] > 1
        if len(ln) > 1 and length < 1.3 * width and (a != b):  # short, attached at one end only: a whisker
            continue
        out.append(ln)
    return out


def read_svg(path: str | Path) -> list[Page]:
    """Ink saved as SVG paths (exports from drawing and note apps): every path, line and polyline is a stroke."""
    import re
    import xml.etree.ElementTree as ET

    path = Path(path)
    root = ET.parse(path).getroot()
    strokes: list[Stroke] = []
    num = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")

    def width(el) -> float:
        m = re.search(r"stroke-width\s*[:=]\s*\"?([\d.]+)", ET.tostring(el, encoding="unicode"))
        return float(m.group(1)) if m else 2.0

    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1]
        runs: list[list[tuple[float, float]]] = []
        if tag == "path" and el.get("d"):
            cur: list[tuple[float, float]] = []
            x = y = 0.0
            for cmd, args in re.findall(r"([MmLlHhVvCcQqSsTtZzAa])([^MmLlHhVvCcQqSsTtZzAa]*)", el.get("d")):
                v = [float(t) for t in num.findall(args)]
                rel = cmd.islower()
                c = cmd.upper()
                if c == "A":  # an arc: drawn as a straight line to where it ends
                    v, c = [t for i in range(0, len(v) - 6, 7) for t in v[i + 5 : i + 7]], "L"
                step = {"M": 2, "L": 2, "T": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4}.get(c, 0)
                if c == "Z":
                    if cur:
                        cur.append(cur[0])
                    continue
                for i in range(0, len(v) - step + 1, step or 1):
                    a = v[i : i + step]
                    if c == "H":
                        x = a[0] + (x if rel else 0)
                    elif c == "V":
                        y = a[0] + (y if rel else 0)
                    else:
                        pts = [(a[j] + (x if rel else 0), a[j + 1] + (y if rel else 0)) for j in range(0, step, 2)]
                        if c in ("C", "S", "Q") and cur:  # sample the curve
                            p0 = np.array(cur[-1])
                            ctrl = [np.array(p) for p in pts]
                            for t in np.linspace(0, 1, 7)[1:-1]:
                                if c == "C":
                                    q = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * ctrl[0] + 3 * (1 - t) * t**2 * ctrl[1] + t**3 * ctrl[2]
                                else:
                                    q = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl[0] + t**2 * ctrl[-1]
                                cur.append((float(q[0]), float(q[1])))
                        x, y = pts[-1]
                    if c == "M" and i == 0:
                        if cur:
                            runs.append(cur)
                        cur = []
                    cur.append((x, y))
            if cur:
                runs.append(cur)
        elif tag in ("polyline", "polygon") and el.get("points"):
            v = [float(t) for t in num.findall(el.get("points"))]
            runs.append(list(zip(v[0::2], v[1::2])))
        elif tag == "line":
            runs.append([(float(el.get("x1", 0)), float(el.get("y1", 0))), (float(el.get("x2", 0)), float(el.get("y2", 0)))])
        w = width(el) if runs else 0.0
        for r in runs:
            if len(r) >= 1:
                strokes.append(Stroke(np.array(r, float), w, order=len(strokes)))
    if strokes:  # bring to roughly a page in points, so sizes are comparable with other sources
        allp = np.vstack([s.points for s in strokes])
        k = 595.0 / max(float(allp[:, 0].max()), 1.0) if allp[:, 0].max() > 900 else 1.0
        strokes = [Stroke(s.points * k, max(s.width * k, 0.3), s.color, s.order) for s in strokes]
    return [Page(source=str(path), index=0, strokes=strokes)]


def read_scan(path: str | Path, kind: str | None = None) -> list[Page]:
    from skimage.morphology import skeletonize

    path = Path(path)
    kind = kind or path.suffix.lower()
    pages = []
    vector: dict[int, list[Stroke]] = {}
    if kind == ".pdf":
        import pymupdf

        with pymupdf.open(path) as doc:
            if doc.needs_pass:
                raise ValueError(f"{path.name} is password-protected: export an unlocked copy and use that")
            for i, pg in enumerate(doc):
                strokes = vector_strokes(pg)
                if strokes:
                    vector[i] = strokes
            n_pages = len(doc)
        if len(vector) == n_pages:  # every page is ink shapes: no need to render pixels at all
            return [Page(source=str(path), index=i, strokes=st) for i, st in sorted(vector.items())]
    for idx, (rgb, k) in enumerate(_images(path, kind)):
        if idx in vector:
            pages.append(Page(source=str(path), index=idx, strokes=vector[idx]))
            continue
        mask = ink_mask(rgb, k)
        page = Page(source=str(path), index=idx, image=rgb, image_scale=k)
        if 50 < mask.sum() < 0.35 * mask.size and not os.environ.get("INKHABIT_NO_UPSCALE"):
            d0 = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
            pen_px = float(np.percentile(d0[mask], 75) * 2)
            if pen_px < float(os.environ.get("INKHABIT_UPSCALE_BELOW", 7)) and max(rgb.shape[:2]) <= 3200:  # strokes only a few pixels wide thin badly: work at twice the size
                big = cv2.resize(rgb, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                rgb_t, k_t = big, k * 2
                mask = ink_mask(rgb_t, k_t)
            else:
                rgb_t, k_t = rgb, k
        else:
            rgb_t, k_t = rgb, k
        if 50 < mask.sum() < 0.35 * mask.size:  # a mostly-"ink" page is a cover or a photo, not writing
            mask = _tidy(mask)
            dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
            skel = skeletonize(mask)
            width = float(np.median(dist[skel]) * 2) if skel.any() else 2.0
            for line in _prune(_trace(skel), width):
                ij = np.clip(line[:: max(1, len(line) // 9)].astype(int), 0, [rgb_t.shape[1] - 1, rgb_t.shape[0] - 1])
                colour = np.median(rgb_t[ij[:, 1], ij[:, 0]], axis=0) / 255
                page.strokes.append(Stroke(_smooth(line) / k_t, width / k_t, (*colour.tolist(), 1.0), order=-1))
        pages.append(page)
    return pages
