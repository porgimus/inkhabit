# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read pen strokes out of .goodnotes notebook files.

A .goodnotes file is a zip. Each file under notes/ is one page: a run of
length-prefixed protobuf records. Stroke records carry an Apple-LZ4 ("bv41")
blob whose payload is a small typed pack ("tpl"):

    v u A(v) A(S(uu)) A(S(uuuu)) v A(f)
    |  |  |    |         |
    |  |  |    |         quadratic segments: control x,y then end x,y
    |  |  |    start point(s)
    |  |  op per element: 0 = move, 1 = curve
    |  pen width (float32)
    version

This layout was worked out by inspection of Goodnotes 7.1 backups, so the
reader is defensive: anything it does not recognise is skipped, not guessed.
"""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

TPL_STROKE = b"vuA(v)A(S(uu))A(S(uuuu))vA(f)"


from .ink import Page, Stroke


def _varint(b: bytes, i: int) -> tuple[int, int]:
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 127) << s
        s += 7
        if c < 128:
            return r, i


def _pb(b: bytes) -> list[tuple[int, int, Any]]:
    """Minimal protobuf wire decoder: [(field, wiretype, value)]."""
    i, out = 0, []
    while i < len(b):
        k, i = _varint(b, i)
        f, w = k >> 3, k & 7
        if w == 0:
            v, i = _varint(b, i)
        elif w == 1:
            v, i = b[i : i + 8], i + 8
        elif w == 2:
            n, i = _varint(b, i)
            v, i = b[i : i + n], i + n
        elif w == 5:
            v, i = b[i : i + 4], i + 4
        else:
            raise ValueError(f"unsupported wire type {w}")
        if i > len(b):
            raise ValueError("truncated protobuf")
        out.append((f, w, v))
    return out


def _lz4_block(src: bytes) -> bytes:
    dst, i, n = bytearray(), 0, len(src)
    while i < n:
        t = src[i]
        i += 1
        lit = t >> 4
        if lit == 15:
            while True:
                c = src[i]
                i += 1
                lit += c
                if c != 255:
                    break
        dst += src[i : i + lit]
        i += lit
        if i >= n:
            break
        off = src[i] | src[i + 1] << 8
        i += 2
        m = t & 15
        if m == 15:
            while True:
                c = src[i]
                i += 1
                m += c
                if c != 255:
                    break
        m += 4
        st = len(dst) - off
        if off == 0 or st < 0:
            raise ValueError("damaged compressed block")
        if off >= m:
            dst += dst[st : st + m]
        else:
            for k in range(m):
                dst.append(dst[st + k])
    return bytes(dst)


def _bv4(b: bytes) -> bytes:
    """Decode Apple's LZ4 frame format (bv41 / bv4- / bv4$ blocks)."""
    out, i = b"", 0
    while i < len(b):
        magic = b[i : i + 4]
        if magic == b"bv41":
            _usize, csize = struct.unpack_from("<II", b, i + 4)
            out += _lz4_block(b[i + 12 : i + 12 + csize])
            i += 12 + csize
        elif magic == b"bv4-":
            (usize,) = struct.unpack_from("<I", b, i + 4)
            out += b[i + 8 : i + 8 + usize]
            i += 8 + usize
        elif magic == b"bv4$":
            break
        else:
            raise ValueError(f"bad bv4 magic {magic!r}")
    return out


def _quad(p0, c, p1, flat: float = 0.35) -> np.ndarray:
    """Sample a quadratic bezier; more points where the curve bends more."""
    bend = np.linalg.norm(p0 - 2 * c + p1)
    n = int(np.clip(np.ceil(np.sqrt(bend / flat)), 1, 12))
    t = np.linspace(0, 1, n + 1)[1:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t**2 * p1


def _parse_tpl(d: bytes) -> tuple[float, list[np.ndarray]] | None:
    if d[:4] != b"tpl\0":
        return None
    end = d.index(b"\0", 8)
    if d[8:end] != TPL_STROKE:
        return None
    b, o = d, end + 1 + 2  # skip version
    (width,) = struct.unpack_from("<f", b, o)
    o += 4
    (n_ops,) = struct.unpack_from("<I", b, o)
    o += 4
    ops = np.frombuffer(b, "<u2", n_ops, o)
    o += 2 * n_ops
    (n_start,) = struct.unpack_from("<I", b, o)
    o += 4
    starts = np.frombuffer(b, "<f4", 2 * n_start, o).reshape(-1, 2)
    o += 8 * n_start
    (n_seg,) = struct.unpack_from("<I", b, o)
    o += 4
    segs = np.frombuffer(b, "<f4", 4 * n_seg, o).reshape(-1, 4)
    paths, cur, si, qi = [], None, 0, 0
    for op in ops:
        if op == 0:
            if cur is not None:
                paths.append(np.vstack(cur))
            cur = [starts[si][None, :].astype(float)]
            si += 1
        elif op == 1 and cur is not None:
            s = segs[qi].astype(float)
            qi += 1
            cur.append(_quad(cur[-1][-1], s[:2], s[2:]))
        else:
            return None
    if cur is not None:
        paths.append(np.vstack(cur))
    return float(width), paths


def read_notebook(path: str | Path) -> list[Page]:
    path = Path(path)
    pages: list[Page] = []
    with zipfile.ZipFile(path) as z:
        order = _page_order(z)
        for idx, name in enumerate(order):
            raw = z.read(name)
            page = Page(source=str(path), index=idx)
            i = 0
            while i < len(raw):
                try:
                    n, i = _varint(raw, i)
                except IndexError:  # the record list is cut short
                    break
                msg, i = raw[i : i + n], i + n
                try:
                    stroke_parts = _stroke_from_record(msg)
                except (ValueError, IndexError, struct.error):
                    continue
                for pts, width, color in stroke_parts:
                    page.strokes.append(Stroke(pts, width, color, len(page.strokes)))
            pages.append(page)
    return pages


def _page_order(z: zipfile.ZipFile) -> list[str]:
    names = [n for n in z.namelist() if n.startswith("notes/") and z.getinfo(n).file_size]
    try:
        listed = []
        raw, i = z.read("index.notes.pb"), 0
        while i < len(raw):
            n, i = _varint(raw, i)
            for f, w, v in _pb(raw[i : i + n]):
                if f == 2 and w == 2:
                    listed.append(v.decode())
            i += n
        ordered = [n for n in listed if n in names]
        return ordered + [n for n in names if n not in ordered]
    except (KeyError, ValueError, IndexError, UnicodeDecodeError):
        return names


def _stroke_from_record(msg: bytes):
    out = []
    for f, w, v in _pb(msg):
        if f != 7 or w != 2:
            continue
        blob, color = None, (0.0, 0.0, 0.0, 1.0)
        for f2, w2, v2 in _pb(v):
            if f2 == 2 and w2 == 2 and v2[:3] == b"bv4":
                blob = v2
            elif f2 == 4 and w2 == 2:
                c = {k: struct.unpack("<f", x)[0] for k, wt, x in _pb(v2) if wt == 5}
                color = (c.get(1, 0.0), c.get(2, 0.0), c.get(3, 0.0), c.get(4, 1.0))
        if blob is None:
            continue
        parsed = _parse_tpl(_bv4(blob))
        if parsed is None:
            continue
        width, paths = parsed
        for pts in paths:
            if len(pts) >= 1 and np.isfinite(pts).all():
                out.append((pts, width, color))
    return out
