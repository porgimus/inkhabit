# SPDX-License-Identifier: AGPL-3.0-or-later
"""Load ink pages from whatever someone hands over.

Supported: Goodnotes notebooks, PDFs (note-app exports, scans, mixed), pictures (PNG, JPEG, TIFF incl.
multi-page, HEIC, BMP, WebP, GIF; any bit depth, greyscale, CMYK, transparent backgrounds), SVG ink, zip
archives of any of these (including a Goodnotes backup), and folders. Files without an extension are
recognised by their contents.
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

from .ink import Page

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".heic", ".heif", ".bmp", ".webp", ".gif", ".jp2"}
SCAN_EXT = IMAGE_EXT | {".pdf"}
KNOWN = SCAN_EXT | {".goodnotes", ".svg", ".zip"}
_unpacked: dict[str, tempfile.TemporaryDirectory] = {}


class Unsupported(ValueError):
    pass


def kind(path: Path) -> str | None:
    """The file's real type: by extension, or by its first bytes when the extension is missing or unknown."""
    ext = path.suffix.lower()
    if ext in KNOWN:
        return ext
    try:
        head = path.read_bytes()[:16]
    except OSError:
        return None
    if head.startswith(b"%PDF"):
        return ".pdf"
    if head.startswith(b"\x89PNG"):
        return ".png"
    if head.startswith(b"\xff\xd8"):
        return ".jpg"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return ".tif"
    if head.startswith(b"PK"):
        try:
            with zipfile.ZipFile(path) as z:
                return ".goodnotes" if any(n.startswith("notes/") for n in z.namelist()) else ".zip"
        except zipfile.BadZipFile:
            return None
    if head[4:12] in (b"ftypheic", b"ftypheix", b"ftypmif1"):
        return ".heic"
    return None


def load(path: str | Path) -> list[Page]:
    path = Path(path)
    k = kind(path)
    if k == ".goodnotes":
        from .goodnotes import read_notebook

        return read_notebook(path)
    if k == ".svg":
        from .scan import read_svg

        return read_svg(path)
    if k in SCAN_EXT:
        from .scan import read_scan

        return read_scan(path, k)
    raise Unsupported(f"{path.name}: not a kind of file handwriting can be read from. Give Goodnotes notebooks, PDFs, "
                      "pictures (png, jpg, tif, heic, bmp, webp), SVG ink, or a zip or folder of those.")


def discover(root: str | Path, exclude: tuple[str, ...] = ()) -> list[Path]:
    """Every readable source under a file, folder or zip. Zips are unpacked to a temporary folder."""
    root = Path(root)
    if not root.exists():
        raise Unsupported(f"{root}: no such file or folder")
    found: list[Path] = []

    def visit(p: Path, top: bool):
        if p.is_dir():
            for q in sorted(p.rglob("*")):
                if q.is_file() and not q.name.startswith(".") and "__MACOSX" not in q.parts:
                    visit(q, False)
            return
        k = kind(p) if (top or p.suffix.lower() in KNOWN or p.suffix == "") else None
        if k == ".zip":
            tmp = _unpacked.setdefault(str(p.resolve()), tempfile.TemporaryDirectory(prefix="inkhabit-"))
            dest = Path(tmp.name)
            if not any(dest.iterdir()):
                with zipfile.ZipFile(p) as z:
                    for info in z.infolist():  # never write outside the temporary folder
                        target = (dest / info.filename).resolve()
                        if target.is_relative_to(dest.resolve()) and target != dest.resolve() and not info.is_dir():
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_bytes(z.read(info))
            visit(dest, False)
        elif k:
            found.append(p)
        elif top:
            raise Unsupported(f"{p.name}: not a kind of file handwriting can be read from. Give Goodnotes notebooks, PDFs, "
                              "pictures (png, jpg, tif, heic, bmp, webp), SVG ink, or a zip or folder of those.")

    visit(root, True)
    return [p for p in found if not any(x.lower() in str(p).lower() for x in exclude)]
