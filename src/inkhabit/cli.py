# SPDX-License-Identifier: AGPL-3.0-or-later
"""inkhabit: your handwriting as a font, and as a pen an agent can write with."""

from __future__ import annotations

import argparse
import json

import numpy as np
from pathlib import Path


def _drops(spec: list[str]) -> dict[str, list[int]]:
    out = {}
    for item in spec:  # "K:0,2" ; the character is everything before the last colon
        ch, _, idx = item.rpartition(":")
        if ch.upper().startswith("U+"):  # e.g. U+002D:4 for the hyphen, which a shell would take as an option
            ch = chr(int(ch[2:], 16))
        out[ch] = [int(i) for i in idx.split(",") if i != ""]
    return out


def _human(level: float | None, settings: list[str], base):
    h = base
    if level is not None:
        h = h.scaled(level)
    try:
        return h.with_settings(dict(item.split("=", 1) for item in settings))
    except ValueError as e:
        raise SystemExit(f"--set takes name=number, e.g. --set word_spacing=0.8 ({e})") from None


SET_HELP = ("fine control, repeatable: letter_spacing word_spacing line_spacing letter_height letter_width slant weight "
            "(1.0 = as you write; slant in degrees) and any variation setting, e.g. --set word_spacing=0.8 --set slant=5")


def _text(arg: str) -> str:
    """The text to write: "@file" reads a file; anything else (including "@Sam, see you at 5") is the text."""
    if arg.startswith("@") and Path(arg[1:]).is_file():
        return Path(arg[1:]).read_text()
    return arg.replace("\\n", "\n")


def _hex(colour: str) -> tuple[int, int, int]:
    h = colour.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        if len(h) != 6:
            raise ValueError
        return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        raise SystemExit(f"--color takes a hex colour such as #1a1a2e, not {colour!r}") from None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="inkhabit", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="sources (.goodnotes, scanned .pdf, images) -> library + font")
    b.add_argument("inputs", nargs="+")
    b.add_argument("-o", "--out", default="inkhabit-out")
    b.add_argument("--exclude", action="append", default=[], help="skip paths containing this text (repeatable)")
    b.add_argument("--family", help="name of the font; default: the name it was built with")
    b.add_argument("--vocab")
    b.add_argument("--transcripts", metavar="DIR",
                   help="folder of corrected transcripts (<source name>.md, from `inkhabit transcript`) to use as the labels")
    b.add_argument("--ink", choices=["lighter", "darker"],
                   help="on filled-in printed pages, keep only the lighter (pencil) or darker ink and ignore the print")
    b.add_argument("--lang", help='language of the handwriting, e.g. "es-ES", "fr-FR" (default: English)')
    b.add_argument("--force", action="store_true", help="build even when there is too little handwriting")

    pr = sub.add_parser("profile", help="redo profile.md for an existing build, keeping its reviewed library")
    pr.add_argument("inputs", nargs="+")
    pr.add_argument("-o", "--out", required=True)
    pr.add_argument("--exclude", action="append", default=[])
    pr.add_argument("--family", help="name of the font; default: the name it was built with")

    r = sub.add_parser("review", help="drop bad variants seen on specimen.png, then rebuild the font")
    r.add_argument("out")
    r.add_argument("--drop", nargs="+", default=[], metavar="CHAR:IDX,IDX")
    r.add_argument("--word-gap", type=float, help="space between words in x-heights, for the font and the writer")
    r.add_argument("--drop-pair", nargs="+", default=[], metavar="PAIR:IDX,IDX", help="same, for pairs.png")
    r.add_argument("--family", help="name of the font; default: the name it was built with")

    rp = sub.add_parser("review-page", help="open a local page to drop bad samples by clicking them")
    rp.add_argument("out")
    rp.add_argument("--family", help="name of the font; default: the name it was built with")
    rp.add_argument("--port", type=int, default=8765)

    sh = sub.add_parser("sheet", help="make a practice sheet for characters the library is short of")
    sh.add_argument("out", help="the build folder")
    sh.add_argument("-o", "--pdf", help="where to write the sheet (default: <out>/practice-sheet.pdf)")
    sh.add_argument("--chars", help="characters to include (default: the missing and thin ones)")
    sh.add_argument("--reps", type=int, default=6)

    sb = sub.add_parser("sheet-read", help="add a filled-in practice sheet to the library")
    sb.add_argument("out", help="the build folder")
    sb.add_argument("filled", help="the sheet with writing on it (.goodnotes or exported .pdf)")
    sb.add_argument("--sheet", help="the sheet's .json (default: <out>/practice-sheet.pdf.json)")
    sb.add_argument("--family", help="name of the font; default: the name it was built with")

    jd = sub.add_parser("judge", help="optional: have TypeSafe's Jev model rank samples by clarity (sends letter images off this machine)")
    jd.add_argument("out")
    jd.add_argument("sources", nargs="*", help="give the sources too, to re-pick weak letters from them")
    jd.add_argument("--family", help="name of the font; default: the name it was built with")
    jd.add_argument("--model", default="jev-latest")
    jd.add_argument("--prune", nargs="?", type=float, const=0.3, metavar="MIN_CLARITY",
                    help="also drop samples judged unclear (default threshold 0.3): an automatic first review, best for scans")

    pg2 = sub.add_parser("playground", help="open a local page to type text and try settings live")
    pg2.add_argument("out")
    pg2.add_argument("--port", type=int, default=8766)

    tn = sub.add_parser("tune", help="drop samples that make words unreadable, measured by reading the writing back")
    tn.add_argument("out")
    tn.add_argument("--family", help="name of the font; default: the name it was built with")

    w = sub.add_parser("write", help="render text in the handwriting (.png, .svg or .pdf)")
    w.add_argument("library")
    w.add_argument("text", help="the text, or @file.txt")
    w.add_argument("-o", "--out", default="written.png")
    w.add_argument("--seed", type=int, default=None)
    w.add_argument("--x-height", type=float, default=14)
    w.add_argument("--width", type=float, default=900, help="wrap width, same units as x-height")
    w.add_argument("--human", type=float, default=None, help="0 = mechanical, 1 = natural, 2 = messy")
    w.add_argument("--style", choices=["careful", "clean", "neat", "natural", "rushed"], default="clean",
                   help="careful: their best, for forms; clean (default); natural: as the notes are written; rushed")
    w.add_argument("--for", dest="situation", choices=["text", "form", "official", "heading", "label", "letter", "note", "scribble"],
                   help="what the writing is for; sets care, joins and size together (overrides --style)")
    w.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help=SET_HELP)
    w.add_argument("--line-gap", type=float, default=3.2, help="distance between lines, in x-heights")
    w.add_argument("--justify", action="store_true", help="spread each full line to the wrap width")
    w.add_argument("--scale", type=float, default=1.0, help="multiply the picture's pixel size, e.g. 3 for print (PNG only)")
    w.add_argument("--color", default="#1a1a2e")

    t = sub.add_parser("typeset", help="set text with the .ttf itself (what typing in an app gives)")
    t.add_argument("ttf")
    t.add_argument("text", help="the text, or @file.txt")
    t.add_argument("-o", "--out", default="typeset.png")
    t.add_argument("--size", type=float, default=48)

    pg = sub.add_parser("pages", help="render source pages to PNG in their own colours, to look at the notes")
    pg.add_argument("source")
    pg.add_argument("-o", "--out", default="pages")
    pg.add_argument("--max", type=int, default=6)

    rd = sub.add_parser("transcript", help="draft transcript of a source, to correct by eye and give to `build --transcripts`")
    rd.add_argument("source")
    rd.add_argument("-o", "--out", help="output .md (default: print)")
    rd.add_argument("--lang", help='language of the handwriting, e.g. "es-ES"')
    rd.add_argument("--library", help="a inkhabit library.json; primes recognition with the writer's vocabulary")

    dg = sub.add_parser("diagrams", help="find the drawings in someone's notes (graphs, flow charts, sketches) and save them, numbered")
    dg.add_argument("inputs", nargs="+")
    dg.add_argument("-o", "--out", required=True, help="the build folder: writes diagrams.json and diagrams/NNN.png")
    dg.add_argument("--library", help="library.json, to prime label reading with their vocabulary (default: <out>/library.json if there)")
    dg.add_argument("--exclude", action="append", default=[])

    dr = sub.add_parser("diagram", help="draw one of their saved diagrams again, optionally with new labels, size or colours")
    dr.add_argument("out_dir", help="the folder `inkhabit diagrams` wrote to")
    dr.add_argument("id", type=int)
    dr.add_argument("-o", "--out", default="diagram.png", help=".png, .svg or .pdf")
    dr.add_argument("--label", action="append", default=[], metavar="N=TEXT", help="rewrite label N (the red number in its picture) in their handwriting")
    dr.add_argument("--width", type=float, help="new overall width, in the source's units (default: as drawn)")
    dr.add_argument("--vary", type=float, default=1.0, help="0 = exact copy, 1 = a natural second attempt, 2 = looser")
    dr.add_argument("--colour", action="append", default=[], metavar="NAME", choices=["blue", "black", "red", "green", "pencil"], help="one ink colour for everything: blue, black, red, green, pencil")
    dr.add_argument("--seed", type=int, default=0)
    dr.add_argument("--library", help="library.json for rewriting labels (default: <out_dir>/library.json)")

    an = sub.add_parser("annotate", help="mark up a PDF by hand: underline, circle, arrows, notes in the margin")
    an.add_argument("library")
    an.add_argument("pdf")
    an.add_argument("items", help="JSON file: [{type, find|rect, ...}]")
    an.add_argument("-o", "--out", default="annotated.pdf")
    an.add_argument("--seed", type=int, default=0)
    an.add_argument("--colour", default="blue", choices=["blue", "black", "red", "green", "pencil"], help="blue, black, red, green, pencil")

    nt = sub.add_parser("notes", help="write an outline as a page of notes in the person's own note-taking style")
    nt.add_argument("library")
    nt.add_argument("outline", help="outline text file (syntax: references/notes-and-maths.md in the skill), or @- for stdin")
    nt.add_argument("-o", "--out", default="notes.pdf")
    nt.add_argument("--paper", choices=["dark", "white", "ruled"], default="white")
    nt.add_argument("--profile", help="profile.json from build (default: next to the library)")
    nt.add_argument("--colours", help="comma-separated ink colours to use instead of the profile's, e.g. blue,pink,black")
    nt.add_argument("--seed", type=int, default=0)
    nt.add_argument("--x-height", type=float, help="default: as in their notes (from the profile)")
    nt.add_argument("--line-pitch", type=float, help="default: as in their notes")

    mt = sub.add_parser("math", help="write maths: fractions, powers, roots, integrals, worked steps aligned on =")
    mt.add_argument("library")
    mt.add_argument("expr", help="LaTeX-like text; separate lines of working with \\\\ or newlines; or @file")
    mt.add_argument("-o", "--out", default="maths.png")
    mt.add_argument("--seed", type=int, default=0)
    mt.add_argument("--x-height", type=float, default=22)
    mt.add_argument("--style", choices=["careful", "clean", "neat", "natural", "rushed"], default="careful")
    mt.add_argument("--color", default="#1a1a2e")

    lk = sub.add_parser("render", help="turn the pages of any PDF into PNG images, to look at them")
    lk.add_argument("pdf")
    lk.add_argument("-o", "--out", default="pages")
    lk.add_argument("--dpi", type=int, default=130)

    f = sub.add_parser("fields", help="list form fields, blank lines and labels on a PDF")
    f.add_argument("pdf")

    fl = sub.add_parser("fill", help="write entries onto a PDF")
    fl.add_argument("library")
    fl.add_argument("pdf")
    fl.add_argument("entries", help="JSON file: [{page, rect, text} | {field, text}]")
    fl.add_argument("-o", "--out", default="filled.pdf")
    fl.add_argument("--seed", type=int, default=0)
    fl.add_argument("--human", type=float, default=None, help="scale irregularity: 0 mechanical .. 2 messy")
    fl.add_argument("--style", choices=["careful", "clean", "neat", "natural", "rushed"], help="default: careful")
    fl.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help=SET_HELP)

    a = ap.parse_args(argv)
    if a.cmd == "build":
        from . import pipeline

        old = Path(a.out) / "library.json"
        if old.exists():  # a build replaces the library, and with it any review: keep the old one
            (Path(a.out) / "library.before-build.json").write_bytes(old.read_bytes())
            print(f"{old} already existed; kept as library.before-build.json")
        pipeline.build(a.inputs, a.out, tuple(a.exclude), a.family, a.vocab, force=a.force, transcripts=a.transcripts, lang=a.lang, ink=a.ink)
    elif a.cmd == "profile":
        from . import pipeline

        pipeline.profile_only(a.inputs, a.out, tuple(a.exclude), family=a.family)
    elif a.cmd == "review":
        from . import library, pipeline

        src = Path(a.out) / "library.json"
        (Path(a.out) / "library.before-review.json").write_bytes(src.read_bytes())  # one step back, if a drop was wrong
        lib = library.drop_variants(library.load(src), _drops(a.drop), _drops(a.drop_pair))
        if lib.get("filled"):  # a thin library: letters just rejected get filled in again from what is left
            from . import extend

            lib = extend.fill(lib)
            f = lib["filled"]
            print(f"real: {f['real']} | turned or flipped: {f['derived'] or 'none'} | neutral stand-ins: {f['neutral'] or 'none'}")
        if a.word_gap is not None:
            lib["meta"]["word_gap_use"] = a.word_gap
        pipeline.finish(lib, Path(a.out), a.family)
    elif a.cmd == "review-page":
        from . import reviewpage

        reviewpage.serve(a.out, a.family, a.port)
    elif a.cmd == "sheet":
        from . import library, sheet

        lib = library.load(Path(a.out) / "library.json")
        chars = a.chars or sheet.wanted(lib)
        pdf = Path(a.pdf or Path(a.out) / "practice-sheet.pdf")
        pages = sheet.make(chars, pdf, a.reps)
        print(f"wrote {pdf}: {pages} page(s) for {len(chars)} characters: {chars}")
    elif a.cmd == "sheet-read":
        from . import library, pipeline, sheet

        out = Path(a.out)
        lib = library.load(out / "library.json")
        library.save(lib, out / "library.before-sheet.json")
        sheet.read_back(a.filled, a.sheet or out / "practice-sheet.pdf.json", lib)
        library.unify(lib)
        pipeline.finish(lib, out, a.family)
    elif a.cmd == "judge":
        from . import judge, library, pipeline

        out = Path(a.out)
        lib = library.load(out / "library.json")
        library.save(lib, out / "library.before-judge.json")
        report = {"scored": judge.run(lib, a.model)}
        if a.sources:
            from concurrent.futures import ProcessPoolExecutor

            from . import segment, sources

            weak = "".join(ch for ch, e in lib["glyphs"].items() if (ch.islower() or ch.isdigit()) and e["variants"]
                           and (np.mean([v.get("clarity", 1.0) for v in e["variants"]]) < 0.5 or len(e["variants"]) < 6))
            samples = []
            with ProcessPoolExecutor(4) as ex:
                for _, ws in ex.map(pipeline._read, [f for s in a.sources for f in sources.discover(s)]):
                    samples += ws
            aligned, _ = segment.run(samples, pipeline.load_vocab(), rounds=1, log=lambda *x: None,
                                     seed_models=segment.models_from_library(lib))
            report["reselected"] = judge.reselect(lib, aligned, weak, model=a.model)
        if a.prune is not None:
            dropped = judge.prune(lib, a.prune)
            report["pruned"] = dropped
            print(f"pruned {sum(dropped.values())} unclear samples from {len(dropped)} characters")
            if lib.get("filled") is not None or any(c not in lib["glyphs"] for c in "abcdefghijklmnopqrstuvwxyz"):
                from . import extend

                lib = extend.fill(lib)
        (out / "judge.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
        pipeline.finish(lib, out, a.family)
    elif a.cmd == "playground":
        from . import playground

        playground.serve(a.out, a.port)
    elif a.cmd == "tune":
        from . import library, pipeline, tune

        out = Path(a.out)
        lib = library.load(out / "library.json")
        library.save(lib, out / "library.before-tune.json")
        report = tune.run(lib)
        (out / "tune.json").write_text(json.dumps(report, indent=1))
        pipeline.finish(lib, out, a.family)
    elif a.cmd == "write":
        from . import library, render

        text = _text(a.text)
        lib = library.load(a.library)
        base, size = render.situation(lib, a.situation) if a.situation else (render.style(lib, a.style), 1.0)
        writer = render.Writer(lib, a.seed, _human(a.human, a.set, base))
        ink = writer.write(text, a.x_height * size, a.width, line_gap=a.line_gap, justify=a.justify)
        rgb = _hex(a.color)
        out = Path(a.out)
        if out.suffix == ".svg":
            out.write_text(render.to_svg(ink, a.color))
        elif out.suffix == ".pdf":
            import pymupdf

            doc = pymupdf.open()
            page = doc.new_page(width=ink.size[0] + 72, height=ink.size[1] + 72)
            render.onto_pdf_page(page, ink, (36, 36), tuple(c / 255 for c in rgb))
            doc.save(out)
        else:
            render.to_image(ink, rgb, scale=a.scale).save(out)
        unit = "px" if out.suffix not in (".svg", ".pdf") else "pt"
        print(f"wrote {out}: {ink.size[0] * (a.scale if unit == 'px' else 1):.0f} x {ink.size[1] * (a.scale if unit == 'px' else 1):.0f} {unit}"
              + (f"; no glyph for: {''.join(sorted(ink.missing))}" if ink.missing else "; every character had a shape"))
    elif a.cmd == "typeset":
        from . import font

        text = _text(a.text)
        font.typeset(a.ttf, text, a.out, a.size)
        print(f"wrote {a.out}")
    elif a.cmd == "pages":
        from . import sources
        from .ink import render_colour

        Path(a.out).mkdir(parents=True, exist_ok=True)
        for p in sources.load(a.source)[: a.max]:
            dest = Path(a.out) / f"{Path(a.source).stem}-{p.index + 1}.png"
            if p.image is not None:  # a scan or photo: show the page itself, which is what a person would read
                from PIL import Image

                im = Image.fromarray(p.image)
                im.thumbnail((2000, 2000))
                im.save(dest)
                traced = dest.with_name(dest.stem + "-traced.png")
                render_colour(p, 3.0).save(traced)
                print(f"wrote {dest} (the page) and {traced.name} (the pen paths recovered from it)")
            else:
                render_colour(p).save(dest)
                print(f"wrote {dest}")
    elif a.cmd == "transcript":
        from . import library, transcribe

        text, rep = transcribe.transcribe(a.source, library.load(a.library) if a.library else None, lang=a.lang)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(text)
            print(f"wrote {a.out}: {rep['lines']} lines, {rep['unsure_lines']} flagged [?] as unsure")
        else:
            print(text)
    elif a.cmd == "diagrams":
        from . import diagrams, library, sources
        from .words import page_words

        lib_path = Path(a.library) if a.library else Path(a.out) / "library.json"
        custom = sorted(library.load(lib_path).get("words", {})) if lib_path.exists() else None
        found, n_pages = [], 0
        for path in [p for root in a.inputs for p in sources.discover(root, tuple(a.exclude))]:
            for page in sources.load(path):
                n_pages += 1
                found += diagrams.find(page, page_words(page, custom_words=custom))
                page.image = None
        rep = diagrams.catalogue(found, a.out)
        print(f"{rep['diagrams']} drawings found on {n_pages} pages; pictures in {a.out}/diagrams/ (red numbers are label numbers)")
    elif a.cmd == "diagram":
        from . import annotate, diagrams, library

        found = json.loads((Path(a.out_dir) / "diagrams.json").read_text())
        if not 0 <= a.id < len(found):
            raise SystemExit(f"no diagram {a.id}: there are {len(found)} (0 to {len(found) - 1})")
        labels = dict(item.split("=", 1) for item in a.label)
        lib_path = Path(a.library) if a.library else Path(a.out_dir) / "library.json"
        lib = library.load(lib_path) if labels and lib_path.exists() else None
        colours = {"all": list(annotate.COLOURS[a.colour[0]])} if a.colour else None
        drawing = diagrams.redraw(found[a.id], lib, labels, a.seed, a.vary, a.width, colours)
        diagrams.save(drawing, a.out)
        print(f"wrote {a.out}" + (f"; no shape for: {' '.join(drawing['missing'])}" if drawing["missing"] else "")
              + (f"; written smaller to fit: {', '.join(drawing['tight'])}" if drawing["tight"] else ""))
    elif a.cmd == "annotate":
        from . import annotate, library

        rep = annotate.annotate(a.pdf, annotate.load_items(a.items), library.load(a.library), a.out, a.seed, a.colour)
        print(json.dumps(rep, indent=1, ensure_ascii=False))
        print(f"wrote {a.out}")
    elif a.cmd == "notes":
        import sys

        from . import library, notes

        text = sys.stdin.read() if a.outline == "@-" else Path(a.outline).read_text()
        prof = notes.load_profile(a.profile or Path(a.library).parent / "profile.json")
        cat = Path(a.library).parent / "diagrams.json"
        rep = notes.write_notes(text, library.load(a.library), a.out, prof, a.paper, a.seed, x_height=a.x_height,
                                diagrams=json.loads(cat.read_text()) if cat.exists() else None,
                                line_pitch=a.line_pitch, colours=a.colours.split(",") if a.colours else None)
        print(f"page {rep['page_size'][0]:.0f} x {rep['page_size'][1]:.0f} pt, writing {rep['x_height']} pt high, lines {rep['line_pitch']} pt apart "
              + ("(from their profile)" if rep["from_profile"] else "(defaults: no profile.json beside the library)"))
        print(f"wrote {a.out}: {rep['pages']} page(s), {rep['lines']} lines, inks: {', '.join(rep['colours'])}"
              + (f"; no shape for: {rep['missing']}" if rep["missing"] else ""))
    elif a.cmd == "math":
        from . import library, mathwrite, render

        lib = library.load(a.library)
        text = Path(a.expr[1:]).read_text() if a.expr.startswith("@") else a.expr
        ink = mathwrite.write_maths(lib, text, a.x_height, a.seed, render.style(lib, a.style))
        rgb = _hex(a.color)
        out = Path(a.out)
        if out.suffix == ".svg":
            out.write_text(render.to_svg(ink, a.color))
        elif out.suffix == ".pdf":
            import pymupdf

            doc = pymupdf.open()
            page = doc.new_page(width=ink.size[0] + 72, height=ink.size[1] + 72)
            render.onto_pdf_page(page, ink, (36, 36), tuple(c / 255 for c in rgb))
            doc.save(out)
        else:
            render.to_image(ink, rgb).save(out)
        print(f"wrote {out}" + (f"; no shape for: {''.join(sorted(ink.missing))}" if ink.missing else ""))
    elif a.cmd == "render":
        import pymupdf

        from .forms import open_doc

        Path(a.out).mkdir(parents=True, exist_ok=True)
        with open_doc(a.pdf) as doc:
            for i, page in enumerate(doc):
                dest = Path(a.out) / f"{Path(a.pdf).stem}-{i + 1}.png"
                page.get_pixmap(dpi=a.dpi).save(dest)
                print(f"wrote {dest}  (1 PDF point = {a.dpi / 72:.3f} px; page is {page.rect.width:.0f} x {page.rect.height:.0f} pt)")
    elif a.cmd == "fields":
        from . import forms

        print(json.dumps(forms.fields(a.pdf), indent=1))
    elif a.cmd == "fill":
        from . import forms, library, render

        lib = library.load(a.library)
        rep = forms.fill(a.pdf, forms.load_entries(a.entries), lib, a.out, a.seed,
                         human=_human(a.human, a.set, render.style(lib, a.style) if a.style else render.form_preset(lib)))
        print(json.dumps(rep, indent=1))
        print(f"wrote {a.out}")
