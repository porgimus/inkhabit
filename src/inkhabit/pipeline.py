# SPDX-License-Identifier: AGPL-3.0-or-later
"""End to end: source files -> library, font, specimen, profile."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import extend, font, library, profile, segment, sources
from .ink import Page
from .render import Writer, to_image
from .words import WordSample, page_words

PANGRAM = "The quick brown fox jumps over the lazy dog.\nPack my box with five dozen liquor jugs, 0123456789."


TRANSCRIPTS: Path | None = None  # folder of corrected transcripts, <source name>.md
LANG: str | None = None
INK: str | None = None  # "lighter" / "darker": keep only the ink on that side of the page's two ink shades


def _keep_ink(page: Page, side: str) -> None:
    """A filled-in printed form has two inks: the print and the pen. Keep the pen.

    Splits the page's strokes into a darker and a lighter group (Otsu on brightness, weighted by stroke
    length) and keeps one side, e.g. "lighter" for pencil answers on a black printed exam.
    """
    strokes = [s for s in page.strokes if len(s.points) > 1]
    if len(strokes) < 20:
        return
    lum = np.array([0.299 * s.color[0] + 0.587 * s.color[1] + 0.114 * s.color[2] for s in strokes])
    wt = np.array([np.linalg.norm(np.diff(s.points, axis=0), axis=1).sum() for s in strokes])
    grid = np.linspace(lum.min(), lum.max(), 64)[1:-1]
    best, cut = -1.0, None
    for t in grid:
        lo, hi = lum <= t, lum > t
        if wt[lo].sum() == 0 or wt[hi].sum() == 0:
            continue
        m0, m1 = np.average(lum[lo], weights=wt[lo]), np.average(lum[hi], weights=wt[hi])
        score = wt[lo].sum() * wt[hi].sum() * (m0 - m1) ** 2
        if score > best:
            best, cut = score, t
    if cut is None:
        return
    lo, hi = lum <= cut, lum > cut
    apart = np.average(lum[hi], weights=wt[hi]) - np.average(lum[lo], weights=wt[lo])
    if apart < 0.2 or min(wt[lo].sum(), wt[hi].sum()) < 0.12 * wt.sum():
        return  # only one kind of ink on this page (an ordinary page of notes): leave it alone
    keep = {id(s) for s, v in zip(strokes, lum) if (v > cut) == (side == "lighter")}
    page.strokes = [s for s in page.strokes if id(s) in keep]


def _read(path: Path) -> tuple[list[Page], list[WordSample]]:
    try:
        pages = sources.load(path)
        labels = {}
        if TRANSCRIPTS and (TRANSCRIPTS / f"{path.stem}.md").exists():
            from .transcribe import parse_transcript

            labels = parse_transcript((TRANSCRIPTS / f"{path.stem}.md").read_text())
        words = []
        for p in pages:
            if INK:
                _keep_ink(p, INK)
            ws = page_words(p, lang=LANG)
            if p.index in labels:
                from .words import apply_transcript

                n = apply_transcript(ws, labels[p.index])
                print(f"{path.name} page {p.index + 1}: {n} of {len(ws)} words labelled from the transcript")
                ws = [w for w in ws if w.trusted]  # with a transcript, anything it does not cover is left out
            words += ws
            p.image = None  # not needed again, and too large to pass between processes
    except Exception as e:  # one unreadable file should not end the whole build
        print(f"skipped {path.name}: {e}")
        return [], []
    return pages, words


def load_vocab(path: str | None = None) -> set[str]:
    for cand in [path, "/usr/share/dict/words", str(Path(__file__).parent / "words.txt")]:
        if cand and Path(cand).exists():
            return {w.strip().lower() for w in Path(cand).read_text().splitlines() if w.strip()}
    raise FileNotFoundError("no word list found; pass --vocab FILE (one word per line)")


def word_gap(samples: list[WordSample], xh: float) -> float:
    lines: dict[tuple, list[WordSample]] = {}
    for s in samples:
        lines.setdefault((s.source, s.page, s.line), []).append(s)
    gaps = []
    for ws in lines.values():
        ws.sort(key=lambda w: w.box[0])
        ext = [np.vstack(w.strokes)[:, 0] for w in ws]  # OCR boxes are padded; measure the ink itself
        gaps += [(b.min() - a.max()) / xh for a, b in zip(ext[:-1], ext[1:])]
    g = np.array([v for v in gaps if 0.2 < v < 5])
    return float(np.median(g)) if len(g) > 10 else 1.2


def build(inputs: list[str], out: str, exclude: tuple[str, ...] = (), family: str = "My Handwriting",
          vocab: str | None = None, workers: int = 4, log=print, force: bool = False,
          transcripts: str | None = None, lang: str | None = None, ink: str | None = None) -> dict:
    global TRANSCRIPTS, LANG, INK
    TRANSCRIPTS, LANG, INK = (Path(transcripts) if transcripts else None), lang, ink
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    files = [f for i in inputs for f in sources.discover(i, exclude)]
    if not files:
        raise SystemExit("no supported files found (.goodnotes, .pdf, .png, .jpg, .tif)")
    log(f"reading {len(files)} files")
    pages, samples = [], []
    if TRANSCRIPTS or LANG or INK:  # these settings live in this process, so read here instead of in workers
        for pg, ws in map(_read, files):
            pages += pg
            samples += ws
    else:
        with ProcessPoolExecutor(workers) as ex:
            for pg, ws in ex.map(_read, files):
                pages += pg
                samples += ws
    per_file: dict[str, int] = {}
    for s in samples:
        per_file[s.source] = per_file.get(s.source, 0) + 1
    nothing = [f.name for f in files if per_file.get(str(f), 0) == 0]
    if nothing:
        log(f"no handwriting found in {len(nothing)} of {len(files)} files: {', '.join(nothing[:8])}{' ...' if len(nothing) > 8 else ''}")
    slant = segment.estimate_slant(samples)
    if abs(slant) >= 4 and not os.environ.get("INKHABIT_NO_DESLANT"):  # leaning writing is stood upright for cutting; the lean is stored and put back when writing
        samples = segment.deslant(samples, slant)
        log(f"writing leans {slant:.0f} degrees; stood upright for cutting")
    else:
        slant = 0.0
    log(f"{len(pages)} pages, {sum(len(p.strokes) for p in pages):,} strokes, {len(samples):,} words read")
    # the bundled word list is English; for another language pass --vocab, or every word is accepted
    words = load_vocab(vocab) if (vocab or not lang or lang.lower().startswith("en")) else None
    aligned, meta = segment.run(samples, words, log=log)
    usable = [a for a in aligned if len(a.prep.text) >= 2]
    if len(usable) < 30:
        raise SystemExit(f"only {len(usable)} words could be read and split into letters; that is too few to build "
                         "a font. Add more pages of handwriting (roughly 20+ pages works well).")
    pens = [s.width for p in pages for s in p.strokes if s.width <= 4] or [s.width for p in pages for s in p.strokes]
    meta["slant"] = round(slant, 1)
    meta["pen_ratio"] = round(float(np.median(pens) / meta["xheight"]), 3)
    q1, q3 = np.percentile(pens, [25, 75])  # how much the pen's thickness really varies from stroke to stroke
    meta["pen_var"] = round(float((q3 - q1) / 1.349 / max(np.median(pens), 1e-6)), 3)
    # as thick as their pen really is relative to their letters, up to the point where loops start to fill in
    meta["font_weight"] = round(float(np.clip(meta["pen_ratio"], 0.11, 0.20)), 3)
    meta["word_gap"] = round(word_gap(samples, meta["xheight"]), 3)
    # Words labelled from a corrected transcript are known to be right, so a looser fit is acceptable;
    # recogniser-only labels need the strict fit to keep misread words out.
    labelled = sum(s.trusted for s in samples)
    lib = library.build(aligned, meta, word_cost=6.0 if labelled > 0.5 * max(1, len(aligned)) else 2.5)
    solid = [c for c, g in lib["glyphs"].items() if c.islower() and g["count"] >= extend.MIN_REAL]
    if len(solid) < 8 and not force:
        raise SystemExit(
            f"Not enough handwriting yet: {len(usable)} words were read cleanly, giving only {len(solid)} of 26 small "
            f"letters with usable samples ({''.join(sorted(solid)) or 'none'}). Filling in the rest would give a "
            "generic font, not this person's. Add more pages, or pass --force to build anyway.")
    if len(solid) < 22:  # some real letters, but not an alphabet: fill the gaps from what there is
        lib = extend.fill(lib)
        f = lib["filled"]
        log(f"thin sample: {len(f['real'])} characters are real handwriting ({f['real']}); "
            f"{len(f['derived'])} turned or flipped from real letters ({f['derived'] or 'none'}); "
            f"{len(f['neutral'])} are a neutral shape fitted to this writer ({f['neutral'] or 'none'})")
    lib["pair_gap"] = library.pair_gaps(aligned)
    real = library.raw_spread(aligned)
    if real:
        lib["measured"]["real"] = {k: round(float(v), 4) for k, v in real.items()}  # what `tune` calibrates against
    sizes, _ = segment.size_map(samples)  # how much whole words differ in size within one line
    if labelled:
        n = library.add_whole_words(lib, samples, sizes)
        log(f"{n} whole words kept exactly as written (from the transcript)")
    own = {i: v for i, s in enumerate(samples) if (v := segment.word_xheight(s))}
    ratio = np.array([own[i] / sizes[i] for i in own])
    if len(ratio) > 50:
        q1, q3 = np.percentile(ratio, [25, 75])
        lib["measured"]["word_size"] = round(float(np.clip((q3 - q1) / 1.349, 0.02, 0.15)), 3)
    write_profile(pages, samples, lib, words, outdir)
    return finish(lib, outdir, family, log)


def write_profile(pages, samples, lib: dict, words: set[str], outdir: Path, harvest: bool = True) -> None:
    from . import extend

    if harvest:  # not when redoing a profile: the symbols in a reviewed library have been checked and must stay as they are
        sizes, _ = segment.size_map(samples)
        got = extend.harvest(samples, sizes, lib)  # the writer's own stand-alone symbols (= + > ...)
        lib["glyphs"].update(got)
        lib["glyphs"] = dict(sorted(lib["glyphs"].items()))
    prof = profile.build(pages, samples, lib)
    prof["habits"] = profile.habits(pages, samples, lib, words)
    if "line_slope_spread" in prof["habits"]:  # write with this writer's own line tilt
        lib["measured"]["slope"] = float(np.clip(prof["habits"]["line_slope_spread"], 0.1, 2.0))
        lib["measured"]["slope_bias"] = float(np.clip(prof["habits"]["line_slope_degrees"], -2.0, 2.0))
    (outdir / "profile.json").write_text(json.dumps(prof, indent=1))
    (outdir / "profile.md").write_text(profile.to_markdown(prof) + "\n" + profile.habits_markdown(prof["habits"]))


def profile_only(inputs: list[str], out: str, exclude: tuple[str, ...] = (), vocab: str | None = None,
                 family: str = "My Handwriting", workers: int = 4, log=print) -> None:
    """Redo the profile for an existing, reviewed library without rebuilding (and so without losing the review)."""
    outdir = Path(out)
    lib = library.load(outdir / "library.json")
    files = [f for i in inputs for f in sources.discover(i, exclude)]
    pages, samples = [], []
    with ProcessPoolExecutor(workers) as ex:
        for pg, ws in ex.map(_read, files):
            pages += pg
            samples += ws
    log(f"{len(pages)} pages, {len(samples):,} words read")
    slant = lib.get("meta", {}).get("slant") or 0.0
    if slant:  # the library's letters were cut from writing stood upright: measure against the same
        samples = segment.deslant(samples, slant)
    words = load_vocab(vocab)
    # re-cut the words using the library's own letters as the model, to measure spacing against the sources
    aligned, _ = segment.run(samples, words, rounds=1, log=lambda *a: None, seed_models=segment.models_from_library(lib))
    lib["pair_gap"] = library.pair_gaps(aligned)
    real = library.raw_spread(aligned)
    if real:
        lib.setdefault("measured", {})["real"] = {k: round(float(v), 4) for k, v in real.items()}
    log(f"spacing measured for {len(lib['pair_gap'])} letter pairs")
    write_profile(pages, samples, lib, words, outdir, harvest=False)
    finish(lib, outdir, family, log)


def finish(lib: dict, outdir: Path, family: str | None = None, log=print) -> dict:
    """(Re)write everything derived from the library: json, specimen, font, sample."""
    family = family or lib.get("meta", {}).get("family") or "My Handwriting"  # keep the name it was built with
    lib.setdefault("meta", {})["family"] = family
    if not lib.get("meta", {}).get("unified"):
        library.unify(lib, log)
    library.save(lib, outdir / "library.json")
    library.specimen(lib, outdir / "specimen.png")
    library.pair_specimen(lib, outdir / "pairs.png")
    info = font.build_font(lib, outdir / "handwriting.ttf", family)
    w = Writer(lib, seed=1)
    to_image(w.write(PANGRAM, x_height=16, max_width=1000)).save(outdir / "sample.png")
    want = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:;?!'()-"
    table, borrowed = font.with_fallbacks(lib)
    report = {
        "characters": "".join(sorted(lib["glyphs"])),
        "borrowed_or_drawn": "".join(borrowed),
        "missing": "".join(c for c in want if c not in table),
        "thin": {c: g["count"] for c, g in lib["glyphs"].items() if g["count"] < 15},
        "whole_words": len(lib["words"]),
        "filled_because_too_few_pages": lib.get("filled"),
        "font": info,
    }
    (outdir / "coverage.json").write_text(json.dumps(report, indent=1))
    log(f"wrote {outdir}/: library.json, handwriting.ttf, specimen.png, sample.png, coverage.json")
    log(f"missing characters: {report['missing'] or 'none'}; borrowed: {report['borrowed_or_drawn'] or 'none'}")
    return report
