# inkhabit

Your handwriting as a font, and as a pen an agent can write with.

Give it your notes (Goodnotes notebooks, scanned PDFs or photos). It reads the words, splits them into
letters, keeps several real samples of every letter, and gives you:

- `handwriting.ttf`: an installable font whose letters rotate through your real variants
- `library.json`: the pen strokes behind it, used to write text that is different every time
- `specimen.png`, `coverage.json`: what was found, for review
- `profile.md`: how you write and lay out notes, measured

```bash
uv run inkhabit build ~/notes -o out/me --family "My Hand"
uv run inkhabit build ~/apuntes -o out/ana --lang es-ES --transcripts corrected/   # other languages, few pages
uv run inkhabit review out/me --drop 'K:0,2'          # after looking at specimen.png
uv run inkhabit playground out/me                      # type text and try settings live in the browser
uv run inkhabit review-page out/me --family "My Hand"  # click bad samples in the browser, press Apply
uv run inkhabit sheet out/me                          # practice sheet for missing and thin characters
uv run inkhabit sheet-read out/me filled-sheet.goodnotes
uv run inkhabit tune out/me --family "My Hand"         # drop samples that read back wrong; prints before/after
uv run inkhabit write out/me/library.json "Hello there" -o hello.png --seed 4 --style neat --set slant=4
uv run inkhabit write out/me/library.json "Chapter notes" -o h.png --for heading
uv run inkhabit notes out/me/library.json examples/outline.txt -o notes.pdf   # a page of notes in your layout style
uv run inkhabit math out/me/library.json 'T = 2\pi \sqrt{\frac{L}{g}}' -o formula.png
uv run inkhabit annotate out/me/library.json examples/book-page.pdf examples/annotations.json -o marked.pdf
uv run inkhabit fields examples/form.pdf
uv run inkhabit fill out/me/library.json examples/form.pdf examples/entries.json -o filled.pdf
uv run inkhabit typeset out/me/handwriting.ttf "Typed with the font itself" -o typed.png
uv run inkhabit profile ~/notes -o out/me             # redo profile.md, keep the reviewed library
uv run inkhabit pages ~/notes/lecture.goodnotes -o pages/
uv run inkhabit diagrams ~/notes -o out/me              # find every drawing in the notes
uv run inkhabit diagram out/me 12 -o graph.png --label 0=Wage   # draw one again, relabelled
uv run pytest
```

## Install

Needs a Mac (recognition uses Apple Vision) and [uv](https://docs.astral.sh/uv/).

As a skill for your coding agent (Claude Code, Codex, Cursor, OpenCode and others), one command:

```bash
npx skills add porgimus/inkhabit -g
```

Then ask the agent to "make a font of my handwriting" and point it at your notes. The skill fetches and
runs the program itself; what it builds goes in `~/inkhabit/`.

To run it yourself without cloning:

```bash
uvx --from git+https://github.com/porgimus/inkhabit inkhabit --help
```

To work on it:

```bash
git clone https://github.com/porgimus/inkhabit && cd inkhabit
uv sync                # installs everything into .venv
uv run pytest          # all on made-up handwriting
ln -s "$PWD/skills/handwriting-font" ~/.claude/skills/handwriting-font   # use the skill from this copy
```

The tool keeps your notes and everything built from them on your disk (`data/` and `out/` in a clone, which
git ignores) and uploads nothing by itself. Two things do send pictures of your writing elsewhere: the agent you run the skill
with looks at page and letter images, and those go to whichever model provider runs that agent; and the
optional `inkhabit judge` step sends letter images to TypeSafe, only if you call it with your key.
Share pictures of the results if you like, but keep `library.json` and the `.ttf` to yourself: they let
anyone produce writing that passes for yours.

## How it works

1. **Ink** (`goodnotes.py`, `scan.py`): Goodnotes files and note-app PDFs hold real pen strokes, read
   directly. Scans and photos are cleaned, thinned to a one-pixel skeleton and traced back into strokes.
2. **Words** (`words.py`, `ocr.py`): Apple Vision reads each page and gives a box per word; the strokes
   under each box become a labelled word. Only dictionary words and numbers are kept.
3. **Letters** (`segment.py`): each word is cut into small pieces at gaps and thin joins, then the pieces
   are grouped into the word's letters by a forced alignment that learns letter shapes as it goes.
4. **Library** (`library.py`): the cleanest, most varied samples of each letter, plus whole common words,
   joined letter pairs ("th", "he") exactly as written, and how irregular the writing measurably is.
   Everything is then unified: brought to one scale and baseline, so pieces from headings and small
   print sit together evenly.
5. **Font** (`font.py`) and **writer** (`render.py`, `forms.py`). The writer has styles (careful, clean,
   natural, rushed) and named settings for spacing, letter width and height, slant, line spacing and pen weight.

6. **Annotation** (`annotate.py`): underlines, circles, brackets, highlights, arrows and margin notes
   drawn on a PDF with the same pen.

## Benchmarks

The benchmarks live in `bench/` and use only invented writers and made-up documents:

```bash
uv run python bench/run.py            # font accuracy: six invented hands as pen strokes, scans and photos
uv run python bench/usecases.py out/me/library.json   # documents filled or annotated, then scored
uv run python bench/formats.py        # one page delivered in about thirty input formats
```

`run.py` reports, stage by stage, where accuracy is lost (words found, right ink, right letter, likeness,
read-back). `usecases.py` reports whether writing fits its place, stays off the print and reads back.
Run them before and after any change to the pipeline.

## Limits

- macOS only (Apple Vision for recognition).
- Characters never written are harvested from stand-alone marks in the notes, composed from real pieces, or
  extrapolated from a plain recipe restyled to the writer (`extend.py`); the last kind is a guess. See
  `coverage.json`.
- With only a few pages the alphabet is filled in: real letters are kept, missing ones are turned or
  flipped from real ones (u from n, q from b), and the rest are a neutral shape fitted to the writer.
  `coverage.json` says which is which. Below 8 real small letters the build stops.
- PDFs exported from note apps keep their pen strokes and are read directly (tested on 77 real exported
  pages: a full, legible font). Scans and photos have only pixels, so pen paths are traced back out of
  them, which is rougher. Tested on a real scan of four pencil-written pages of a printed exam at 150 dpi,
  with a corrected transcript and `--ink lighter`: 22 real characters and 90 whole words, legible in parts.
  One photographed page of fast joined-up writing gave whole words only, no usable letters.
- Tested end to end by giving a fresh agent only the skill and ten deliberately messy pages by an invented
  writer with a cramped, slanted hand (skewed scans, phone photos, a worksheet, another person's page): it
  produced a recognisable, mostly readable imitation in which about half the kept samples were the right
  letter. Scans and photos of difficult writing remain the weak case; the practice sheet is the fix.
- For joined-up writing, words from a corrected transcript are kept whole and reused; only words the
  person has actually written come out as theirs.
- The Goodnotes format was worked out by inspection of Goodnotes 7.1 backups and may change.
- Use it only with handwriting you own or have permission to use. Not for signatures.

The agent skill is in `skills/handwriting-font/`. Personal data (`data/`, `out/`) is git-ignored.

## Fair use of it

This is for writing in your own handwriting. Do not use it to copy someone else's writing without their
agreement, to produce signatures, or to make any document that is meant to deceive. What you make with it
is your responsibility.

Goodnotes, Notability, OneNote, TypeSafe and Apple are names of their owners' products, used here only to
say what the tool reads or calls. This project is not affiliated with or endorsed by any of them.

## Licence

Copyright (C) 2026 Porg

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero
General Public License as published by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version. It is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
`LICENSE` for the full text.

In short: use, change and share it freely. If you pass on a changed version, or run one as a service that
other people use over a network, you must make your source available under the same licence.

It builds on PyMuPDF (AGPL-3.0), installed as a dependency and not included here. Every other dependency is
under an MIT, BSD or Apache licence.
