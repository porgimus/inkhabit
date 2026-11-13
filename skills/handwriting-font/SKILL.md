---
name: handwriting-font
description: Turn a person's own handwritten notes (Goodnotes notebooks, scanned PDFs, photos) into a font and handwriting library, then write text or fill PDF forms in that handwriting with natural, adjustable variation. Use for "make a font of my handwriting", "write this in my handwriting", "fill this form by hand", "make it look more handwritten". Not for copying anyone else's writing or producing signatures.
---

# Handwriting font

Every `inkhabit ...` command in this skill and its references is run one of two ways:

- Inside a copy of the inkhabit repo (the folder two above this file's real location has a
  `pyproject.toml`; if the skill folder is a symlink, follow it first):
  `uv run --project <repo> inkhabit ...`, and the work folder `<work>` is `<repo>/out`.
- Anywhere else (the skill was installed on its own):
  `uvx --from git+https://github.com/porgimus/inkhabit inkhabit ...`, and `<work>` is `~/inkhabit`.
  This needs [uv](https://docs.astral.sh/uv/) and fetches the program the first time.

macOS only: reading the handwriting uses Apple's on-device text recognition.

Finished builds are in `<work>/<name>/`; look there before building again. Put outputs, page pictures and
transcripts under `<work>/<name>/` unless the user names another place.

## Boundaries

- Build only from handwriting the user owns or has permission to use. Do not produce signatures, and do
  not fill documents where the person's own hand or a wet signature is legally required.
  Reason: the output passes for the person's own writing.
- Sources and outputs are personal data. Never commit or upload notebooks, libraries, specimens or fonts
  unless the user asks. The tool itself uploads nothing, except `inkhabit judge` (letter pictures go to an
  outside service; only with the user's say-so). Pictures you look at are seen by the model running you.
- Confirm form values with the user before filling. Never invent personal details.

## 1. Build

First look at what you were given; a folder of someone's notes is rarely clean. For each file run
`inkhabit pages <file> -o <dir> --max 2` and look at the images (for a zip or a big folder, sample a few
notebooks from across it, not just the first). For a scan or photo this writes the page
itself plus a `-traced.png` showing the pen paths recovered from it; for notebook files it draws the ink.
Then decide, and tell the user what you decided:

- Leave out (`--exclude "text in the path"`) typed documents and anything written by someone else. A
  different hand is usually obvious side by side: slant, roundness, letter height.
- A page that is mostly the person's but has a few words in another hand, crossings-out or doodles is
  fine to keep when you use a transcript (below): whatever the transcript omits is left out.
- Printed pages with handwriting on them (worksheets, exams, annotated handouts) need care: see below.
- Note which sources are notebook files or note-app PDFs (pen strokes kept: best) and which are scans or
  photos (pixels only: much rougher). Say so to the user before building.

```bash
inkhabit build <folder-or-files...> -o <out> --family "<Name> Hand" [--exclude "text in path"]
```

Give it whatever they have; point `build` at files, folders or zips, mixed freely:

| They have | Give | Quality |
|---|---|---|
| Goodnotes | the backup zip as it is (Menu > Cloud & Backup > Manual Backup), or `.goodnotes` files | best: real pen strokes |
| another note app (Notability, OneNote, an iPad) | a PDF export, or SVG if the app offers it | as good: strokes are kept as shapes |
| paper | a scanned PDF, or photos (jpg, heic, png, tif incl. multi-page, bmp, webp) | rougher: paths are traced from pixels |
| a mix, or a folder they never tidied | the folder or a zip of it | each file is read as what it is |

`build` prints which files it skipped (damaged, password-protected) and which had no handwriting; relay
that to the user. Any paper and ink colour works; ruled lines, printed text and covers are ignored.

Read `references/hard-sources.md` before building when any of these is true: the sources are scans or
photos; there are fewer than about 20 pages; the writing is not English; pages are printed with handwriting
on them; the writing is fast and fully joined; or a build came out poor. It covers transcripts
(`--transcripts`), `--lang`, `--ink`, what to expect, and what `build` does when there is too little.

## 2. Review (do not skip)

Recognition misreads some words, so some variants are the wrong letter or two letters.

1. Read `<out>/specimen.png` (one row per character, variants numbered from 0, sample count under the
   label) and `<out>/pairs.png` (joined letter pairs such as "th").
2. Read `<out>/coverage.json`: `missing`, `thin` (few samples: inspect hardest), `borrowed_or_drawn`.
3. Drop everything that is not clearly that character, in one command, then read the sheets again
   (indices change after every build or review):

```bash
inkhabit review <out> --family "<Name> Hand" --drop 'K:0,2' ',:1' --drop-pair 'en:0,3'
```

4. Tune: `inkhabit tune <out> --family "<Name> Hand"` drops samples that make words unreadable and prints
   read-back accuracy per style (0.9+ is a clean library, under 0.5 means the letters are the problem).
5. Tell the user what is really theirs, what is thin, and what is a stand-in.

Read `references/review-and-tune.md` when most of a row is wrong, a letter is missing or thin, words come
out garbled, tune does not help, or you want the person to review by eye (`review-page`), try it
themselves (`playground`), fix gaps with a practice sheet (`sheet`), or have samples judged by an outside
model (`judge`, only with their say-so).

## 3. Use

Font, for typing in any app: `<out>/handwriting.ttf` (copy to `~/Library/Fonts` to install; ask first).
Each letter rotates through up to 8 real variants when "contextual alternates" is on (the usual default).
To set text with the font file yourself: `inkhabit typeset <out>/handwriting.ttf "text" -o out.png`.

Writing, different every time (preferred whenever the result is an image or PDF):

```bash
inkhabit write <out>/library.json "text or @file.txt" -o note.png|.svg|.pdf \
  --for <situation> --seed 7 [--x-height 24 --width 1400] [--set name=value ...]
```

Decide from the situation; do not ask the user which style they want. Write clean unless the situation
or the user says otherwise, and pass `--for` so care, joining and size are set together:

| The text is... | Use | Why |
|---|---|---|
| anything not covered below | `--for text` (the default look) | best-case writing is what people expect to receive |
| going into a form, an application, an envelope, anything official or that a stranger must read | `inkhabit fill` (careful by default), or `--for official` | people write their best here: only their clearest letter shapes, letters kept apart, level and evenly spaced. Never roughen this on purpose |
| a title, a name on a cover, a section header | `--for heading` | people write headings larger, firmer and unjoined |
| a caption, a figure label, a short margin note | `--for label` (what `annotate` notes use) | small and plain so it reads at a glance |
| a card, a letter, a paragraph meant to feel personal | `--for letter` | continuous prose with a little more life |
| a whole page of notes or a revision sheet | `inkhabit notes` (below) | their layout habits as well as their letters |
| a line or two meant to look like their own notes | `--for note` | their real spacing, joins and variation |
| a quick reminder, a sticky note, "jotted down" | `--for scribble` | hurried on purpose |

If the user then asks for a change in plain words ("bigger", "more spread out", "leaning", "messier"),
keep the situation and add `--set`. Only go messier than clean when asked or when the situation is one
of the last two rows.

More control, when asked for:
- `--style` is for when no `--for` fits; it has no effect when `--for` is given. `clean` (the default) is
  the person's writing at its best: their most usual letter shapes, steady and evenly spaced. `careful` is
  their most careful hand, which forms and `--for official` already use. `natural` writes the way the notes
  really look, looser and more varied; `rushed` is hurried. `neat` is an older name for `clean`.
- `--human` scales all irregularity together (0 = mechanical, 1 = as measured, 2 = messy).
- `--set` changes one thing when the user asks for it. Dimensions (1.0 = as they write):
  `letter_spacing`, `word_spacing`, `line_spacing`, `letter_height`, `letter_width`, `weight`,
  `slant` (degrees). Irregularity: `size`, `height_var`, `width_var`, `word_size`, `tilt`, `bounce`,
  `spacing`, `word_spacing_var`, `line_spacing_var`, `margin`, `drift`, `slope`, `wobble`, `pressure`,
  `joins`, `whole_words`, `typical`. Example: "taller and more spread out" ->
  `--set letter_height=1.15 --set word_spacing=1.2`.
- Same seed gives the same result; change the seed for another take. Look at the output before
  delivering it, and try another seed if a word came out badly.
- Pen thickness follows the person's own pen relative to their letters (capped where loops would fill in),
  and in documents never goes below about a 0.35 mm line however small the writing, as with a real pen.
  `--set weight=1.2` makes it heavier, `0.8` finer.
- Size: `--x-height` (default 14) and `--width` (wrap width, same units). For a sample someone will judge,
  use `--x-height 24 --width 1400` or larger; the default is small.

`references/use-cases.md` has a one-line recipe for each common request (forms, worksheets, maths answers,
tables, planners, cards, envelopes, labels, notes, to-do lists, annotation): start there when the request
names one.

Other things it can make. Read the reference before using each:

| The job | Command | Reference |
|---|---|---|
| fill a form, application, worksheet, table, card, envelope, diagram labels (tick boxes, one-character cells, ruled lines), whether a PDF, a scan or a photo | `inkhabit fields`, `inkhabit fill` | `references/documents.md` |
| mark up a document: underline, circle, highlight, bracket, margin notes with arrows | `inkhabit annotate` | `references/documents.md` |
| a whole page of notes in their layout style, with sketches | `inkhabit notes` | `references/notes-and-maths.md` |
| draw one of their own diagrams again (graph, flow chart, sketch), with new labels | `inkhabit diagrams`, `inkhabit diagram` | `references/notes-and-maths.md` |
| maths: fractions, powers, roots, integrals, worked steps | `inkhabit math` | `references/notes-and-maths.md` |
| describe how they write and take notes | `<out>/profile.md`, `inkhabit pages` | `references/profile.md` |

Whatever you make, look at it before saying it is done: `inkhabit render file.pdf -o <dir>` turns any PDF
into PNGs (`inkhabit pages` is for handwriting sources, not printed documents). Check the command's report
for anything that did not fit or has no shape; a report of `ok` means it fitted, not that it reads well.
`write` sizes are in pixels for PNG (`--x-height 24` is 24 px letters); add `--scale 3` for print.
