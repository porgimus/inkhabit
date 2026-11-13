# Notes pages, sketches and maths

Taking notes for them (study notes, a summary of a reading, a revision sheet), laid out the way they lay
out their own notes and not just in their letters:

```bash
inkhabit notes <out>/library.json outline.txt -o notes.pdf [--paper white|dark|ruled] [--colours blue,pink,black] [--seed 4]
```

Write the outline as plain text, using what you know of how the person takes notes (`<out>/profile.md`,
and `style-notes.md` if it exists):

```
# Heading                           underlined, a little larger
A line of text                      one idea per line; keep lines short, as notes are
  - an indented point               two spaces per level; "-" gives a dash bullet
    a consequence -> a result       "->" becomes a drawn arrow
> a side remark                     goes to the right of the line above, with an arrow to it
! a key line                        gets a box drawn round it
                                    a blank line starts a new block of thought, in a new ink colour
```

Lines starting `@` draw a small sketch in the same pen: `@ axes Amount Month` followed by
`@ curve up Daylight`, `@ curve down Frost` (also `hump`, an arch that comes down to the axis at both
ends; `valley`, a U; and `flat`) and `@ point Planting`, which marks the middle of the plot with dashed
lines to both axes (it is for where two curves cross, not for an arbitrary point). Or
`@ flow cause -> effect -> result` for a chain of boxes joined by arrows, or `@ theirs N` for one of their
own drawings (below). Any other `@` line is an error. Lines starting `$` are maths.

Page size, writing size, line spacing, ink colours, indent depth and how often indented points get a
hooked lead-in arrow all come from their profile, so the page is proportioned like their own; colour changes by block of thought, not by importance. Use `--paper dark` only if their own notes
are on dark paper. Write the content as notes, not prose: short lines, terms followed by a colon and an
explanation, sub-points indented under the idea they belong to. Render the PDF and look at it (`inkhabit render notes.pdf -o <dir>`).
The command prints the page size, writing height and line spacing it used and whether they came from the
person's profile. Without a `profile.json` beside the library it uses plain defaults; override with
`--x-height` (points; about 5 is small notes, 7 is comfortable) and `--line-pitch` (points between lines).
To keep to one page, shorten the outline or reduce `--line-pitch`; it runs onto further pages otherwise.
A `>` side remark belongs after an ordinary line or a line of maths, not after a sketch.

## Their own drawings, drawn again

Graphs, flow charts, pyramids, little pictures: anything they drew in their notes can be kept and drawn
again, with new labels in their handwriting.

```bash
inkhabit diagrams <their notes: files, folders, zips> -o <out>      # once: finds every drawing
inkhabit diagram <out> <N> -o again.png|.svg|.pdf [--label 3=Wage --label "7=Euro area"] \
  [--vary 1] [--width 300] [--colour blue] [--seed 2]
```

`diagrams` writes `<out>/diagrams.json` and one picture per drawing in `<out>/diagrams/NNN.png`. Look at
the pictures to choose one: the small red numbers are label numbers. Do not choose by the label text in
the JSON, which is the machine's reading and often wrong. `diagram` redraws drawing N as a second attempt
would look (`--vary 0` is an exact copy, 2 is looser). A label you do not rewrite keeps its original ink,
so it is their real writing; a rewritten label is written at the size and place of the old one, smaller
if it would run into the next thing (the command says so). Only words the recogniser found are labels:
a single letter such as an axis "Q" is usually part of the drawing and cannot be rewritten.

In a notes page, `@ theirs 34 0=W 1=S` puts drawing 34 there at the size of the page's writing.

Limits: it replays a drawing they made and changes its labels, size and colours; it does not invent a
new kind of drawing in their style (for a new graph use `@ axes`, `@ curve`, `@ flow` above). It needs
notebook files or note-app PDFs, where pen strokes are kept; it is untested on scans. Some finds are
worked maths or a table, not a diagram: skip those. Look at every result before using it.

## Maths

Maths, in their hand:

```bash
inkhabit math <out>/library.json 'T = 2\pi \sqrt{\frac{L}{g}}' -o formula.png [--x-height 22]
```

It reads a forgiving subset of LaTeX: `^` and `_`, `\frac{}{}`, `\sqrt{}`, `\int_a^b`, `\sum_{i=1}^{n}`,
`\lim_{x \to \infty}`, brackets that grow, `\times \div \pm \leq \geq \neq \to \approx \cdot`,
`\pi \theta \Delta \infty \alpha \mu \lambda \sigma`, `\text{words}`, and sin, cos, tan, log, ln. Lines separated
by `\\` or newlines are worked steps and are aligned on their first equals sign. The same maths works:
- in notes: a line starting with `$` is maths (consecutive ones are worked steps), and `$...$` inside an
  ordinary line sets that part as maths;
- in forms: `{"rect": [...], "text": "\\frac{3}{4}", "math": true}`.
Digits, letters and signs are the person's own; bars, root signs, tall brackets and the Greek letters are
drawn to match. Look at the result: anything beyond this subset is written out literally.
