# Filling forms and other documents, and marking them up

Forms:

```bash
inkhabit fields form.pdf      # form fields, blank lines, boxes and printed labels, with positions
inkhabit fill <out>/library.json form.pdf entries.json -o filled.pdf --seed 3 [--set ...]
```

The document to write on can be a PDF, a scanned PDF, or a photo or scan of a page (jpg, png, heic...):
`fields`, `fill`, `annotate` and `render` take any of them. For a picture, places are found by looking at
it (lines, boxes, tick boxes, rows of cells) and labels by reading it, so check the list against the
rendered page with extra care; the result is saved as a PDF with the picture as its page. On a scanned
reading, `annotate` finds the words to mark by reading the page, and copes with two columns.

Look at the document first: `inkhabit render form.pdf -o <dir>` writes each page as a PNG and prints the
scale (points to pixels), so you can read positions off the picture when you need to.

`fields` lists, with positions in PDF points: `field` (real form fields), `line` (a blank line to write
on), `box`, `tickbox`, `cells` (a row of one-character cells), and `label` (each printed line of text).
Pair every place with the label beside or above it. Where an item has a `"use"` hint, copy it into your
entry (`{"check": true}`, `{"comb": 8}`, `{"valign": "middle"}`). Table cells drawn as ruled lines are found as `box` items, one per cell. It can still miss places drawn in unusual
ways: compare the list with the rendered page, and add a rect by hand for anything missing.

`entries.json`: `[{"page":0,"rect":[x0,y0,x1,y1],"text":"..."}, {"field":"email","text":"..."}]`.
An entry may carry `"x_height"` and `"style": {"slant": 4}` to override settings for that entry only.
Before filling, settle anything the form leaves open with the user: the order of a date in cells
(day-month-year or month-day-year), which of several boxes to tick, and whether a box marked "initials"
or "signature" is theirs to write (never sign for them; initials only if they say so).

| The place is... | Entry | 
|---|---|
| a tick box | `{"rect": [...], "check": true}` (add `"mark": "cross"` for an X) |
| a row of one-character cells (dates, postcodes, IDs) | `{"rect": [whole row], "text": "14032005", "comb": 8}` |
| ruled lines to write along (worksheets, notebook paper) | `{"rect": [...], "text": "...", "line_pitch": 24}`; the rect's top is one pitch above the first line |
| a label to centre on a short line, or text set to the right | add `"align": "center"` or `"right"` |
| a single line inside a small closed box | add `"valign": "middle"` so it does not sit on the bottom edge |
| not a form field at all (a card message, a heading, a caption, a sticky note) | add `"for": "letter" | "heading" | "label" | "note" | "scribble"` (or `"official"` for an address on an envelope) |

Use explicit line breaks (`\n`) in the text for addresses. The same command therefore covers cards,
envelopes, diagram labels, worksheet answers and notebook pages, not only forms.
Check the report (`status: check` means it did not fit or a character is missing), then `inkhabit render`
the result and look at every entry: `ok` means it fitted, not that it is easy to read. If a long value
came out small, give it more room (a wider rect, or two lines) before accepting it.

Marking up a document, the way someone annotates a book or handout:

```bash
inkhabit annotate <out>/library.json page.pdf items.json -o marked.pdf [--colour blue|black|red|green|pencil]
```

`items.json` is a list. Every item is on the first page unless it says `"page": N` (counted from 0). Marks on printed text name it with `"find": "the printed words"` (or give a
`"rect"`): `underline`, `double-underline`, `strike`, `circle`, `box`, `bracket` (`"side"`), `highlight`
(`"colour"`), `star`, `tick`. `{"type": "arrow", "from": [x, y], "to": [x, y]}` draws a free arrow.
`{"type": "note", "text": "...", "find": "words it refers to", "side": "right", "arrow": true}` writes a
note in the margin beside the column those words are in, sized to fit. Its arrow points at the line the
words start on, stopping at the edge of the text so it never runs across the print (it reaches the words
themselves only when they are at that edge). Margins are narrow: keep notes to a few words, and if the
report says `check` with "too small to read", shorten the note or use `"side": "bottom"`. If there is
no room on that side (a neighbouring column), it moves to the other side and the report says `moved_to`.
`"side": "bottom"` or `"top"` puts it below or above the text. Read the document first so `find` strings match the print exactly,
check the report for `target not found` or `no room in that margin`, and render the page to look at it.
