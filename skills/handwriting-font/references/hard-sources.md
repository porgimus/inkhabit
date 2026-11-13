# Hard sources: scans, photos, few pages, other languages, printed pages, joined-up writing

What to expect from scans and photos, measured on a test with ten messy pages of a cramped, slanted
hand: about a third of the samples build picks are the right letter before review, and about half after a
careful review, with many letters left on one to three samples. That makes a recognisable imitation that
is mostly readable, not a faithful font. Say this plainly, and offer the two things that help most: the
practice sheet (`inkhabit sheet`), which fixes it at the source, and, if they agree to letter images
being sent to an outside service, `inkhabit judge <out> --prune`, which reviews the samples automatically. Notebook files and note-app PDFs do far better (nine in ten or more).

Other languages: pass `--lang` with the language tag (`es-ES`, `fr-FR`, `de-DE` ...) to `build` and `transcript`.
Any alphabet written left to right in separate letters works (accents are composed from the plain letter
when there are too few samples, and ¿ ¡ from ? !). There is no word list outside English, so misread
words are not filtered out: use a corrected transcript (next paragraph) for anything but large samples.
Scripts without separable letters (Chinese, Japanese, Arabic) are not supported.

Few pages, hard handwriting, or another language: label the words yourself instead of trusting the
recogniser. For each source file run `inkhabit transcript <source> [--lang ..] -o <dir>/<name>.md`, where `<name>` is the
file's name without its ending (`lecture.goodnotes` -> `lecture.md`; for a zip, one per file inside it), look at the
page, and correct the file so it says exactly what is written, misspellings included (the labels must
match the ink, not the dictionary). Keep the `# title` and `## Page N` lines; write one line of text per
written line. Then `inkhabit build ... --transcripts <dir>`. Line order does not have to match for tables.
This typically triples the usable words on a small sample. What to leave out of a transcript:
- any word you cannot read, and anything crossed out;
- words with ink tangled through them (circled, struck through). A title with a line drawn under it is fine;
- words in someone else's hand, arrows, doodles and drawings;
- a character whose ink does not look like that character (a colon written as one dot, a decimal point
  floating mid-height): drop the mark, or the whole token if it sits inside a number.
`inkhabit transcript` flags almost nothing as unsure even when a quarter of the words are wrong, so read every
line against the page; do not trust an unflagged line.

Handwriting on printed pages (a filled-in exam, a worksheet, an annotated handout): a scan has no way to
tell print from pen by itself, and print would end up in the font. Always give such a page a transcript
containing only the handwritten words; that alone keeps the print out. `--ink lighter` (pencil or pale pen
on black print) or `--ink darker` (dark pen on grey print) also removes the other ink before reading, which
helps when print and writing overlap. `--ink` applies to the whole build but only acts on pages that
really have two kinds of ink, so it is safe in a mixed folder; if a build gets worse with it, drop it.

Fast, fully joined-up writing: the letters inside a word are not separately formed, so they cannot be
cut out, and a letter font from it will be poor however it is labelled. With `--transcripts`, every
word in the transcript is also kept whole, exactly as written, and the writer reuses those
(`--set whole_words=1` to prefer them). A word is reused whole only when it matches one they wrote exactly:
same capitals, three letters or more, no punctuation attached. Every other word falls back to letters and
stand-ins. Say which is which.

Best with roughly 20+ pages. Scans need more than digital ink, since tracing loses detail, and fully
joined-up writing needs more again.

With fewer pages `build` fills the alphabet from what there is and prints exactly how: letters with
enough real samples are kept; missing ones are made by turning or flipping a real letter (u from n,
q from b, p from d); the rest are a neutral shape fitted to the writer's letter heights, width, slant and
irregularity. `coverage.json` records this under `filled_because_too_few_pages`. Always tell the user
how many characters are really theirs: a filled font matches their writing, it is not all their writing,
and each extra page replaces stand-ins with real letters. Below 8 real small letters `build` stops,
because the result would be a generic font; ask for more pages instead of using `--force`.
Photos of paper work (shadows, ruled lines, show-through, iPhone HEIC), best when flat and evenly lit.
