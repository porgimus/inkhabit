# Reviewing, tuning and filling gaps in a library

For a character the shell would misread (a hyphen), give its code point: `--drop 'U+002D:4'`.
Indices are only valid for the specimen you just read. What review can and cannot do:
- The sheet shows the (up to eight) samples that were kept. The number under the label is how many were
  found in the sources, not how many are left; dropping does not bring in replacements.
- When most of a row is wrong, drop the wrong ones anyway. One or two true samples beat eight with five
  wrong; the writer blends and varies them. Tell the user which letters ended up thin.
- Dropping every sample of a letter makes it a stand-in (a plain shape fitted to their proportions). That
  is the right trade when every sample is wrong or misleading (an "l" cut to half height, an "m" that
  reads as another letter), because it restores readability; but it is then not their letter, so say so.
- Whole words lifted from the sources cannot be reviewed on a sheet. If words come out garbled or patchy
  when you write a sample, turn them off with `--set whole_words=0` (0 = never, 1 = whenever available).
  For scans and photos built without a corrected transcript, prefer `whole_words=0` unless a sample shows
  they help.

`review` keeps the library as it was in `library.before-review.json`. After any `build` or `review`, read the sheets
again before dropping more: a rebuild can reorder variants (recognition is not perfectly repeatable).
The person can also review by eye, which is the best check there is for "does this look like my
writing": `inkhabit review-page <out> --family "<Name> Hand"` serves a page on this machine only; they
click the samples that are wrong and press Apply. Run it in the background, give them the address, and
stop it when they are done.

Optional, and only with the user's say-so because it sends small pictures of their letters to an outside
service and costs money: `inkhabit judge <out> [sources...]` asks TypeSafe's Jev model how clearly each sample reads as its
letter and which look-alike it most resembles. Samples are then ordered clearest first, which is what
clean writing draws on. Given the sources as well, it goes back to them for the weak letters, has dozens
of candidates judged, and keeps the clearest. It needs `TYPESAFE_API_KEY` in the environment (the user sets it; never ask for the key in chat) and costs a
few cents. Jev's image input is labelled experimental by its makers: treat its answers as a ranking, and
let the person's own review overrule it.
For fonts built from scans or photos, add `--prune` as the last thing on the line
(`inkhabit judge <out> <sources...> --prune`): it also drops the samples judged unclear, which is
an automatic first review. On benchmark scans of three difficult invented hands this took the share of
words read back correctly from 7-22% to 50-80%, by removing fragments and wrong letters and leaving the
clearest samples first. It is the most effective single step for scans; still review by eye afterwards,
and say which letters ended up as stand-ins.

To let the person try it themselves: `inkhabit playground <out>` serves a page on this machine only where
they type text, pick what it is for, move sliders (size, spacing, height, slant, thickness, irregularity),
ask for another take, and see the exact command that reproduces what is on screen. Run it in the
background, give them the address, and stop it when they are done.

Characters that are missing or thin are best fixed at the source: `inkhabit sheet <out>` makes a practice
sheet (a box per sample, with a baseline) for exactly those characters. The person writes on it in their
note app or on paper, and `inkhabit sheet-read <out> <filled file>` adds what they wrote. No recognition
is involved, so these are the most reliable samples in the library.

Then tune, which checks what your eye cannot: `inkhabit tune <out> --family "<Name> Hand"` writes words
with each sample of each letter, each joined pair and each whole word, reads them back with the
recogniser, and drops the samples that make words unreadable. It then calibrates the natural style against the
notes: generated words are measured with the same instrument as the real ones, and the settings are
adjusted until baseline stray, height variation, letter gap and share of joined letters match. It
prints read-back accuracy before and after for each style and keeps `library.before-tune.json`.
Rebuilding from the same sources again does not help once this is done; more or cleaner source pages do. Run it after your own review, not instead of
it: it measures readability, not whether a sample looks like the person's writing.

Characters the writer never produced are filled in three ways, best first, and `coverage.json` lists them
under `borrowed_or_drawn`: harvested (their own stand-alone marks such as = + → found in the notes; these
appear on the specimen and need the same review), composed from their real pieces (; from their dot and
comma, ← ↑ ↓ from their →, ≥ from their > and -), and extrapolated (a plain recipe for * # & [ ] { } and
similar, restyled to their proportions and irregularity). Extrapolated ones are guesses at how they would
write it: say so if the user asks, and prefer more source pages when a character matters.
The space between words defaults to tighter than in loose notes; change it with
`inkhabit review <out> --word-gap 1.0` (x-heights) or per request with `--set word_spacing=`.
