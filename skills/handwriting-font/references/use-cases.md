# Recipes by use case

Each is measured in `bench/usecases.py` on made-up documents. Use the recipe, then render and look.

| The request | How | Notes |
|---|---|---|
| "Fill in this form" (PDF, scan or photo) | `fields` -> pair places with labels -> `fill` | Careful style by default. Copy each item's `"use"` hint. Ask about date order and anything left open; never sign. |
| "Answer this worksheet / homework sheet" | `fill` with `"line_pitch"` on ruled answer lines, boxes otherwise | Keep answers the length a student would write; 2 to 3 lines fit a 3-line gap. |
| "Do the maths questions" | `fill` with `"math": true` in each answer box | Worked steps separated by `\\\\`; start at `"x_height": 8`. Cannot be read back by machine: look closely. |
| "Fill in this table / sign-up sheet / register" | `fields` gives one `box` per cell -> `fill` with `"valign": "middle"` | Shrink the rect 2 pt inside the cell so ink clears the rules. |
| "Fill in my planner / calendar" | `fill`, one entry per day box, `"for": "note"`, `\n` between items | Short phrases; `"x_height": 6.5` for small cells. |
| "Write a card / postcard / gift tag" | `fill` on the card PDF with `"for": "letter"`, or `write --for letter` for an image | Greeting, message and sign-off as three entries; right-align the sign-off. |
| "Address this envelope" | `fill` with `"for": "official"`, explicit line breaks | Recipient large (`"x_height": 10`), return address small in the top left. |
| "Label this diagram / figure" | `fill` with `"for": "label"`, `"align": "center"` | One entry per label line. |
| "Write up these notes / make a revision sheet" | `notes` from an outline (see notes-and-maths.md) | Notes, not prose. `$` for maths, `@` for sketches. |
| "Write this out on lined paper" | `fill` on a ruled page with `"line_pitch"`, `"for": "note"` | Heading as a separate `"for": "heading"` entry. |
| "Make a to-do list / sticky note" | `fill` with `"for": "scribble"`, `"check": true` on done items | Heading via `"for": "heading"`. |
| "Annotate / mark up this reading" (PDF, scan or photo) | `annotate` with `find` strings copied from the text | Margin notes of a few words; `"side": "bottom"` for a longer comment. |

Measured with one writer's font (share of the intended words a recogniser reads back from the finished
document; "fits" and "stays off the print" were 100% and under 4% of ink throughout, except worksheet
lines where descenders cross the rules):
form 93%, worksheet 100%, card 92%, envelope 94%, diagram labels 86%, ruled page 96%, to-do note 88%,
table 92%, planner 87%, photographed form 93%. Short labels and scribbles read worst; if one matters,
write it larger or in the careful style.
