# SPDX-License-Identifier: AGPL-3.0-or-later
"""Describe how someone writes and lays out notes, from measured ink."""

from __future__ import annotations

from collections import Counter

import numpy as np

from .ink import Page
from .words import WordSample


def _slant(pages: list[Page], limit: int = 20000) -> float:
    """Typical lean of near-vertical pen movement, degrees right of upright."""
    angs, wts, n = [], [], 0
    for p in pages:
        for s in p.strokes:
            if len(s.points) < 4:
                continue
            d = np.diff(s.points[:: max(1, len(s.points) // 12)], axis=0)
            ln = np.linalg.norm(d, axis=1)
            a = np.degrees(np.arctan2(d[:, 0] * -np.sign(d[:, 1]), np.abs(d[:, 1]) + 1e-9))
            keep = (np.abs(a) < 40) & (ln > 0)
            angs += a[keep].tolist()
            wts += ln[keep].tolist()
            n += 1
            if n > limit:
                break
    return float(np.average(angs, weights=wts)) if angs else 0.0


def _colour_name(c) -> str:
    r, g, b = c[:3]
    mx, mn = max(r, g, b), min(r, g, b)
    if mx - mn < 0.12:
        return "white" if mx > 0.85 else "black" if mx < 0.25 else "grey"
    h = (np.degrees(np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)) + 360) % 360
    for lim, name in [(20, "red"), (45, "orange"), (70, "yellow"), (165, "green"), (200, "cyan"),
                      (255, "blue"), (290, "purple"), (345, "pink"), (361, "red")]:
        if h < lim:
            return name
    return "red"


def _iqr_sd(v) -> float:
    q1, q3 = np.percentile(v, [25, 75])
    return float((q3 - q1) / 1.349)


def habits(pages: list[Page], samples: list[WordSample], lib: dict, vocab: set[str] | None) -> dict:
    """Measured writing and note-taking habits beyond the basics."""
    from .segment import ASC, DESC, word_xheight

    xh = lib["meta"]["xheight"]
    vocab = vocab or set()  # no word list for this language: every short word counts as shorthand
    out: dict = {}
    ink = {id(s): np.vstack(s.strokes) for s in samples}
    by_page: dict[tuple, list[WordSample]] = {}
    lines: dict[tuple, list[WordSample]] = {}
    for s in samples:
        by_page.setdefault((s.source, s.page), []).append(s)
        lines.setdefault((s.source, s.page, s.line), []).append(s)
    for ws in lines.values():
        ws.sort(key=lambda w: ink[id(w)][:, 0].min())

    # does a line climb or sink as it goes right?
    slopes = []
    for ws in lines.values():
        flat = [w for w in ws if not set(w.text) & DESC]
        if len(flat) >= 4:
            x = np.array([ink[id(w)][:, 0].mean() for w in flat])
            y = np.array([np.percentile(ink[id(w)][:, 1], 97) for w in flat])
            if np.ptp(x) > 15 * xh:
                slopes.append(-np.degrees(np.arctan(np.polyfit(x, y, 1)[0])))
    if len(slopes) > 20:
        out["line_slope_degrees"] = round(float(np.median(slopes)), 2)
        out["line_slope_spread"] = round(_iqr_sd(slopes), 2)

    # does the writing shrink or grow down a page?
    drift, per_page_words, per_page_lines, levels, steps = [], [], [], [], []
    for key, ws in by_page.items():
        per_page_words.append(len(ws))
        starts = [l[0] for k, l in lines.items() if k[:2] == key]
        per_page_lines.append(len(starts))
        sz = [(ink[id(w)][:, 1].mean(), word_xheight(w)) for w in ws]
        sz = np.array([(y, v) for y, v in sz if v])
        if len(sz) > 25 and np.ptp(sz[:, 0]) > 100 * xh / 4.5:
            a = np.polyfit(sz[:, 0], sz[:, 1], 1)[0] * np.ptp(sz[:, 0]) / np.median(sz[:, 1])
            drift.append(a)
        if len(starts) >= 6:
            left = np.array([ink[id(w)][:, 0].min() for w in starts])
            lv = np.round((left - np.percentile(left, 5)) / (3 * xh)).clip(0, 12).astype(int)
            used_levels = sorted(l for l, n in Counter(lv.tolist()).items() if n >= 2)
            levels.append(len(used_levels))
            if len(used_levels) > 1:
                steps.append(float(np.median(np.diff(used_levels))) * 3)
    if drift:
        out["size_change_down_page"] = round(float(np.median(drift)), 3)
    out["words_per_page"] = float(np.median(per_page_words)) if per_page_words else None
    out["lines_per_page"] = float(np.median(per_page_lines)) if per_page_lines else None
    if levels:
        out["indent_levels_per_page"] = float(np.median(levels))
        out["indent_step_x_heights"] = round(float(np.median(steps)), 1) if steps else None

    # what is each colour used for?
    roles = {}
    page_size = {k: np.median([v for v in (word_xheight(w) for w in ws) if v] or [xh]) for k, ws in by_page.items()}
    line_first = {id(l[0]) for l in lines.values()}
    by_colour: dict[str, list[WordSample]] = {}
    for s in samples:
        by_colour.setdefault(_colour_name(s.color), []).append(s)
    for name, ws in by_colour.items():
        if len(ws) < 0.02 * len(samples):
            continue
        rel = [v / page_size[(w.source, w.page)] for w in ws if (v := word_xheight(w))]
        roles[name] = {
            "share_of_words": round(len(ws) / len(samples), 3),
            "relative_size": round(float(np.median(rel)), 2) if rel else None,
            "starts_a_line": round(float(np.mean([id(w) in line_first for w in ws])), 2),
            "capitalised": round(float(np.mean([w.text[:1].isupper() for w in ws])), 2),
            "numeric": round(float(np.mean([any(c.isdigit() for c in w.text) for w in ws])), 2),
        }
    out["colour_roles"] = roles

    # how colour changes: line by line, or in blocks?
    runs = []
    for key in by_page:
        seq = [_colour_name(Counter(w.color for w in l).most_common(1)[0][0])
               for k, l in sorted(lines.items(), key=lambda kv: np.mean([ink[id(w)][:, 1].mean() for w in kv[1]])) if k[:2] == key]
        run = 1
        for a, b in zip(seq, seq[1:]):
            if a == b:
                run += 1
            else:
                runs.append(run)
                run = 1
        if seq:
            runs.append(run)
    out["lines_per_colour_block"] = round(float(np.mean(runs)), 1) if runs else None
    mixed = [len({_colour_name(w.color) for w in l}) > 1 for l in lines.values() if len(l) >= 3]
    out["lines_mixing_colours"] = round(float(np.mean(mixed)), 2) if mixed else None

    # line patterns: "term: definition", dash bullets, arrows
    texts = [" ".join(w.text for w in l) for l in lines.values() if len(l) >= 2]
    if texts:
        out["term_colon_lines"] = round(float(np.mean([":" in " ".join(t.split()[:3]) for t in texts])), 2)
        out["dash_bullet_lines"] = round(float(np.mean([t.lstrip()[:1] in "-•–" for t in texts])), 2)
        out["lines_with_arrows"] = round(float(np.mean([any(c in t for c in "→↑↓←↳⇒") for t in texts])), 2)
        out["question_lines"] = round(float(np.mean([t.rstrip().endswith("?") for t in texts])), 2)

    # marks that are not words: underlines, enclosures, lead-in marks before indented lines
    in_words = {(round(float(p[0, 0]), 2), round(float(p[0, 1]), 2), len(p)) for s in samples for p in s.strokes}
    underlines = boxes = 0
    for p in pages:
        for st in p.strokes:
            if st.width > 4 or len(st.points) < 2:
                continue
            if (round(float(st.points[0, 0]), 2), round(float(st.points[0, 1]), 2), len(st.points)) in in_words:
                continue
            x0, y0, x1, y1 = st.bbox
            if x1 - x0 > 3 * xh and y1 - y0 < 0.6 * xh:
                underlines += 1
            elif x1 - x0 > 3 * xh and y1 - y0 > 2 * xh and np.linalg.norm(st.points[0] - st.points[-1]) < 1.5 * xh:
                boxes += 1
    n_pages = max(1, len(by_page))
    out["underlines_per_page"] = round(underlines / n_pages, 1)
    out["enclosures_per_page"] = round(boxes / n_pages, 1)

    # shorthand, capitals, maths
    alpha = [s.text for s in samples if s.text.isalpha()]
    short = Counter(t.lower() for t in alpha if 2 <= len(t) <= 4 and t.lower() not in vocab)
    marks = Counter(s.text for s in samples if not any(c.isalnum() for c in s.text) and len(s.text) <= 2)
    out["shorthand"] = [w for w, n in short.most_common(12) if n >= 5]
    out["frequent_marks"] = [w for w, n in marks.most_common(8) if n >= 5]
    out["all_caps_share"] = round(float(np.mean([t.isupper() and len(t) > 1 for t in alpha])), 3) if alpha else None
    firsts = [l[0].text for l in lines.values() if l[0].text[:1].isalpha()]
    out["lines_starting_capital"] = round(float(np.mean([t[0].isupper() for t in firsts])), 2) if firsts else None
    out["numeric_share"] = round(float(np.mean([any(c.isdigit() for c in s.text) for s in samples])), 3)

    # letterforms, from the chosen samples
    g = lib["glyphs"]

    def tops(chars):
        v = [-min(p[1] for st in var["strokes"] for p in st) for c in chars if c in g for var in g[c]["variants"]]
        return round(float(np.median(v)), 2) if v else None

    def bottoms(chars):
        v = [max(p[1] for st in var["strokes"] for p in st) for c in chars if c in g for var in g[c]["variants"]]
        return round(float(np.median(v)), 2) if v else None

    out["ascender_height"] = tops("bdhkl")
    out["capital_height"] = tops("ABDEFHLPR")
    out["digit_height"] = tops("0123456789")
    out["descender_depth"] = bottoms("gpqy")
    lifts = {c: float(np.mean([len(v["strokes"]) for v in g[c]["variants"]])) for c in g if c.isalpha() and c.islower()}
    out["one_stroke_letters"] = "".join(sorted(c for c, n in lifts.items() if n <= 1.15))
    out["multi_stroke_letters"] = "".join(sorted(c for c, n in lifts.items() if n >= 1.8))
    spread = {c: float(np.std([v["width"] for v in g[c]["variants"]]) / max(np.mean([v["width"] for v in g[c]["variants"]]), 1e-6))
              for c in lifts if len(g[c]["variants"]) >= 5 and c not in "il"}
    order = sorted(spread, key=spread.get)
    out["most_consistent_letters"] = "".join(order[:5])
    out["most_variable_letters"] = "".join(order[-5:])

    # look-alike letters: whose chosen samples sit closest to another letter's samples?
    from .segment import feature

    feats, labels = [], []
    for c, entry in g.items():
        if c.isalnum():
            for var in entry["variants"]:
                feats.append(feature([np.asarray(st, float) for st in var["strokes"]], 0.0, 1.0))
                labels.append(c)
    if len(feats) > 20:
        x = np.array(feats)
        d = ((x[:, None] - x[None]) ** 2).sum(-1)
        np.fill_diagonal(d, np.inf)
        near = [labels[j] for j in d.argmin(1)]
        mix = Counter(tuple(sorted((a, b))) for a, b in zip(labels, near) if a != b and a.lower() != b.lower())
        out["look_alike_pairs"] = [f"{a}/{b}" for (a, b), n in mix.most_common(10) if n >= 2]
        same_case = Counter(a.lower() for a, b in zip(labels, near) if a != b and a.lower() == b.lower())
        out["capital_like_lowercase"] = "".join(sorted(c for c, n in same_case.items() if n >= 3))

    # blank gaps between blocks, and marks leading into indented lines
    gaps, lead, indented_n = [], 0, 0
    loose_by_src: dict[tuple, list] = {}
    for p in pages:
        loose_by_src[(p.source, p.index)] = []
    for p in pages:
        for st in p.strokes:
            if st.width <= 4 and len(st.points) >= 2 and (round(float(st.points[0, 0]), 2), round(float(st.points[0, 1]), 2), len(st.points)) not in in_words:
                loose_by_src[(p.source, p.index)].append(st.bbox)
    for key, ws in by_page.items():
        ls = sorted((l for k, l in lines.items() if k[:2] == key), key=lambda l: np.mean([ink[id(w)][:, 1].mean() for w in l]))
        ys = [np.mean([ink[id(w)][:, 1].mean() for w in l]) for l in ls]
        gaps += (np.diff(ys) / xh).tolist()
        if len(ls) < 6:
            continue
        lefts = np.array([ink[id(l[0])][:, 0].min() for l in ls])
        base = np.percentile(lefts, 5)
        boxes_ = np.array(loose_by_src.get(key, [])).reshape(-1, 4)
        for l, left, y in zip(ls, lefts, ys):
            if left > base + 3 * xh:
                indented_n += 1
                if len(boxes_) and np.any((boxes_[:, 2] > left - 5 * xh) & (boxes_[:, 2] < left + 0.5 * xh)
                                           & (boxes_[:, 1] < y + 1.5 * xh) & (boxes_[:, 3] > y - 3 * xh)):
                    lead += 1
    gaps = np.array([v for v in gaps if 1.5 < v < 40])
    if len(gaps) > 50:
        usual = float(np.median(gaps))
        out["block_gap_share"] = round(float(np.mean(gaps > 1.5 * usual)), 2)
        out["block_gap_size"] = round(float(np.median(gaps[gaps > 1.5 * usual]) / usual), 1) if np.any(gaps > 1.5 * usual) else None
    if indented_n:
        out["indented_lines_with_lead_in_mark"] = round(lead / indented_n, 2)

    # by folder (course / notebook group)
    groups: dict[str, list[tuple]] = {}
    for key in by_page:
        parts = key[0].replace("\\", "/").split("/")
        groups.setdefault(parts[-2] if len(parts) > 1 else "", []).append(key)
    table = {}
    for name, keys in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:8]:
        ws = [w for k in keys for w in by_page[k]]
        table[name] = {"pages": len(keys), "words_per_page": round(len(ws) / len(keys)),
                       "numeric": round(float(np.mean([any(c.isdigit() for c in w.text) for w in ws])), 2),
                       "main_colour": Counter(_colour_name(w.color) for w in ws).most_common(1)[0][0]}
    out["by_folder"] = table
    return out


def habits_markdown(h: dict) -> str:
    def pct(v):
        return "n/a" if v is None else f"{v * 100:.0f}%"

    L = ["## Lines and pages"]
    if "line_slope_degrees" in h:
        d = h["line_slope_degrees"]
        way = "climb" if d > 0.15 else "sink" if d < -0.15 else "stay level"
        L.append(f"- Lines {way} as they go right: {d:+.2f} degrees typically, varying by about {h['line_slope_spread']} degrees.")
    if "size_change_down_page" in h:
        d = h["size_change_down_page"]
        L.append(f"- Writing gets {'larger' if d > 0 else 'smaller'} down a page by about {abs(d) * 100:.0f}%.")
    if h.get("words_per_page") is not None and h.get("lines_per_page") is not None:
        L.append(f"- A typical page has {h['words_per_page']:.0f} readable words on {h['lines_per_page']:.0f} lines.")
    if "indent_levels_per_page" in h:
        L.append(f"- Outline depth: {h['indent_levels_per_page']:.0f} indent levels on a typical page, stepping in about {h['indent_step_x_heights']} x-heights each time.")
    L.append(f"- {h['underlines_per_page']} underline-like strokes and {h['enclosures_per_page']} boxes or circles per page.")
    L += ["", "## What each colour is used for", "", "| Colour | Share of words | Size vs page | Starts a line | Capitalised | Has digits |", "|---|---|---|---|---|---|"]
    for name, r in sorted(h["colour_roles"].items(), key=lambda kv: -kv[1]["share_of_words"]):
        L.append(f"| {name} | {pct(r['share_of_words'])} | {r['relative_size']}x | {pct(r['starts_a_line'])} | {pct(r['capitalised'])} | {pct(r['numeric'])} |")
    L += ["", f"Colour changes in blocks of about {h.get('lines_per_colour_block')} lines; {pct(h.get('lines_mixing_colours'))} of lines mix colours. "
          "If size, line starts and capitals look the same across colours above, colour marks a change of topic, not a heading level.",
          "", "## Line patterns",
          f"- {pct(h.get('term_colon_lines'))} of lines are 'term: explanation'; {pct(h.get('dash_bullet_lines'))} start with a dash or bullet; "
          f"{pct(h.get('lines_with_arrows'))} contain an arrow; {pct(h.get('question_lines'))} end in a question mark."]
    if "block_gap_share" in h:
        L.append(f"- {pct(h['block_gap_share'])} of line breaks are a wider gap separating blocks (about {h['block_gap_size']}x the usual line spacing); "
                 f"{pct(h.get('indented_lines_with_lead_in_mark'))} of indented lines are led in by a drawn mark such as an arrow or dash.")
    L += ["", "## Letterforms",
          f"- Look-alike pairs (samples of one sit closest to samples of the other): {', '.join(h.get('look_alike_pairs', [])) or 'none'}.",
          f"- Capitals written like the lowercase letter: {' '.join(h.get('capital_like_lowercase', '')) or 'none'}.",
          f"- Tall letters (b, d, h, k, l) reach {h['ascender_height']} x-heights; capitals {h['capital_height']}; digits {h['digit_height']}; tails (g, p, q, y) drop {h['descender_depth']}.",
          f"- Written in one stroke: {' '.join(h['one_stroke_letters']) or 'none'}. Usually two or more strokes: {' '.join(h['multi_stroke_letters']) or 'none'}.",
          f"- Most consistent letters: {' '.join(h['most_consistent_letters'])}. Most variable: {' '.join(h['most_variable_letters'])}.",
          "", "## Wording",
          f"- {pct(h['lines_starting_capital'])} of lines start with a capital; {pct(h['all_caps_share'])} of words are in all capitals; {pct(h['numeric_share'])} of tokens contain digits.",
          f"- Shorthand and non-dictionary short words: {', '.join(h['shorthand']) or 'none found'}.",
          f"- Frequent stand-alone marks: {' '.join(h['frequent_marks']) or 'none found'}.",
          "", "## By folder", "", "| Folder | Pages | Words per page | Tokens with digits | Main colour |", "|---|---|---|---|---|"]
    for name, r in h["by_folder"].items():
        L.append(f"| {name} | {r['pages']} | {r['words_per_page']} | {pct(r['numeric'])} | {r['main_colour']} |")
    return "\n".join(L) + "\n"


def build(pages: list[Page], samples: list[WordSample], lib: dict) -> dict:
    xh = lib["meta"]["xheight"]
    n_strokes = sum(len(p.strokes) for p in pages)
    used = [p for p in pages if len(p.strokes) > 20]
    text_strokes = sum(len(s.strokes) for s in samples)

    colours, per_page_colours, widths = Counter(), [], Counter()
    for p in used:
        names = Counter(_colour_name(s.color) for s in p.strokes if s.width <= 4)
        colours.update(names)
        per_page_colours.append(sum(1 for n in names.values() if n > 0.05 * sum(names.values())))
        widths.update(round(s.width, 1) for s in p.strokes)
    highlighter = sum(n for w, n in widths.items() if w > 4)

    lines: dict[tuple, list[WordSample]] = {}
    for s in samples:
        lines.setdefault((s.source, s.page, s.line), []).append(s)
    word_gaps, words_per_line, indents, line_tops = [], [], {}, {}
    for key, ws in lines.items():
        ws.sort(key=lambda w: w.box[0])
        ext = [np.vstack(w.strokes)[:, 0] for w in ws]  # OCR boxes are padded; measure the ink itself
        words_per_line.append(len(ws))
        indents.setdefault(key[:2], []).append(ws[0].box[0])
        line_tops.setdefault(key[:2], []).append(np.mean([w.box[3] for w in ws]))
        for a, b in zip(ext[:-1], ext[1:]):
            g = (b.min() - a.max()) / xh
            if 0.2 < g < 6:
                word_gaps.append(g)
    line_gaps, indented = [], []
    for key, tops in line_tops.items():
        d = np.diff(np.sort(tops)) / xh
        line_gaps += d[(d > 1.5) & (d < 12)].tolist()
        left = np.array(indents[key])
        if len(left) >= 4:
            indented.append(float(np.mean(left > np.percentile(left, 10) + 3 * xh)))

    joins = lib.get("join_rate", {})
    letter_pairs = {k: v for k, v in joins.items() if k.isalpha()}
    top_joined = sorted(letter_pairs.items(), key=lambda kv: -kv[1])[:12]
    from .segment import word_xheight

    sizes = [v for v in (word_xheight(s) for s in samples) if v]

    return {
        "sources": len({p.source for p in pages}),
        "pages_with_ink": len(used),
        "strokes": n_strokes,
        "words_read": len(samples),
        "characters_in_library": len(lib["glyphs"]),
        "x_height_units": round(xh, 2),
        "page_width_units": round(float(np.median([max(s.bbox[2] for s in p.strokes) for p in used])) + 12.0, 1) if used else None,
        "pen_width_to_x_height": round(float(np.median([s.width for p in used for s in p.strokes if s.width <= 4] or [0.0]) / xh), 2),
        "slant_degrees": round(_slant(used), 1) + 0.0,
        "letter_gap_x_heights": lib["spacing"]["letter_gap_mean"],
        "word_gap_x_heights": round(float(np.median(word_gaps)), 2) if word_gaps else None,
        "line_gap_x_heights": round(float(np.median(line_gaps)), 2) if line_gaps else None,
        "words_per_line_median": float(np.median(words_per_line)) if words_per_line else None,
        "size_variation": round(float(np.std(sizes) / np.mean(sizes)), 2) if sizes else None,
        "joined_pair_share": round(float(np.mean(list(letter_pairs.values()))), 2) if letter_pairs else None,
        "most_joined_pairs": top_joined,
        "share_of_lines_indented": round(float(np.mean(indented)), 2) if indented else None,
        "ink_not_in_words_share": round(1 - text_strokes / max(1, n_strokes), 2),
        "ink_colours": {k: round(v / sum(colours.values()), 3) for k, v in colours.most_common(8)},
        "colours_per_page_median": float(np.median(per_page_colours)) if per_page_colours else None,
        "highlighter_strokes": int(highlighter),
        "frequent_words": [w for w, _ in Counter(s.text.lower() for s in samples if s.text.isalpha() and len(s.text) > 3).most_common(25)],
    }


def to_markdown(p: dict) -> str:
    def pct(v):
        return "n/a" if v is None else f"{v * 100:.0f}%"

    def num(v, fmt=".0f"):
        return "n/a" if v is None else format(v, fmt)

    style = ("mostly print (separate letters)" if (p["joined_pair_share"] or 0) < 0.2
             else "mixed print and joined-up" if (p["joined_pair_share"] or 0) < 0.55 else "mostly joined-up (cursive)")
    L = [
        "# Handwriting and note-taking profile", "",
        f"Measured from {p['sources']} sources, {p['pages_with_ink']} pages, {p['strokes']:,} pen strokes, {p['words_read']:,} words read.", "",
        "## Letterforms",
        f"- Style: {style}; on average {pct(p['joined_pair_share'])} of letter pairs are joined.",
        f"- Most often joined: {', '.join(f'{k} ({v * 100:.0f}%)' for k, v in p['most_joined_pairs'])}.",
        f"- Slant: {p['slant_degrees']} degrees right of upright.",
        f"- Pen width is {p['pen_width_to_x_height']} of the x-height (the height of a letter like 'x').",
        f"- Letter size varies by {pct(p['size_variation'])} from word to word (headings included).",
        "", "## Spacing",
        f"- Between letters: {p['letter_gap_x_heights']} x-heights. Between words: {p['word_gap_x_heights']}. Between lines: {p['line_gap_x_heights']}.",
        f"- About {num(p['words_per_line_median'])} words per line.",
        "", "## Note layout",
        f"- {pct(p['share_of_lines_indented'])} of lines are indented under a parent line (outline style).",
        f"- {pct(p['ink_not_in_words_share'])} of strokes are not part of readable words (diagrams, arrows, maths, underlines).",
        f"- Colours: {', '.join(f'{k} {v * 100:.0f}%' for k, v in p['ink_colours'].items())}; a typical page uses {num(p['colours_per_page_median'])}.",
        f"- Highlighter strokes: {p['highlighter_strokes']}.",
        "", "## Vocabulary", f"- Frequent words: {', '.join(p['frequent_words'])}.", "",
    ]
    return "\n".join(L)
