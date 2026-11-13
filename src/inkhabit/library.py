# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pick the best letter samples and save them as a reusable handwriting library.

Units in the library: 1.0 = the writer's x-height, origin on the baseline at
the letter's left edge, y pointing down (so ascenders are negative).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from .segment import Aligned, _feat

VARIANTS = 8
WORD_VARIANTS = 4
MAX_WORDS = 600
MAX_PAIRS = 150
PAIR_VARIANTS = 4


def _norm(pieces: list[np.ndarray], x0: float, baseline: float, xh: float) -> list[list[list[float]]]:
    out = []
    for p in pieces:
        q = (p - [x0, baseline]) / xh
        if len(q) > 2:
            q = cv2.approxPolyDP(q.astype(np.float32)[:, None], 0.012, False)[:, 0]
        out.append(np.round(q, 3).tolist())
    return out


def _join_pieces(pieces: list[np.ndarray]) -> list[np.ndarray]:
    """Re-join pieces that were one pen stroke before cutting."""
    out: list[np.ndarray] = []
    for p in pieces:
        if out and np.allclose(out[-1][-1], p[0], atol=1e-6):
            out[-1] = np.vstack([out[-1], p[1:]])
        else:
            out.append(p)
    return out


def _diverse(feats: np.ndarray, k: int, rng: np.random.Generator) -> list[int]:
    """Indices of up to k samples spread across the natural variation."""
    if len(feats) <= k:
        return list(range(len(feats)))
    centres = feats[rng.choice(len(feats), k, replace=False)]
    for _ in range(10):
        lab = ((feats[:, None] - centres[None]) ** 2).sum(-1).argmin(1)
        centres = np.array([feats[lab == c].mean(0) if (lab == c).any() else centres[c] for c in range(k)])
    d = ((feats[:, None] - centres[None]) ** 2).sum(-1)
    sizes = Counter(d.argmin(1).tolist())
    picks = [int(d[:, c].argmin()) for c, _ in sizes.most_common()]
    return list(dict.fromkeys(picks))


def _max_width(ch: str) -> float:
    if ch in "mwMW":
        return 2.6
    return 2.0 if ch.isupper() or ch in "%&@#" else 1.5


def drop_variants(lib: dict, drops: dict[str, list[int]], pair_drops: dict[str, list[int]] | None = None) -> dict:
    """Remove reviewed-and-rejected variants, e.g. {"K": [0, 2]}; an empty char is removed."""
    for pair, idxs in (pair_drops or {}).items():
        if pair in lib.get("pairs", {}):
            lib["pairs"][pair] = [v for i, v in enumerate(lib["pairs"][pair]) if i not in set(idxs)]
            if not lib["pairs"][pair]:
                del lib["pairs"][pair]
    for ch, idxs in drops.items():
        if ch in lib["glyphs"]:
            vs = lib["glyphs"][ch]["variants"]
            lib["glyphs"][ch]["variants"] = [v for i, v in enumerate(vs) if i not in set(idxs)]
            if not lib["glyphs"][ch]["variants"]:
                del lib["glyphs"][ch]
                lib.setdefault("rejected", []).append(ch)  # so a later harvest does not bring it back
    return lib


def build(aligned: list[Aligned], meta: dict, word_cost: float = 2.5, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    good = [a for a in aligned if np.mean(a.costs) <= word_cost and max(a.costs) <= 3 * word_cost]
    cands: dict[str, list[tuple[float, Aligned, int]]] = defaultdict(list)
    gaps, joins = [], Counter()
    pairs: dict[str, list[tuple[float, Aligned, int]]] = defaultdict(list)
    bounce, heights = [], []
    for a in good:
        p = a.prep
        for k, (ch, (i, j)) in enumerate(zip(p.text, a.groups)):
            cands[ch].append((a.costs[k], a, k))
            ys = np.concatenate([q[:, 1] for g in p.prims[i:j] for q in g])
            if ch.isalpha() and ch not in "gjpqyQ":
                bounce.append((ys.max() - p.baseline) / p.xh)  # how far the letter's foot is from the baseline
            if ch in "acemnorsuvwxz":
                heights.append(np.ptp(ys) / p.xh)
            if k + 1 < len(p.text):
                ni = a.groups[k + 1][0]
                gaps.append((p.bounds[ni:a.groups[k + 1][1], 0].min() - p.bounds[i:j, 1].max()) / p.xh)
                joined = np.isclose(p.bounds[ni, 0], p.bounds[j - 1, 1], atol=1e-6)
                joins[(ch + p.text[k + 1], bool(joined))] += 1
                if joined and (ch + p.text[k + 1]).isalpha() and max(a.costs[k], a.costs[k + 1]) < 1.2:
                    pairs[ch + p.text[k + 1]].append((a.costs[k] + a.costs[k + 1], a, k))
    glyphs = {}
    for ch, items in cands.items():
        if not ch.isascii() and len(items) < 5:
            continue  # too few samples of an accented letter: it is composed from the plain one instead
        items.sort(key=lambda t: t[0])
        # a "letter" much wider than letters get has swallowed its neighbour
        widths = np.array([np.ptp(a.prep.bounds[a.groups[k][0]:a.groups[k][1]]) / a.prep.xh for _, a, k in items])
        cap = min(_max_width(ch), 1.45 * float(np.median(widths)) + 0.15)
        items = [it for it, w in zip(items, widths) if w <= cap and it[0] < 30]
        if not items:
            continue
        items = items[: max(VARIANTS, int(len(items) * 0.4))][:400]
        feats = np.array([_feat(a.prep, *a.groups[k]) for _, a, k in items])
        variants = []
        for ix in _diverse(feats, VARIANTS, rng):
            cost, a, k = items[ix]
            p, (i, j) = a.prep, a.groups[k]
            pieces = _join_pieces([q for g in p.prims[i:j] for q in g])
            x0 = min(q[:, 0].min() for q in pieces)
            x1 = max(q[:, 0].max() for q in pieces)
            variants.append({
                "strokes": _norm(pieces, x0, p.baseline, p.xh),
                "width": round(float((x1 - x0) / p.xh), 3),
                "cost": round(float(cost), 3),
                "from": p.text,
                "pos": "s" if len(p.text) == 1 else "i" if k == 0 else "f" if k == len(p.text) - 1 else "m",
            })
        glyphs[ch] = {"count": len(cands[ch]), "variants": variants}

    words: dict[str, list] = defaultdict(list)
    word_freq = Counter(a.prep.text for a in good)
    for a in sorted(good, key=lambda a: np.mean(a.costs)):
        p = a.prep
        if len(p.text) < 2 or not p.text.isalpha() or len(words[p.text]) >= WORD_VARIANTS:
            continue
        pieces = _join_pieces([q for g in p.prims for q in g])
        x0 = min(q[:, 0].min() for q in pieces)
        x1 = max(q[:, 0].max() for q in pieces)
        words[p.text].append({"strokes": _norm(pieces, x0, p.baseline, p.xh), "width": round((x1 - x0) / p.xh, 3)})

    # joined pairs exactly as written ("th", "he"...): real joins beat drawn ones
    bigrams = {}
    for pair, items in sorted(pairs.items(), key=lambda kv: -len(kv[1]))[:MAX_PAIRS]:
        if len(items) < 6:
            continue
        items.sort(key=lambda t: t[0])
        items = items[: PAIR_VARIANTS * 4]
        w = np.array([(a.prep.bounds[a.groups[k + 1][1] - 1, 1] - a.prep.bounds[a.groups[k][0], 0]) / a.prep.xh for _, a, k in items])
        keep = [it for it, wi in zip(items, w) if abs(wi - np.median(w)) <= 0.35 * np.median(w) + 0.1]
        feats = np.array([_feat(a.prep, a.groups[k][0], a.groups[k + 1][1]) for _, a, k in keep])
        out = []
        for ix in _diverse(feats, PAIR_VARIANTS, rng):
            _, a, k = keep[ix]
            p, i, j = a.prep, a.groups[k][0], a.groups[k + 1][1]
            pieces = _join_pieces([q for g in p.prims[i:j] for q in g])
            x0 = min(q[:, 0].min() for q in pieces)
            x1 = max(q[:, 0].max() for q in pieces)
            out.append({"strokes": _norm(pieces, x0, p.baseline, p.xh), "width": round(float((x1 - x0) / p.xh), 3)})
        bigrams[pair] = out

    def spread(v, lo, hi):
        v = np.asarray(v)
        if len(v) < 30:
            return None
        q1, q3 = np.percentile(v, [25, 75])  # robust to the occasional mis-split letter
        return round(float(np.clip((q3 - q1) / 1.349, lo, hi)), 3)

    measured = {"bounce": spread(bounce, 0.01, 0.15),
                "size": spread(np.asarray(heights) / max(np.median(heights), 1e-6), 0.01, 0.15) if heights else None}

    join_rate = {}
    for (pair, joined), n in joins.items():
        tot = joins[(pair, True)] + joins[(pair, False)]
        if tot >= 5:
            join_rate[pair] = round(joins[(pair, True)] / tot, 2)
    g = np.array(gaps or [0.1])
    return {
        "format": "inkhabit-library/1",
        "meta": {**meta, "words_used": len(good), "words_aligned": len(aligned)},
        "spacing": {"letter_gap_mean": round(float(np.median(g)), 3), "letter_gap_sd": round(float(g.std()), 3)},
        "join_rate": join_rate,
        "measured": {k: v for k, v in measured.items() if v is not None},
        "pairs": dict(sorted(bigrams.items())),
        "glyphs": dict(sorted(glyphs.items())),
        "words": dict(sorted(sorted(words.items(), key=lambda kv: -word_freq[kv[0]])[:MAX_WORDS])),
    }


def save(lib: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(lib, separators=(",", ":")))


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def pair_specimen(lib: dict, path: str | Path) -> None:
    """Contact sheet of the joined letter pairs."""
    if lib.get("pairs"):
        specimen({"glyphs": {k: {"count": len(v), "variants": v} for k, v in lib["pairs"].items()}}, path, cell=64, cols=3)


def specimen(lib: dict, path: str | Path, cell: int = 90, cols: int = 2) -> None:
    """Contact sheet: every character with all its variants, for a human or agent to review."""
    chars = list(lib["glyphs"])
    sc = cell / 3.6
    rows = []
    for ch in chars:
        nvar = max(len(g["variants"]) for g in lib["glyphs"].values())
        row = np.full((cell, cell * (nvar + 1), 3), 255, np.uint8)
        cv2.putText(row, ch, (6, cell - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0 if len(ch) == 1 else 0.7, (180, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(row, str(lib["glyphs"][ch]["count"]), (10, cell - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.line(row, (cell, int(2.2 * sc)), (row.shape[1], int(2.2 * sc)), (225, 225, 225), 1)
        cv2.line(row, (cell, int(1.2 * sc)), (row.shape[1], int(1.2 * sc)), (240, 240, 240), 1)
        for v, var in enumerate(lib["glyphs"][ch]["variants"]):
            ox = cell * (v + 1) + (cell - var["width"] * sc) / 2
            for s in var["strokes"]:
                q = (np.array(s) * sc + [ox, 2.2 * sc]) * 16
                cv2.polylines(row, [q.astype(np.int32)], False, (0, 0, 0), 2, cv2.LINE_AA, 4)
            cv2.putText(row, str(v), (cell * (v + 1) + 3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
        rows.append(row)
    if not rows:
        return
    per = (len(rows) + cols - 1) // cols
    rows += [np.full_like(rows[0], 255)] * (per * cols - len(rows))
    cv2.imwrite(str(path), np.hstack([np.vstack(rows[c * per : (c + 1) * per]) for c in range(cols)]))


def add_whole_words(lib: dict, samples, sizes: dict[int, float], per_word: int = WORD_VARIANTS) -> int:
    """Keep every word whose text we trust as one unit, exactly as written.

    For joined-up writing this matters more than letters: the letters inside a fast, joined word are
    not separately formed, so they cannot be cut out, but the whole word can be reused as it stands.
    Returns how many word samples were added.
    """
    from .segment import _baseline, _resample

    added = 0
    words = lib.setdefault("words", {})
    for i, s in enumerate(samples):
        if not s.trusted or len(s.text) < 1 or not any(c.isalnum() for c in s.text):
            continue
        strokes = [np.asarray(p, float) for p in s.strokes if len(p)]
        if not strokes:
            continue
        xh = sizes[i]
        # the baseline is the foot of the band where most of the ink sits; this ignores stray marks
        # (bits of ruled line, a neighbour's descender) that a simple "lowest point" would follow
        dense = np.vstack([_resample(p, xh / 14) for p in strokes])
        base = _baseline(dense, "g", xh)
        strokes = [p for p in strokes if base - 2.4 * xh < (p[:, 1].min() + p[:, 1].max()) / 2 < base + 1.2 * xh]
        if not strokes:
            continue
        pts = np.vstack(strokes)
        if np.ptp(pts[:, 0]) > 40 * xh:
            continue  # ink from elsewhere on the line got attached; not a clean word
        x0, x1 = pts[:, 0].min(), pts[:, 0].max()
        entry = {"strokes": _norm(strokes, x0, base, xh), "width": round(float((x1 - x0) / xh), 3), "source": "whole"}
        bucket = words.setdefault(s.text, [])
        if len(bucket) < per_word:
            bucket.append(entry)
            added += 1
    lib.setdefault("meta", {})["whole_words_from_transcript"] = added
    return added


def raw_spread(aligned: list[Aligned], word_cost: float = 2.5) -> dict:
    """How irregular the real writing is, measured from well-aligned words (no clipping):
    bounce = how far letter feet stray from the baseline; size = how much small letters vary in height."""
    bounce, heights, gaps, joined = [], [], [], 0
    for a in aligned:
        if np.mean(a.costs) > word_cost:
            continue
        p = a.prep
        for k in range(len(p.text) - 1):
            (i0, j0), (i1, j1) = a.groups[k], a.groups[k + 1]
            gaps.append((p.bounds[i1:j1, 0].min() - p.bounds[i0:j0, 1].max()) / p.xh)
            joined += bool(np.isclose(p.bounds[i1, 0], p.bounds[j0 - 1, 1], atol=1e-6))
        for ch, (i, j) in zip(p.text, a.groups):
            ys = np.concatenate([q[:, 1] for g in p.prims[i:j] for q in g])
            if ch.isalpha() and ch not in "gjpqyQ":
                bounce.append((ys.max() - p.baseline) / p.xh)
            if ch in "acemnorsuvwxz":
                heights.append(np.ptp(ys) / p.xh)

    def sd(v):
        q1, q3 = np.percentile(v, [25, 75])
        return float((q3 - q1) / 1.349)

    if len(bounce) < 20 or len(heights) < 20:
        return {}  # too little to measure
    h = np.asarray(heights)
    return {"bounce": sd(bounce), "size": sd(h / np.median(h)), "gap": float(np.median(gaps)) if gaps else 0.25,
            "joined": joined / max(1, len(gaps)), "n": len(bounce)}


MARKS = {".": (-0.12, 0.02), ",": (-0.14, 0.3), ":": (-0.78, 0.02), ";": (-0.78, 0.3), "'": (-1.75, -1.25), '"': (-1.75, -1.25)}


def canon(lib: dict) -> tuple[dict[str, tuple[float, float]], dict]:
    """Each character's canonical top and bottom: the median of its own samples. Small marks are placed by
    rule, because measured from scraps of ink they come out hanging at odd heights."""
    out: dict[str, tuple[float, float]] = {}
    for ch, e in lib["glyphs"].items():
        ext = [(min(p[1] for s in v["strokes"] for p in s), max(p[1] for s in v["strokes"] for p in s))
               for v in e["variants"] if v["strokes"]]
        if ext:
            out[ch] = (float(np.median([a for a, _ in ext])), float(np.median([b for _, b in ext])))
    out.update({c: v for c, v in MARKS.items() if c in out})
    return out, MARKS


def fit(entry: dict, text: str, canon: dict, marks: dict, always: bool = False) -> dict | None:
    """Rescale and shift one sample so its height is what its characters' canonical measurements predict.
    Returns None when it is so far off that it is probably not a clean sample (unless always=True)."""
    known = [canon[c] for c in text if c in canon]
    ys = [p[1] for s in entry["strokes"] for p in s]
    if not known or not ys:
        return entry
    want_top, want_bot = min(a for a, _ in known), max(b for _, b in known)
    top, bot = min(ys), max(ys)
    if (want_bot - want_top < 0.5 and text not in marks) or bot - top < 0.15:  # dots, dashes: too flat to scale by; just place them
        k, dy = 1.0, (want_top + want_bot) / 2 - (top + bot) / 2
    else:
        k = (want_bot - want_top) / (bot - top)
        if not 0.55 <= k <= 1.8 and not always:
            return None
        dy = want_bot - bot * k
    xs = [p[0] * k for s in entry["strokes"] for p in s]
    x0 = min(xs)
    return {**entry, "strokes": [[[round(x * k - x0, 3), round(y * k + dy, 3)] for x, y in s] for s in entry["strokes"]],
            "width": round(max(xs) - x0, 3)}


def unify(lib: dict, log=print) -> dict:
    """Bring every component to one scale and one baseline.

    Samples are cut from all over the sources: body text, headings, margin notes, each normalised by a
    size estimate that can be a quarter out. Used as they are, a word lifted from a heading comes out
    huge beside letters taken from small print. Here each character gets canonical measurements (the
    median top and bottom of its own samples), and every letter sample, joined pair, whole word and
    symbol is rescaled and shifted so that its height is what its letters say it should be.
    Irregularity is then added back deliberately by the writer, evenly, instead of arriving by accident.
    """
    g = lib["glyphs"]
    cn, marks = canon(lib)

    def fit_(entry, text, always=False):
        return fit(entry, text, cn, marks, always)

    stats = {"letters": 0, "pairs_dropped": 0, "words_dropped": 0}
    for ch, e in g.items():
        e["variants"] = [fit_(v, ch, always=True) for v in e["variants"]]  # a letter is never dropped here, only resized
        stats["letters"] += len(e["variants"])
    for key, label in (("pairs", "pairs_dropped"), ("words", "words_dropped")):
        for text in list(lib.get(key, {})):
            fitted = [fit_(v, text) for v in lib[key][text]]
            stats[label] += sum(f is None for f in fitted)
            lib[key][text] = [f for f in fitted if f is not None]
            if not lib[key][text]:
                del lib[key][text]
    lib["format"] = "inkhabit-library/2"
    lib.setdefault("meta", {})["unified"] = True
    log(f"unified {stats['letters']} letter samples; dropped {stats['pairs_dropped']} joined pairs and "
        f"{stats['words_dropped']} whole words that were too far from the expected size")
    return lib


def pair_gaps(aligned: list[Aligned], word_cost: float = 2.5, at_least: int = 8) -> dict[str, float]:
    """The usual gap between each pair of neighbouring characters, in x-heights, from well-cut words.
    People do not space letters evenly: "ra" sits closer than "ll". Only pairs seen often enough are kept."""
    seen: dict[str, list[float]] = defaultdict(list)
    for a in aligned:
        if np.mean(a.costs) > word_cost:
            continue
        p = a.prep
        for k in range(len(p.text) - 1):
            (i0, j0), (i1, j1) = a.groups[k], a.groups[k + 1]
            seen[p.text[k : k + 2]].append((p.bounds[i1:j1, 0].min() - p.bounds[i0:j0, 1].max()) / p.xh)
    return {pair: round(float(np.median(v)), 3) for pair, v in seen.items() if len(v) >= at_least}
