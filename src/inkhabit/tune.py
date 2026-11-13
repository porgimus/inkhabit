# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measure how readable the generated handwriting is, and tune the library with that measurement.

The test is a round trip: write words with the library, read them back with the same recogniser used
to read the notes, and count how many come back right. Run once per variant of each character (that
variant forced, everything else as usual), it shows which samples make words unreadable. Those are
usually mis-cut or mislabelled samples that a contact sheet does not always reveal.
"""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np

from . import render
from .ocr import recognize

COMMON = ("the and that have with this from they would there their what about which when make can like time just him know "
          "take people into year your good some could them see other than then now look only come its over think also back "
          "after use two how our work first well way even new want because any these give day most us quick brown fox jumps "
          "lazy dog size zero quiz jazz exam box next six mix very every give move keep kind black week joke major object "
          "enjoy equal quite quote question").split()
CAPS = ("Monday Tuesday April June July August October November December Friday Sunday January March Paris London Berlin "
        "Canada Texas Kenya Ghana Italy Japan Vienna Zurich Xavier Quebec Oslo Egypt Yemen Utah Rome Hello Good Dear Yes "
        "World Winter Zoo King Queen Africa Asia Boston Brazil China Cuba Denver Dublin England Europe France Florida "
        "Germany Greece Hawaii Holland India Ireland Jordan Jamaica Korea Kansas Lima Lisbon Mexico Madrid Norway Nevada "
        "Ohio Ontario Peru Poland Qatar Quito Russia Rwanda Spain Sweden Turkey Tokyo Uganda Ukraine Venice Vermont "
        "Wales Warsaw Xerox Xenon York Yukon Zambia Zagreb").split()


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _words_for(ch: str, rng: np.random.Generator, n: int) -> list[str]:
    if ch.isdigit():
        return ["".join(rng.choice(list("0123456789"), 3)) + ch if i % 2 else ch + "".join(rng.choice(list("0123456789"), 3))
                for i in range(n)]
    pool = [w for w in (CAPS if ch.isupper() else COMMON) if ch in w]
    if not pool:
        return []
    return [str(w) for w in rng.choice(pool, min(n, len(pool)), replace=False)]


def read_back(lib: dict, words: list[str], force: dict[str, int] | None = None, seed: int = 0, human: render.Human | None = None) -> float:
    """Share of `words` that are read back correctly when written with this library."""
    if not words:
        return float("nan")
    h = human or replace(render.Human.from_library(lib).scaled(0.4), whole_words=0.0, joins=0.0, typical=1.0)
    w = render.Writer(lib, seed, h)
    w.force = force or {}
    ink = w.write("   ".join(words), x_height=26, max_width=1500, line_gap=3.6)
    got = [_norm(x.text) for x in recognize(render.to_image(ink, supersample=2))]
    want = [_norm(x) for x in words]
    return sum(1 for x in want if x in got) / len(want)


def score_variants(lib: dict, chars: str | None = None, per_variant: int = 8, seed: int = 0, log=print, rounds: int = 3) -> dict[str, list[float]]:
    """{char: [readability of each variant]} for letters and digits."""
    rng = np.random.default_rng(seed)
    out = {}
    for ch, entry in lib["glyphs"].items():
        if not ch.isalnum() or (chars and ch not in chars) or len(entry["variants"]) < 2:
            continue
        words = _words_for(ch, rng, per_variant)
        if len(words) < 3:
            continue
        out[ch] = [read_back(lib, words, {ch: i}, seed + i) for i in range(len(entry["variants"]))]
        for r in range(1, rounds):  # different words and different neighbours each round, then average
            more = _words_for(ch, rng, per_variant)
            nxt = [read_back(lib, more, {ch: i}, seed + 100 * r + i) for i in range(len(entry["variants"]))]
            out[ch] = [a + b for a, b in zip(out[ch], nxt)]
        out[ch] = [round(v / rounds, 3) for v in out[ch]]
        log(f"{ch}: " + " ".join(f"{s:.2f}" for s in out[ch]))
    return out


def apply(lib: dict, scores: dict[str, list[float]], keep_at_least: int = 3) -> dict[str, list[int]]:
    """Drop variants that read clearly worse than their siblings, and put the most readable first
    (the first three are the ones the neat style prefers). Returns what was dropped."""
    dropped = {}
    for ch, sc in scores.items():
        vs = lib["glyphs"][ch]["variants"]
        order = sorted(range(len(vs)), key=lambda i: -sc[i])
        best = sc[order[0]]
        keep = [i for i in order if sc[i] >= max(0.5 * best, best - 0.35)]
        keep = (keep + [i for i in order if i not in keep])[: max(keep_at_least, len(keep))]
        dropped[ch] = sorted(set(range(len(vs))) - set(keep))
        lib["glyphs"][ch]["variants"] = [vs[i] for i in keep]
    return {c: d for c, d in dropped.items() if d}


def legibility(lib: dict, seeds=(1, 2, 3, 4, 5, 6), style: str = "clean") -> float:
    """Overall readability of ordinary writing in a style: share of common words read back correctly."""
    words = COMMON[:90]
    h = replace(render.style(lib, style), whole_words=0.0)
    return float(np.mean([read_back(lib, words[i % 3 :: 3], None, s, h) for i, s in enumerate(seeds)]))


def score_pairs(lib: dict, per_variant: int = 8, seed: int = 0, rounds: int = 2, log=print) -> dict[str, list[float]]:
    """Readability of each joined-pair sample ("th", "he"...), tested inside words with the pair forced."""
    rng = np.random.default_rng(seed)
    h = replace(render.Human.from_library(lib).scaled(0.4), whole_words=0.0, joins=50.0, typical=1.0)
    out = {}
    for pair, variants in lib.get("pairs", {}).items():
        pool = [w for w in COMMON + CAPS if pair in w]
        if len(pool) < 2:
            continue
        only = {p: 0.0 for p in lib["join_rate"]}  # switch every other pair off, so only this one is under test
        saved, lib["join_rate"] = lib["join_rate"], {**only, pair: 1.0}
        try:
            tot = [0.0] * len(variants)
            for r in range(rounds):
                words = [str(w) for w in rng.choice(pool, min(per_variant, len(pool)), replace=False)]
                for i in range(len(variants)):
                    tot[i] += read_back(lib, words, {"p:" + pair: i}, seed + 10 * r + i, h)
            plain = float(np.mean([read_back(lib, [str(w) for w in pool[:per_variant]], None, seed + k,
                                             replace(h, joins=0.0)) for k in range(2)]))
        finally:
            lib["join_rate"] = saved
        out[pair] = [round(t / rounds, 3) for t in tot] + [round(plain, 3)]  # last entry: the same words without the pair
        log(f"{pair}: " + " ".join(f"{s:.2f}" for s in out[pair][:-1]) + f" | unjoined {plain:.2f}")
    return out


def apply_pairs(lib: dict, scores: dict[str, list[float]]) -> dict[str, list[int]]:
    """Drop joined-pair samples that read worse than writing the two letters separately."""
    dropped = {}
    for pair, sc in scores.items():
        *each, plain = sc
        keep = [i for i, s in enumerate(each) if s >= plain - 0.13]
        dropped[pair] = [i for i in range(len(each)) if i not in keep]
        lib["pairs"][pair] = [lib["pairs"][pair][i] for i in sorted(keep, key=lambda i: -each[i])]
        if not lib["pairs"][pair]:
            del lib["pairs"][pair]
    return {p: d for p, d in dropped.items() if d}


def tune_words(lib: dict, seed: int = 0, batch: int = 10) -> int:
    """Drop whole-word samples that are not read back as that word. Returns how many were dropped."""
    h = replace(render.Human.from_library(lib).scaled(0.3), whole_words=1.0, joins=0.0)
    items = [(w, i) for w, vs in lib.get("words", {}).items() if w.isalpha() and len(w) > 1 for i in range(len(vs))]
    bad: set[tuple[str, int]] = set()
    k = -batch
    while (k := k + batch) < len(items):  # items can grow below
        chunk = items[k : k + batch]
        if len({w for w, _ in chunk}) < len(chunk):  # the same word twice in one line cannot be told apart
            chunk = list({w: (w, i) for w, i in chunk}.values())
            items += [it for it in items[k : k + batch] if it not in chunk]
        wr = render.Writer(lib, seed + k, h)
        wr.force = {"w:" + w: i for w, i in chunk}
        ink = wr.write("   ".join(w for w, _ in chunk), x_height=26, max_width=1700, line_gap=3.6)
        got = {_norm(x.text) for x in recognize(render.to_image(ink, supersample=2))}
        bad |= {(w, i) for w, i in chunk if _norm(w) not in got}
    for w in list(lib.get("words", {})):
        lib["words"][w] = [v for i, v in enumerate(lib["words"][w]) if (w, i) not in bad]
        if not lib["words"][w]:
            del lib["words"][w]
    return len(bad)


def run(lib: dict, log=print) -> dict:
    """The whole tuning pass. Returns a before/after report; `lib` is changed in place."""
    before = {st: round(legibility(lib, style=st), 3) for st in ("clean", "natural", "rushed")}
    dropped = apply(lib, score_variants(lib, log=lambda *a: None))
    pairs = apply_pairs(lib, score_pairs(lib, log=lambda *a: None))
    words = tune_words(lib)
    calibrate(lib, log=lambda *a: None)  # match the real notes' irregularity, when the build recorded it
    after = {st: round(legibility(lib, style=st), 3) for st in ("clean", "natural", "rushed")}
    report = {"read_back_before": before, "read_back_after": after, "letter_variants_dropped": dropped,
              "pair_samples_dropped": sum(len(v) for v in pairs.values()), "whole_word_samples_dropped": words}
    log(f"read-back accuracy before {before} -> after {after}")
    log(f"dropped: {sum(len(v) for v in dropped.values())} letter variants, {report['pair_samples_dropped']} joined-pair samples, "
        f"{words} whole-word samples")
    return report


def measure_generated(lib: dict, human: render.Human, n_words: int = 360, seed: int = 0) -> dict:
    """Write words and measure them with the same instrument used on the real notes
    (same size estimate, same letter cutting, same spread statistics), so the two can be compared."""
    from . import library, segment
    from .words import WordSample

    rng = np.random.default_rng(seed)
    wr = render.Writer(lib, seed, replace(human, whole_words=0.0))
    wr.base_slant = 0.0  # measured upright, like the stored samples
    words = [str(w) for w in rng.choice(COMMON, n_words)]
    samples, unit = [], 10.0
    for k, w in enumerate(words):
        if k % 5 == 0:  # a new line: lines differ a little in size, as real ones do
            line_scale = 1 + rng.normal(0, human.line_size)
        strokes, _, missing = wr._word(w)
        if missing or not strokes:
            continue
        pts = [np.asarray(s, float) * unit * line_scale + [k % 5 * 400.0, k // 5 * 100.0] for s in strokes]
        samples.append(WordSample(w, 1.0, pts, [1.0] * len(pts), (0, 0, 0, 0), "generated", 0, k // 5))
    sizes, _ = segment.size_map(samples)
    models = segment.models_from_library(lib)
    aligned = []
    for i, s in enumerate(samples):
        p = segment.prepare(s, s.text, sizes[i])
        a = segment.align(p, models) if p else None
        if a:
            aligned.append(a)
    return library.raw_spread(aligned, word_cost=1e9)


def calibrate(lib: dict, target: dict | None = None, rounds: int = 5, log=print) -> dict:
    """Match the natural style to the real notes on the two things that can be measured reliably: the gap
    between letters and how often letters are joined. Generated words are measured with the same
    instrument as the real ones and the settings adjusted until they agree.
    target = library.raw_spread() of the real aligned words (stored by `build` as lib["measured"]["real"])."""
    target = target or lib.get("measured", {}).get("real")
    if not target:
        return {}
    base = render.Human.from_library(lib)
    v = {"gap": base.letter_spacing, "joined": base.joins}
    lim = {"gap": (0.7, 1.5), "joined": (0.3, 1.6)}
    got = {}
    for r in range(rounds):
        got = measure_generated(lib, replace(base, letter_spacing=v["gap"], joins=v["joined"]), seed=r)
        if not got:
            return {}
        log(f"round {r}: " + "; ".join(f"{k} {got[k]:.3f} (real {target[k]:.3f})" for k in v))
        for k in v:  # damped steps: part of what the instrument sees is its own error
            v[k] = float(np.clip(v[k] * (target[k] / max(got[k], 1e-3)) ** 0.6, *lim[k]))
    m = lib.setdefault("measured", {})
    m.pop("bounce_use", None)
    m.pop("size_use", None)
    m.update({"letter_spacing_use": round(v["gap"], 3), "joins_use": round(v["joined"], 3)})
    return {"settings": v, "generated": got, "real": target}

