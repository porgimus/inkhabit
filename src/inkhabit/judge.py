# SPDX-License-Identifier: AGPL-3.0-or-later
"""Optional: ask TypeSafe's Jev model how clearly each sample reads as its character.

This is the one part of inkhabit that sends anything off the machine: a small picture of each
letter sample goes to api.typesafe.ai. It is never run unless asked for (`inkhabit judge`), and it
needs the user's own key in the TYPESAFE_API_KEY environment variable.

Jev answers typed questions about an image with probabilities, which suits this exactly:
"is this clearly a lowercase e?" and "which of e, c, o is it?". The answers rank each letter's samples
by clarity, so clean writing can prefer the clearest ones, and they expose mislabelled samples.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

URL = "https://api.typesafe.ai/v1/systemone"
# letters that are easily mistaken for one another in handwriting
LOOKALIKES = ["ecoa", "tfl", "nuhr", "mnw", "vuy", "gyq9", "bh6", "dao", "il1jI", "sS5", "zZ2", "O0oD", "CcG", "UuV", "KkR", "pP",
              "xX", "wW", "B8", "T7", "Zz", "A4", "E3F", "Mm", "Nn", "Hh", "Ll", "Jj", "Yy", "Qq", "Ww", ".,", ":;", "!1", "?2"]


def _names(ch: str) -> str:
    if ch.isalpha():
        return f"{'lowercase' if ch.islower() else 'capital'} letter {ch}"
    return f"digit {ch}" if ch.isdigit() else f"character {ch}"


def picture(variant: dict, size: int = 320) -> bytes:  # the service needs at least 280 px a side
    """The sample drawn on a baseline, with a faint line at small-letter height so size is visible."""
    im = np.full((size, size, 3), 255, np.uint8)
    s = size / 4.6
    base = int(size * 0.66)
    cv2.line(im, (8, base), (size - 8, base), (190, 190, 190), 1)
    cv2.line(im, (8, int(base - s)), (size - 8, int(base - s)), (225, 225, 225), 1)
    ox = size / 2 - variant["width"] * s / 2
    for st in variant["strokes"]:
        q = np.round((np.asarray(st, float) * s + [ox, base]) * 16).astype(np.int32)
        if len(q) == 1:
            cv2.circle(im, tuple(int(v) for v in q[0]), 4 * 16, (20, 20, 20), -1, cv2.LINE_AA, 4)
        else:
            cv2.polylines(im, [q], False, (20, 20, 20), 5, cv2.LINE_AA, 4)
    ok, buf = cv2.imencode(".png", im)
    return buf.tobytes()


def ask(ch: str, png: bytes, key: str, model: str, timeout: float = 30.0) -> dict:
    options = {c for group in LOOKALIKES if ch in group for c in group} | {ch}
    ref = str(uuid.uuid4())  # the image is referred to by a key that cannot occur in the text by accident
    body = {
        "model": model,
        "state": {"sample": ref, "about": "One handwritten character. The darker grey line is the writing baseline; the "
                                           "lighter line above it is the height of a small letter such as x."},
        "assets": {ref: {"type": "image", "format": "png", "data_base64": base64.b64encode(png).decode("ascii")}},
        "questions": {
            "clear": {"type": "noul", "instructions": f"Is the handwritten character in `sample` clearly a {_names(ch)}?",
                      "criteria": {"true": f"A reader would read it as {ch} without hesitation",
                                   "false": "It is ambiguous, incomplete, has stray marks, or looks like a different character"}},
            "which": {"type": "choice", "instructions": "Which character is written in `sample`?",
                      "criteria": {**{c: _names(c) for c in sorted(options)}, "other": "None of these, or not a single clear character"}},
        },
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json", "TypeSafe-Danger-Zone-Images": "true"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    a = out["answers"]
    probs = a["which"].get("probabilities", {})
    return {"clear": float(a["clear"]["noul"]), "which": a["which"].get("choice"), "p_right": float(probs.get(ch, 0.0)),
            "tokens": int(out.get("usage", {}).get("input_tokens", 0))}


def run(lib: dict, model: str = "jev-latest", limit: int | None = None, workers: int = 6, log=print) -> dict:
    """Score every real sample of every letter and digit. Adds "clarity" to each variant, orders variants
    clearest first, and returns a report. Stops at the first refusal from the service."""
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise SystemExit("TYPESAFE_API_KEY is not set. This step sends letter images to TypeSafe and needs your key.")
    jobs = [(ch, i, v) for ch, e in lib["glyphs"].items() if ch.isalnum() and e.get("count", 0) > 0
            for i, v in enumerate(e["variants"]) if v.get("source") in (None, "sheet", "harvested")]
    jobs = jobs[:limit] if limit else jobs
    results: dict[tuple[str, int], dict] = {}

    def one(job):
        ch, i, v = job
        try:
            return (ch, i), ask(ch, picture(v), key, model)
        except urllib.error.HTTPError as e:
            return (ch, i), {"error": f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"}
        except Exception as e:  # network trouble on one sample should not lose the rest
            return (ch, i), {"error": repr(e)[:200]}

    if not jobs:
        raise SystemExit("nothing to judge: the library has no samples cut from the person's own writing")
    first = one(jobs[0])  # one request first: if the key or the model is wrong, find out before sending the rest
    if "error" in first[1]:
        raise SystemExit(f"TypeSafe refused the first request, nothing more was sent: {first[1]['error']}")
    results[first[0]] = first[1]
    with ThreadPoolExecutor(workers) as ex:
        for k, r in ex.map(one, jobs[1:]):
            results[k] = r
    tokens = sum(r.get("tokens", 0) for r in results.values())
    errors = [k for k, r in results.items() if "error" in r]
    wrong = []
    for (ch, i), r in results.items():
        if "error" in r:
            continue
        lib["glyphs"][ch]["variants"][i]["clarity"] = round(r["clear"] * (0.5 + 0.5 * r["p_right"]), 3)
        if r["which"] not in (ch, None) and r["p_right"] < 0.2:
            wrong.append((ch, i, r["which"], round(r["clear"], 2)))
    for ch in {c for c, _ in results}:
        vs = lib["glyphs"][ch]["variants"]
        lib["glyphs"][ch]["variants"] = sorted(vs, key=lambda v: -v.get("clarity", 0.5))  # clearest first
    log(f"judged {len(results) - len(errors)} samples, {tokens:,} input tokens, {len(errors)} failed; "
        f"{len(wrong)} look like a different character")
    return {"judged": len(results) - len(errors), "input_tokens": tokens, "failed": len(errors),
            "read_as_another_character": wrong, "first_error": results[errors[0]]["error"] if errors else None}


def reselect(lib: dict, aligned, chars: str, per_char: int = 40, keep: int = 8, model: str = "jev-latest", workers: int = 6,
             log=print) -> dict:
    """For letters whose chosen samples read poorly, go back to the sources: take many candidate cuts of
    that letter, have each judged, and keep the clearest. `aligned` is the re-cut words (segment.run).
    A letter's samples are only replaced when the new set is clearer than the old one."""
    from . import library
    from .library import _join_pieces, _max_width, _norm

    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise SystemExit("TYPESAFE_API_KEY is not set.")
    canon, marks = library.canon(lib)
    pool: dict[str, list] = {c: [] for c in chars}
    for a in aligned:
        if np.mean(a.costs) > 2.5:
            continue
        p = a.prep
        for k, (ch, (i, j)) in enumerate(zip(p.text, a.groups)):
            if ch in pool:
                pool[ch].append((a.costs[k], a, k))
    report = {}
    for ch, items in pool.items():
        items.sort(key=lambda t: t[0])
        cands = []
        for cost, a, k in items[: per_char * 3]:
            p, (i, j) = a.prep, a.groups[k]
            pieces = _join_pieces([q for g in p.prims[i:j] for q in g])
            x0, x1 = min(q[:, 0].min() for q in pieces), max(q[:, 0].max() for q in pieces)
            if (x1 - x0) / p.xh > _max_width(ch):
                continue
            entry = {"strokes": _norm(pieces, x0, p.baseline, p.xh), "width": round(float((x1 - x0) / p.xh), 3), "cost": round(float(cost), 3),
                     "from": p.text, "pos": "s" if len(p.text) == 1 else "i" if k == 0 else "f" if k == len(p.text) - 1 else "m"}
            fitted = library.fit(entry, ch, canon, marks, always=True)
            cands.append(fitted)
            if len(cands) >= per_char:
                break
        if len(cands) < keep:
            continue

        def one(v):
            try:
                r = ask(ch, picture(v), key, model)
                return round(r["clear"] * (0.5 + 0.5 * r["p_right"]), 3), r["tokens"]
            except Exception:
                return None, 0

        with ThreadPoolExecutor(workers) as ex:
            scored = list(ex.map(one, cands))
        ranked = sorted(((s, v) for (s, _), v in zip(scored, cands) if s is not None), key=lambda t: -t[0])[:keep]
        old = [v.get("clarity", 0.0) for v in lib["glyphs"].get(ch, {"variants": []})["variants"]]
        new = [s for s, _ in ranked]
        better = len(ranked) >= 3 and np.mean(new) > (np.mean(old) if old else 0) + 0.03
        if better:
            lib["glyphs"][ch]["variants"] = [{**v, "clarity": s} for s, v in ranked]
        report[ch] = {"candidates": len(cands), "old": [round(float(x), 2) for x in old], "new": [round(float(x), 2) for x in new],
                      "replaced": bool(better), "tokens": sum(t for _, t in scored)}
        log(f"{ch}: {len(cands)} candidates; clarity {np.mean(old) if old else 0:.2f} -> {np.mean(new):.2f}"
            + ("  replaced" if better else "  kept the old samples"))
    return report


def prune(lib: dict, min_clarity: float = 0.3, keep_at_least: int = 1) -> dict[str, int]:
    """After `run`: drop samples judged unclear, keeping each letter's best. An automatic first review,
    most useful for scans and photos, where many samples are fragments or the wrong letter.
    A letter whose best sample is still very unclear is dropped altogether, so that it is filled in
    honestly as a stand-in instead of being written wrongly. Returns {char: samples dropped}."""
    dropped = {}
    for ch, e in list(lib["glyphs"].items()):
        scored = [v for v in e["variants"] if "clarity" in v]
        if not scored:
            continue
        keep = [v for v in e["variants"] if v.get("clarity", 1.0) >= min_clarity]
        best = max(scored, key=lambda v: v["clarity"])
        if not keep and best["clarity"] >= min_clarity * 0.5:
            keep = [best][:keep_at_least]
        n = len(e["variants"]) - len(keep)
        if n:
            dropped[ch] = n
        if keep:
            e["variants"] = keep
        else:
            del lib["glyphs"][ch]
            lib.setdefault("rejected", []).append(ch)
    return dropped
