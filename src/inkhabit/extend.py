# SPDX-License-Identifier: AGPL-3.0-or-later
"""Characters the writer never gave us: find them, compose them, or work them out.

Three sources, best first:

1. Harvested. Symbols like = + - > that stand alone in the notes were read by the
   recogniser as one-character "words". Those are the writer's real marks.
2. Composed from real pieces. A semicolon is this writer's dot over their comma;
   a double quote is their apostrophe twice; a dollar is their S with a bar.
3. Extrapolated. For anything else (an asterisk, a hash, brackets, arrows) we start
   from a plain pen recipe and restyle it to the writer: their letter proportions,
   slant, how round or angular their strokes are, and their natural irregularity.
   These are educated guesses and are labelled as such.
"""

from __future__ import annotations

import math

import numpy as np

HARVEST = set("=+-><*#&~^|/\\[]{}@%$\"'") | set("→←↑↓•×÷±≈≠≤≥✓°")


def _arc(cx, cy, rx, ry, a0, a1, n=14):
    return [[cx + rx * math.cos(math.radians(a0 + (a1 - a0) * i / n)), cy + ry * math.sin(math.radians(a0 + (a1 - a0) * i / n))]
            for i in range(n + 1)]


# Plain pen recipes. Units: x-height = 1, baseline y = 0, y down (so up is negative). "T" marks a height
# that should follow the writer's own tall-letter height instead of a fixed number.
T = -1.5
RECIPES: dict[str, list[list[list[float]]]] = {
    ".": [[[0.0, -0.04], [0.04, 0.0]]],
    ",": [[[0.08, -0.06], [0.0, 0.3]]],
    ":": [[[0.0, -0.72], [0.04, -0.68]], [[0.0, -0.04], [0.04, 0.0]]],
    ";": [[[0.06, -0.72], [0.1, -0.68]], [[0.1, -0.06], [0.0, 0.3]]],
    "!": [[[0.03, T], [0.02, -0.4]], [[0.0, -0.04], [0.04, 0.0]]],
    "?": [[[0.0, -1.25], [0.15, T], [0.4, -1.35], [0.38, -1.05], [0.2, -0.75], [0.2, -0.42]], [[0.18, -0.04], [0.22, 0.0]]],
    "'": [[[0.04, T - 0.1], [0.0, -1.2]]],
    '"': [[[0.04, T - 0.1], [0.0, -1.2]], [[0.26, T - 0.1], [0.22, -1.2]]],
    "(": [_arc(0.5, -0.7, 0.5, 1.1, 125, 235, 10)],
    ")": [_arc(-0.2, -0.7, 0.5, 1.1, 55, -55, 10)],
    "-": [[[0.0, -0.48], [0.5, -0.5]]],
    "_": [[[0.0, 0.12], [0.9, 0.1]]],
    "/": [[[0.0, 0.15], [0.6, T - 0.2]]],
    "=": [[[0.0, -0.68], [0.6, -0.7]], [[0.0, -0.3], [0.6, -0.32]]],
    "+": [[[0.0, -0.5], [0.6, -0.52]], [[0.3, -0.82], [0.31, -0.2]]],
    "@": [_arc(0.55, -0.5, 0.2, 0.22, 20, 380, 18) + [[0.78, -0.3]] + _arc(0.55, -0.5, 0.5, 0.52, 15, -300, 22)],
    "*": [[[0.3, -1.25], [0.3, -0.55]], [[0.0, -1.08], [0.6, -0.72]], [[0.6, -1.08], [0.0, -0.72]]],
    "#": [[[0.28, -1.3], [0.16, -0.1]], [[0.62, -1.3], [0.5, -0.1]], [[0.0, -0.9], [0.8, -0.92]], [[-0.04, -0.48], [0.76, -0.5]]],
    "&": [[[0.75, 0.0], [0.2, -1.0], [0.18, -1.3], [0.36, -1.45], [0.52, -1.28], [0.4, -1.0], [0.08, -0.55], [0.05, -0.25],
           [0.25, -0.02], [0.5, -0.1], [0.75, -0.6]]],
    "~": [[[0.0, -0.5], [0.15, -0.66], [0.3, -0.62], [0.45, -0.46], [0.6, -0.42], [0.75, -0.58]]],
    "^": [[[0.0, -1.1], [0.25, -1.5], [0.5, -1.1]]],
    "<": [[[0.6, -1.0], [0.0, -0.55], [0.6, -0.1]]],
    ">": [[[0.0, -1.0], [0.6, -0.55], [0.0, -0.1]]],
    "[": [[[0.3, T - 0.1], [0.0, T - 0.1], [0.0, 0.25], [0.3, 0.25]]],
    "]": [[[0.0, T - 0.1], [0.3, T - 0.1], [0.3, 0.25], [0.0, 0.25]]],
    "{": [[[0.4, T - 0.1], [0.22, T], [0.2, -0.9], [0.0, -0.62], [0.2, -0.35], [0.22, 0.15], [0.4, 0.25]]],
    "}": [[[0.0, T - 0.1], [0.18, T], [0.2, -0.9], [0.4, -0.62], [0.2, -0.35], [0.18, 0.15], [0.0, 0.25]]],
    "|": [[[0.0, T - 0.1], [0.02, 0.3]]],
    "\\": [[[0.0, T - 0.1], [0.6, 0.15]]],
    "`": [[[0.0, -1.6], [0.15, -1.3]]],
    "•": [_arc(0.15, -0.5, 0.1, 0.1, 0, 420, 16)],
    "°": [_arc(0.22, -1.3, 0.2, 0.2, -90, 290, 14)],
    "→": [[[0.0, -0.5], [1.0, -0.52]], [[0.72, -0.78], [1.0, -0.52], [0.72, -0.26]]],
    "←": [[[1.0, -0.5], [0.0, -0.52]], [[0.28, -0.78], [0.0, -0.52], [0.28, -0.26]]],
    "↑": [[[0.25, 0.0], [0.27, -1.3]], [[0.0, -1.0], [0.27, -1.3], [0.52, -1.0]]],
    "↓": [[[0.25, -1.3], [0.27, 0.0]], [[0.0, -0.3], [0.27, 0.0], [0.52, -0.3]]],
    "✓": [[[0.0, -0.45], [0.22, -0.05], [0.75, -1.2]]],
    "×": [[[0.0, -0.85], [0.55, -0.15]], [[0.55, -0.85], [0.0, -0.15]]],
    "±": [[[0.0, -0.72], [0.6, -0.74]], [[0.3, -1.05], [0.31, -0.42]], [[0.0, -0.12], [0.6, -0.14]]],
    "≈": [[[0.0, -0.7], [0.15, -0.84], [0.3, -0.78], [0.45, -0.66], [0.6, -0.72]],
          [[0.0, -0.32], [0.15, -0.46], [0.3, -0.4], [0.45, -0.28], [0.6, -0.34]]],
    "≠": [[[0.0, -0.68], [0.6, -0.7]], [[0.0, -0.3], [0.6, -0.32]], [[0.42, -1.0], [0.18, 0.0]]],
    "≤": [[[0.6, -1.05], [0.0, -0.68], [0.6, -0.32]], [[0.0, -0.08], [0.6, -0.1]]],
    "≥": [[[0.0, -1.05], [0.6, -0.68], [0.0, -0.32]], [[0.0, -0.08], [0.6, -0.1]]],
    "£": [[[0.62, -1.25], [0.45, -1.48], [0.25, -1.3], [0.24, -0.4], [0.05, 0.0], [0.7, -0.02]], [[0.02, -0.72], [0.5, -0.74]]],
    "€": [_arc(0.42, -0.72, 0.4, 0.72, -55, -305, 16), [[-0.06, -0.86], [0.5, -0.88]], [[-0.06, -0.56], [0.46, -0.58]]],
}


RECIPES.update({  # letters and signs used in maths and science that few people's notes contain as separate words
    "π": [[[0.0, -0.85], [0.15, -1.0], [0.85, -0.98]], [[0.27, -0.98], [0.2, 0.0]], [[0.62, -0.98], [0.64, -0.12], [0.78, 0.0]]],
    "θ": [_arc(0.3, -0.85, 0.28, 0.85, -90, 270, 18), [[0.03, -0.85], [0.57, -0.87]]],
    "Δ": [[[0.0, 0.0], [0.45, T], [0.45, T], [0.9, 0.0], [0.9, 0.0], [0.0, 0.02], [0.0, 0.02]]],
    "∞": [[[0.5, -0.5], [0.72, -0.78], [0.98, -0.7], [1.0, -0.4], [0.78, -0.24], [0.5, -0.5], [0.25, -0.78], [0.03, -0.66], [0.0, -0.38],
           [0.22, -0.22], [0.5, -0.5]]],
    "α": [[[0.72, -0.95], [0.48, -0.12], [0.22, 0.0], [0.02, -0.45], [0.2, -0.95], [0.46, -0.85], [0.7, -0.08], [0.82, 0.0]]],
    "μ": [[[0.0, 0.5], [0.06, -0.95]], [[0.06, -0.3], [0.22, 0.0], [0.46, -0.12], [0.56, -0.95]], [[0.56, -0.25], [0.66, 0.0]]],
    "λ": [[[0.05, T], [0.2, T], [0.62, 0.0]], [[0.36, -0.75], [0.0, 0.0]]],
    "σ": [_arc(0.3, -0.45, 0.3, 0.45, -90, 270, 16) + [[0.3, -0.9], [0.75, -0.92]]],
    "Σ": [[[0.75, T + 0.1], [0.05, T + 0.1], [0.42, -0.7], [0.05, 0.0], [0.8, 0.0]]],
    "∫": [[[0.55, T - 0.3], [0.42, T - 0.42], [0.32, T - 0.2], [0.24, 0.35], [0.13, 0.6], [0.0, 0.48]]],
    "√": [[[0.0, -0.45], [0.14, -0.32], [0.3, 0.05], [0.55, T], [1.0, T - 0.02]]],
    "·": [[[0.0, -0.5], [0.04, -0.47]]],
    "′": [[[0.06, T], [0.0, -1.15]]],
})


def _traits(lib: dict) -> dict:
    """What we can measure about the writer that should carry over to new characters."""
    g = {c: {**e, "variants": [v for v in e["variants"] if v.get("source") not in ("derived", "neutral")]} for c, e in lib["glyphs"].items()}

    def top(chars):
        v = [-min(p[1] for s in var["strokes"] for p in s) for c in chars if c in g for var in g[c]["variants"]]
        return float(np.median(v)) if v else None

    tall = top("bdhklABDEFHLPR") or 1.5
    paren = top("()")  # brackets and bars should stand as tall as this writer's own parentheses
    # how round is the writing? ratio of the straight-line distance to the path length over all strokes:
    # near 1 = straight and angular, lower = loopy
    ratios, widths = [], []
    for c, e in g.items():
        if c.isalpha() and c.islower():
            for var in e["variants"]:
                widths.append(var["width"])
                for s in var["strokes"]:
                    p = np.asarray(s, float)
                    if len(p) > 3:
                        L = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
                        if L > 0.4:
                            ratios.append(np.linalg.norm(p[-1] - p[0]) / L)
    m = lib.get("measured", {})
    body = top("acemnorsuvwxz") or 1.0  # how tall this writer's small letters really are, in nominal x-heights
    tall = float(np.clip(tall, 1.3, 2.2))  # from very few letters these estimates can be wild; keep them human
    return {"tall": tall, "body": float(np.clip(body, 0.55, 1.25)), "bracket": float(np.clip((paren or tall * 1.15) / 1.6, 0.85, 1.4)),
            "width": float(np.clip(np.median(widths) / 0.62, 0.75, 1.35)) if widths else 1.0,
            "round": 1 - float(np.median(ratios)) if ratios else 0.4,
            "wild": float(np.clip(m.get("size", 0.08), 0.03, 0.15)), "slant": 0.0}  # shapes are stored upright; the writer adds the lean


def _hand(points, rng, tr, tall_fix):
    """Redraw one recipe stroke the way a hand would: rounded corners, uneven, in the writer's proportions."""
    p = np.asarray(points, float).copy()
    cut = -1.05 * min(1.0, tr.get("body", 1.0)) if tall_fix != tr["tall"] / 1.5 else -1.05
    p[:, 1] = np.where(p[:, 1] < cut, p[:, 1] * tall_fix, p[:, 1])  # tall parts follow the writer's tall letters
    p[:, 0] *= tr["width"]
    if len(p) >= 3:  # round the corners in proportion to how round the writing is (Chaikin corner cutting)
        for _ in range(2 if tr["round"] > 0.25 else 1):
            q = [p[0]]
            for a, b in zip(p[:-1], p[1:]):
                q += [a * 0.75 + b * 0.25, a * 0.25 + b * 0.75]
            p = np.array(q + [p[-1]])
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    t = np.linspace(0, seg[-1], max(len(p), int(seg[-1] / 0.07) + 2))
    p = np.c_[np.interp(t, seg, p[:, 0]), np.interp(t, seg, p[:, 1])]
    amp = 0.35 * tr["wild"]
    f, ph = rng.uniform(1.5, 3.5, 2), rng.uniform(0, 6.28, 2)
    p[:, 0] += amp * np.sin(t * f[0] + ph[0])
    p[:, 1] += amp * np.sin(t * f[1] + ph[1])
    p += rng.normal(0, 0.25 * tr["wild"], 2)  # strokes do not land exactly where intended
    p[:, 0] -= p[:, 1] * math.tan(math.radians(tr["slant"]))
    return p


def extrapolate(lib: dict, chars: str | None = None, variants: int = 4, seed: int = 7) -> dict:
    """Variants for characters missing from lib, restyled to the writer. {char: [variant, ...]}."""
    tr = _traits(lib)
    tall_fix = tr["tall"] / 1.5
    out = {}
    for ch, recipe in RECIPES.items():
        if ch in lib["glyphs"] or (chars is not None and ch not in chars):
            continue
        rng = np.random.default_rng(seed + ord(ch))
        vs = []
        fix = tr["bracket"] if ch in "[]{}|\\" else tall_fix
        for _ in range(variants):
            scale = 1 + rng.normal(0, tr["wild"])
            strokes = [_hand(s, rng, tr, fix) * scale for s in recipe]
            x0 = min(s[:, 0].min() for s in strokes)
            vs.append({"strokes": [np.round(s - [x0, 0], 3).tolist() for s in strokes],
                       "width": round(float(max(s[:, 0].max() for s in strokes) - x0), 3), "source": "extrapolated"})
        out[ch] = vs
    return out


def compose(lib: dict) -> dict:
    """Characters built from the writer's real pieces."""
    g, out = lib["glyphs"], {}

    def shifted(var, dx=0.0, dy=0.0):
        return [[[x + dx, y + dy] for x, y in s] for s in var["strokes"]]

    def pairs(a, b):
        return list(zip(g[a]["variants"], g[b]["variants"] * len(g[a]["variants"])))

    if ";" not in g and "." in g and "," in g:
        out[";"] = [{"strokes": shifted(d, dy=-0.75) + shifted(c), "width": max(d["width"], c["width"]), "source": "composed"}
                    for d, c in pairs(".", ",")]
    if "'" not in g and "," in g:
        out["'"] = [{"strokes": shifted(c, dy=-1.45), "width": c["width"], "source": "composed"} for c in g[","]["variants"]]
    if '"' not in g and "," in g:
        out['"'] = [{"strokes": shifted(c, dy=-1.45) + shifted(c, dx=c["width"] + 0.2, dy=-1.45), "width": 2 * c["width"] + 0.2,
                     "source": "composed"} for c in g[","]["variants"]]
    if "$" not in g and "S" in g:
        out["$"] = [{"strokes": v["strokes"] + [[[v["width"] / 2 + 0.03, -1.95], [v["width"] / 2, 0.2]]], "width": v["width"],
                     "source": "composed"} for v in g["S"]["variants"]]
    if "÷" not in g and "." in g:
        out["÷"] = [{"strokes": [[[0.0, -0.5], [0.6, -0.52]]] + shifted(d, dx=0.28, dy=-0.85) + shifted(d, dx=0.28, dy=-0.15),
                     "width": 0.6, "source": "composed"} for d in g["."]["variants"]]
    def turned(var, kind):
        """A real mark mirrored or rotated about its own centre: an arrow pointing another way."""
        pts = np.vstack([np.asarray(s, float) for s in var["strokes"]])
        c = (pts.min(0) + pts.max(0)) / 2
        m = {"mirror": [[-1, 0], [0, 1]], "up": [[0, 1], [-1, 0]], "down": [[0, -1], [1, 0]]}[kind]
        strokes = [(np.asarray(s, float) - c) @ np.array(m, float).T for s in var["strokes"]]
        lo = np.vstack(strokes).min(0)
        hi = np.vstack(strokes).max(0)
        dy = c[1] if kind == "mirror" else -0.6 - (lo[1] + hi[1]) / 2  # upright arrows sit mid-height
        return {"strokes": [np.round(s + [-lo[0], dy], 3).tolist() for s in strokes], "width": round(float(hi[0] - lo[0]), 3),
                "source": "composed"}

    def stacked(top, bottom, gap=0.12):
        tp = np.vstack([np.asarray(s, float) for s in top["strokes"]])
        bp = np.vstack([np.asarray(s, float) for s in bottom["strokes"]])
        dy = tp[:, 1].max() + gap - bp[:, 1].min()
        dx = (top["width"] - bottom["width"]) / 2
        return {"strokes": top["strokes"] + shifted(bottom, dx, dy), "width": max(top["width"], bottom["width"]), "source": "composed"}

    # accented letters: the writer's own letter with the mark drawn above it
    marks = {"\u0301": [[0.1, -0.18], [0.32, -0.42]], "\u0300": [[0.32, -0.18], [0.1, -0.42]], "\u0308": None,
             "\u0303": [[0.0, -0.22], [0.12, -0.36], [0.26, -0.24], [0.4, -0.38]], "\u0302": [[0.05, -0.18], [0.22, -0.42], [0.4, -0.18]]}
    import unicodedata

    for ch in "áéíóúàèìòùâêîôûäëïöüñãõÁÉÍÓÚÑÜçÇ":
        base, *mark = unicodedata.normalize("NFD", ch)
        if ch in g or base not in g or not mark or mark[0] not in marks:
            continue
        vs = []
        for v in g[base]["variants"]:
            top = min(p[1] for s in v["strokes"] for p in s)
            cx = v["width"] / 2 - 0.2
            if marks[mark[0]] is None:  # dieresis: two dots
                extra = [[[cx + 0.05, top - 0.25], [cx + 0.08, top - 0.22]], [[cx + 0.32, top - 0.25], [cx + 0.35, top - 0.22]]]
            else:
                extra = [[[round(cx + x, 3), round(top + y, 3)] for x, y in marks[mark[0]]]]
            strokes = v["strokes"] if ch not in "íìîï" else v["strokes"][:1]  # the mark replaces the dot of an i
            vs.append({"strokes": strokes + extra, "width": v["width"], "source": "composed"})
        out[ch] = vs
    for ch, src in (("¿", "?"), ("¡", "!")):  # the same mark upside down, hanging below the line
        if ch not in g and src in g:
            vs = []
            for v in g[src]["variants"]:
                pts = np.vstack([np.asarray(s, float) for s in v["strokes"]])
                lo, hi = pts[:, 1].min(), pts[:, 1].max()
                vs.append({"strokes": [[[round(v["width"] - x, 3), round(lo + hi - y + 0.55, 3)] for x, y in s] for s in v["strokes"]],
                           "width": v["width"], "source": "composed"})
            out[ch] = vs

    real = {c for c, e in g.items() if e["variants"]}
    for ch, src, kind in (("←", "→", "mirror"), ("↑", "→", "up"), ("↓", "→", "down"), ("<", ">", "mirror")):
        if ch not in g and src in real:
            out[ch] = [turned(v, kind) for v in g[src]["variants"]]
    if "≥" not in g and {">", "-"} <= real:
        out["≥"] = [stacked(a, b) for a, b in pairs(">", "-")]
    if "≤" not in g and {">", "-"} <= real:
        out["≤"] = [stacked(turned(a, "mirror"), b) for a, b in pairs(">", "-")]
    if "±" not in g and {"+", "-"} <= real:
        out["±"] = [stacked(a, b, 0.08) for a, b in pairs("+", "-")]
    if "≠" not in g and "=" in real:
        out["≠"] = [{"strokes": v["strokes"] + [[[v["width"] * 0.5 + 0.38, -1.15], [v["width"] * 0.5 - 0.38, 0.12]]], "width": v["width"],
                     "source": "composed"} for v in g["="]["variants"]]
    if "×" not in g and "x" in g:
        out["×"] = [{"strokes": [[[x * 0.8, y * 0.8 - 0.15] for x, y in s] for s in v["strokes"]], "width": v["width"] * 0.8,
                     "source": "composed"} for v in g["x"]["variants"] if len(v["strokes"]) >= 2] or None
        if not out["×"]:
            del out["×"]
    return out


def harvest(samples, sizes: dict[int, float], lib: dict, per_char: int = 6, seed: int = 0) -> dict:
    """The writer's own stand-alone symbols, taken from words the recogniser read as a single mark."""
    from .library import _diverse, _norm
    from .segment import feature

    rng = np.random.default_rng(seed)
    lines: dict[tuple, list[int]] = {}
    for i, s in enumerate(samples):
        lines.setdefault((s.source, s.page, s.line), []).append(i)
    found: dict[str, list] = {}
    for idxs in lines.values():
        if len(idxs) < 3:
            continue
        base = [np.percentile(np.vstack(samples[i].strokes)[:, 1], 92) for i in idxs
                if len(samples[i].text) > 2 and not set(samples[i].text) & set("gjpqy,;()")]
        if len(base) < 2:
            continue
        baseline = float(np.median(base))
        for i in idxs:
            s = samples[i]
            if (len(s.text) == 1 and s.text in HARVEST and s.text not in lib["glyphs"]
                    and s.text not in lib.get("rejected", []) and s.conf >= 0.5):
                xh = sizes[i]
                pts = np.vstack(s.strokes)
                w, h = np.ptp(pts[:, 0]) / xh, np.ptp(pts[:, 1]) / xh
                if 0.08 < max(w, h) < 2.6 and len(s.strokes) <= 6:
                    found.setdefault(s.text, []).append(([np.asarray(p, float) for p in s.strokes], baseline, xh))
    out = {}
    for ch, items in found.items():
        if len(items) < 6:
            continue
        feats = np.array([feature(p, b, x) for p, b, x in items])
        d = ((feats - np.median(feats, 0)) ** 2).sum(1)
        keep = np.argsort(d)[: max(per_char, len(items) // 2)]  # the half most like the rest
        vs = []
        for k in _diverse(feats[keep], per_char, rng):
            pieces, b, x = items[int(keep[k])]
            x0 = min(p[:, 0].min() for p in pieces)
            x1 = max(p[:, 0].max() for p in pieces)
            vs.append({"strokes": _norm(pieces, x0, b, x), "width": round(float((x1 - x0) / x), 3), "source": "harvested"})
        out[ch] = {"count": len(items), "variants": vs}
    return out


# ---------------------------------------------------------------------------
# Not enough handwriting: fill the alphabet from what there is
# ---------------------------------------------------------------------------

# Letters that are another letter turned or flipped. From a real sample of the source we get the
# target in the writer's own stroke. (kind, source): "turn" = rotated half a turn, "flip" = mirrored.
RELATED = {
    "u": [("turn", "n")], "n": [("turn", "u")],
    "b": [("turn", "q"), ("flip", "d")], "q": [("turn", "b"), ("flip", "p")],
    "d": [("turn", "p"), ("flip", "b")], "p": [("turn", "d"), ("flip", "q")],
    "m": [("turn", "w")], "w": [("turn", "m")],
    "l": [("same", "I"), ("same", "1")], "I": [("same", "l"), ("same", "1")], "1": [("same", "l"), ("same", "I")],
    "0": [("same", "O")], "O": [("same", "0")],
}
MIN_REAL = 3  # with fewer samples than this a letter is too unreliable to trust
NEUTRAL_FONTS = [("/System/Library/Fonts/Noteworthy.ttc", 0), ("/System/Library/Fonts/Supplemental/Chalkboard.ttc", 0),
                 ("/System/Library/Fonts/Supplemental/Comic Sans MS.ttf", 0)]
ALNUM = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def derive(lib: dict, trusted: set[str]) -> dict:
    """Missing letters made by turning or flipping the writer's real letters (n -> u, b -> q, d -> p ...)."""
    g, out = lib["glyphs"], {}
    for ch, options in RELATED.items():
        if ch in trusted:
            continue
        for kind, src in options:
            if src not in trusted:
                continue
            vs = []
            for v in g[src]["variants"]:
                w = v["width"]
                if kind == "turn":  # half turn about the middle of the x-height box
                    strokes = [[[round(w - x, 3), round(-1.0 - y, 3)] for x, y in s] for s in v["strokes"]]
                elif kind == "flip":
                    strokes = [[[round(w - x, 3), y] for x, y in s] for s in v["strokes"]]
                else:
                    strokes = v["strokes"]
                vs.append({"strokes": strokes, "width": w, "source": "derived"})
            out[ch] = vs
            break
    return out


_SKELETONS: dict[str, list] = {}


def _neutral_skeleton(ch: str):
    """Pen path of a character in a plain, neutral hand (a system font thinned to its centre line)."""
    if ch in _SKELETONS:
        return _SKELETONS[ch]
    from pathlib import Path

    import cv2
    from PIL import Image, ImageDraw, ImageFont
    from skimage.morphology import skeletonize

    from .scan import _smooth, _trace

    font = next((ImageFont.truetype(p, 300, index=i) for p, i in NEUTRAL_FONTS if Path(p).exists()), None)
    if font is None:
        _SKELETONS[ch] = None
        return None

    def mask(c):
        im = Image.new("L", (700, 700), 0)
        ImageDraw.Draw(im).text((200, 150), c, fill=255, font=font)
        return np.asarray(im) > 128

    x = mask("x")
    ys = np.flatnonzero(x.any(1))
    baseline, xh = ys.max(), ys.max() - ys.min()
    m = mask(ch)
    if not m.any():
        _SKELETONS[ch] = None
        return None
    x0 = np.flatnonzero(m.any(0)).min()
    strokes = []
    for line in _trace(skeletonize(m)):
        if len(line) >= 2 and np.linalg.norm(np.diff(line, axis=0), axis=1).sum() < 0.12 * xh and len(strokes) > 0:
            continue  # skeleton whiskers
        p = (_smooth(line) - [x0, baseline]) / xh
        if len(p) > 2:
            p = cv2.approxPolyDP(p.astype(np.float32)[:, None], 0.02, False)[:, 0]
        strokes.append(p.tolist())
    _SKELETONS[ch] = strokes or None
    return _SKELETONS[ch]


def neutral(lib: dict, chars: str, variants: int = 4, seed: int = 11) -> dict:
    """Letters the writer gave us no usable sample of, drawn in a plain hand and fitted to the writer:
    their letter heights, width, slant and irregularity. Not their handwriting; a stand-in that matches it."""
    tr = _traits(lib)
    out = {}
    for ch in chars:
        skel = _neutral_skeleton(ch)
        if not skel:
            continue
        rng = np.random.default_rng(seed + ord(ch))
        vs = []
        for _ in range(variants):
            scale = 1 + rng.normal(0, tr["wild"])
            # size the stand-in like the writer's own letters: small letters to their body height,
            # tall parts to their tall-letter height
            b = tr["body"]
            strokes = [_hand(np.asarray(s, float) * b, rng, {**tr, "round": 0.0}, tr["tall"] / (1.75 * b)) * scale for s in skel]
            x0 = min(s[:, 0].min() for s in strokes)
            vs.append({"strokes": [np.round(s - [x0, 0], 3).tolist() for s in strokes],
                       "width": round(float(max(s[:, 0].max() for s in strokes) - x0), 3), "source": "neutral"})
        out[ch] = vs
    return out


def fill(lib: dict) -> dict:
    """Complete a thin library. Keeps every letter with enough real samples; replaces the rest, best source
    first: turned or flipped from a real letter, else a neutral shape fitted to the writer. Records what was
    done in lib["filled"] so nothing is passed off as real."""
    g = lib["glyphs"]
    trusted = {c for c, e in g.items() if e["count"] >= MIN_REAL and e["variants"]
               and e["variants"][0].get("source") not in ("derived", "neutral")}
    weak = [c for c in ALNUM if c not in trusted]
    derived = derive(lib, trusted)
    report = {"real": "".join(c for c in ALNUM if c in trusted), "derived": "", "neutral": ""}
    for c in weak:
        if c in derived:
            g[c] = {"count": 0, "variants": derived[c]}
            report["derived"] += c
    still = [c for c in weak if c not in derived]
    # case borrowing (C from c ...) happens later in font.with_fallbacks; leave those to it when possible
    from .font import FALLBACK

    borrowable = {c for c in still if c in FALLBACK and FALLBACK[c][0] in trusted}
    for c in borrowable:
        g.pop(c, None)
    for c, vs in neutral(lib, "".join(c for c in still if c not in borrowable)).items():
        g[c] = {"count": 0, "variants": vs}
        report["neutral"] += c
    lib["glyphs"] = dict(sorted(g.items()))
    sp = lib["spacing"]  # spacing measured from a handful of letters is unreliable too
    gap = sp.get("letter_gap_mean", 0.25)
    sp["letter_gap_mean"] = float(np.clip(gap if np.isfinite(gap) else 0.25, 0.18, 0.4))
    sp["letter_gap_sd"] = float(np.clip(sp.get("letter_gap_sd", 0.1) if np.isfinite(sp.get("letter_gap_sd", 0.1)) else 0.1, 0.03, 0.15))
    if len(report["real"]) < 15:  # joined pairs and words cut by alignment from so little are not trustworthy
        lib["pairs"] = {}
        lib["words"] = {w: [v for v in vs if v.get("source") == "whole"] for w, vs in lib.get("words", {}).items()}
        lib["words"] = {w: vs for w, vs in lib["words"].items() if vs}
    lib["filled"] = report
    return lib
