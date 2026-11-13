# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np
import pytest

from inkhabit import diagrams
from inkhabit.ink import Page, Stroke
from inkhabit.words import WordSample
from test_inkhabit import tiny_lib


def page_with_a_drawing():
    """Two written words (each two short strokes) beside a circle with an arrow into it."""
    def scribble(x, y):
        return [np.array([[x, y], [x + 4, y + 8], [x + 8, y]], float), np.array([[x + 10, y], [x + 14, y + 8]], float)]

    t = np.linspace(0, 2 * np.pi, 40)
    circle = np.c_[100 + 30 * np.cos(t), 100 + 30 * np.sin(t)]
    arrow = np.array([[170, 60], [150, 75], [135, 88]], float)
    w1, w2, far = scribble(150, 50), scribble(90, 95), scribble(400, 400)
    strokes = [Stroke(p, 0.8, (0.1, 0.2, 0.9, 1.0)) for p in [circle, arrow, *w1, *w2, *far]]
    words = [WordSample("supply", 1.0, w1, [0.8, 0.8], (150, 48, 166, 60), "doc", 0, 0),
             WordSample("core", 1.0, w2, [0.8, 0.8], (90, 93, 106, 105), "doc", 0, 1),
             WordSample("elsewhere", 1.0, far, [0.8, 0.8], (400, 398, 416, 410), "doc", 0, 2)]
    return Page("doc", 0, strokes), words


def test_a_drawing_is_found_with_the_words_in_it_as_labels():
    found = diagrams.find(*page_with_a_drawing())
    assert len(found) == 1
    d = found[0]
    assert sorted(lab["text"] for lab in d["labels"]) == ["core", "supply"]
    assert sum(s["label"] is None for s in d["strokes"]) == 2 and len(d["strokes"]) == 6


def test_writing_alone_is_not_a_drawing():
    page, words = page_with_a_drawing()
    page.strokes = page.strokes[2:]
    assert diagrams.find(page, words) == []


def test_redraw_copies_exactly_at_zero_and_differs_a_little_otherwise(tmp_path):
    d = diagrams.find(*page_with_a_drawing())[0]
    exact = diagrams.redraw(d, vary=0)
    x0, y0 = d["box"][:2]
    assert np.allclose(exact["strokes"][0]["points"], np.asarray(d["strokes"][0]["points"]) - [x0, y0])
    again = diagrams.redraw(d, vary=1, seed=3)
    moved = np.abs(np.asarray(again["strokes"][0]["points"]) - exact["strokes"][0]["points"]).max()
    assert 0 < moved < 0.6 * d["unit"]
    twice = diagrams.redraw(d, width=2 * (d["box"][2] - d["box"][0]), vary=0)
    assert twice["size"][0] == pytest.approx(2 * exact["size"][0])
    for ext in ("png", "svg", "pdf"):
        diagrams.save(again, tmp_path / f"d.{ext}")
        assert (tmp_path / f"d.{ext}").stat().st_size > 100


def test_a_label_is_rewritten_in_place_and_others_keep_their_ink():
    d = diagrams.find(*page_with_a_drawing())[0]
    i = next(k for k, lab in enumerate(d["labels"]) if lab["text"] == "core")
    out = diagrams.redraw(d, tiny_lib(), {str(i): "ab"}, vary=0)
    kept = [s for s in d["strokes"] if s["label"] != i]
    assert len(out["strokes"]) > len(kept) and out["missing"] == []
    new = np.vstack([np.asarray(s["points"]) for s in out["strokes"][len(kept):]])
    old = np.vstack([np.asarray(s["points"]) for s in d["strokes"] if s["label"] == i]) - d["box"][:2]
    assert abs(new[:, 0].min() - old[:, 0].min()) < 1 and abs(new[:, 1].max() - old[:, 1].max()) < d["unit"]
    with pytest.raises(SystemExit):
        diagrams.redraw(d, tiny_lib(), {"99": "ab"})
