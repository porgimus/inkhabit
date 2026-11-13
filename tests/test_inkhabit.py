# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests use only synthetic ink, so they run without anyone's real handwriting."""

import json
import time
import struct

import numpy as np
import pymupdf
import pytest

from inkhabit import font, forms, goodnotes, library, render, segment
from inkhabit.words import WordSample

SHAPES = {  # crude letters in x-height units, y down, baseline 0
    "a": [[[0.6, -0.9], [0.1, -0.8], [0.0, -0.2], [0.5, 0.0], [0.6, -0.9], [0.65, 0.0]]],
    "b": [[[0.0, -1.9], [0.0, 0.0], [0.5, -0.1], [0.55, -0.7], [0.0, -0.8]]],
    "c": [[[0.6, -0.8], [0.1, -0.9], [0.0, -0.2], [0.6, 0.0]]],
}


def tiny_lib():
    def var(ch, dx):
        return {"strokes": [[[x + dx * 0.03, y] for x, y in s] for s in SHAPES[ch]], "width": 0.65, "cost": 0.1, "from": ch}

    return {
        "format": "inkhabit-library/1",
        "meta": {"xheight": 5.0, "font_weight": 0.12, "word_gap": 1.0, "word_gap_use": 1.0},
        "spacing": {"letter_gap_mean": 0.25, "letter_gap_sd": 0.1},
        "join_rate": {"ab": 1.0},
        "glyphs": {ch: {"count": 3, "variants": [var(ch, i) for i in range(3)]} for ch in SHAPES},
        "words": {"cab": [{"strokes": SHAPES["c"] + SHAPES["a"], "width": 1.5}]},
    }


def word(text, xh=5.0, gap=0.3):
    strokes, x = [], 0.0
    for ch in text:
        strokes += [(np.array(s) + [x, 0]) * xh + [100, 200] for s in SHAPES[ch]]
        x += 0.65 + gap
    return WordSample(text, 1.0, strokes, [1.0] * len(strokes), (0, 0, 0, 0), "t", 0, 0)


def test_bv4_and_stroke_blob():
    body = struct.pack("<Hf", 2, 1.5) + struct.pack("<I2H", 2, 0, 1) + struct.pack("<I2f", 1, 10, 20)
    body += struct.pack("<I4f", 1, 15, 20, 20, 30) + struct.pack("<HI", 1, 0)
    tpl = b"tpl\0" + struct.pack("<I", len(body)) + goodnotes.TPL_STROKE + b"\0" + body
    raw = b"bv4-" + struct.pack("<I", len(tpl)) + tpl + b"bv4$"
    width, paths = goodnotes._parse_tpl(goodnotes._bv4(raw))
    assert width == 1.5
    assert np.allclose(paths[0][0], [10, 20]) and np.allclose(paths[0][-1], [20, 30])


def test_lz4_block_overlapping_copy():
    # literal "ab", then copy 6 bytes from offset 2 -> "abababab"
    assert goodnotes._lz4_block(bytes([0x22, ord("a"), ord("b"), 2, 0])) == b"abababab"


def test_alignment_splits_separate_letters():
    prep = segment.prepare(word("abc"), "abc", 5.0)
    assert prep and len(prep.prims) == 3
    aligned = segment.align(prep, None)
    assert aligned.groups == [(0, 1), (1, 2), (2, 3)]


def test_alignment_rejects_text_longer_than_ink():
    assert segment.prepare(word("a"), "abc", 5.0) is None


def test_vocab_filter():
    vocab = {"cab"}
    assert segment.usable_text("cab,", vocab) == "cab,"
    assert segment.usable_text("cxb", vocab) is None
    assert segment.usable_text("2024", vocab) == "2024"


def test_drop_variants_removes_empty_char():
    lib = library.drop_variants(tiny_lib(), {"a": [0, 1, 2], "b": [1]})
    assert "a" not in lib["glyphs"] and len(lib["glyphs"]["b"]["variants"]) == 2


def test_font_rotates_alternates(tmp_path):
    hb = pytest.importorskip("uharfbuzz")
    path = tmp_path / "t.ttf"
    info = font.build_font(tiny_lib(), path, "Test Hand")
    assert info["characters"] >= 4
    f = hb.Font(hb.Face(hb.Blob.from_file_path(str(path))))
    buf = hb.Buffer()
    buf.add_str("aaa")
    buf.guess_segment_properties()
    hb.shape(f, buf, {"calt": True})
    names = [f.glyph_to_string(i.codepoint) for i in buf.glyph_infos]
    assert names == ["a", "a.alt1", "a.alt2"]


def test_writer_is_reproducible_and_varies():
    lib = tiny_lib()
    a = render.Writer(lib, seed=1).write("abc cab abc")
    b = render.Writer(lib, seed=1).write("abc cab abc")
    c = render.Writer(lib, seed=2).write("abc cab abc")
    assert all(np.array_equal(x, y) for x, y in zip(a.strokes, b.strokes))
    assert len(a.strokes) != len(c.strokes) or any(x.shape != y.shape or not np.allclose(x, y) for x, y in zip(a.strokes, c.strokes))
    assert render.Writer(lib, seed=1).write("ab\u0416").missing == {"\u0416"}  # no sample, no recipe, no stand-in


def test_same_letter_never_repeats_same_variant():
    w = render.Writer(tiny_lib(), seed=0)
    picks = [w._pick("a", 3) for _ in range(50)]
    assert all(x != y for x, y in zip(picks, picks[1:]))


def test_wrapping_respects_width():
    ink = render.Writer(tiny_lib(), seed=0).write("abc " * 12, x_height=10, max_width=150)
    assert ink.size[0] <= 150 + 20 and ink.size[1] > 60


def test_fill_writes_vector_ink_and_reports(tmp_path):
    src, out = tmp_path / "f.pdf", tmp_path / "o.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.draw_line((100, 115), (400, 115))
    doc.save(src)
    assert any(f["kind"] == "line" for f in forms.fields(src))
    rep = forms.fill(src, [{"page": 0, "rect": [100, 99, 400, 115], "text": "abc cab"},
                           {"field": "nope", "text": "a"}], tiny_lib(), out)
    assert rep[0]["status"] == "ok" and rep[1]["status"] == "no such field"
    assert len(pymupdf.open(out)[0].get_drawings()) > 2
    json.dumps(rep)


def _extent(ink):
    pts = np.vstack(ink.strokes)
    return np.ptp(pts[:, 0]), np.ptp(pts[:, 1])


def test_dimension_settings_change_the_writing():
    lib, text = tiny_lib(), "abc abc abc"
    calm = render.Human(whole_words=0).scaled(0)
    base = _extent(render.Writer(lib, 0, calm).write(text))
    wide = _extent(render.Writer(lib, 0, calm.with_settings({"word_spacing": 2.0})).write(text))
    tall = _extent(render.Writer(lib, 0, calm.with_settings({"letter_height": 1.5})).write(text))
    loose = _extent(render.Writer(lib, 0, calm.with_settings({"letter_spacing": 3.0})).write(text))
    assert wide[0] > base[0] * 1.2 and loose[0] > base[0] * 1.2
    assert tall[1] == pytest.approx(base[1] * 1.5, rel=0.02) and tall[0] == pytest.approx(base[0], rel=0.02)
    two = render.Writer(lib, 0, calm).write("abc\nabc")
    far = render.Writer(lib, 0, calm.with_settings({"line_spacing": 2.0})).write("abc\nabc")
    assert far.size[1] > two.size[1] * 1.3


def test_slant_leans_right():
    lib = tiny_lib()
    calm = render.Human(whole_words=0, joins=0).scaled(0)
    up = np.vstack(render.Writer(lib, 0, calm).write("b").strokes)
    lean = np.vstack(render.Writer(lib, 0, calm.with_settings({"slant": 20})).write("b").strokes)
    top_up, top_lean = up[up[:, 1].argmin()], lean[lean[:, 1].argmin()]
    assert top_lean[0] > top_up[0] + 1.0  # the top of the 'b' stem moves right


def test_unknown_setting_is_an_error():
    with pytest.raises(ValueError, match="unknown setting"):
        render.Human().with_settings({"wordspacing": 2})


def test_scaled_zero_is_mechanical_but_keeps_dimensions():
    h = render.Human(slant=5, word_spacing=1.4).scaled(0)
    assert h.size == 0 and h.wobble == 0 and h.slant == 5 and h.word_spacing == 1.4


def test_no_join_from_a_dot():
    w = render.Writer(tiny_lib(), 0)
    w.join_rate = {"ab": 1.0}
    dot = np.array([[0.0, -1.4], [0.02, -1.38]])
    nxt = np.array([[0.3, -1.0], [0.3, 0.0]])
    assert not w._should_join("ab", dot, nxt)


def test_apostrophe_sits_high():
    lib = tiny_lib()
    lib["glyphs"][","] = {"count": 9, "variants": [{"strokes": [[[0.0, -0.05], [0.05, 0.35]]], "width": 0.05}]}
    table, borrowed = font.with_fallbacks(lib)
    assert "'" in borrowed and max(y for _, y in table["'"][0][0]["strokes"][0]) < -1.0
    assert table["'"][0][0]["source"] == "composed"


def test_underscore_blanks_listed_once(tmp_path):
    src = tmp_path / "u.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Name: ________")
    page.insert_text((72, 300), "Date: ________")
    doc.save(src)
    assert sum(f["kind"] == "line" for f in forms.fields(src)) == 2


def test_capitals_are_never_written_small():
    lib = tiny_lib()
    lib["glyphs"]["B"] = {"count": 3, "variants": [{"strokes": [[[0, -0.8], [0, 0], [0.4, -0.4]]], "width": 0.4}]}
    calm = render.Human(whole_words=0, joins=0).scaled(0)
    pts = np.vstack(render.Writer(lib, 0, calm).write("B", x_height=1.0).strokes)
    assert np.ptp(pts[:, 1]) >= 1.35


def test_styles_and_dollar():
    lib = tiny_lib()
    assert render.style(lib, "neat").typical > 0 and render.style(lib, "rushed").slant > 0
    with pytest.raises(ValueError):
        render.style(lib, "fancy")
    lib["glyphs"]["S"] = {"count": 3, "variants": [{"strokes": [[[0.5, -1.6], [0.0, -1.0], [0.5, 0.0]]], "width": 0.5}]}
    table, borrowed = font.with_fallbacks(lib)
    assert "$" in borrowed and len(table["$"][0][0]["strokes"]) == 2


def test_typical_sticks_to_usual_shapes():
    lib = tiny_lib()
    lib["glyphs"]["a"]["variants"] *= 3  # nine variants
    w = render.Writer(lib, 0, render.Human(typical=1.0))
    assert max(w._pick("a", 9) for _ in range(60)) < 3


def test_typeset_uses_the_font_file(tmp_path):
    import cv2

    ttf, out = tmp_path / "t.ttf", tmp_path / "t.png"
    font.build_font(tiny_lib(), ttf, "Test Hand")
    font.typeset(ttf, "abc\ncab", out, size=40)
    im = cv2.imread(str(out))
    assert im is not None and (im < 128).any()


def test_pdf_scan_ignores_printed_text_and_traces_ink(tmp_path):
    from inkhabit.scan import read_scan

    src = tmp_path / "s.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((30, 50), "PRINTED SLIDE TEXT", fontsize=14)
    page.draw_polyline([(40, 120), (60, 150), (80, 120), (100, 150)], width=1.2)  # pen zigzag
    doc.save(src)
    strokes = read_scan(src)[0].strokes
    pts = np.vstack([s.points for s in strokes])
    assert len(strokes) >= 1 and pts[:, 1].min() > 100  # nothing traced from the printed line
    assert 35 < pts[:, 0].min() < 45 and 95 < pts[:, 0].max() < 105  # coordinates are in PDF points


def test_heic_photo_is_read(tmp_path):
    import shutil
    import subprocess

    from PIL import Image, ImageDraw

    from inkhabit.scan import read_scan

    if not shutil.which("sips"):
        pytest.skip("needs macOS sips")
    im = Image.new("RGB", (600, 400), (235, 232, 225))
    ImageDraw.Draw(im).line([(100, 300), (200, 100), (300, 300), (400, 100)], fill=(20, 20, 60), width=6)
    png, heic = tmp_path / "p.png", tmp_path / "p.heic"
    im.save(png)
    subprocess.run(["sips", "-s", "format", "heic", str(png), "--out", str(heic)], check=True, capture_output=True)
    strokes = read_scan(heic)[0].strokes
    assert strokes and np.ptp(np.vstack([s.points for s in strokes])[:, 0]) > 50


def test_transcript_keeps_indentation(tmp_path):
    pytest.importorskip("Vision")
    from PIL import Image, ImageDraw, ImageFont

    from inkhabit.transcribe import transcribe

    im = Image.new("RGB", (1400, 500), "white")
    d = ImageDraw.Draw(im)
    font_ = ImageFont.load_default(size=44)
    d.text((60, 60), "Marketing strategy", fill="black", font=font_)
    d.text((260, 180), "consumer needs", fill="black", font=font_)
    d.text((60, 300), "Second topic here", fill="black", font=font_)
    src = tmp_path / "note.png"
    im.save(src)
    text, rep = transcribe(src)
    lines = [l for l in text.splitlines() if l.strip() and not l.startswith("#")]
    assert rep["lines"] == 3 and "Marketing" in lines[0] and "consumer" in lines[1]
    assert lines[1].startswith("  ") and not lines[0].startswith(" ") and not lines[2].startswith(" ")


def test_too_little_handwriting_stops_with_explanation(tmp_path):
    pytest.importorskip("Vision")
    from PIL import Image, ImageDraw, ImageFont

    from inkhabit import pipeline

    im = Image.new("RGB", (1200, 300), "white")
    ImageDraw.Draw(im).text((60, 100), "only a few words here", fill="black", font=ImageFont.load_default(size=44))
    im.save(tmp_path / "p.png")
    with pytest.raises(SystemExit, match="too few|Not enough"):
        pipeline.build([str(tmp_path / "p.png")], str(tmp_path / "out"), workers=1, log=lambda *_: None)


def test_missing_symbols_are_extrapolated_in_the_writers_proportions():
    from inkhabit import extend

    lib = tiny_lib()
    got = extend.extrapolate(lib, "*[")
    assert set(got) == {"*", "["} and len(got["*"]) == 4 and len(got["*"][0]["strokes"]) == 3
    assert got["*"][0]["strokes"] != got["*"][1]["strokes"]  # variants differ
    tall = lambda g: -min(y for s in g["["][0]["strokes"] for _, y in s)
    lib2 = tiny_lib()
    for v in lib2["glyphs"]["b"]["variants"]:  # a writer with much shorter tall letters
        v["strokes"] = [[[x, y * 0.72] for x, y in s] for s in v["strokes"]]
    assert tall(extend.extrapolate(lib2, "[")) < tall(got) * 0.9
    table, added = font.with_fallbacks(lib)
    assert "*" in table and "*" in added and "a" not in added


def test_typed_word_gap_is_tightened_and_overridable():
    lib = tiny_lib()
    del lib["meta"]["word_gap_use"]
    lib["meta"]["word_gap"] = 2.0
    assert font.word_gap(lib) == pytest.approx(1.2)
    lib["meta"]["word_gap_use"] = 0.9
    assert font.word_gap(lib) == 0.9


def test_thin_library_is_filled_and_labelled():
    from inkhabit import extend

    lib = tiny_lib()
    lib["glyphs"]["n"] = {"count": 5, "variants": [{"strokes": [[[0.0, -1.0], [0.0, 0.0]], [[0.0, -0.7], [0.4, -1.0], [0.5, 0.0]]], "width": 0.5}]}
    lib["glyphs"]["c"]["count"] = 1  # too few samples to trust
    lib = extend.fill(lib)
    f = lib["filled"]
    assert set(f["real"]) == {"a", "b", "n"}
    assert "u" in f["derived"] and "q" in f["derived"] and "d" in f["derived"]
    # u is n turned half a turn: its stem now hangs from the top of the x-height instead of standing on the baseline
    u = lib["glyphs"]["u"]["variants"][0]
    assert u["source"] == "derived" and u["strokes"][0] == [[0.5, 0.0], [0.5, -1.0]]
    # q is b turned: the ascender becomes a descender
    assert max(y for s in lib["glyphs"]["q"]["variants"][0]["strokes"] for _, y in s) > 0.5
    if f["neutral"]:  # needs a macOS system font
        k = lib["glyphs"]["k"]["variants"]
        assert k[0]["source"] == "neutral" and len(k) == 4 and k[0]["strokes"] != k[1]["strokes"]
    out = render.Writer(lib, 0).write("a bunk quad")
    assert not out.missing or not f["neutral"]


def test_transcript_relabels_words_in_lines_and_tables():
    from inkhabit.words import apply_transcript

    def w(text, line, x, y):
        return WordSample(text, 0.5, [np.zeros((2, 2))], [1.0], (x, y, x + 10, y + 5), "t", 0, line)

    page = [w("Hola", 0, 0, 0), w("mvndo", 0, 20, 0), w("comcr", 1, 0, 20), w("zzzz", 2, 0, 40), w("vivir", 3, 50, 60)]
    n = apply_transcript(page, ["Hola mundo", "comer", "vivir extra"])
    assert [s.text for s in page] == ["Hola", "mundo", "comer", "zzzz", "vivir"] and n == 4
    assert [s.trusted for s in page] == [True, True, True, False, True]


def test_accents_and_inverted_marks_are_composed():
    lib = tiny_lib()
    table, added = font.with_fallbacks(lib)
    assert {"á", "¿", "¡"} <= set(added)
    plain = table["a"][0][0]["strokes"]
    accented = table["á"][0][0]["strokes"]
    assert len(accented) == len(plain) + 1 and max(y for _, y in accented[-1]) < min(y for s in plain for _, y in s)
    assert font.glyph_name("á") == "uni00E1" and font.glyph_name("a") == "a"
    assert segment.usable_text("mañana,", None) == "mañana," and segment.usable_text("¿Porque", None) == "¿Porque"


def test_form_preset_is_steadier_than_notes():
    lib = tiny_lib()
    notes, form = render.Human.from_library(lib), render.form_preset(lib)
    assert form.drift < notes.drift and form.tilt < notes.tilt and form.joins < notes.joins and form.typical > 0


def test_pen_strokes_are_read_straight_from_a_vector_pdf(tmp_path):
    from inkhabit.scan import read_scan

    src = tmp_path / "v.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=300)
    page.draw_rect(pymupdf.Rect(0, 0, 300, 300), fill=(0.2, 0.2, 0.2), color=None)  # dark paper
    page.draw_line((0, 50), (300, 50), color=(0.4, 0.4, 0.4), width=0.5)  # a ruled line
    for i in range(25):  # pen strokes stored as lines with a width
        x = 20 + i * 10
        page.draw_polyline([(x, 100), (x + 4, 120), (x + 8, 100)], color=(0, 0.5, 1), width=0.9)
    doc.save(src)
    strokes = read_scan(src)[0].strokes
    assert len(strokes) == 25  # the paper and the rule are not ink
    assert np.allclose(strokes[0].points[0], [20, 100], atol=0.01) and strokes[0].width == pytest.approx(0.9, abs=0.01)
    assert strokes[0].color[:3] == pytest.approx((0, 0.5, 1), abs=0.01)


def test_filled_stroke_outlines_are_thinned_one_at_a_time(tmp_path):
    from inkhabit.scan import read_scan

    src = tmp_path / "f.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=400, height=300)
    for i in range(24):  # each stroke stored as the outline of a thick bar, overlapping its neighbour
        x = 20 + i * 14
        page.draw_quad(pymupdf.Quad((x, 100), (x + 3, 100), (x, 160), (x + 3, 160)), fill=(1, 1, 1), color=None)
    doc.save(src)
    strokes = read_scan(src)[0].strokes
    assert len(strokes) == 24
    span = [np.ptp(s.points[:, 1]) for s in strokes]
    assert min(span) > 45 and max(np.ptp(s.points[:, 0]) for s in strokes) < 3  # centre lines, not outlines


def test_whole_words_from_a_transcript_are_kept_and_used():
    lib = tiny_lib()
    lib["words"] = {}
    s = word("cab")
    s.trusted = True
    stray = np.array([[100.0, 260.0], [140.0, 260.5]])  # a bit of ruled line far below the word
    s.strokes.append(stray)
    n = library.add_whole_words(lib, [s, word("abc")], {0: 5.0, 1: 5.0})
    assert n == 1 and list(lib["words"]) == ["cab"]  # only the trusted word; the untrusted one is ignored
    kept = lib["words"]["cab"][0]
    assert kept["source"] == "whole" and len(kept["strokes"]) == len(s.strokes) - 1  # the stray mark is dropped
    ys = [y for st in kept["strokes"] for _, y in st]
    assert -2.2 < min(ys) < -0.7 and -0.2 < max(ys) < 0.3  # sits on the baseline
    w = render.Writer(lib, 0, render.Human(whole_words=1.0))
    w.write("cab cab")
    assert w.used_words == 2


def test_ink_filter_keeps_pencil_and_drops_print():
    from inkhabit import pipeline
    from inkhabit.ink import Page, Stroke

    line = np.array([[0.0, 0.0], [10.0, 0.0]])
    page = Page("t", 0, [Stroke(line + [0, i], 1.0, (0.05, 0.05, 0.05, 1.0)) for i in range(30)]
                + [Stroke(line + [20, i], 1.0, (0.55, 0.55, 0.55, 1.0)) for i in range(15)])
    pipeline._keep_ink(page, "lighter")
    assert len(page.strokes) == 15 and all(s.color[0] > 0.5 for s in page.strokes)


def test_tuning_drops_unreadable_samples_and_orders_the_rest():
    from inkhabit import tune

    lib = tiny_lib()
    lib["glyphs"]["a"]["variants"] = [{"strokes": [[[0, 0], [0.1 * i, -1]]], "width": 0.5, "id": i} for i in range(6)]
    dropped = tune.apply(lib, {"a": [0.9, 0.2, 1.0, 0.8, 0.1, 0.95]})
    assert dropped == {"a": [1, 4]}
    assert [v["id"] for v in lib["glyphs"]["a"]["variants"]] == [2, 5, 0, 3]  # most readable first
    lib2 = tiny_lib()
    assert tune.apply(lib2, {"a": [0.3, 0.2, 0.1]}) == {}  # never fewer than three left
    lib["pairs"] = {"ab": [{"strokes": [], "width": 1}, {"strokes": [], "width": 1}], "bc": [{"strokes": [], "width": 1}]}
    assert tune.apply_pairs(lib, {"ab": [0.9, 0.4, 0.8], "bc": [0.2, 0.9]}) == {"ab": [1], "bc": [0]}
    assert list(lib["pairs"]) == ["ab"] and len(lib["pairs"]["ab"]) == 1  # worse than unjoined: removed


def test_forced_variant_is_used():
    w = render.Writer(tiny_lib(), 0)
    w.force = {"a": 2}
    assert {w._pick("a", 3) for _ in range(10)} == {2}


def test_calibration_needs_a_real_measurement_and_sets_the_defaults():
    from inkhabit import tune

    lib = tiny_lib()
    assert tune.calibrate(lib, log=lambda *a: None) == {}  # nothing recorded from real notes: leave settings alone
    lib["measured"] = {"letter_spacing_use": 1.2, "joins_use": 0.8, "bounce": 0.3, "size": 0.3}
    h = render.Human.from_library(lib)
    assert (h.letter_spacing, h.joins) == (1.2, 0.8)
    assert h.bounce == render.Human().bounce and h.size == render.Human().size  # noisy measurements are not copied


def test_unify_brings_samples_to_one_scale_and_baseline():
    lib = tiny_lib()
    big = lib["glyphs"]["a"]["variants"][0]
    big["strokes"] = [[[x * 2, y * 2 + 0.4] for x, y in s] for s in big["strokes"]]  # one sample twice the size and sunk
    lib["words"]["cab"][0]["strokes"] = [[[x * 1.5, y * 1.5] for x, y in s] for s in lib["words"]["cab"][0]["strokes"]]
    library.unify(lib, log=lambda *a: None)

    def extent(v):
        ys = [y for s in v["strokes"] for _, y in s]
        return min(ys), max(ys)

    tops, bottoms = zip(*(extent(v) for v in lib["glyphs"]["a"]["variants"]))
    assert max(tops) - min(tops) < 0.02 and max(bottoms) - min(bottoms) < 0.02
    word_top, word_bottom = extent(lib["words"]["cab"][0])
    assert word_bottom == pytest.approx(0.0, abs=0.02) and word_top == pytest.approx(-1.9, abs=0.05)  # as tall as its tallest letter, b
    assert lib["meta"]["unified"] and lib["format"].endswith("/2")


def test_blend_makes_a_new_shape_between_two_samples():
    lib = tiny_lib()
    a, b = lib["glyphs"]["a"]["variants"][0], lib["glyphs"]["a"]["variants"][2]
    w = render.Writer(lib, 0, render.Human(morph=1.0))
    strokes, width = w._blend(a, [a, b])
    xs = np.asarray(strokes[0])[:, 0]
    assert min(np.asarray(a["strokes"][0])[:, 0].min(), 0) <= xs.min() and xs.max() <= np.asarray(b["strokes"][0])[:, 0].max() + 1e-6
    assert not np.allclose(np.asarray(strokes[0])[0], a["strokes"][0][0]) and not np.allclose(np.asarray(strokes[0])[0], b["strokes"][0][0])
    assert render.Writer(lib, 0, render.Human(morph=0.0))._blend(a, [a, b])[0] is a["strokes"]


def test_write_to_fit_tightens_then_shrinks():
    w = render.Writer(tiny_lib(), 0, render.Human().scaled(0))
    roomy, xh1 = w.write_to_fit("abc cab abc", 400, x_height=10, multiline=False)
    tight, xh2 = w.write_to_fit("abc cab abc", roomy.size[0] * 0.6, x_height=10, multiline=False)
    assert xh1 == 10 and xh2 < 10 and tight.size[0] <= roomy.size[0] * 0.6
    assert w.h.letter_spacing == 1.0  # the writer's own settings are left as they were


def test_situations():
    lib = tiny_lib()
    heading, big = render.situation(lib, "heading")
    label, small = render.situation(lib, "label")
    assert big > 1 > small and heading.whole_words == 0 and label.joins < render.Human().joins
    clean = render.style(lib, "clean")
    assert render.situation(lib, "text")[0] == clean and render.Writer(lib, 0).h == clean  # clean unless told otherwise
    assert render.style(lib, "neat") == clean
    assert render.situation(lib, "scribble")[0].wobble > clean.wobble
    with pytest.raises(ValueError):
        render.situation(lib, "poster")


def test_annotate_marks_printed_text_and_writes_margin_notes(tmp_path):
    from inkhabit import annotate

    src, out = tmp_path / "p.pdf", tmp_path / "o.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=432, height=648)
    page.insert_textbox(pymupdf.Rect(80, 100, 330, 400), "A market is a habit. Demand is elastic. " * 6, fontsize=10)
    doc.save(src)
    before = len(pymupdf.open(src)[0].get_drawings())
    rep = annotate.annotate(src, [{"type": "underline", "find": "A market is a habit"}, {"type": "circle", "find": "elastic"},
                                  {"type": "note", "find": "elastic", "side": "right", "text": "abc cab", "arrow": True},
                                  {"type": "strike", "find": "no such words here"}], tiny_lib(), out)
    assert [r["status"] for r in rep] == ["ok", "ok", "ok", "target not found"]
    drawings = pymupdf.open(out)[0].get_drawings()
    assert len(drawings) > before + 3
    assert any(d["rect"].x0 > 330 for d in drawings)  # the note sits in the right margin, outside the text


def test_pen_thickness_is_constant_unless_the_source_pen_varies():
    lib = tiny_lib()
    ink = render.Writer(lib, 0).write("abc cab abc")
    assert len(set(ink.widths)) == 1  # a fixed-width pen stays fixed
    lib["meta"]["pen_var"] = 0.1
    ink = render.Writer(lib, 0).write("abc cab abc")
    assert len(set(ink.widths)) > 1


def test_review_page_drops_marked_samples_only_with_its_token(tmp_path):
    import threading
    import urllib.error
    import urllib.request

    from inkhabit import reviewpage

    library.save(tiny_lib(), tmp_path / "library.json")
    port = 8799
    t = threading.Thread(target=reviewpage.serve, args=(tmp_path, "Test Hand", port, "secret"), daemon=True)
    t.start()
    for _ in range(50):
        try:
            page = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1).read().decode()
            break
        except OSError:
            time.sleep(0.05)
    assert "Review: Test Hand" in page and '"token": "secret"' in page

    def post(token):
        req = urllib.request.Request(f"http://127.0.0.1:{port}/apply", data=json.dumps({"drops": [{"kind": "glyph", "ch": "a", "i": 1}]}).encode(),
                                     headers={"Content-Type": "application/json", "X-Token": token})
        return urllib.request.urlopen(req, timeout=20).read()

    with pytest.raises(urllib.error.HTTPError):
        post("wrong")
    assert len(library.load(tmp_path / "library.json")["glyphs"]["a"]["variants"]) == 3
    assert b"Dropped 1 sample" in post("secret")
    assert len(library.load(tmp_path / "library.json")["glyphs"]["a"]["variants"]) == 2
    assert list(tmp_path.glob("library.before-review-*.json")) and (tmp_path / "handwriting.ttf").exists()


def test_pair_spacing_and_punctuation():
    lib = tiny_lib()
    calm = render.Human(whole_words=0, joins=0, morph=0).scaled(0)

    def width(text, pair_gap=None):
        lib["pair_gap"] = pair_gap or {}
        return render.Writer(lib, 0, calm)._word(text)[1]

    assert width("ab", {"ab": 0.6}) > width("ab") > width("ab", {"ab": 0.0})  # a pair's own gap moves the letters
    lib["glyphs"]["."] = {"count": 5, "variants": [{"strokes": [[[0.0, -0.04], [0.04, 0.0]]], "width": 0.04}]}
    assert width("a.", {"a.": 1.2}) < width("ab")  # a full stop stays close even if the notes space it widely


def test_justify_fills_the_line_but_not_the_last_one():
    lib = tiny_lib()
    calm = render.Human(whole_words=0).scaled(0)
    text = "abc cab abc cab abc cab abc"

    def line_ends(justify):
        ink = render.Writer(lib, 0, calm).write(text, x_height=10, max_width=185, justify=justify)
        rows: dict[int, float] = {}
        for s in ink.strokes:
            row = int(s[:, 1].mean() // 30)
            rows[row] = max(rows.get(row, 0), float(s[:, 0].max()))
        return [rows[k] for k in sorted(rows)]

    ragged, full = line_ends(False), line_ends(True)
    assert len(full) >= 2 and full[0] > ragged[0] and full[-1] == pytest.approx(ragged[-1], abs=1.0)


def test_library_records_pair_gaps():
    al = segment.align(segment.prepare(word("abc", gap=0.5), "abc", 5.0), None)
    gaps = library.pair_gaps([al] * 8)
    assert set(gaps) == {"ab", "bc"} and gaps["ab"] == pytest.approx(0.5, abs=0.1)


def test_practice_sheet_round_trip(tmp_path):
    from inkhabit import sheet

    pdf = tmp_path / "sheet.pdf"
    assert sheet.make("zQ8", pdf, reps=3) == 1
    boxes = sheet.layout("zQ8", 3)
    filled = tmp_path / "filled.pdf"
    doc = pymupdf.open(pdf)
    page = doc[0]
    for b in boxes:  # "write" a zigzag in every box, sitting on the baseline, one x-height tall
        x0, base = b["rect"][0] + 20, b["baseline"]
        for k in range(8):  # eight strokes per box so the page reads as ink shapes
            page.draw_polyline([(x0 + k, base - sheet.XH), (x0 + 12 + k, base - sheet.XH), (x0 + k, base), (x0 + 12 + k, base)],
                               color=(0.05, 0.1, 0.4), width=1.0)
    doc.save(filled)
    lib = tiny_lib()
    added = sheet.read_back(filled, str(pdf) + ".json", lib, log=lambda *a: None)
    assert added == {"z": 3, "Q": 3, "8": 3}
    v = lib["glyphs"]["z"]["variants"][0]
    ys = [y for s in v["strokes"] for _, y in s]
    assert v["source"] == "sheet" and min(ys) == pytest.approx(-1.0, abs=0.02) and max(ys) == pytest.approx(0.0, abs=0.02)
    assert "z" in sheet.wanted(tiny_lib()) and "a" in sheet.wanted(tiny_lib())  # everything is thin in the tiny library


def test_judge_builds_a_valid_request_and_ranks_by_clarity(monkeypatch):
    import io

    from inkhabit import judge

    lib = tiny_lib()
    png = judge.picture(lib["glyphs"]["a"]["variants"][0])
    im = cv2_imdecode(png)
    assert im.shape[0] >= 280 and im.shape[1] >= 280  # the service rejects anything smaller
    seen = []

    def fake_urlopen(req, timeout=0):
        body = json.loads(req.data)
        seen.append((req.full_url, dict(req.header_items()), body))
        (ref,) = body["assets"]
        assert body["state"]["sample"] == ref and "`sample`" in body["questions"]["clear"]["instructions"]
        n = len(seen)
        answers = {"clear": {"type": "noul", "noul": 0.9 if n == 2 else 0.2},
                   "which": {"type": "choice", "choice": "a", "probabilities": {"a": 0.8}}}
        return io.BytesIO(json.dumps({"model": "jev-latest", "answers": answers, "usage": {"input_tokens": 100}}).encode())

    class Ctx(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(judge.urllib.request, "urlopen", lambda req, timeout=0: Ctx(fake_urlopen(req).read()))
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        judge.run(lib)  # never sends anything without the user's key
    assert not seen
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    only_a = {**lib, "glyphs": {"a": lib["glyphs"]["a"]}}
    rep = judge.run(only_a, workers=1, log=lambda *a: None)
    assert rep["judged"] == 3 and rep["input_tokens"] == 300
    url, headers, _ = seen[0]
    assert url == "https://api.typesafe.ai/v1/systemone" and headers["Authorization"] == "Bearer test-key"
    assert {k.lower(): v for k, v in headers.items()}["typesafe-danger-zone-images"] == "true"
    clar = [v["clarity"] for v in only_a["glyphs"]["a"]["variants"]]
    assert clar == sorted(clar, reverse=True) and clar[0] > 0.7  # clearest first


def cv2_imdecode(png: bytes):
    import cv2

    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)


def test_playground_writes_what_is_asked_and_shows_the_command(tmp_path):
    import threading
    import urllib.parse
    import urllib.request

    from inkhabit import playground

    library.save(tiny_lib(), tmp_path / "library.json")
    port = 8798
    threading.Thread(target=playground.serve, args=(tmp_path, port), daemon=True).start()
    for _ in range(50):
        try:
            page = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1).read().decode()
            break
        except OSError:
            time.sleep(0.05)
    assert "Try the handwriting" in page and '"heading"' in page

    def write(**q):
        return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/write?" + urllib.parse.urlencode(q), timeout=5).read())

    a = write(text="abc cab", seed=1)
    b = write(text="abc cab", seed=1, slant=15, **{"for": "heading"})
    assert a["svg"].startswith("<svg") and a["svg"] != b["svg"]
    assert "--for heading" in b["command"] and "--set slant=15" in b["command"] and "--seed 1" in b["command"]
    assert write(text="abЖ")["missing"] == "Ж"
    assert library.load(tmp_path / "library.json") == tiny_lib()  # it only reads


def test_slanted_writing_is_stood_upright_and_lean_is_put_back():
    lean = np.tan(np.radians(15))
    samples = []
    for k in range(40):
        s = word("abc")
        s.strokes = [np.c_[p[:, 0] - (p[:, 1] - 200) * lean, p[:, 1]] for p in s.strokes] * 3
        samples.append(s)
    got = segment.estimate_slant(samples)
    assert 9 < got < 21
    upright = segment.deslant(samples, got)
    assert abs(segment.estimate_slant(upright)) < got / 2  # most of the lean is gone
    lib = tiny_lib()
    calm = render.Human(whole_words=0, joins=0, morph=0).scaled(0)
    straight = np.vstack(render.Writer(lib, 0, calm).write("b").strokes)
    lib["meta"]["slant"] = 15.0
    leaning = np.vstack(render.Writer(lib, 0, calm).write("b").strokes)
    assert leaning[leaning[:, 1].argmin(), 0] > straight[straight[:, 1].argmin(), 0] + 1.0


def test_ink_filter_leaves_single_ink_pages_alone():
    from inkhabit import pipeline
    from inkhabit.ink import Page, Stroke

    line = np.array([[0.0, 0.0], [10.0, 0.0]])
    page = Page("t", 0, [Stroke(line + [0, i], 1.0, (0.1 + 0.002 * i, 0.1, 0.3, 1.0)) for i in range(40)])
    pipeline._keep_ink(page, "lighter")
    assert len(page.strokes) == 40


def test_scan_ink_is_tidied_before_thinning():
    from inkhabit.scan import _tidy

    m = np.zeros((60, 200), bool)
    m[20:32, 10:190] = True  # a thick stroke
    m[25, 50] = m[26, 120] = False  # pinholes
    out = _tidy(m)
    assert out[25, 50] and out[26, 120] and out.sum() >= m.sum() * 0.98  # pinholes filled; corners may round


def test_fill_tick_comb_ruled_and_aligned(tmp_path):
    src, out = tmp_path / "f.pdf", tmp_path / "o.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(src)
    lib = tiny_lib()
    lib["glyphs"]["1"] = {"count": 5, "variants": [{"strokes": [[[0.1, -1.6], [0.1, 0.0]]], "width": 0.2}]}
    entries = [{"page": 0, "rect": [100, 100, 112, 112], "check": True},
               {"page": 0, "rect": [100, 140, 180, 160], "text": "1111", "comb": 4},
               {"page": 0, "rect": [100, 200, 400, 300], "text": "abc cab " * 14, "line_pitch": 24},
               {"page": 0, "rect": [100, 340, 400, 360], "text": "abc", "align": "right"},
               {"page": 0, "rect": [100, 380, 400, 410], "text": "abc", "for": "heading"}]
    rep = forms.fill(src, entries, lib, out, human=render.Human(whole_words=0, joins=0, morph=0).scaled(0))
    assert [r["status"] for r in rep] == ["ok"] * 5
    d = pymupdf.open(out)[0].get_drawings()
    xs = sorted((dr["rect"].x0 + dr["rect"].x1) / 2 for dr in d if 138 < dr["rect"].y0 < 162)
    assert len(xs) == 4 and all(abs((x - 100) % 20 - 10) < 4 for x in xs)  # one character centred in each cell
    ruled = [dr["rect"] for dr in d if 195 < dr["rect"].y0 < 300]
    bottoms = sorted({round(r.y1) for r in ruled})
    assert max(bottoms) - min(bottoms) >= 20  # it wrapped onto a second ruled line
    right = [dr["rect"] for dr in d if 335 < dr["rect"].y0 < 362]
    assert max(r.x1 for r in right) > 385  # pushed to the right edge
    assert rep[4]["x_height"] > rep[3]["x_height"]  # a heading is written larger


def test_notes_outline_and_layout(tmp_path):
    from inkhabit import notes

    outline = "# abc\n\ncab abc\n  - abc cab -> abc\n> cab\n\n! abc cab\n"
    items = notes.parse(outline)
    assert [i["kind"] for i in items] == ["heading", "line", "line", "aside", "line"]
    assert items[2]["level"] == 1 and items[2]["bullet"] and "→" in items[2]["text"]
    assert items[4]["key"] and items[4]["new_block"] and not items[2]["new_block"]
    out = tmp_path / "n.pdf"
    prof = {"ink_colours": {"blue": 0.6, "pink": 0.3, "white": 0.1}, "habits": {"indent_step_x_heights": 4.0}}
    rep = notes.write_notes(outline, tiny_lib(), out, prof, paper="white", seed=1)
    assert rep["pages"] == 1 and "black" in rep["colours"] and "white" not in rep["colours"]  # white ink becomes black on white paper
    d = pymupdf.open(out)[0].get_drawings()
    colours = {tuple(round(c, 2) for c in dr["color"]) for dr in d if dr.get("color")}
    assert len(colours) >= 2  # blocks are written in different inks
    assert min(dr["width"] for dr in d if dr.get("width")) >= 0.94  # never a hairline
    long = "\n".join(["abc cab abc"] * 60)
    assert notes.write_notes(long, tiny_lib(), tmp_path / "long.pdf", None)["pages"] >= 2


def test_documents_never_get_hairlines(tmp_path):
    src, out = tmp_path / "f.pdf", tmp_path / "o.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(src)
    forms.fill(src, [{"page": 0, "rect": [100, 100, 300, 112], "text": "abc cab", "x_height": 3.2}], tiny_lib(), out)
    assert min(dr["width"] for dr in pymupdf.open(out)[0].get_drawings() if dr.get("width")) >= 0.94


def test_careful_style_is_their_best_and_is_what_forms_use():
    lib = tiny_lib()
    careful, clean, natural = render.style(lib, "careful"), render.style(lib, "clean"), render.style(lib, "natural")
    assert careful.typical == 1.0 and careful.joins == 0 and careful.whole_words == 0 and careful.drift == 0
    assert careful.bounce < clean.bounce < natural.bounce and careful.bounce > 0  # steadier, but not mechanical
    assert render.form_preset(lib) == careful and render.situation(lib, "official")[0] == careful
    lib["glyphs"]["a"]["variants"] *= 3
    w = render.Writer(lib, 0, careful)
    assert max(w._pick("a", 9) for _ in range(40)) < 3  # only the clearest shapes


def _maths_lib():
    lib = tiny_lib()
    for ch in "x12=+":
        lib["glyphs"][ch] = {"count": 5, "variants": [{"strokes": [[[0.0, -1.0 if ch != "1" else -1.6], [0.5, 0.0]]], "width": 0.5}]}
    return lib


def test_maths_layout_fraction_power_root_and_alignment():
    from inkhabit import mathwrite

    lib = _maths_lib()
    w = render.Writer(lib, 0, render.Human(whole_words=0, joins=0, morph=0).scaled(0))
    m = mathwrite.Maths(w, lib, 0)
    plain, power = m.expression("x"), m.expression("x^2")
    assert power.a > plain.a + 0.3 and power.w > plain.w  # the power sits raised, to the right
    frac = m.expression(r"\frac{1}{2}")
    assert frac.a > 1.5 and frac.d > 0.8 and len(frac.strokes) == 3  # numerator, bar, denominator
    root = m.expression(r"\sqrt{x}")
    assert root.a > plain.a + 0.3 and len(root.strokes) == 2
    big = m.expression(r"\frac{\frac{1}{2}}{x}")
    assert big.a > frac.a
    assert m.expression("(x)").w < m.expression(r"(\frac{\frac{1}{2}}{2})").w  # tall contents get drawn brackets
    ink = mathwrite.write_maths(lib, "x + 1 = 2\nx = 1", x_height=10, human=render.Human(whole_words=0, joins=0, morph=0).scaled(0))
    mid = ink.size[1] / 2
    top = min(st[:, 0].min() for st in ink.strokes if st[:, 1].mean() < mid)
    bottom = min(st[:, 0].min() for st in ink.strokes if st[:, 1].mean() >= mid)
    assert top < 1 and bottom > top + 5  # the shorter second line is pushed right so the equals signs line up
    assert not m.missing


def test_notes_accept_maths_lines(tmp_path):
    from inkhabit import notes

    items = notes.parse("abc\n$ x = 1\n$ x + 1 = 2\n")
    assert [i["kind"] for i in items] == ["line", "math", "math"]
    rep = notes.write_notes("abc $x^2$\n$ x = 1\n$ x + 1 = 2\n", _maths_lib(), tmp_path / "m.pdf", None)
    assert rep["pages"] == 1 and rep["missing"] == ""


def test_notes_draw_sketches(tmp_path):
    from inkhabit import notes

    outline = "abc\n@ axes abc cab\n@ curve up abc\n@ curve down cab\n@ point abc\n\n@ flow abc -> cab -> abc\n"
    items = notes.parse(outline)
    assert [i["kind"] for i in items] == ["line", "diagram", "diagram", "diagram", "diagram", "diagram"]
    rep = notes.write_notes(outline, tiny_lib(), tmp_path / "s.pdf", None)
    d = pymupdf.open(tmp_path / "s.pdf")[0].get_drawings()
    assert rep["pages"] == 1 and len(d) > 40  # axes, two curves, dashes, three boxes, arrows and labels


def test_judge_prune_keeps_the_clear_and_drops_the_hopeless():
    from inkhabit import judge

    lib = tiny_lib()
    for v, c in zip(lib["glyphs"]["a"]["variants"], (0.9, 0.1, 0.5)):
        v["clarity"] = c
    for v in lib["glyphs"]["b"]["variants"]:
        v["clarity"] = 0.2  # none clear, but the best is not hopeless: keep one
    for v in lib["glyphs"]["c"]["variants"]:
        v["clarity"] = 0.05  # hopeless: better an honest stand-in than a wrong letter
    dropped = judge.prune(lib, 0.3)
    assert dropped == {"a": 1, "b": 2, "c": 3}
    assert len(lib["glyphs"]["a"]["variants"]) == 2 and len(lib["glyphs"]["b"]["variants"]) == 1
    assert "c" not in lib["glyphs"] and "c" in lib["rejected"]


def test_fields_finds_tick_boxes_cells_and_small_boxes(tmp_path):
    src = tmp_path / "f.pdf"
    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text((60, 100), "Date of birth", fontsize=10)
    for k in range(8):
        p.draw_rect(pymupdf.Rect(170 + k * 20, 90, 190 + k * 20, 110), width=0.6)
    p.insert_text((60, 150), "Small", fontsize=10)
    p.draw_rect(pymupdf.Rect(100, 140, 112, 152), width=0.7)
    p.insert_text((130, 150), "Large", fontsize=10)
    p.draw_rect(pymupdf.Rect(170, 140, 182, 152), width=0.7)
    p.draw_rect(pymupdf.Rect(60, 200, 120, 218), width=0.6)
    doc.save(src)
    got = forms.fields(src)
    kinds = [f["kind"] for f in got]
    assert kinds.count("cells") == 1 and kinds.count("tickbox") == 2 and kinds.count("box") == 1
    cells = next(f for f in got if f["kind"] == "cells")
    assert cells["use"] == {"comb": 8} and cells["rect"][0] == 170 and cells["rect"][2] == 330
    assert {f["text"] for f in got if f["kind"] == "label"} == {"Date of birth", "Small", "Large"}  # one label per printed item


def test_dots_are_drawn_as_round_marks(tmp_path):
    ink = render.Ink([np.array([[10.0, 10.0], [10.2, 10.1]]), np.array([[20.0, 10.0], [60.0, 10.0]])], [1.0, 1.0], (70, 20))
    svg = render.to_svg(ink)
    assert svg.count("<circle") == 1 and svg.count("<path") == 1
    doc = pymupdf.open()
    page = doc.new_page()
    render.onto_pdf_page(page, ink, (0, 0))
    filled = [d for d in page.get_drawings() if d.get("fill")]
    assert len(filled) == 1 and filled[0]["rect"].width > 1.2  # wider than the pen: a visible dot


def test_inputs_of_many_kinds(tmp_path):
    import zipfile

    from PIL import Image

    from inkhabit import scan, sources

    rgba = np.zeros((40, 40, 4), np.uint8)
    rgba[10:30, 18:22] = (20, 20, 20, 255)  # dark ink on a transparent background
    got = scan._to_rgb(Image.fromarray(rgba, "RGBA"))
    assert got[0, 0].tolist() == [255, 255, 255] and got[20, 20].max() < 60  # placed on white paper
    rgba[10:30, 18:22] = (250, 250, 250, 255)  # white ink: gets dark paper instead
    assert scan._to_rgb(Image.fromarray(rgba, "RGBA"))[0, 0].max() < 60
    deep = np.full((20, 20), 60000, np.uint16)
    deep[5:15, 5:15] = 2000
    eight = scan._to_rgb(Image.fromarray(deep))
    assert eight.dtype == np.uint8 and eight[0, 0, 0] > 200 and eight[10, 10, 0] < 30
    svg = tmp_path / "ink.svg"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg"><path d="M10,10 L50,10 l0,30" stroke-width="2"/>'
                   '<polyline points="5,5 6,9 9,12"/><line x1="0" y1="0" x2="4" y2="4"/></svg>')
    strokes = sources.load(svg)[0].strokes
    assert len(strokes) == 3 and strokes[0].points[-1].tolist() == [50.0, 40.0] and strokes[0].width == 2.0
    noext = tmp_path / "scan_from_phone"
    Image.new("RGB", (30, 30), "white").save(noext, "JPEG")
    assert sources.kind(noext) == ".jpg"
    z = tmp_path / "bundle.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(svg, "deep/folder/ink.svg")
        zf.writestr("deep/readme.txt", "not handwriting")
        zf.writestr("../escape.svg", "<svg/>")  # must not be written outside the temporary folder
    found = sources.discover(z)
    assert [p.name for p in found] == ["ink.svg"]
    assert not (tmp_path.parent / "escape.svg").exists()
    with pytest.raises(sources.Unsupported, match="not a kind of file"):
        sources.discover(tmp_path / "bundle.zip".replace("bundle.zip", "x.txt")) if (tmp_path / "x.txt").write_text("hi") else None
    locked = tmp_path / "locked.pdf"
    d = pymupdf.open()
    d.new_page()
    d.save(locked, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    with pytest.raises(ValueError, match="password-protected"):
        sources.load(locked)
    tif = tmp_path / "two.tif"
    Image.new("RGB", (300, 300), "white").save(tif, save_all=True, append_images=[Image.new("RGB", (300, 300), "white")])
    from inkhabit import scan

    assert len(scan._images(tif)) == 2  # both pages of a multi-page tif are read


def test_scanned_form_fields_are_found_by_looking(tmp_path, monkeypatch):
    import cv2

    from inkhabit import forms as F

    doc = pymupdf.open()
    p = doc.new_page()
    p.draw_line((170, 115), (540, 115), width=0.8)  # a line to write on
    for k in range(6):
        p.draw_rect(pymupdf.Rect(170 + k * 20, 200, 190 + k * 20, 220), width=0.8)  # a row of cells
    p.draw_rect(pymupdf.Rect(230, 300, 242, 312), width=0.9)  # a tick box
    p.draw_rect(pymupdf.Rect(60, 400, 540, 500), width=0.8)  # a big answer box
    for r in range(4):  # a table drawn as lines
        p.draw_line((60, 560 + r * 30), (360, 560 + r * 30), width=0.8)
    for c in (60, 160, 260, 360):
        p.draw_line((c, 560), (c, 650), width=0.8)
    px = p.get_pixmap(dpi=140)
    im = np.frombuffer(px.samples, np.uint8).reshape(px.height, px.width, 3)
    jpg = tmp_path / "form.jpg"
    cv2.imwrite(str(jpg), cv2.GaussianBlur(im, (0, 0), 0.6))
    monkeypatch.setattr(F, "ocr_words", lambda page: [])  # no recogniser needed for shapes
    monkeypatch.setattr("inkhabit.scan._upright", lambda rgb: rgb)
    got = F.fields(jpg)
    kinds = [f["kind"] for f in got]
    assert kinds.count("cells") == 1 and next(f for f in got if f["kind"] == "cells")["cells"] == 6
    assert kinds.count("tickbox") == 1 and kinds.count("box") == 1 + 9  # the answer box and nine table cells
    lines = [f for f in got if f["kind"] == "line"]
    assert len(lines) == 1 and abs(lines[0]["rect"][3] - 115) < 4
    rep = F.fill(jpg, [{"page": 0, "rect": lines[0]["rect"], "text": "abc cab"}], tiny_lib(), tmp_path / "out.pdf")
    assert rep[0]["status"] == "ok" and len(pymupdf.open(tmp_path / "out.pdf")[0].get_images()) == 1
