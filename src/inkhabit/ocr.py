# SPDX-License-Identifier: AGPL-3.0-or-later
"""Word-level handwriting recognition.

Backend: Apple Vision (macOS, on-device). It returns each text line with a
confidence, and a box for any character range, which we use per word.
Other backends only need to return the same Word records.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class Word:
    text: str
    conf: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in image pixels, y down
    line: int


def recognize(im: Image.Image, custom_words: list[str] | None = None, lang: str | None = None) -> list[Word]:
    """lang: a language tag such as "es-ES" or "fr-FR"; default is the system's (usually English)."""
    import Quartz
    import Vision

    rgb = np.ascontiguousarray(np.asarray(im.convert("RGBA")))
    h, w = rgb.shape[:2]
    provider = Quartz.CGDataProviderCreateWithData(None, rgb.tobytes(), rgb.nbytes, None)
    cg = Quartz.CGImageCreate(
        w, h, 8, 32, 4 * w, Quartz.CGColorSpaceCreateDeviceRGB(),
        Quartz.kCGImageAlphaNoneSkipLast, provider, None, False, Quartz.kCGRenderingIntentDefault,
    )
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(True)
    if lang:
        req.setRecognitionLanguages_([lang])
    if custom_words:  # the writer's own vocabulary: course terms, names, acronyms
        req.setCustomWords_(custom_words[:2000])
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, None)
    ok, err = handler.performRequests_error_([req], None)
    if not ok:
        raise RuntimeError(f"Vision OCR failed: {err}")
    words: list[Word] = []
    for li, obs in enumerate(req.results() or []):
        cand = obs.topCandidates_(1)[0]
        text = str(cand.string())
        for m in re.finditer(r"\S+", text):
            rect, _ = cand.boundingBoxForRange_error_((m.start(), m.end() - m.start()), None)
            if rect is None:
                continue
            b = rect.boundingBox()
            x0, y1 = b.origin.x * w, (1 - b.origin.y) * h
            words.append(
                Word(m.group(), float(cand.confidence()),
                     (x0, y1 - b.size.height * h, x0 + b.size.width * w, y1), li)
            )
    return words
