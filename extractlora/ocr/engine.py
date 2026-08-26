"""OCR wrapper. Uses EasyOCR (pure pip install, no external binary required)."""

from functools import lru_cache

import numpy as np
from PIL import Image


@lru_cache(maxsize=1)
def _reader():
    import easyocr

    # quantize=False: dynamic quantization (torch.quantization.quantize_dynamic)
    # segfaults at inference time with this torch build on Windows/Python 3.14.
    return easyocr.Reader(["fr", "en"], gpu=False, verbose=False, quantize=False)


def extract_text(image_path: str) -> str:
    """Runs OCR on an image and returns the recognized text, reading order top-to-bottom."""
    img = Image.open(image_path).convert("RGB")
    results = _reader().readtext(np.array(img))
    # sort by vertical position (top of bounding box) to approximate reading order
    results.sort(key=lambda r: r[0][0][1])
    lines = [text for _bbox, text, _conf in results]
    return "\n".join(lines)
