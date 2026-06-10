"""
Preprocessing shared by training and on-device inference.

CRITICAL: training and inference MUST use the exact same preprocessing, or the
model sees different inputs in the field than it trained on. Both import this
one module. It depends only on cv2 + numpy (no TensorFlow), so the Raspberry Pi
needs only tflite-runtime + opencv + numpy to run inference.
"""
import cv2
import numpy as np
import config as C


def _resize_to_width(gray):
    if C.RESIZE_WIDTH is None:
        return gray
    h, w = gray.shape[:2]
    if w == C.RESIZE_WIDTH:
        return gray
    scale = C.RESIZE_WIDTH / float(w)
    return cv2.resize(gray, (C.RESIZE_WIDTH, max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA)


def _crop_single(gray):
    if C.BOARD_CROP is None:
        return gray
    x, y, w, h = C.BOARD_CROP
    H, W = gray.shape[:2]
    x = max(0, min(x, W - 1)); y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x)); h = max(1, min(h, H - y))
    return gray[y:y + h, x:x + w]


def _compose_pins(gray):
    """Tile each pin region side-by-side into one strip, then it gets resized
    to the square input. Forces the model to look only at the pins."""
    if not C.PIN_REGIONS:
        return gray
    side = C.INPUT_SIZE
    n = len(C.PIN_REGIONS)
    pw = max(8, side // n)
    H, W = gray.shape[:2]
    patches = []
    for (x, y, w, h) in C.PIN_REGIONS:
        x = max(0, min(x, W - 1)); y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x)); h = max(1, min(h, H - y))
        p = gray[y:y + h, x:x + w]
        if p.size == 0:
            p = np.zeros((side, pw), np.uint8)
        else:
            p = cv2.resize(p, (pw, side), interpolation=cv2.INTER_AREA)
        patches.append(p)
    return np.hstack(patches)


def _clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=C.CLAHE_CLIP,
                            tileGridSize=(C.CLAHE_GRID, C.CLAHE_GRID))
    return clahe.apply(gray)


def preprocess_bgr(bgr):
    """
    BGR frame -> (INPUT_SIZE, INPUT_SIZE, 3) float32 in [0, 1].
    Grayscale (lighting-colour invariant) -> resize -> crop/compose ->
    resize to square -> CLAHE -> optional per-image standardisation -> 3ch.
    """
    if bgr is None or bgr.size == 0:
        raise ValueError("empty image")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = _resize_to_width(gray)
    gray = _compose_pins(gray) if C.CROP_MODE == "pins" else _crop_single(gray)
    gray = cv2.resize(gray, (C.INPUT_SIZE, C.INPUT_SIZE),
                      interpolation=cv2.INTER_AREA)
    if C.USE_CLAHE:
        gray = _clahe(gray)

    g = gray.astype(np.float32)
    if C.PER_IMAGE_STD:
        m, s = float(g.mean()), float(g.std()) + 1e-6
        g = (g - m) / s
        g = (g - g.min()) / (g.max() - g.min() + 1e-6)   # back to [0,1]
    else:
        g = g / 255.0

    return np.stack([g, g, g], axis=-1).astype(np.float32)


def preprocess_file(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError("cannot read image: " + path)
    return preprocess_bgr(bgr)
