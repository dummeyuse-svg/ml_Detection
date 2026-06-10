"""
RUN THIS FIRST, before training anything.

It answers the two questions that decide whether ML can work at all:

  1. Is the seating signal actually visible in your cropped images?
     -> It builds a contact sheet (good row vs bad row) so you can eyeball it.
        If you cannot see the difference between a GOOD and a marginal BAD in
        that sheet, the signal is not in your pixels and NO model will find it
        (this is the false-negative trap you already hit). Fix the optics, not
        the model.

  2. Is lighting leaking into the labels?
     -> It measures how well raw crop brightness ALONE separates good/bad. If
        brightness alone is a strong separator, your model will learn lighting
        instead of seating. It warns you loudly so you can fix data collection
        (mix good + bad in the same sessions/lighting).

Depends only on cv2 + numpy (no TensorFlow, no sklearn).
Usage:  python sanity.py
"""
import os, glob
import numpy as np
import cv2
import config as C
from preprocess import preprocess_file, _resize_to_width, _crop_single, _compose_pins


def _items():
    items = []
    for idx, label in enumerate(C.LABELS):
        for p in sorted(glob.glob(os.path.join(C.DATA_DIR, label, "*"))):
            if os.path.isfile(p):
                items.append((p, idx))
    return items


def _crop_gray(path):
    """The exact grayscale crop the model sees, BEFORE CLAHE/normalisation,
    so brightness leakage is measured on the real input region."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = _resize_to_width(gray)
    gray = _compose_pins(gray) if C.CROP_MODE == "pins" else _crop_single(gray)
    return gray


def _auc(scores, labels):
    """ROC AUC via Mann-Whitney (no sklearn). labels: 0/1."""
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float); ranks[order] = np.arange(1, len(scores) + 1)
    auc = (ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    return float(auc)


def dataset_report(items):
    print("\n=== DATASET REPORT ===")
    n_good = sum(1 for _, y in items if y == 0)
    n_bad  = sum(1 for _, y in items if y == 1)
    print("good: {}    bad: {}    total: {}".format(n_good, n_bad, len(items)))
    if n_good == 0 or n_bad == 0:
        print("!! You need BOTH classes. Put images in dataset/good and dataset/bad.")
        return

    sessions = {}
    for p, y in items:
        s = C.parse_session(p)
        sessions.setdefault(s, [0, 0])[y] += 1
    print("\nsessions: {}".format(len(sessions)))
    single_class = 0
    for s, (g, b) in sorted(sessions.items()):
        flag = ""
        if g == 0 or b == 0:
            flag = "   <-- single-class session (lighting can leak!)"
            single_class += 1
        print("  {:<28} good={:<4} bad={:<4}{}".format(s, g, b, flag))
    if single_class:
        print("\n!! {} session(s) contain only one class. Collect good AND bad "
              "in the SAME sessions so the model can't learn 'which day'."
              .format(single_class))
    if len(sessions) < 4:
        print("\n!! Only {} session(s). Train/val split by session needs several. "
              "Collect across more days/batches.".format(len(sessions)))


def lighting_leakage_check(items):
    print("\n=== LIGHTING LEAKAGE CHECK ===")
    bright, ys = [], []
    for p, y in items:
        g = _crop_gray(p)
        if g is None:
            continue
        bright.append(float(g.mean())); ys.append(y)
    if len(set(ys)) < 2:
        print("need both classes"); return
    auc = _auc(bright, ys); auc = max(auc, 1 - auc)
    print("brightness-only separability (AUC) = {:.3f}".format(auc))
    if auc > 0.70:
        print("!! WARNING: raw crop brightness alone separates good/bad well.")
        print("   Your model is likely to learn LIGHTING, not seating.")
        print("   Fix: collect good AND bad under the same lighting/sessions,")
        print("   interleaved, so brightness is useless as a discriminator.")
    else:
        print("OK: brightness alone is a weak separator. Good - the model is")
        print("    pushed toward the actual seating signal.")


def contact_sheet(items, per_class=12, out="model_out/contact_sheet.png"):
    """Good row(s) vs bad row(s) of the preprocessed crops -> eyeball/blink test."""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    good = [p for p, y in items if y == 0][:per_class]
    bad  = [p for p, y in items if y == 1][:per_class]
    if not good or not bad:
        print("contact sheet skipped (need both classes)"); return

    def tile(paths):
        cells = []
        for p in paths:
            try:
                im = (preprocess_file(p)[:, :, 0] * 255).astype(np.uint8)
            except Exception:
                im = np.zeros((C.INPUT_SIZE, C.INPUT_SIZE), np.uint8)
            im = cv2.copyMakeBorder(im, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=40)
            cells.append(im)
        return np.hstack(cells) if cells else None

    grow, brow = tile(good), tile(bad)
    width = max(grow.shape[1], brow.shape[1])
    def pad(row):
        if row.shape[1] < width:
            row = cv2.copyMakeBorder(row, 0, 0, 0, width - row.shape[1],
                                     cv2.BORDER_CONSTANT, value=0)
        return row
    grow, brow = pad(grow), pad(brow)
    label_h = 22
    gtag = np.zeros((label_h, width), np.uint8)
    btag = np.zeros((label_h, width), np.uint8)
    cv2.putText(gtag, "GOOD", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
    cv2.putText(btag, "BAD",  (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
    sheet = np.vstack([gtag, grow, btag, brow])
    cv2.imwrite(out, sheet)
    print("\ncontact sheet written -> {}".format(out))
    print("OPEN IT. If you cannot tell GOOD from a marginal BAD by eye, the")
    print("signal is not in the pixels and ML cannot help. Fix the optics.")


if __name__ == "__main__":
    items = _items()
    if not items:
        print("No images found under '{}/good' and '{}/bad'.".format(C.DATA_DIR, C.DATA_DIR))
        raise SystemExit(1)
    dataset_report(items)
    lighting_leakage_check(items)
    contact_sheet(items)
    print("\nDone. Review the warnings and the contact sheet before training.")
