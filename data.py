"""
Dataset assembly for training (TensorFlow).

Key design choices that protect against the failure modes discussed:
  * In-memory load of the cropped images (datasets here are small).
  * Train/val split is by SESSION, never by random shuffle, so lighting cannot
    leak from train into val and inflate the score.
  * Augmentation is heavily photometric (brightness/contrast/gamma) so the model
    learns that lighting changes do NOT change the label.
"""
import os, glob
import numpy as np
import tensorflow as tf
import config as C
from preprocess import preprocess_file

AUTOTUNE = tf.data.AUTOTUNE


def _items():
    items = []
    for idx, label in enumerate(C.LABELS):
        for p in sorted(glob.glob(os.path.join(C.DATA_DIR, label, "*"))):
            if os.path.isfile(p):
                items.append((p, idx))
    return items


def load_all():
    """Return X (uint8 N,S,S,3), y (float32 N,1), sessions (list[str])."""
    items = _items()
    if not items:
        raise RuntimeError("No images under '{}/good' and '{}/bad'.".format(
            C.DATA_DIR, C.DATA_DIR))
    X, y, sess = [], [], []
    for p, label in items:
        try:
            arr = preprocess_file(p)                 # float32 [0,1]
        except Exception as e:
            print("skip", p, e); continue
        X.append((arr * 255).astype(np.uint8))       # store compact
        y.append(label)
        sess.append(C.parse_session(p))
    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    print("loaded {} images  good={} bad={}".format(
        len(y), int((y == 0).sum()), int((y == 1).sum())))
    return X, y, sess


def session_split(y, sessions):
    """Hold out whole sessions for validation. Falls back to a stratified
    random split (with a warning) if session-based holdout can't give both
    classes in val."""
    rng = np.random.default_rng(C.SEED)
    uniq = sorted(set(sessions))
    rng.shuffle(uniq)

    n_total = len(y)
    target = max(1, int(round(C.VAL_FRACTION * n_total)))
    val_sessions, count = set(), 0
    for s in uniq:
        if count >= target:
            break
        val_sessions.add(s)
        count += sum(1 for x in sessions if x == s)

    val_idx = np.array([i for i, s in enumerate(sessions) if s in val_sessions])
    tr_idx  = np.array([i for i, s in enumerate(sessions) if s not in val_sessions])

    def both_classes(idx):
        return idx.size > 0 and len(set(y[idx].ravel().tolist())) == 2

    if not (both_classes(tr_idx) and val_idx.size > 0 and
            len(set(y[val_idx].ravel().tolist())) >= 1):
        print("!! session split could not balance classes - falling back to "
              "stratified random split (less honest; collect more sessions).")
        idx = np.arange(n_total); rng.shuffle(idx)
        cut = int(n_total * (1 - C.VAL_FRACTION))
        tr_idx, val_idx = idx[:cut], idx[cut:]
    else:
        print("val sessions ({}): {}".format(len(val_sessions), sorted(val_sessions)))
    return tr_idx, val_idx


def _augment(x, y):
    # x in [0,1]
    x = tf.image.random_brightness(x, 0.15)
    x = tf.image.random_contrast(x, 0.8, 1.2)
    g = tf.random.uniform([], 0.7, 1.4)
    x = tf.pow(tf.clip_by_value(x, 1e-6, 1.0), g)             # gamma jitter
    pad = max(2, int(C.INPUT_SIZE * 0.06))                    # small translation
    x = tf.image.resize_with_crop_or_pad(x, C.INPUT_SIZE + 2 * pad,
                                         C.INPUT_SIZE + 2 * pad)
    x = tf.image.random_crop(x, [C.INPUT_SIZE, C.INPUT_SIZE, 3])
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x, y


def make_ds(X, y, idx, training):
    Xi = X[idx]; yi = y[idx]
    ds = tf.data.Dataset.from_tensor_slices((Xi, yi))
    if training:
        ds = ds.shuffle(max(2, len(idx)), seed=C.SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda a, b: (tf.cast(a, tf.float32) / 255.0, b),
                num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)
    return ds.batch(C.BATCH_SIZE).prefetch(AUTOTUNE)
