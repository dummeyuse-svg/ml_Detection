"""
Train the seating classifier and export it for the Pi.

Run sanity.py FIRST. If the contact sheet shows no visible good/bad difference,
or the lighting-leakage AUC is high, stop and fix data/optics - training will
only produce a confident-but-wrong model.

Usage:
    python train.py

Outputs (in model_out/):
    pcb_seating.keras        full Keras model
    pcb_seating.tflite       quantised model for the Pi
    gradcam/*.png            where the model looked (verify it's the pins/edge)
    eval_report.txt          confusion + threshold sweep + per-session accuracy
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import config as C
import data as D


# ── Model ────────────────────────────────────────────────────────────────────
def build_model():
    inp = layers.Input((C.INPUT_SIZE, C.INPUT_SIZE, 3))
    x = layers.Rescaling(2.0, offset=-1.0)(inp)          # [0,1] -> [-1,1]

    if C.BACKBONE == "mobilenet":
        backbone = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet",
            input_shape=(C.INPUT_SIZE, C.INPUT_SIZE, 3))
        backbone.trainable = False
        x = backbone(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
    else:  # "simple" - fully offline, needs more data
        for f in (16, 32, 64):
            x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
            x = layers.Conv2D(f, 3, padding="same", activation="relu")(x)
            x = layers.MaxPool2D()(x)
            x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)        # P(bad)
    return Model(inp, out)


# ── Evaluation ───────────────────────────────────────────────────────────────
def evaluate(model, X, y, val_idx, sessions, report_path):
    lines = []
    def out(s):
        print(s); lines.append(s)

    xv = (X[val_idx].astype(np.float32) / 255.0)
    yv = y[val_idx].ravel().astype(int)
    p_bad = model.predict(xv, verbose=0).ravel()

    out("\n=== VALIDATION ({} images) ===".format(len(yv)))
    out("threshold sweep (FN = bad board that PASSED = worst error):")
    out("  thr   acc    FP   FN")
    for t in (0.30, 0.40, 0.50, 0.60, 0.70):
        pred = (p_bad >= t).astype(int)
        fp = int(((pred == 1) & (yv == 0)).sum())
        fn = int(((pred == 0) & (yv == 1)).sum())
        acc = float((pred == yv).mean()) if len(yv) else 0.0
        mark = "  <- DECISION_THRESHOLD" if abs(t - C.DECISION_THRESHOLD) < 1e-6 else ""
        out("  {:.2f}  {:.3f}  {:>3}  {:>3}{}".format(t, acc, fp, fn, mark))

    t = C.DECISION_THRESHOLD
    pred = (p_bad >= t).astype(int)
    tp = int(((pred == 1) & (yv == 1)).sum()); tn = int(((pred == 0) & (yv == 0)).sum())
    fp = int(((pred == 1) & (yv == 0)).sum()); fn = int(((pred == 0) & (yv == 1)).sum())
    out("\nconfusion @ thr={:.2f}:  TP={} TN={} FP={} FN={}".format(t, tp, tn, fp, fn))

    # per-session accuracy (honesty check: should be similar across sessions)
    out("\nper-session accuracy:")
    vs = [sessions[i] for i in val_idx]
    for s in sorted(set(vs)):
        m = np.array([x == s for x in vs])
        if m.any():
            acc = float((pred[m] == yv[m]).mean())
            out("  {:<28} n={:<4} acc={:.3f}".format(s, int(m.sum()), acc))

    os.makedirs(C.MODEL_DIR, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return p_bad, yv


# ── Grad-CAM (best effort) ───────────────────────────────────────────────────
def gradcam(model, X, val_idx, n=8):
    """Save heatmaps of where the model looks. If it attends to the pins/edge,
    good. If it attends to background, it learned lighting -> distrust it.
    Wrapped in try/except because nested-backbone CAM is fragile across TF
    versions; it's a diagnostic, not part of inference."""
    try:
        import cv2
        out_dir = os.path.join(C.MODEL_DIR, "gradcam"); os.makedirs(out_dir, exist_ok=True)

        # find the last 4D (conv) feature map, including inside a nested backbone
        last_conv = None; host = model
        def scan(m):
            lc = None
            for lyr in m.layers:
                try:
                    if len(lyr.output.shape) == 4:
                        lc = lyr
                except Exception:
                    pass
            return lc
        for lyr in model.layers:
            if isinstance(lyr, tf.keras.Model):
                inner = scan(lyr)
                if inner is not None:
                    last_conv, host = inner, lyr
        if last_conv is None:
            last_conv = scan(model); host = model
        if last_conv is None:
            print("grad-cam skipped (no conv layer found)"); return

        grad_model = tf.keras.models.Model(model.inputs,
                                           [last_conv.output, model.output])
        idx = list(val_idx)[:n]
        for k, i in enumerate(idx):
            img = (X[i].astype(np.float32) / 255.0)[None, ...]
            with tf.GradientTape() as tape:
                conv, pred = grad_model(img)
                loss = pred[:, 0]
            grads = tape.gradient(loss, conv)[0]
            conv = conv[0]
            weights = tf.reduce_mean(grads, axis=(0, 1))
            cam = tf.reduce_sum(conv * weights, axis=-1).numpy()
            cam = np.maximum(cam, 0)
            cam = cam / (cam.max() + 1e-8)
            cam = cv2.resize(cam, (C.INPUT_SIZE, C.INPUT_SIZE))
            heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
            base = (X[i]).astype(np.uint8)
            overlay = cv2.addWeighted(base, 0.6, heat, 0.4, 0)
            cv2.imwrite(os.path.join(out_dir, "cam_{:02d}.png".format(k)), overlay)
        print("grad-cam overlays -> {} (check the model looks at the pins/edge)".format(out_dir))
    except Exception as e:
        print("grad-cam skipped:", e)


# ── Export ───────────────────────────────────────────────────────────────────
def export_tflite(model):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]      # dynamic-range quant
    tfl = conv.convert()
    os.makedirs(C.MODEL_DIR, exist_ok=True)
    with open(C.TFLITE_MODEL, "wb") as f:
        f.write(tfl)
    print("tflite written -> {} ({:.1f} KB)".format(
        C.TFLITE_MODEL, os.path.getsize(C.TFLITE_MODEL) / 1024.0))


def main():
    X, y, sessions = D.load_all()
    tr_idx, val_idx = D.session_split(y, sessions)
    print("train={}  val={}".format(len(tr_idx), len(val_idx)))

    n_good = int((y[tr_idx] == 0).sum()); n_bad = int((y[tr_idx] == 1).sum())
    total = max(1, n_good + n_bad)
    class_weight = {0: total / (2.0 * max(1, n_good)),
                    1: total / (2.0 * max(1, n_bad))}
    print("class_weight:", {k: round(v, 3) for k, v in class_weight.items()})

    train_ds = D.make_ds(X, y, tr_idx, training=True)
    val_ds   = D.make_ds(X, y, val_idx, training=False) if len(val_idx) else None

    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(C.LEARNING_RATE),
                  loss="binary_crossentropy",
                  metrics=["accuracy",
                           tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall")])

    cbs = [tf.keras.callbacks.EarlyStopping(
                monitor="val_loss" if val_ds else "loss",
                patience=8, restore_best_weights=True)]
    model.fit(train_ds, validation_data=val_ds, epochs=C.EPOCHS,
              class_weight=class_weight, callbacks=cbs, verbose=2)

    os.makedirs(C.MODEL_DIR, exist_ok=True)
    model.save(C.KERAS_MODEL)
    print("keras model -> {}".format(C.KERAS_MODEL))

    if len(val_idx):
        evaluate(model, X, y, val_idx, sessions,
                 os.path.join(C.MODEL_DIR, "eval_report.txt"))
        gradcam(model, X, val_idx)

    export_tflite(model)
    print("\nDone. Verify: (1) FN count acceptable at your threshold, "
          "(2) per-session accuracy is uniform, (3) grad-cam looks at the "
          "pins/edge - not the background.")


if __name__ == "__main__":
    main()
