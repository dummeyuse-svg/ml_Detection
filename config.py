"""
Central configuration for the PCB seating ML inspector.

Everything here is offline. Train on a PC (the only step that may need a
one-time internet hit to download MobileNet weights); run inference on the Pi
with tflite-runtime, fully offline.

The two most important settings are CROP (where the model is allowed to look)
and the SESSION filename convention (how train/val are split honestly).
"""
import os

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR     = "dataset"            # dataset/good/*.jpg  and  dataset/bad/*.jpg
MODEL_DIR    = "model_out"
KERAS_MODEL  = os.path.join(MODEL_DIR, "pcb_seating.keras")
TFLITE_MODEL = os.path.join(MODEL_DIR, "pcb_seating.tflite")
LABELS       = ["good", "bad"]      # index 0 = good, 1 = bad  (model outputs P(bad))

# ── Resize + crop ────────────────────────────────────────────────────────────
# The captured image is first resized to RESIZE_WIDTH (matching your main app's
# TARGET_WIDTH so crop coordinates line up), THEN cropped. The model only ever
# sees the crop -> it physically cannot use the background or overall scene
# brightness as a shortcut. This is the single most important anti-cheat.
RESIZE_WIDTH = 1280                 # set None to skip resizing

# CROP_MODE:
#   "single" -> one rectangle (BOARD_CROP). Easiest; matches your existing ROI.
#   "pins"   -> tile several small pin rectangles (PIN_REGIONS) into one image,
#               forcing the model to look ONLY at the pins. Most robust, but you
#               must locate each pin. Label stays board-level (good/bad).
CROP_MODE = "single"

# (x, y, w, h) in pixels AT RESIZE_WIDTH. Use the same box as your app's ROI.
# Your app's ROI was X 50->1230, Y 200->500  =>  (50, 200, 1180, 300).
BOARD_CROP = (660,55,127,211)   # set None to use the full frame (not advised)

# Used only when CROP_MODE == "pins". One (x, y, w, h) per pin region.
PIN_REGIONS = [
    # (x, y, w, h),
]

INPUT_SIZE = 160                    # model input is INPUT_SIZE x INPUT_SIZE

# ── Preprocessing (lighting robustness) ──────────────────────────────────────
USE_CLAHE     = True                # local contrast equalisation; flattens lighting
CLAHE_CLIP    = 3.0
CLAHE_GRID    = 8
PER_IMAGE_STD = True                # subtract mean / divide std; kills global brightness

# ── Model ────────────────────────────────────────────────────────────────────
# "mobilenet": transfer learning. Best for small datasets. Needs ONE-TIME
#              internet on the TRAINING machine to fetch ImageNet weights; after
#              that everything (including the exported .tflite) is offline.
# "simple":    small CNN from scratch. 100% offline always, but needs more data.
BACKBONE = "mobilenet"

# ── Training ─────────────────────────────────────────────────────────────────
EPOCHS        = 40
BATCH_SIZE    = 16
LEARNING_RATE = 1e-3
VAL_FRACTION  = 0.2                 # fraction of SESSIONS held out (not images)
SEED          = 42

# Manufacturing asymmetry: a BAD board PASSING (false negative) is the costly
# error. DECISION_THRESHOLD is on P(bad). Lower it to catch more bads (fewer
# false negatives, at the cost of more false positives). 0.5 is neutral.
DECISION_THRESHOLD = 0.5

# ── Session parsing (honest train/val split) ─────────────────────────────────
# Images are grouped into "sessions" so a session/day never spans train AND val.
# Recommended filename:  good__2025-06-01__batch3__017.jpg
#   -> session id = "2025-06-01__batch3"
# collect.py writes this convention for you.
SESSION_FROM_FILENAME = True
SESSION_DELIM         = "__"
SESSION_FIELDS        = (1, 2)      # which delimited fields form the session id


def parse_session(path):
    """Derive a session id from a filename, per the convention above."""
    if not SESSION_FROM_FILENAME:
        return "all"
    name = os.path.splitext(os.path.basename(path))[0]
    parts = name.split(SESSION_DELIM)
    try:
        return SESSION_DELIM.join(parts[i] for i in SESSION_FIELDS)
    except IndexError:
        return "unknown"
