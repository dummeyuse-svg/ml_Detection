"""
Offline inference on the Raspberry Pi (or any machine) using the .tflite model.

Needs only:  tflite-runtime  (or full tensorflow)  +  opencv  +  numpy.
No internet, no cloud.

Pi install (lightweight):  pip install tflite-runtime opencv-python-headless numpy

CLI:
    python infer.py --image path/to/frame.jpg
    python infer.py --dir   path/to/folder           # batch; metrics if good/ bad/ subdirs

Integration with your existing Tk + GPIO app:
    from infer import Classifier
    clf = Classifier()                       # loads model_out/pcb_seating.tflite
    label, p_bad = clf.predict_bgr(frame_bgr)   # label = "PASS" or "FAIL"
    gpio.signal_result(label == "PASS")
"""
import os, glob, argparse
import numpy as np
import config as C
from preprocess import preprocess_bgr, preprocess_file

# # tflite-runtime on the Pi; full TF elsewhere
# try:
#     from tflite_runtime.interpreter import Interpreter
# except ImportError:
#     from tensorflow.lite import Interpreter

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


class Classifier:
    def __init__(self, model_path=None, threshold=None):
        self.threshold = C.DECISION_THRESHOLD if threshold is None else threshold
        path = model_path or C.TFLITE_MODEL
        if not os.path.exists(path):
            raise FileNotFoundError(
                "model not found: {} (run train.py first)".format(path))
        self.interp = Interpreter(model_path=path)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.outp = self.interp.get_output_details()[0]

    def _p_bad(self, arr):
        x = arr.astype(np.float32)[None, ...]            # (1,S,S,3) in [0,1]
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        return float(self.interp.get_tensor(self.outp["index"]).ravel()[0])

    def predict_bgr(self, bgr):
        p_bad = self._p_bad(preprocess_bgr(bgr))
        return ("FAIL" if p_bad >= self.threshold else "PASS"), p_bad

    def predict_file(self, path):
        p_bad = self._p_bad(preprocess_file(path))
        return ("FAIL" if p_bad >= self.threshold else "PASS"), p_bad


def _run_dir(clf, d):
    # if it has good/ and bad/ subdirs, also report accuracy
    subdirs = {lbl: os.path.join(d, lbl) for lbl in C.LABELS
               if os.path.isdir(os.path.join(d, lbl))}
    if subdirs:
        tp = tn = fp = fn = 0
        for idx, lbl in enumerate(C.LABELS):
            if lbl not in subdirs:
                continue
            for p in sorted(glob.glob(os.path.join(subdirs[lbl], "*"))):
                if not os.path.isfile(p):
                    continue
                res, pb = clf.predict_file(p)
                true_bad = (idx == 1)
                pred_bad = (res == "FAIL")
                tp += int(pred_bad and true_bad); tn += int((not pred_bad) and (not true_bad))
                fp += int(pred_bad and (not true_bad)); fn += int((not pred_bad) and true_bad)
                print("  {:<40} {}  p_bad={:.3f}".format(os.path.basename(p), res, pb))
        n = tp + tn + fp + fn
        if n:
            print("\nTP={} TN={} FP={} FN={}  acc={:.3f}  (FN = bad passed!)".format(
                tp, tn, fp, fn, (tp + tn) / n))
    else:
        for p in sorted(glob.glob(os.path.join(d, "*"))):
            if os.path.isfile(p):
                try:
                    res, pb = clf.predict_file(p)
                    print("  {:<40} {}  p_bad={:.3f}".format(os.path.basename(p), res, pb))
                except Exception as e:
                    print("  skip", p, e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--dir")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    clf = Classifier(threshold=args.threshold)
    print("loaded {}  threshold={:.2f}".format(C.TFLITE_MODEL, clf.threshold))

    if args.image:
        res, pb = clf.predict_file(args.image)
        print("{}  ->  {}   p_bad={:.3f}".format(args.image, res, pb))
    elif args.dir:
        _run_dir(clf, args.dir)
    else:
        print("give --image <file> or --dir <folder>")
