# """
# Capture and label images into the dataset, with the session-encoded filename
# the rest of the toolkit expects:

#     dataset/good/good__2025-06-01__batch3__017.jpg
#     dataset/bad/ bad__2025-06-01__batch3__004.jpg

# Why the convention matters: train/val are split by session ("2025-06-01__batch3")
# so lighting cannot leak across the split. ALWAYS capture good AND bad in the
# same session, interleaved.

# Controls (a preview window opens):
#     g  = save current frame as GOOD
#     b  = save current frame as BAD
#     q  = quit

# Usage:
#     python collect.py --batch batch3                 # webcam (default index 0)
#     python collect.py --batch batch3 --camera pi     # Raspberry Pi camera
#     python collect.py --batch batch3 --webcam 1      # specific webcam index
# """
# import os, argparse
# from datetime import datetime
# import cv2
# import config as C


# def _next_index(folder, label, session):
#     os.makedirs(folder, exist_ok=True)
#     prefix = "{}{}{}".format(label, C.SESSION_DELIM, session)
#     existing = [f for f in os.listdir(folder) if f.startswith(prefix)]
#     return len(existing)


# def _save(frame_bgr, label, session):
#     folder = os.path.join(C.DATA_DIR, label)
#     idx = _next_index(folder, label, session)
#     name = "{lab}{d}{sess}{d}{idx:04d}.jpg".format(
#         lab=label, d=C.SESSION_DELIM, sess=session, idx=idx)
#     path = os.path.join(folder, name)
#     cv2.imwrite(path, frame_bgr)
#     return path


# class _PiCam:
#     def __init__(self):
#         from picamera2 import Picamera2
#         self.cam = Picamera2()
#         self.cam.configure(self.cam.create_preview_configuration(
#             main={"size": (1280, 720), "format": "RGB888"}))
#         self.cam.start()
#     def read(self):
#         import cv2 as _cv2
#         rgb = self.cam.capture_array("main")
#         return True, _cv2.cvtColor(rgb, _cv2.COLOR_RGB2BGR)
#     def release(self):
#         try: self.cam.stop()
#         except Exception: pass


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--batch", default="batch1", help="batch tag for the session id")
#     ap.add_argument("--camera", choices=["webcam", "pi"], default="webcam")
#     ap.add_argument("--webcam", type=int, default=0)
#     args = ap.parse_args()

#     date = datetime.now().strftime("%Y-%m-%d")
#     session = "{}{}{}".format(date, C.SESSION_DELIM, args.batch)
#     print("session id =", session)
#     print("g = save GOOD, b = save BAD, q = quit")

#     if args.camera == "pi":
#         cap = _PiCam()
#     else:
#         cap = cv2.VideoCapture(args.webcam)
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#     n_good = n_bad = 0
#     try:
#         while True:
#             ok, frame = cap.read()
#             if not ok:
#                 print("camera read failed"); break
#             view = frame.copy()
#             cv2.putText(view, "g=GOOD  b=BAD  q=quit   good:{} bad:{}".format(
#                 n_good, n_bad), (10, 28),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#             cv2.imshow("collect - " + session, view)
#             k = cv2.waitKey(1) & 0xFF
#             if k == ord("q"):
#                 break
#             elif k == ord("g"):
#                 p = _save(frame, "good", session); n_good += 1; print("GOOD ->", p)
#             elif k == ord("b"):
#                 p = _save(frame, "bad", session); n_bad += 1; print("BAD  ->", p)
#     finally:
#         cap.release()
#         cv2.destroyAllWindows()
#         print("saved this session: good={} bad={}".format(n_good, n_bad))


# if __name__ == "__main__":
#     main()


"""
Capture and label images into the dataset, with the session-encoded filename.

Example output:

    dataset/good/good__2025-06-01__batch3__0001.jpg
    dataset/bad/bad__2025-06-01__batch3__0001.jpg

Controls:
    g = save current frame as GOOD
    b = save current frame as BAD
    q = quit

Usage:

    python collect.py
    python collect.py --batch batch2
    python collect.py --webcam 1
"""

import os
import argparse
from datetime import datetime

import cv2
import config as C


def _next_index(folder, label, session):
    os.makedirs(folder, exist_ok=True)

    prefix = f"{label}{C.SESSION_DELIM}{session}"

    existing = [
        f for f in os.listdir(folder)
        if f.startswith(prefix)
    ]

    return len(existing)


def _save(frame_bgr, label, session):
    folder = os.path.join(C.DATA_DIR, label)

    idx = _next_index(folder, label, session)

    filename = (
        f"{label}"
        f"{C.SESSION_DELIM}"
        f"{session}"
        f"{C.SESSION_DELIM}"
        f"{idx:04d}.jpg"
    )

    path = os.path.join(folder, filename)

    cv2.imwrite(path, frame_bgr)

    return path


class _PiCam:
    def __init__(self):
        from picamera2 import Picamera2

        self.cam = Picamera2()

        self.cam.configure(
            self.cam.create_preview_configuration(
                main={
                    "size": (1280, 720),
                    "format": "RGB888"
                }
            )
        )

        self.cam.start()

    def read(self):
        rgb = self.cam.capture_array("main")

        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        return True, frame

    def release(self):
        try:
            self.cam.stop()
        except Exception:
            pass


def get_camera(webcam_index=0):
    """
    Try Raspberry Pi Camera first.
    If unavailable, fall back to USB webcam.
    """

    try:
        print("\nTrying Raspberry Pi Camera...")

        cam = _PiCam()

        ok, _ = cam.read()

        if ok:
            print("✓ Raspberry Pi Camera detected")
            return cam

    except Exception as e:
        print(f"Pi Camera unavailable: {e}")

    print("\nTrying USB Webcam...")

    cap = cv2.VideoCapture(webcam_index)

    if cap.isOpened():

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("✓ USB Webcam detected")

        return cap

    raise RuntimeError(
        "No camera found. Check Pi Camera connection or USB webcam."
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--batch",
        default="batch1",
        help="Batch/session tag"
    )

    ap.add_argument(
        "--webcam",
        type=int,
        default=0,
        help="USB webcam index"
    )

    args = ap.parse_args()

    date = datetime.now().strftime("%Y-%m-%d")

    session = (
        f"{date}"
        f"{C.SESSION_DELIM}"
        f"{args.batch}"
    )

    print("\nSession ID:", session)
    print("g = GOOD")
    print("b = BAD")
    print("q = QUIT")

    cap = get_camera(args.webcam)

    n_good = 0
    n_bad = 0

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                print("Camera read failed")
                break

            view = frame.copy()

            # Draw ROI box
            if C.BOARD_CROP is not None:

                x, y, w, h = C.BOARD_CROP

                cv2.rectangle(
                    view,
                    (x, y),
                    (x + w, y + h),
                    (0, 0, 255),   # Red ROI
                    2
                )

                cv2.putText(
                    view,
                    "MODEL ROI",
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            cv2.putText(
                view,
                f"g=GOOD  b=BAD  q=QUIT   good:{n_good} bad:{n_bad}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow(
                f"Collect Dataset - {session}",
                view
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("g"):

                path = _save(
                    frame,
                    "good",
                    session
                )

                n_good += 1

                print(f"GOOD -> {path}")

            elif key == ord("b"):

                path = _save(
                    frame,
                    "bad",
                    session
                )

                n_bad += 1

                print(f"BAD  -> {path}")

    finally:

        try:
            cap.release()
        except Exception:
            pass

        cv2.destroyAllWindows()

        print(
            f"\nSaved this session:"
            f" good={n_good}"
            f" bad={n_bad}"
        )
        
if __name__ == "__main__":
    main()
