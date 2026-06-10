import cv2


class PiCam:
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
        except:
            pass


def get_camera(webcam_index=0):

    try:
        print("Trying Raspberry Pi Camera...")

        cam = PiCam()

        ok, _ = cam.read()

        if ok:
            print("✓ Raspberry Pi Camera detected")
            return cam

    except Exception as e:
        print(f"Pi Camera unavailable: {e}")

    print("Trying USB Webcam...")

    cap = cv2.VideoCapture(webcam_index)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("✓ USB Webcam detected")
        return cap

    raise RuntimeError("No camera found")


def main():

    cap = get_camera()

    print("\nControls:")
    print("SPACE = Freeze frame and select ROI")
    print("Q     = Quit")

    frame = None

    while True:

        ok, frame = cap.read()

        if not ok:
            print("Camera read failed")
            break

        display = frame.copy()

        cv2.putText(
            display,
            "SPACE = Select ROI | Q = Quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("Live Camera", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == 32:  # SPACE
            break

    if frame is not None:

        roi = cv2.selectROI(
            "Draw ROI and Press ENTER",
            frame,
            showCrosshair=True,
            fromCenter=False
        )

        x, y, w, h = roi

        print("\nROI Selected")
        print("---------------------")
        print(f"x      = {x}")
        print(f"y      = {y}")
        print(f"width  = {w}")
        print(f"height = {h}")

        print("\nCopy into config.py:")
        print(f"BOARD_CROP = ({x}, {y}, {w}, {h})")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
