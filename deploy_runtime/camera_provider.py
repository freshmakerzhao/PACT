import numpy as np


class TopCameraProvider:
    def __init__(self, source=0, width=640, height=480):
        self.source = source
        self.width = width
        self.height = height
        self._cap = None

    def connect(self):
        try:
            import cv2
        except ImportError as exc:
            raise ImportError('opencv-python is required for TopCameraProvider') from exc

        self._cv2 = cv2
        self._cap = cv2.VideoCapture(self.source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def get_top_image(self):
        if self._cap is None:
            self.connect()

        ok, frame = self._cap.read()
        if not ok or frame is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        return frame

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
