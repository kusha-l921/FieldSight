"""
Thread-safe OpenCV video capture and camera stream manager for FieldSight-Lite.
Includes frame rate throttling, device probing, and graceful hardware error handling.
"""

from __future__ import annotations

import threading
import time
from typing import Generator, Optional, Tuple, Union
import cv2
import numpy as np


class CameraCapture:
    """
    Thread-safe camera stream manager with frame rate limiter and error resilience.
    Supports USB webcams, Raspberry Pi Camera modules (via V4L2/GStreamer), and video files.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        target_fps: float = 15.0,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None
    ):
        self.source = source
        self.target_fps = max(1.0, float(target_fps))
        self.frame_interval = 1.0 / self.target_fps
        self.frame_width = frame_width
        self.frame_height = frame_height

        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._is_open = False
        self._last_frame_time = 0.0

    def open(self) -> bool:
        """Initializes the video capture device or file."""
        with self._lock:
            if self._is_open and self._cap is not None and self._cap.isOpened():
                return True

            if isinstance(self.source, int):
                self._cap = cv2.VideoCapture(self.source)
            else:
                self._cap = cv2.VideoCapture(str(self.source))

            if not self._cap.isOpened():
                self._is_open = False
                return False

            if self.frame_width is not None:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            if self.frame_height is not None:
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

            self._is_open = True
            self._last_frame_time = time.perf_counter()
            return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads the next frame, applying rate throttling to maintain target FPS.
        Returns (success: bool, frame: Optional[np.ndarray]).
        """
        with self._lock:
            if not self._is_open or self._cap is None or not self._cap.isOpened():
                return False, None

            # Frame rate throttling
            now = time.perf_counter()
            elapsed = now - self._last_frame_time
            if elapsed < self.frame_interval:
                sleep_time = self.frame_interval - elapsed
                time.sleep(sleep_time)

            ret, frame = self._cap.read()
            self._last_frame_time = time.perf_counter()

            if not ret or frame is None:
                return False, None

            return True, frame

    def stream_frames(self, max_frames: Optional[int] = None) -> Generator[np.ndarray, None, None]:
        """Generator yielding frames until EOF or max_frames reached."""
        if not self.open():
            raise RuntimeError(f"Failed to open video source: {self.source}")

        count = 0
        try:
            while True:
                if max_frames is not None and count >= max_frames:
                    break
                ret, frame = self.read_frame()
                if not ret or frame is None:
                    break
                yield frame
                count += 1
        finally:
            self.release()

    def release(self) -> None:
        """Closes the video capture and releases hardware resources."""
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self._is_open = False

    def __enter__(self) -> CameraCapture:
        if not self.open():
            raise RuntimeError(f"Could not open camera/video source: {self.source}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
