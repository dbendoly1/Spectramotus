import time
import cv2
import numpy as np
from typing import Optional


class PresenceDetector:
    def __init__(self, cfg: dict, downscale_width: int = 320):
        self.min_area = cfg.get("min_area", 800)
        self.linger_seconds = cfg.get("linger_seconds", 3.0)
        self.downscale_width = downscale_width

        self.prev = None
        self.present = False
        self.present_since = None
        self.linger_sent = False

    def _preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        return blur

    def update(self, frame) -> Optional[str]:
        """Process frame and return 'presence' or 'linger' or None."""
        proc = self._preprocess(frame)

        if self.prev is None:
            self.prev = proc
            return None

        diff = cv2.absdiff(self.prev, proc)
        _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        self.prev = proc

        # find contours
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = sum(cv2.contourArea(c) for c in contours)

        now = time.time()
        if area >= self.min_area:
            if not self.present:
                self.present = True
                self.present_since = now
                self.linger_sent = False
                return "presence"
            else:
                # check linger
                if (not self.linger_sent) and (now - (self.present_since or now) >= self.linger_seconds):
                    self.linger_sent = True
                    return "linger"
        else:
            # reset
            self.present = False
            self.present_since = None
            self.linger_sent = False

        return None
