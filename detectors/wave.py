import time
import cv2
import numpy as np
from collections import deque
from typing import Optional


class WaveDetector:
    def __init__(self, cfg: dict, downscale_width: int = 320):
        self.min_magnitude = cfg.get("min_magnitude", 6.0)
        self.min_oscillations = int(cfg.get("min_oscillations", 2))
        self.history_seconds = cfg.get("history_seconds", 2.0)
        self.downscale_width = downscale_width

        self.history = deque()

    def _find_largest_centroid(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 200:
            return None
        M = cv2.moments(c)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        return cx

    def update(self, frame) -> Optional[str]:
        now = time.time()
        cx = self._find_largest_centroid(frame)
        if cx is None:
            # clear history older than window
            while self.history and now - self.history[0][0] > self.history_seconds:
                self.history.popleft()
            return None

        self.history.append((now, cx))
        # trim history
        while self.history and now - self.history[0][0] > self.history_seconds:
            self.history.popleft()

        # compute oscillations: count sign changes in dx where magnitude > min_magnitude
        xs = [p[1] for p in self.history]
        if len(xs) < 3:
            return None

        dxs = [xs[i+1] - xs[i] for i in range(len(xs)-1)]
        signs = [1 if d > self.min_magnitude else (-1 if d < -self.min_magnitude else 0) for d in dxs]
        # count non-zero sign changes
        filtered = [s for s in signs if s != 0]
        if len(filtered) < 2:
            return None
        oscillations = sum(1 for i in range(len(filtered)-1) if filtered[i] != filtered[i+1])
        if oscillations >= self.min_oscillations:
            # clear history to avoid immediate re-trigger
            self.history.clear()
            return "wave"

        return None
