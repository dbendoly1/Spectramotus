import time
import cv2
import numpy as np
from typing import Optional


class PresenceDetector:
    def __init__(self, cfg: dict, downscale_width: int = 320):
        self.min_area = cfg.get("min_area", 800)
        self.linger_seconds = cfg.get("linger_seconds", 3.0)
        self.downscale_width = downscale_width

        # Use background subtractor to detect stationary + moving objects
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )

        self.present = False
        self.present_since = None
        self.linger_sent = False

    def _preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        return blur

    def update(self, frame) -> Optional[str]:
        """
        Detect presence (stationary or moving person) using background subtraction.
        Return 'presence' on initial detection, 'linger' after N seconds of presence.
        """
        gray = self._preprocess(frame)

        # Apply background subtractor
        mask = self.bg_sub.apply(gray)
        
        # Remove shadows (gray pixels in the mask)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours and compute total foreground area
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = sum(cv2.contourArea(c) for c in contours)

        now = time.time()
        if area >= self.min_area:
            if not self.present:
                # Transition from no-presence to presence
                self.present = True
                self.present_since = now
                self.linger_sent = False
                return "presence"
            else:
                # Presence is ongoing; check if linger time has elapsed
                elapsed = now - (self.present_since or now)
                if (not self.linger_sent) and (elapsed >= self.linger_seconds):
                    self.linger_sent = True
                    return "linger"
        else:
            # Presence has ended; reset state
            self.present = False
            self.present_since = None
            self.linger_sent = False

        return None
