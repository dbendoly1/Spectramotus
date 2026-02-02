import threading
import logging
from pathlib import Path
from typing import Optional, Callable
from queue import Queue

import cv2


class MediaManager:
    """
    Non-blocking media playback. Manages a background thread for clip playback.
    Returns to base image between clips.
    """
    
    def __init__(self, cfg: dict, display_callback: Optional[Callable] = None):
        """
        Args:
            cfg: Config dict with 'clips', 'base_image', etc.
            display_callback: Optional callback to display frames (for headless or custom rendering).
        """
        self.cfg = cfg
        self.root = Path(__file__).parent
        self.clips = cfg.get("clips", {})
        self.base_image_path = Path(cfg.get("base_image", "assets/background.jpg"))
        if not self.base_image_path.is_absolute():
            self.base_image_path = self.root / self.base_image_path

        self.logger = logging.getLogger("media")
        self._base_image = None
        self._load_base_image()

        self.display_callback = display_callback
        self.playback_queue = Queue(maxsize=1)
        self.stop_event = threading.Event()
        
        # start playback thread
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.playback_thread.start()

    def _load_base_image(self):
        if self.base_image_path.exists():
            im = cv2.imread(str(self.base_image_path))
            if im is not None:
                self.logger.info(f"Loaded base image: {self.base_image_path}")
                self._base_image = im
            else:
                self.logger.warning(f"Failed to load base image: {self.base_image_path}")
        else:
            self.logger.warning(f"Base image not found: {self.base_image_path}")

    def _playback_worker(self):
        """Background thread: consume queue and play clips."""
        window = "display"
        if not self.display_callback:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        while not self.stop_event.is_set():
            try:
                clip_path = self.playback_queue.get(timeout=0.5)
            except:
                # show base image while idle
                if self._base_image is not None and not self.display_callback:
                    cv2.imshow(window, self._base_image)
                    cv2.waitKey(1)
                continue

            # play video
            cap = cv2.VideoCapture(str(clip_path))
            if not cap.isOpened():
                self.logger.warning(f"Unable to open clip: {clip_path}")
                continue

            self.logger.info(f"Playing clip: {clip_path}")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if self.display_callback:
                    self.display_callback(frame)
                else:
                    cv2.imshow(window, frame)
                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        break

            cap.release()
            self.logger.info(f"Finished clip: {clip_path}")

            # return to base image
            if self._base_image is not None:
                if self.display_callback:
                    self.display_callback(self._base_image)
                else:
                    cv2.imshow(window, self._base_image)
                    cv2.waitKey(1)

    def play(self, trigger: str):
        """Queue a clip for playback (non-blocking)."""
        clip_rel = self.clips.get(trigger)
        if not clip_rel:
            self.logger.warning(f"No clip mapped for trigger '{trigger}'")
            return

        clip_path = Path(clip_rel)
        if not clip_path.is_absolute():
            clip_path = self.root / clip_path

        if not clip_path.exists():
            self.logger.warning(f"Clip not found: {clip_path}")
            return

        # try to queue; if full, drop (only one clip queued at a time)
        try:
            self.playback_queue.put_nowait(clip_path)
        except:
            self.logger.warning(f"Playback queue full, dropping trigger: {trigger}")

    def shutdown(self):
        """Gracefully stop playback thread."""
        self.stop_event.set()
        if self.playback_thread.is_alive():
            self.playback_thread.join(timeout=2)
        cv2.destroyAllWindows()
