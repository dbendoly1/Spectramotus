import threading
import logging
from pathlib import Path
from typing import Optional, Callable, Tuple
from queue import Queue, Empty

import cv2


class MediaManager:
    """
    Non-blocking media playback. Manages a background thread for clip playback.
    Returns to base image between clips.
    Fullscreen "cover" rendering (fills screen, preserves aspect ratio; crops if needed).
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

        # Display config
        self.display_callback = display_callback
        self.window = cfg.get("window_name", "display")
        self._display_inited = False
        self._base_image_full = None  

        # If you know your screen size, set in config:
        # cfg["screen_w"]=..., cfg["screen_h"]=...
        self.screen_w = int(cfg.get("screen_w", 0))
        self.screen_h = int(cfg.get("screen_h", 0))

        # Playback pacing
        # If your clips are 30fps, 33ms is right. We'll also try to read FPS from file.
        self.default_frame_delay_ms = int(cfg.get("frame_delay_ms", 33))

        self._base_image = None
        self._load_base_image()

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

    def _init_display(self):
        if self.display_callback:
            return
        if getattr(self, "_display_inited", False):
            return

        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)

        # IMPORTANT: force window size BEFORE fullscreen
        if self.screen_w <= 0 or self.screen_h <= 0:
            # Set these in cfg ideally. Fallback if you must:
            self.screen_w = int(self.cfg.get("screen_w", 1024))
            self.screen_h = int(self.cfg.get("screen_h", 600))

        cv2.resizeWindow(self.window, self.screen_w, self.screen_h)

        # Then request fullscreen
        cv2.setWindowProperty(self.window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        if self._base_image is not None and self._base_image_full is None:
            self._base_image_full = self._cover_resize(self._base_image, self.screen_w, self.screen_h)
        self._display_inited = True
        
    @staticmethod
    def _cover_resize(img, target_w: int, target_h: int):
        """
        Scale-to-cover + center crop. Preserves aspect ratio; crops overflow.
        """
        h, w = img.shape[:2]
        if w == 0 or h == 0:
            return img

        scale = max(target_w / w, target_h / h)
        nw, nh = int(w * scale), int(h * scale)

        # Resize (INTER_AREA is good for downscale; INTER_LINEAR ok for upscale)
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(img, (nw, nh), interpolation=interp)

        x0 = max(0, (nw - target_w) // 2)
        y0 = max(0, (nh - target_h) // 2)
        return resized[y0:y0 + target_h, x0:x0 + target_w]

    def _show(self, frame):
        if frame is None:
            return

        if self.display_callback:
            self.display_callback(frame)
            return

        self._init_display()

        # If it's the base image, use the cached fullscreen version
        if self._base_image is not None and frame is self._base_image and self._base_image_full is not None:
            cv2.imshow(self.window, self._base_image_full)
            cv2.waitKey(1)
            return

        h, w = frame.shape[:2]
        if w == self.screen_w and h == self.screen_h:
            frame2 = frame  # no resize/crop cost
        else:
            frame2 = self._cover_resize(frame, self.screen_w, self.screen_h)

        cv2.imshow(self.window, frame2)
        cv2.waitKey(1)


    def _playback_worker(self):
        """Background thread: consume queue and play clips."""
        # Show base image immediately on startup
        if self._base_image is not None:
            self._show(self._base_image)

        while not self.stop_event.is_set():
            try:
                clip_path = self.playback_queue.get(timeout=0.5)
            except Empty:
                # show base image while idle
                if self._base_image is not None:
                    self._show(self._base_image)
                continue
            except Exception:
                continue

            # play video
            cap = cv2.VideoCapture(str(clip_path))
            if not cap.isOpened():
                self.logger.warning(f"Unable to open clip: {clip_path}")
                continue

            # Try to honor video FPS (helps smooth playback)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 1:
                frame_delay_ms = max(1, int(1000 / fps))
            else:
                frame_delay_ms = self.default_frame_delay_ms

            self.logger.info(f"Playing clip: {clip_path} (delay={frame_delay_ms}ms)")

            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                if self.display_callback:
                    self.display_callback(frame)
                else:
                    # Fullscreen cover draw
                    self._init_display()
                    frame2 = self._cover_resize(frame, self.screen_w, self.screen_h)
                    cv2.imshow(self.window, frame2)

                    # Delay based on FPS; allow quitting with 'q'
                    if cv2.waitKey(frame_delay_ms) & 0xFF == ord('q'):
                        self.stop_event.set()
                        break

            cap.release()
            self.logger.info(f"Finished clip: {clip_path}")

            # return to base image
            if self._base_image is not None:
                self._show(self._base_image)

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
        except Exception:
            self.logger.warning(f"Playback queue full, dropping trigger: {trigger}")

    def shutdown(self):
        """Gracefully stop playback thread."""
        self.stop_event.set()
        if self.playback_thread.is_alive():
            self.playback_thread.join(timeout=2)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
