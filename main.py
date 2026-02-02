import time
import logging
from pathlib import Path

import cv2
import yaml

from media_manager import MediaManager
from detectors.presence import PresenceDetector
from detectors.wave import WaveDetector
from utils.image_utils import resize_width
from utils.cooldown import CooldownManager


def load_config(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(level_str: str):
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s: %(message)s")


def main():
    root = Path(__file__).parent
    cfg = load_config(root / "config.yaml")
    setup_logging(cfg.get("log_level", "INFO"))

    logger = logging.getLogger("main")

    cap = cv2.VideoCapture(int(cfg.get("camera_index", 0)))
    if not cap.isOpened():
        logger.error("Cannot open camera")
        return

    media = MediaManager(cfg)
    cooldown = CooldownManager(cfg.get("cooldowns", {}))

    presence = PresenceDetector(cfg.get("presence", {}), cfg.get("downscale_width", 320))
    wave = WaveDetector(cfg.get("wave", {}), cfg.get("downscale_width", 320))

    fps = cfg.get("frame_rate", 12)
    frame_interval = 1.0 / float(fps)

    logger.info("Starting main loop (press Ctrl+C to quit)")
    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame capture failed, retrying")
                time.sleep(0.1)
                continue

            small = resize_width(frame, cfg.get("downscale_width", 320))

            # Run detectors
            pres_trigger = presence.update(small)
            if pres_trigger and cooldown.is_ready(pres_trigger):
                logger.info(f"Presence trigger: {pres_trigger}")
                media.play(pres_trigger)
                cooldown.record(pres_trigger)

            wave_trigger = wave.update(small)
            if wave_trigger and cooldown.is_ready(wave_trigger):
                logger.info(f"Wave trigger: {wave_trigger}")
                media.play(wave_trigger)
                cooldown.record(wave_trigger)

            # throttle to configured FPS
            dt = time.time() - t0
            if dt < frame_interval:
                time.sleep(frame_interval - dt)

    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        cap.release()
        media.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

