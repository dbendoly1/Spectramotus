import time
from typing import Dict


class CooldownManager:
    """Manages per-trigger cooldown windows to avoid rapid re-triggers."""
    
    def __init__(self, cooldowns: Dict[str, float]):
        """
        Args:
            cooldowns: Dict mapping trigger names to cooldown duration (seconds).
        """
        self.cooldowns = cooldowns
        self.last_trigger_time: Dict[str, float] = {}
    
    def is_ready(self, trigger: str) -> bool:
        """Check if trigger can fire (cooldown elapsed)."""
        cooldown = self.cooldowns.get(trigger, 0)
        last = self.last_trigger_time.get(trigger, 0)
        return time.time() - last >= cooldown
    
    def record(self, trigger: str):
        """Record that trigger has fired now."""
        self.last_trigger_time[trigger] = time.time()
