"""
Security thresholds and status definitions for evaluating NeuroSync encryption schemes.
"""

from enum import Enum
from dataclasses import dataclass

class SecurityStatus(Enum):
    """Security status levels."""
    OK = "ok"
    WARN = "warn"
    BAD = "bad"

@dataclass
class SecurityThresholds:
    """Thresholds for evaluating security checks."""

    ok_threshold: float = 0.2
    warn_threshold: float = 0.4

    def evaluate(self, total_penalty: float) -> SecurityStatus:
        """Evaluates overall security status based on total penalty."""
        if total_penalty <= self.ok_threshold:
            return SecurityStatus.OK
        elif total_penalty <= self.warn_threshold:
            return SecurityStatus.WARN
        else:
            return SecurityStatus.BAD
    
    def evaluate_component(self, penalty: float) -> SecurityStatus:
        """Evaluates individual component security status based on penalty."""
        return self.evaluate(penalty)
    