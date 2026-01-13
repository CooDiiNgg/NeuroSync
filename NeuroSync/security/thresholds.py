from enum import Enum
from dataclasses import dataclass

class SecurityStatus(Enum):
    OK = "ok"
    WARN = "warn"
    BAD = "bad"

@dataclass
class SecurityThresholds:
    ok_threshold: float = 0.2
    warn_threshold: float = 0.4

    def evaluate(self, total_penalty: float) -> SecurityStatus:
        if total_penalty <= self.ok_threshold:
            return SecurityStatus.OK
        elif total_penalty <= self.warn_threshold:
            return SecurityStatus.WARN
        else:
            return SecurityStatus.BAD
    
    def evaluate_component(self, penalty: float) -> SecurityStatus:
        return self.evaluate(penalty)
    