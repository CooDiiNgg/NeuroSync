from dataclasses import dataclass
from typing import Optional


@dataclass
class AdversarialScheduler:
    max_weight: float = 0.15
    min_active_weight: float = 0.02
    increase_rate_high: float = 0.01
    increase_rate_low: float = 0.005
    decrease_rate: float = 0.005
    decrease_rate_recovery: float = 0.02
    
    bob_threshold_inactive: float = 95.0
    bob_threshold_active: float = 98.0
    eve_threshold_high: float = 70.0
    eve_threshold_mid: float = 40.0
    eve_threshold_low: float = 20.0
    
    def step(
        self,
        current_weight: float,
        running_bob_accuracy: float,
        running_eve_accuracy: float,
        maintenance_mode: bool = False,
    ) -> float:
        if maintenance_mode:
            return current_weight
        
        if running_bob_accuracy < self.bob_threshold_inactive:
            return 0.0
        elif running_bob_accuracy < self.bob_threshold_active:
            return max(0.0, current_weight - self.decrease_rate_recovery)
        else:
            if running_eve_accuracy > self.eve_threshold_high:
                return min(self.max_weight, current_weight + self.increase_rate_high)
            elif running_eve_accuracy > self.eve_threshold_mid:
                return min(self.max_weight, current_weight + self.increase_rate_low)
            elif running_eve_accuracy < self.eve_threshold_low:
                return max(self.min_active_weight, current_weight - self.decrease_rate)
            return current_weight


@dataclass
class SecurityScheduler:  
    max_weight: float = 0.1
    increase_rate: float = 0.01
    decrease_rate: float = 0.005
    decrease_rate_recovery: float = 0.01
    
    bob_threshold_inactive: float = 95.0
    bob_threshold_active: float = 98.0
    security_threshold_high: float = 0.3
    security_threshold_low: float = 0.1
    
    def step(
        self,
        current_weight: float,
        running_bob_accuracy: float,
        running_security: float,
        maintenance_mode: bool = False,
    ) -> float:
        if maintenance_mode:
            return current_weight
        
        if running_bob_accuracy < self.bob_threshold_inactive:
            return 0.0
        elif running_bob_accuracy < self.bob_threshold_active:
            return max(0.0, current_weight - self.decrease_rate_recovery)
        else:
            if running_security > self.security_threshold_high:
                return min(self.max_weight, current_weight + self.increase_rate)
            elif running_security < self.security_threshold_low:
                return max(0.0, current_weight - self.decrease_rate)
            return current_weight


@dataclass
class ConfidenceScheduler:
    max_weight: float = 0.3
    low_threshold: float = 90.0
    high_threshold: float = 97.0
    
    def step(self, bit_accuracy: float) -> float:
        if bit_accuracy < self.low_threshold:
            return 0.0
        elif bit_accuracy < self.high_threshold:
            ratio = (bit_accuracy - self.low_threshold) / (self.high_threshold - self.low_threshold)
            return ratio * self.max_weight
        else:
            return self.max_weight


class MaintenanceModeController:
    def __init__(
        self,
        enter_threshold: float = 99.0,
        exit_threshold: float = 95.0,
        consecutive_required: int = 3,
        eve_alert_threshold: float = 60.0,
        security_alert_threshold: float = 0.4,
    ):
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.consecutive_required = consecutive_required
        self.eve_alert_threshold = eve_alert_threshold
        self.security_alert_threshold = security_alert_threshold
        
        self._consecutive_count = 0
        self._in_maintenance = False
    
    @property
    def in_maintenance(self) -> bool:
        return self._in_maintenance
    
    def step(
        self,
        running_bob_accuracy: float,
        running_eve_accuracy: float,
        running_security: float,
    ) -> tuple:
        entered = False
        exited = False
        reason = ""
        
        if not self._in_maintenance:
            if running_bob_accuracy >= self.enter_threshold:
                self._consecutive_count += 1
                if self._consecutive_count >= self.consecutive_required:
                    self._in_maintenance = True
                    entered = True
                    reason = f"accuracy {running_bob_accuracy:.2f}% for {self._consecutive_count} intervals"
            else:
                self._consecutive_count = 0
        else:
            if running_bob_accuracy < self.exit_threshold:
                exited = True
                reason = f"accuracy dropped to {running_bob_accuracy:.1f}%"
            elif (running_eve_accuracy > self.eve_alert_threshold and 
                  running_security > self.security_alert_threshold):
                exited = True
                reason = f"security alert: Eve={running_eve_accuracy:.1f}%, sec={running_security:.3f}"
            
            if exited:
                self._in_maintenance = False
                self._consecutive_count = 0
        
        return entered, exited, reason
    
    def reset(self) -> None:
        self._consecutive_count = 0
        self._in_maintenance = False


class LossScheduler:
    def __init__(
        self,
        plateau_threshold: int = 10,
        bob_accuracy_threshold: float = 90.0,
        eve_accuracy_threshold: float = 30.0,
    ):
        self.plateau_threshold = plateau_threshold
        self.bob_accuracy_threshold = bob_accuracy_threshold
        self.eve_accuracy_threshold = eve_accuracy_threshold
    
    def should_use_smooth_l1_bob(
        self,
        plateau_count: int,
        recent_accuracy: float,
    ) -> bool:
        return plateau_count >= self.plateau_threshold and recent_accuracy < self.bob_accuracy_threshold
    
    def should_use_smooth_l1_eve(
        self,
        running_eve_accuracy: float,
    ) -> bool:
        return running_eve_accuracy < self.eve_accuracy_threshold
