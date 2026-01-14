from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import torch


@dataclass
class TrainingState:
    running_bob_accuracy: float = 0.0
    running_eve_accuracy: float = 0.0
    running_security: float = 0.0
    accuracy_momentum: float = 0.9
    
    perfect_count: int = 0
    total_count: int = 0
    eve_guess_count: int = 0
    correct_bits: int = 0
    total_bits: int = 0
    plateau_count: int = 0
    consecutive_accuracy: int = 0
    
    best_accuracy: float = 0.0
    best_alice_state: Optional[Dict[str, torch.Tensor]] = None
    best_bob_state: Optional[Dict[str, torch.Tensor]] = None
    
    bob_errors: List[float] = field(default_factory=list)
    eve_errors: List[float] = field(default_factory=list)
    
    maintenance_mode: bool = False
    use_smooth_l1: bool = False
    eve_use_smooth_l1: bool = False
    
    adversarial_weight: float = 0.0
    confidence_weight: float = 0.0
    security_weight: float = 0.0
    
    def update_accuracy(self, bob_acc: float, eve_acc: float) -> None:
        self.running_bob_accuracy = (
            self.accuracy_momentum * self.running_bob_accuracy +
            (1 - self.accuracy_momentum) * bob_acc
        )
        self.running_eve_accuracy = (
            self.accuracy_momentum * self.running_eve_accuracy +
            (1 - self.accuracy_momentum) * eve_acc
        )
    
    def update_best(
        self,
        accuracy: float,
        alice_state: Dict[str, torch.Tensor],
        bob_state: Dict[str, torch.Tensor],
    ) -> bool:
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_alice_state = {k: v.clone() for k, v in alice_state.items()}
            self.best_bob_state = {k: v.clone() for k, v in bob_state.items()}
            self.plateau_count = 0
            return True
        self.plateau_count += 1
        return False
    
    def reset_counters(self) -> None:
        self.perfect_count = 0
        self.total_count = 0
        self.eve_guess_count = 0
        self.correct_bits = 0
        self.total_bits = 0
    
    def get_recent_bob_error(self, n: int = 100) -> float:
        if not self.bob_errors:
            return 0.0
        recent = self.bob_errors[-n:]
        return sum(recent) / len(recent)
