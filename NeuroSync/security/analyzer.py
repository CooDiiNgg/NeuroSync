"""
Security analysis module for NeuroSync encryption schemes.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Callable
import torch

from NeuroSync.security.checks import (
    check_leakage,
    check_diversity,
    check_repetition,
    check_key_sensitivity,
)
from NeuroSync.security.thresholds import SecurityThresholds, SecurityStatus

@dataclass
class SecurityReport:
    """Dataclass to hold security analysis report."""

    leakage: float
    diversity: float
    repetition: float
    key_sensitivity: float
    overall_score: float
    status: SecurityStatus
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "leakage": self.leakage,
            "diversity": self.diversity,
            "repetition": self.repetition,
            "key_sensitivity": self.key_sensitivity,
            "overall_score": self.overall_score,
            "status": self.status.value,
        }


class SecurityAnalyzer:
    """Class to perform security analysis on NeuroSync encryption schemes."""

    def __init__(self, thresholds: Optional[SecurityThresholds] = None):
        self.thresholds = thresholds or SecurityThresholds()

    def analyze(
        self,
        alice: Callable,
        plaintext_bits: torch.Tensor,
        ciphertext_bits: torch.Tensor,
        key_batch: torch.Tensor,
    ) -> SecurityReport:
        """
        Runs comprehensive security analysis.
        
        Args:
            alice: Alice network (callable)
            plaintext_bits: Original plaintext tensor
            ciphertext_bits: Encrypted ciphertext tensor
            key_batch: Key used for encryption tensor
        
        Returns:
            SecurityReport with all metrics
        """
        leakage_penalty = check_leakage(plaintext_bits, ciphertext_bits)
        diversity_penalty = check_diversity(ciphertext_bits)
        repetition_penalty = check_repetition(ciphertext_bits)
        key_sensitivity_penalty = check_key_sensitivity(alice, plaintext_bits, key_batch)

        overall_score = (
            leakage_penalty +
            diversity_penalty +
            repetition_penalty +
            key_sensitivity_penalty
        ) / 4.0

        status = self.thresholds.evaluate(overall_score)

        return SecurityReport(
            leakage=leakage_penalty,
            diversity=diversity_penalty,
            repetition=repetition_penalty,
            key_sensitivity=key_sensitivity_penalty,
            overall_score=overall_score,
            status=status
        )