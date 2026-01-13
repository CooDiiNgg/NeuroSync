import torch
import numpy as np
from typing import Tuple, List, Callable

from NeuroSync.crypto.operations import xor


def check_leakage(
    plaintext_bits: torch.Tensor,
    ciphertext_bits: torch.Tensor,
) -> float:
    with torch.no_grad():
        pt_sign = torch.sign(plaintext_bits)
        ct_sign = torch.sign(ciphertext_bits)
        matches = (pt_sign == ct_sign).float().mean().item()
        penalty = max(0.0, (matches - 0.55) * 2.0)
    return penalty

def check_diversity(ciphertext_bits: torch.Tensor) -> float:
    with torch.no_grad():
        variance = torch.var(ciphertext_bits.float(), dim=0).mean().item()
        penalty = max(0.0, (0.20 - variance) * 2.0)
    return penalty

def check_repetition(ciphertext_bits: torch.Tensor) -> float:
    with torch.no_grad():
        num_samples = min(ciphertext_bits.size(0), 10)
        repetitions = []
        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                rep = (ciphertext_bits[i] == ciphertext_bits[j]).float().mean().item()
                repetitions.append(rep)
        avg_repetition = np.mean(repetitions)
        penalty = max(0.0, (avg_repetition - 0.6) * 2.0)
    return penalty

def check_key_sensitivity(
    alice: Callable,
    plaintext_bits: torch.Tensor,
    key_batch: torch.Tensor,
) -> float:
    with torch.no_grad():
        key_flipped = key_batch.clone()
        rand_idx = np.random.randint(0, key_flipped.shape[1])
        key_flipped[:, rand_idx] *= -1.0
        
        ai_original = xor(plaintext_bits, key_batch)
        ai_flipped = xor(plaintext_bits, key_flipped)
        
        ciph_original = alice(ai_original)
        ciph_flipped = alice(ai_flipped)
        
        diff = (torch.sign(ciph_original) != torch.sign(ciph_flipped)).float().mean().item()
        penalty = max(0.0, (0.2 - diff) * 2.0)
    return penalty

def check_total(
    alice: Callable,
    plaintext_bits: torch.Tensor,
    ciphertext_bits: torch.Tensor,
    key_batch: torch.Tensor,
) -> Tuple[float, List[float]]:
    penalties = []
    
    leakage_penalty = check_leakage(plaintext_bits, ciphertext_bits)
    penalties.append(leakage_penalty)
    
    diversity_penalty = check_diversity(ciphertext_bits)
    penalties.append(diversity_penalty)
    
    repetition_penalty = check_repetition(ciphertext_bits)
    penalties.append(repetition_penalty)
    
    key_sensitivity_penalty = check_key_sensitivity(alice, plaintext_bits, key_batch)
    penalties.append(key_sensitivity_penalty)
    
    total_penalty = sum(penalties) / len(penalties)
    
    return total_penalty, penalties