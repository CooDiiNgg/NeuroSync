"""
Evaluation routines for NeuroSync training.
"""

import torch
from typing import List, Tuple, Dict

from NeuroSync.encoding.batch import text_to_bits_batch, bits_to_text_batch
from NeuroSync.crypto.operations import xor
from NeuroSync.security.checks import check_total


def evaluate_accuracy(
    alice,
    bob,
    test_words: List[str],
    key_batch: torch.Tensor,
    device: torch.device,
) -> Tuple[int, int]:
    """
    Evaluates accuracy on test words.
    
    Args:
        alice: Alice network
        bob: Bob network
        test_words: List of test words
        key_batch: Key tensor
        device: Compute device
    
    Returns:
        Tuple of (correct_count, total_count)
    """

    alice.eval()
    bob.eval()
    correct = 0
    with torch.no_grad():
        test_bits = text_to_bits_batch(test_words, device=device)
        
        alice_input = xor(test_bits, key_batch[:len(test_words)])
        ciphertext = alice(alice_input)
        ciphertext = torch.sign(ciphertext)
        
        bob_input = xor(ciphertext, key_batch[:len(test_words)])
        decrypted = bob(bob_input)
        
        decrypted_texts = bits_to_text_batch(decrypted)
        
        for original, decoded in zip(test_words, decrypted_texts):
            if original == decoded:
                correct += 1
    
    alice.train()
    bob.train()
    
    return correct, len(test_words)


def evaluate_security(
    alice,
    test_plaintexts: torch.Tensor,
    key_batch: torch.Tensor,
) -> Dict[str, float]:
    """
    Evaluates security metrics.
    
    Args:
        alice: Alice network
        test_plaintexts: Test plaintext tensor
        key_batch: Key tensor
    
    Returns:
        Dictionary of security metric values
    """

    with torch.no_grad():
        ciphertexts = alice(xor(test_plaintexts, key_batch))
        ciphertexts = torch.sign(ciphertexts)
        
        total, details = check_total(
            alice, test_plaintexts, ciphertexts, key_batch
        )
        
        check_names = ["leakage", "diversity", "repetition", "key_sensitivity"]
        return {
            "total": total,
            **{name: val for name, val in zip(check_names, details)},
        }