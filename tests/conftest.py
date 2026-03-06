"""Shared fixtures for NeuroSync tests."""

import os
import pytest
import torch
import numpy as np
import tempfile
import shutil
from NeuroSync.encoding.constants import BIT_LENGTH, KEY_BIT_LENGTH

from NeuroSync.utils.device import set_device


@pytest.fixture(autouse=True, scope="session")
def force_cpu():
    set_device(torch.device("cpu"))
    yield


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def bit_length():
    return BIT_LENGTH


@pytest.fixture
def hidden_size():
    return 32


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def sample_key(device):
    np.random.seed(42)
    key_np = np.random.choice([-1.0, 1.0], size=KEY_BIT_LENGTH)
    return torch.tensor(key_np, dtype=torch.float32, device=device)


@pytest.fixture
def sample_plaintext_bits(device, batch_size, bit_length):
    torch.manual_seed(42)
    return torch.sign(torch.randn(batch_size, bit_length, device=device))


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def small_word_file(tmp_dir):
    filepath = os.path.join(tmp_dir, "words.txt")
    words = ["hello", "world", "test", "neural", "crypto", "alpha", "beta"]
    with open(filepath, "w") as f:
        for w in words:
            f.write(w + "\n")
    return filepath


@pytest.fixture
def alice_bob_networks(bit_length, hidden_size, device):
    from NeuroSync.core.networks import Alice, Bob
    alice = Alice(bit_length, hidden_size, num_residual_blocks=1).to(device)
    bob = Bob(bit_length, hidden_size, num_residual_blocks=1).to(device)
    alice.eval()
    bob.eval()
    return alice, bob


@pytest.fixture
def key_manager(device):
    from NeuroSync.crypto.keys import KeyManager
    km = KeyManager(device=device)
    km.generate()
    return km


@pytest.fixture
def crypto_session(alice_bob_networks, key_manager, device):
    from NeuroSync.protocol.session import CryptoSession
    alice, bob = alice_bob_networks
    return CryptoSession(alice=alice, bob=bob, key_manager=key_manager, device=device)
