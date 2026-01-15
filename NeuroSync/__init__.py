"""
NeuroSync - Neural Cryptography Library

A neural cryptography library implementing adversarial training for secure communication,
with dynamic key rotation, error correction, and a complete protocol stack.
"""

from NeuroSync.version import __version__

from NeuroSync.interface.cipher import NeuroSync
from NeuroSync.interface.sender import Sender
from NeuroSync.interface.receiver import Receiver
from NeuroSync.interface.visualizer import Visualizer

from NeuroSync.training.trainer import NeuroSyncTrainer
from NeuroSync.training.config import TrainingConfig

from NeuroSync.protocol.packet import Packet
from NeuroSync.protocol.session import CryptoSession

from NeuroSync import core
from NeuroSync import encoding
from NeuroSync import crypto
from NeuroSync import security
from NeuroSync import protocol

__all__ = [
    "__version__",
    "NeuroSync",
    "Sender",
    "Receiver",
    "Visualizer",
    "NeuroSyncTrainer",
    "TrainingConfig",
    "Packet",
    "CryptoSession",
    "core",
    "encoding",
    "crypto",
    "security",
    "protocol",
]
