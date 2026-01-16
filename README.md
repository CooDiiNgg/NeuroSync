# NeuroSync

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.8+](https://img.shields.io/badge/pytorch-2.8+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://github.com/CooDiiNgg/NeuroSync)

A neural cryptography library implementing adversarial training for secure communication. NeuroSync features three neural networks (Alice, Bob, and Eve) that learn to encrypt and decrypt messages while an adversary attempts to break the encryption, resulting in continuously improving security.

## Overview

NeuroSync is inspired by [Google Brain's research on neural cryptography](https://arxiv.org/abs/1610.06918), where neural networks learn to protect communications. The system consists of:

- **Alice** (Encoder): Transforms plaintext into ciphertext using a shared key
- **Bob** (Decoder): Recovers plaintext from ciphertext using the same key
- **Eve** (Adversary): Attempts to decrypt without the key, driving security improvements

Through adversarial training, Alice and Bob learn encryption schemes that Eve cannot break, while the system provides dynamic key rotation, error correction, and a complete protocol stack for real-world applications.

## Features

- **Adversarial Training**: Three-network system that continuously improves encryption security
- **Dynamic Key Rotation**: Automatic key updates with encrypted key exchange
- **Error Correction**: Parity-based error detection and correction for reliable transmission
- **Complete Protocol Stack**: Packet-based communication with checksums, sequencing, and reassembly
- **GPU Acceleration**: Full CUDA support for fast training and inference
- **Simple API**: High-level interface for easy integration

## Installation

### Prerequisites

- Python 3.9 or higher
- PyTorch 2.8.0 or higher
- CUDA-capable GPU (recommended for training)

### From PyPI

```bash
pip install NeuroSync
```

### From Source

```bash
git clone https://github.com/CooDiiNgg/NeuroSync.git
cd NeuroSync
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/CooDiiNgg/NeuroSync.git
cd NeuroSync
pip install -e ".[dev]"
```

### GPU Support

NeuroSync automatically detects and uses CUDA when available. For CPU-only installation:

```bash
pip install NeuroSync
# PyTorch will use CPU if no CUDA is available
```

For specific CUDA versions, install PyTorch first following the [official instructions](https://pytorch.org/get-started/locally/), then install NeuroSync.

## Quick Start

### Basic Encryption/Decryption

```python
from NeuroSync import NeuroSync

# Load from pretrained weights
cipher = NeuroSync.from_pretrained("./weights/")

# Encrypt a message
encrypted = cipher.encrypt("Hello, World!")

# Decrypt it back
decrypted = cipher.decrypt(encrypted)
print(decrypted)  # "Hello, World!"
```

### Training a New System

```python
from NeuroSync import NeuroSync, TrainingConfig

# Train with default configuration
cipher = NeuroSync.train_new()

# Or with custom configuration
config = TrainingConfig(
    training_episodes=10_000_000,
    batch_size=128,
    hidden_size=512,
)
cipher = NeuroSync.train_new(config)

# Save the trained models
cipher.save("./my_weights/")
```

### Using the Full Protocol

```python
from NeuroSync import NeuroSync

cipher = NeuroSync.from_pretrained("./weights/")

# Create sender and receiver
sender = cipher.create_sender()
receiver = cipher.create_receiver()

# Send a message
packets = sender.send("Hello from NeuroSync!")

# Receive and reassemble
for packet in packets:
    message = receiver.receive(packet)
    if message:
        print(f"Received: {message}")
```

## Usage Examples

### Custom Training Configuration

```python
from NeuroSync import NeuroSyncTrainer, TrainingConfig

config = TrainingConfig(
    # Message and key settings
    message_length=16,          # Characters per message chunk
    key_size=16,                # Key size in characters
    
    # Training parameters
    training_episodes=20_000_000,
    batch_size=64,
    learning_rate=0.0005,
    
    # Network architecture
    hidden_size=512,
    num_residual_blocks=3,
    dropout=0.05,
    
    # Adversarial training
    adversarial_max=0.15,       # Maximum adversarial weight
    eve_learning_rate=0.001,
    eve_train_iterations=3,
    
    # Security settings
    security_max=0.1,
    maintenance_threshold=99.0,
    
    # Paths
    data_dir="./data",
    checkpoint_dir="./checkpoints",
    word_list_file="words.txt",
)

trainer = NeuroSyncTrainer(config)
result = trainer.train()
result.save("./trained_models/")

print(f"Final accuracy: {result.final_accuracy:.2f}%")
print(f"Best accuracy: {result.best_accuracy:.2f}%")
```

### Protocol with Key Rotation

```python
from NeuroSync import NeuroSync

cipher = NeuroSync.from_pretrained("./weights/")
sender = cipher.create_sender()
receiver = cipher.create_receiver()

# Send multiple messages
messages = [
    "First message",
    "Second message", 
    "Third message",
]

for msg in messages:
    # Check if key rotation is needed
    key_rotation_packet = sender.check_key_rotation()
    if key_rotation_packet:
        # Transmit key rotation packet
        receiver.receive(key_rotation_packet)
        # Get and handle acknowledgment
        acks = receiver.get_pending_acks()
        for ack in acks:
            sender.handle_ack(ack)
    
    # Send the actual message
    packets = sender.send(msg)
    for packet in packets:
        result = receiver.receive(packet)
        if result:
            print(f"Received: {result}")
```

### Working with Raw Networks

```python
import torch
from NeuroSync.core.networks import Alice, Bob, Eve
from NeuroSync.encoding.codec import text_to_bits, bits_to_text
from NeuroSync.crypto.operations import xor
from NeuroSync.crypto.keys import KeyManager

# Initialize networks
bit_length = 96  # 16 characters * 6 bits
alice = Alice(bit_length, hidden_size=512)
bob = Bob(bit_length, hidden_size=512)

# Load trained weights
alice.load("./weights/alice.pth")
bob.load("./weights/bob.pth")

# Set to evaluation mode
alice.eval()
bob.eval()

# Create key
key_manager = KeyManager()
key = key_manager.generate()

# Encrypt
plaintext = "Hello World!    "  # Pad to 16 chars
plain_bits = torch.tensor(text_to_bits(plaintext), dtype=torch.float32)
alice_input = xor(plain_bits, key)

with torch.no_grad():
    ciphertext = alice(alice_input, single=True)
    ciphertext = torch.sign(ciphertext)
    
    # Decrypt
    bob_input = xor(ciphertext, key)
    decrypted_bits = bob(bob_input, single=True)
    decrypted = bits_to_text(decrypted_bits)

print(f"Original:  '{plaintext}'")
print(f"Decrypted: '{decrypted}'")
```

### Batch Processing

```python
import torch
from NeuroSync import NeuroSync

cipher = NeuroSync.from_pretrained("./weights/")

# Process multiple messages in a batch
messages = [
    "Message one     ",
    "Message two     ",
    "Message three   ",
]

# Create sender for batch processing
sender = cipher.create_sender()

all_packets = []
for msg in messages:
    packets = sender.send(msg)
    all_packets.extend(packets)

print(f"Total packets: {len(all_packets)}")
```

## API Reference

### NeuroSync (Main Interface)

```python
class NeuroSync:
    @classmethod
    def from_pretrained(cls, dirpath: str, device=None) -> "NeuroSync"
        """Load from pretrained weights directory."""
    
    @classmethod  
    def train_new(cls, config=None, device=None) -> "NeuroSync"
        """Train a new NeuroSync system."""
    
    def encrypt(self, plaintext: str) -> torch.Tensor
        """Encrypt a plaintext string."""
    
    def decrypt(self, ciphertext: torch.Tensor) -> str
        """Decrypt an encrypted tensor."""
    
    def create_sender(self) -> Sender
        """Create a Sender for full protocol usage."""
    
    def create_receiver(self) -> Receiver
        """Create a Receiver for full protocol usage."""
    
    def save(self, dirpath: str) -> None
        """Save weights to directory."""
```

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    # Message settings
    message_length: int = 16        # Characters per chunk
    key_size: int = 16              # Key size in characters
    
    # Training
    training_episodes: int = 20_000_000
    batch_size: int = 64
    learning_rate: float = 0.0005
    weight_decay: float = 1e-4
    
    # Architecture
    hidden_size: int = 512
    num_residual_blocks: int = 3
    dropout: float = 0.05
    
    # Adversarial
    adversarial_max: float = 0.15
    eve_learning_rate: float = 0.001
    
    # Paths
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
```

### Sender

```python
class Sender:
    def __init__(self, session, enable_error_correction=True, key_rotation_interval=1000)
    
    def send(self, message: str) -> List[Packet]
        """Prepare message for transmission."""
    
    def check_key_rotation(self) -> Optional[Packet]
        """Check if key rotation is needed."""
    
    def handle_ack(self, packet: Packet) -> None
        """Handle acknowledgment packets."""
```

### Receiver

```python
class Receiver:
    def __init__(self, session, enable_error_correction=True)
    
    def receive(self, packet: Packet) -> Optional[str]
        """Receive and process a packet."""
    
    def get_pending_acks(self) -> List[Packet]
        """Get pending acknowledgment packets."""
    
    def has_pending_data(self) -> bool
        """Check for pending data."""
    
    def reset(self) -> None
        """Reset receiver state."""
```

## Architecture

### Network Architecture

Each network (Alice, Bob, Eve) uses the same base architecture:

```
Input (96 bits) 
    → Linear(96, 512) + LayerNorm + Tanh
    → ResidualBlock × 3
    → Linear(512, 512) + LayerNorm + Tanh  
    → Linear(512, 96) + Tanh/Temperature
Output (96 bits)
```

Each ResidualBlock contains:
- Linear layer with LayerNorm
- Tanh activation
- Dropout (5%)
- Residual connection

### Protocol Stack

```
┌─────────────────────────────────────┐
│           Application               │
│     (NeuroSync / Sender/Receiver)   │
├─────────────────────────────────────┤
│           Session Layer             │
│     (CryptoSession)                 │
├─────────────────────────────────────┤
│           Protocol Layer            │
│  (Packets, Flags, Key/Weight Rot.)  │
├─────────────────────────────────────┤
│           Encoding Layer            │
│  (6-bit encoding, batch processing) │
├─────────────────────────────────────┤
│           Crypto Layer              │
│  (Alice/Bob networks, XOR mixing)   │
└─────────────────────────────────────┘
```

### Packet Structure

```
┌────────────────────────────────────────────┐
│ Header (12 bytes)                          │
│  ├─ Version (1 byte)                       │
│  ├─ Flags (1 byte)                         │
│  ├─ Sequence ID (4 bytes)                  │
│  ├─ Payload Length (2 bytes)               │
│  └─ Checksum (4 bytes)                     │
├────────────────────────────────────────────┤
│ Payload (variable)                         │
│  └─ Encrypted message data                 │
├────────────────────────────────────────────┤
│ Parity (optional)                          │
│  └─ Error correction bits                  │
└────────────────────────────────────────────┘
```

## Training Guide

### Preparing Training Data

Create a `words.txt` file with training words (one per line):

```text
hello
world
neural
crypto
secure
...
```

### Training Process

The training process involves three phases:

1. **Warm-up Phase** (first ~2000 batches): Duplicate messages to establish basic encryption
2. **Main Training**: Full adversarial training with Eve providing security pressure  
3. **Maintenance Mode**: When accuracy reaches 99%+, reduce training to maintain stability

### Monitoring Training

Training logs include:
- Bob's decryption accuracy (target: 99%+)
- Eve's decryption accuracy (should stay low)
- Security check scores (leakage, diversity, repetition, key sensitivity)
- Network temperatures

### Tips for Training

- Start with default configuration for baseline results
- Increase `hidden_size` for more complex encryption patterns
- Adjust `adversarial_max` to balance security vs. stability
- Use a diverse word list for better generalization
- Training typically requires 10-20 million episodes for 99%+ accuracy

## Project Structure

```
NeuroSync/
├── NeuroSync/
│   ├── __init__.py          # Package exports
│   ├── version.py           # Version info
│   ├── core/                # Neural network components
│   │   ├── networks.py      # Alice, Bob, Eve networks
│   │   ├── layers.py        # ResidualBlock, etc.
│   │   ├── activations.py   # StraightThroughSign
│   │   └── losses.py        # Custom loss functions
│   ├── crypto/              # Cryptographic operations
│   │   ├── keys.py          # KeyManager
│   │   ├── weights.py       # WeightManager
│   │   └── operations.py    # XOR and mixing ops
│   ├── encoding/            # Message encoding
│   │   ├── codec.py         # text_to_bits, bits_to_text
│   │   ├── batch.py         # Batch processing
│   │   └── constants.py     # MESSAGE_LENGTH, BIT_LENGTH
│   ├── interface/           # High-level interfaces
│   │   ├── cipher.py        # NeuroSync main class
│   │   ├── sender.py        # Sender interface
│   │   ├── receiver.py      # Receiver interface
│   │   └── visualizer.py    # Training visualization
│   ├── protocol/            # Communication protocol
│   │   ├── packet.py        # Packet structure
│   │   ├── session.py       # CryptoSession
│   │   ├── flags.py         # PacketFlags
│   │   ├── header.py        # PacketHeader
│   │   ├── key_rotation.py  # Key rotation manager
│   │   ├── weight_rotation.py # Weight rotation manager in the future
│   │   ├── error_correction.py # Parity-based ECC
│   │   └── assembler.py     # Packet reassembly
│   ├── security/            # Security analysis
│   │   ├── analyzer.py      # SecurityAnalyzer
│   │   ├── checks.py        # Security check functions
│   │   └── thresholds.py    # SecurityThresholds
│   ├── training/            # Training pipeline
│   │   ├── trainer.py       # NeuroSyncTrainer
│   │   ├── config.py        # TrainingConfig
│   │   ├── state.py         # TrainingState
│   │   ├── evaluation.py    # Accuracy/security evaluation
│   │   └── schedulers.py    # Learning rate schedulers
│   └── utils/               # Utilities
│       ├── device.py        # CUDA/CPU detection
│       ├── io.py            # File I/O helpers
│       ├── logging.py       # Logging setup
│       └── timing.py        # Performance timing
├── tests/                   # Test suite
├── examples/                # Usage examples
├── docs/                    # Documentation
├── pyproject.toml           # Package configuration
└── README.md                # This file
```

## Security Considerations

NeuroSync is a research project in a beta state, demonstrating neural cryptography concepts. While it implements several security features, it still isnt production-ready. Please consider the following:

- **Not for Production**: This is still a beta version so it is an experimental software not audited for production use
- **Key Management**: Secure key storage and transmission are the user's responsibility
- **Side Channels**: Not tested against timing or power analysis attacks
- **Quantum Security**: While it is designed to be quantum-resistant, it has not been formally analyzed against quantum attacks and is not yet ready for such scenarios

Use only for educational and research purposes, or at your own risk. (Before a new stable release is made available.)
For production cryptography needs, use established libraries like [cryptography](https://cryptography.io/) or [NaCl](https://nacl.cr.yp.to/).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

```bash
# Setup development environment
git clone https://github.com/CooDiiNgg/NeuroSync.git
cd NeuroSync
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black NeuroSync/
isort NeuroSync/

# Type checking
mypy NeuroSync/
```

If you only have suggestions or bug reports, please open an issue on GitHub, or feel free to contact me directly. (Contact info is at the bottom of the page.)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use NeuroSync in your research, please cite - it brings me great joy!

```bibtex
@software{neurosync2026,
  author = {Valkanov, Nikolay},
  title = {NeuroSync: Neural Cryptography Library},
  year = {2026},
  url = {https://github.com/CooDiiNgg/NeuroSync}
}
```

## Acknowledgments

- Inspired by [Learning to Protect Communications with Adversarial Neural Cryptography](https://arxiv.org/abs/1610.06918) by Abadi & Andersen (Google Brain)
- Built with [PyTorch](https://pytorch.org/)

## Links

- [GitHub Repository](https://github.com/CooDiiNgg/NeuroSync)
- [Documentation](https://NeuroSync.readthedocs.io)
- [Issue Tracker](https://github.com/CooDiiNgg/NeuroSync/issues)

## Contact
For questions or support, please open an issue on GitHub or contact me at:
- Email: niki@valkanovi.com
- LinkedIn: [Nikolay Valkanov](https://www.linkedin.com/in/nikolay-valkanov-thedumb1/)