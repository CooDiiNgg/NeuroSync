"""
Training configuration for NeuroSync.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    """
    Configuration parameters for NeuroSync training.

    Captures all hyperparameters and settings.
    """

    message_length: int = 16
    key_size: int = 16
    
    training_episodes: int = 20_000_000
    batch_size: int = 64
    
    hidden_size: int = 512
    num_residual_blocks: int = 3
    dropout: float = 0.05
    
    learning_rate: float = 0.0005
    weight_decay: float = 1e-4
    betas: tuple = (0.9, 0.999)
    
    eve_learning_rate: float = 0.001
    eve_train_skip: int = 1
    eve_train_iterations: int = 3
    
    adversarial_max: float = 0.15
    adversarial_weight: float = 0.0
    
    confidence_max: float = 0.3
    confidence_weight: float = 0.0
    confidence_margin: float = 0.7
    
    security_max: float = 0.1
    security_weight: float = 0.0
    
    maintenance_threshold: float = 99.0
    maintenance_threshold_exit: float = 95.0
    consecutive_accuracy_required: int = 3
    
    scheduler_step_size: int = 50000
    scheduler_gamma: float = 0.5
    
    max_grad_norm: float = 1.0
    max_grad_norm_low_loss: float = 0.5
    low_loss_threshold: float = 0.1
    
    log_interval: int = 2500
    test_interval: int = 10000
    
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    word_list_file: str = "words.txt"
    word_list_size: int = 1_000_000
    
    @property
    def bit_length(self) -> int:
        """Gets the message length in bits."""
        return self.message_length * 6
    
    @property
    def key_bit_length(self) -> int:
        """Gets the key size in bits."""
        return self.key_size * 6
    
    def to_dict(self) -> dict:
        """Gets a dictionary representation of the config."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        """Creates a TrainingConfig from a dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
