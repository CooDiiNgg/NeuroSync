"""E2E: train tiny model, save, load, encrypt, decrypt."""
import os
import torch
import pytest
from NeuroSync.training.config import TrainingConfig
from NeuroSync.training.trainer import NeuroSyncTrainer
from NeuroSync.interface.cipher import NeuroSync


class TestTrainingE2E:
    @pytest.fixture(scope="class")
    def trained(self, tmp_path_factory):
        tmp = str(tmp_path_factory.mktemp("e2e"))
        config = TrainingConfig(
            training_episodes=100_000, batch_size=64, hidden_size=512,
            num_residual_blocks=2, dropout=0.05, learning_rate=0.001,
            eve_learning_rate=0.002, eve_train_iterations=2,
            log_interval=1000, test_interval=5000, scheduler_step_size=10000,
            data_dir=os.path.join(tmp, "data"),
            checkpoint_dir=os.path.join(tmp, "ckpt"),
            word_list_size=4_000,
        )
        trainer = NeuroSyncTrainer(config)
        result = trainer.train()
        return result, tmp, trainer

    def test_completes(self, trained):
        result, _, _ = trained
        assert result.alice is not None

    def test_save_load(self, trained):
        result, tmp, trainer = trained
        d = os.path.join(tmp, "w")
        result.save(d)
        trainer.key_manager.save(os.path.join(d, "key.npy"))
        cipher = NeuroSync.from_pretrained(d, device=torch.device("cpu"))
        assert cipher is not None

    def test_encrypt_decrypt(self, trained):
        result, tmp, trainer = trained
        d = os.path.join(tmp, "w")
        if not os.path.exists(os.path.join(d, "alice.pth")):
            result.save(d)
            trainer.key_manager.save(os.path.join(d, "key.npy"))
        cipher = NeuroSync.from_pretrained(d, device=torch.device("cpu"))
        ct = cipher.encrypt("hello world     ")
        assert isinstance(ct, torch.Tensor)
        pt = cipher.decrypt(ct)
        assert isinstance(pt, str) and len(pt) == 16
        assert pt.strip() == "hello world"
