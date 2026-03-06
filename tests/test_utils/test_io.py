"""Tests for NeuroSync.utils.io."""
import os
import torch
import pytest
from NeuroSync.utils.io import save_checkpoint, load_checkpoint, ensure_dir


class TestIO:
    def test_save_load(self, tmp_dir):
        path = os.path.join(tmp_dir, "ckpt.pth")
        save_checkpoint(path, {"w": torch.tensor([1.0, 2.0]), "e": 5})
        loaded = load_checkpoint(path, torch.device("cpu"))
        assert loaded["e"] == 5

    def test_load_missing(self, tmp_dir):
        with pytest.raises(FileNotFoundError):
            load_checkpoint(os.path.join(tmp_dir, "x.pth"))

    def test_ensure_dir(self, tmp_dir):
        p = os.path.join(tmp_dir, "new")
        ensure_dir(p)
        assert os.path.isdir(p)
