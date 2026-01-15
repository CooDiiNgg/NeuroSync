"""
The complete training pipeline for NeuroSync.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Tuple, List
from dataclasses import dataclass

from NeuroSync.training.config import TrainingConfig
from NeuroSync.training.state import TrainingState
from NeuroSync.training.schedulers import (
    AdversarialScheduler,
    SecurityScheduler,
    ConfidenceScheduler,
    MaintenanceModeController,
    LossScheduler,
)
from NeuroSync.training.evaluation import evaluate_accuracy, evaluate_security

from NeuroSync.core.networks import Alice, Bob, Eve
from NeuroSync.core.activations import straight_through_sign
from NeuroSync.core.losses import confidence_loss

from NeuroSync.encoding.batch import text_to_bits_batch, bits_to_text_batch
from NeuroSync.encoding.codec import text_to_bits, bits_to_text

from NeuroSync.crypto.operations import xor
from NeuroSync.crypto.keys import KeyManager

from NeuroSync.security.analyzer import SecurityAnalyzer
from NeuroSync.security.thresholds import SecurityThresholds

from NeuroSync.data.generators import MessageGenerator

from NeuroSync.utils.device import get_device
from NeuroSync.utils.logging import get_logger
from NeuroSync.utils.io import save_checkpoint, load_checkpoint, ensure_dir


@dataclass
class TrainingResult:
    """Results of a completed training session."""
    alice: Alice
    bob: Bob
    eve: Eve
    final_accuracy: float
    best_accuracy: float
    config: TrainingConfig
    
    def save(self, dirpath: str) -> None:
        """Saves the trained models to the specified directory."""
        ensure_dir(dirpath)
        self.alice.save(os.path.join(dirpath, "alice.pth"))
        self.bob.save(os.path.join(dirpath, "bob.pth"))
        self.eve.save(os.path.join(dirpath, "eve.pth"))


class NeuroSyncTrainer:
    """
    Main trainer for NeuroSync neural cryptography system.
    
    This class encapsulates the complete training loop from the POC,
    using library helper modules for cleaner, more maintainable code.
    
    Usage:
        config = TrainingConfig()
        trainer = NeuroSyncTrainer(config)
        result = trainer.train()
        result.save("./trained_models/")
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.device = get_device()
        self.logger = get_logger()
        
        self.alice: Optional[Alice] = None
        self.bob: Optional[Bob] = None
        self.eve: Optional[Eve] = None
        
        self.key_manager = KeyManager(device=self.device)
        self.eve_key_manager = KeyManager(device=self.device)
        
        self.message_generator = MessageGenerator(message_length=self.config.message_length)
        
        self.state = TrainingState()
        
        self.security_analyzer = SecurityAnalyzer(SecurityThresholds())
        
        self.adversarial_scheduler = AdversarialScheduler(
            max_weight=self.config.adversarial_max,
        )
        self.security_scheduler = SecurityScheduler(
            max_weight=self.config.security_max,
        )
        self.confidence_scheduler = ConfidenceScheduler(
            max_weight=self.config.confidence_max,
        )
        self.maintenance_controller = MaintenanceModeController(
            enter_threshold=self.config.maintenance_threshold,
            exit_threshold=self.config.maintenance_threshold_exit,
            consecutive_required=self.config.consecutive_accuracy_required,
        )
        self.loss_scheduler = LossScheduler()
        
        self.alice_bob_optimizer: Optional[optim.Optimizer] = None
        self.eve_optimizer: Optional[optim.Optimizer] = None
        self.alice_bob_scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.eve_scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        
        self.mse_criterion = nn.MSELoss()
        self.smooth_l1_criterion = nn.SmoothL1Loss()
    
    def _init_networks(self) -> None:
        """Initializes Alice, Bob, and Eve networks."""

        bit_length = self.config.bit_length
        hidden_size = self.config.hidden_size
        
        self.alice = Alice(bit_length, hidden_size).to(self.device)
        self.bob = Bob(bit_length, hidden_size).to(self.device)
        self.eve = Eve(bit_length, hidden_size).to(self.device)
        
        self.logger.info("Networks initialized:")
        self.logger.info(f"  Bit length: {bit_length}")
        self.logger.info(f"  Hidden size: {hidden_size}")
    
    def _init_optimizers(self) -> None:
        """Initializes optimizers and learning rate schedulers."""

        cfg = self.config
        
        alice_bob_params = list(self.alice.parameters()) + list(self.bob.parameters())
        self.alice_bob_optimizer = optim.AdamW(
            alice_bob_params,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
        )
        
        self.eve_optimizer = optim.AdamW(
            self.eve.parameters(),
            lr=cfg.eve_learning_rate,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
        )
        
        self.alice_bob_scheduler = optim.lr_scheduler.StepLR(
            self.alice_bob_optimizer,
            step_size=cfg.scheduler_step_size,
            gamma=cfg.scheduler_gamma,
        )
        
        self.eve_scheduler = optim.lr_scheduler.StepLR(
            self.eve_optimizer,
            step_size=cfg.scheduler_step_size,
            gamma=cfg.scheduler_gamma,
        )
    
    def _load_checkpoints(self, checkpoint_dir: str) -> bool:
        """Loads saved models and training state from the specified directory."""

        alice_path = os.path.join(checkpoint_dir, 'alice_test.pth')
        bob_path = os.path.join(checkpoint_dir, 'bob_test.pth')
        eve_path = os.path.join(checkpoint_dir, 'eve_test.pth')
        state_path = os.path.join(checkpoint_dir, 'training_state_test.pth')
        key_path = os.path.join(checkpoint_dir, 'key.npy')
        eve_key_path = os.path.join(checkpoint_dir, 'eve_key.npy')
        
        loaded = False
        
        if os.path.exists(key_path):
            self.key_manager.load(key_path)
            print(f"Loaded {self.key_manager.key_bit_length}-bit key")
            loaded = True
        
        if os.path.exists(eve_key_path):
            self.eve_key_manager.load(eve_key_path)
            print(f"Loaded {self.eve_key_manager.key_bit_length}-bit Eve key")
        
        if os.path.exists(alice_path) and os.path.exists(bob_path):
            print("Loading saved networks...")
            self.alice.load(alice_path, self.device)
            self.bob.load(bob_path, self.device)
            if os.path.exists(eve_path):
                self.eve.load(eve_path, self.device)
            print("Loaded!")
            loaded = True
        
        if os.path.exists(state_path):
            print("Loading training state...")
            checkpoint = load_checkpoint(state_path, self.device)
            self.alice_bob_optimizer.load_state_dict(checkpoint['alice_and_bob_optimizer'])
            if 'eve_optimizer' in checkpoint:
                self.eve_optimizer.load_state_dict(checkpoint['eve_optimizer'])
            if 'best_accuracy' in checkpoint:
                self.state.best_accuracy = checkpoint['best_accuracy']
            print("Loaded training state!")
        
        return loaded
    
    def _save_checkpoints(self, checkpoint_dir: str) -> None:
        """Saves models and training state to the specified directory."""

        ensure_dir(checkpoint_dir)
        
        self.alice.save(os.path.join(checkpoint_dir, 'alice_test.pth'))
        self.bob.save(os.path.join(checkpoint_dir, 'bob_test.pth'))
        self.eve.save(os.path.join(checkpoint_dir, 'eve_test.pth'))
        
        self.key_manager.save(os.path.join(checkpoint_dir, 'key.npy'))
        self.eve_key_manager.save(os.path.join(checkpoint_dir, 'eve_key.npy'))
        
        save_checkpoint(
            os.path.join(checkpoint_dir, 'training_state_test.pth'),
            {
                'alice_and_bob_optimizer': self.alice_bob_optimizer.state_dict(),
                'eve_optimizer': self.eve_optimizer.state_dict(),
                'best_accuracy': self.state.best_accuracy,
            }
        )
    
    def _get_loss_criterion(self, use_smooth_l1: bool):
        """Returns the appropriate loss criterion based on the flag."""

        return self.smooth_l1_criterion if use_smooth_l1 else self.mse_criterion
    
    def _train_alice_bob_batch(
        self,
        plain_bits_batch: torch.Tensor,
        key_batch: torch.Tensor,
        eve_key_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Trains Alice and Bob for one batch and returns relevant outputs and loss."""

        cfg = self.config
        state = self.state
        
        skip = self.maintenance_controller.in_maintenance
        
        if not skip:
            self.alice.train()
            self.bob.train()
        
        self.eve.eval()
        
        alice_input = xor(plain_bits_batch, key_batch)
        ciphertext_batch_original = self.alice(alice_input)
        ciphertext_batch = straight_through_sign(ciphertext_batch_original)
        
        bob_input = xor(ciphertext_batch, key_batch)
        decrypted_bits_batch = self.bob(bob_input)
        
        with torch.no_grad():
            alice_input_eve = xor(plain_bits_batch, eve_key_batch)
            ciphertext_batch_eve = straight_through_sign(self.alice(alice_input_eve))
        
        eve_output = self.eve(ciphertext_batch_eve)
        
        eve_criterion = self._get_loss_criterion(state.eve_use_smooth_l1)
        bob_criterion = self._get_loss_criterion(state.use_smooth_l1)
        
        eve_loss = eve_criterion(eve_output, plain_bits_batch)
        loss = bob_criterion(decrypted_bits_batch, plain_bits_batch)
        state.bob_errors.append(loss.item())
        
        if not skip:
            report = self.security_analyzer.analyze(
                self.alice, plain_bits_batch, ciphertext_batch, key_batch
            )
            state.running_security = 0.9 * state.running_security + 0.05 * report.overall_score
            sec_loss = state.security_weight * report.overall_score
            
            effective_adv_weight = state.adversarial_weight if state.running_bob_accuracy >= 98.0 else 0.0
            
            total_loss = (
                loss 
                + state.confidence_weight * confidence_loss(ciphertext_batch_original, cfg.confidence_margin)
                + sec_loss 
                - effective_adv_weight * eve_loss
            )
            
            self.alice_bob_optimizer.zero_grad()
            total_loss.backward()
            
            max_norm = cfg.max_grad_norm if loss.item() < cfg.low_loss_threshold else cfg.max_grad_norm_low_loss
            torch.nn.utils.clip_grad_norm_(self.bob.parameters(), max_norm)
            torch.nn.utils.clip_grad_norm_(self.alice.parameters(), max_norm)
            
            self.alice_bob_optimizer.step()
            self.alice_bob_scheduler.step()
        
        return ciphertext_batch, decrypted_bits_batch, eve_output, loss.item()
    
    def _train_eve_batch(
        self,
        plain_bits_batch: torch.Tensor,
        eve_key_batch: torch.Tensor,
    ) -> None:
        """Trains Eve for one batch."""

        state = self.state
        
        self.eve.train()
        self.alice.eval()
        self.bob.eval()
        
        eve_criterion = self._get_loss_criterion(state.eve_use_smooth_l1)
        
        for _ in range(self.config.eve_train_iterations):
            with torch.no_grad():
                alice_input_eve = xor(plain_bits_batch, eve_key_batch)
                ciphertext_batch_eve = straight_through_sign(self.alice(alice_input_eve))
            
            eve_output = self.eve(ciphertext_batch_eve)
            eve_loss = eve_criterion(eve_output, plain_bits_batch)
            state.eve_errors.append(eve_loss.item())
            
            self.eve_optimizer.zero_grad()
            eve_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.eve.parameters(), 1.0)
            self.eve_optimizer.step()
        
        self.eve_scheduler.step()
    
    def _update_counters(
        self,
        plaintexts: List[str],
        decrypted_bits_batch: torch.Tensor,
        eve_output: torch.Tensor,
        plain_bits_batch: torch.Tensor,
    ) -> None:
        """Updates training state counters based on the latest batch results."""

        state = self.state
        
        with torch.no_grad():
            decrypted_texts = bits_to_text_batch(decrypted_bits_batch.detach())
            eve_texts = bits_to_text_batch(eve_output.detach())
            
            for pt, dt in zip(plaintexts, decrypted_texts):
                state.total_count += 1
                if pt == dt:
                    state.perfect_count += 1
            
            for pt, et in zip(plaintexts, eve_texts):
                if pt == et:
                    state.eve_guess_count += 1
            
            bit_matches = (torch.sign(decrypted_bits_batch) == torch.sign(plain_bits_batch)).float()
            state.correct_bits += int(bit_matches.sum().item())
            state.total_bits += bit_matches.numel()
    
    def _update_schedulers(self, episode: int, recent_accuracy: float) -> None:
        """Updates all training schedulers based on the latest accuracy metrics."""

        state = self.state
        
        eve_accuracy = (100 * state.eve_guess_count / state.total_count) if state.total_count > 0 else 0.0
        
        state.update_accuracy(recent_accuracy, eve_accuracy)
        
        self.eve_key_manager.generate()
        
        entered, exited, reason = self.maintenance_controller.step(
            state.running_bob_accuracy,
            state.running_eve_accuracy,
            state.running_security,
        )
        
        if entered:
            print(f"\nEntering maintenance mode at episode {episode}: {reason}")
            state.best_alice_state = {k: v.clone() for k, v in self.alice.state_dict().items()}
            state.best_bob_state = {k: v.clone() for k, v in self.bob.state_dict().items()}
        
        if exited:
            print(f"\nExiting maintenance mode at episode {episode}: {reason}")
            if state.best_alice_state and state.best_bob_state and state.running_bob_accuracy < 90.0:
                self.alice.load_state_dict(state.best_alice_state)
                self.bob.load_state_dict(state.best_bob_state)
                state.running_bob_accuracy = 95.0
        
        state.maintenance_mode = self.maintenance_controller.in_maintenance
        
        state.adversarial_weight = self.adversarial_scheduler.step(
            state.adversarial_weight,
            state.running_bob_accuracy,
            state.running_eve_accuracy,
            state.maintenance_mode,
        )
        
        state.security_weight = self.security_scheduler.step(
            state.security_weight,
            state.running_bob_accuracy,
            state.running_security,
            state.maintenance_mode,
        )
        
        state.eve_use_smooth_l1 = self.loss_scheduler.should_use_smooth_l1_eve(state.running_eve_accuracy)
        
        improved = state.update_best(recent_accuracy, self.alice.state_dict(), self.bob.state_dict())
        if not improved and state.plateau_count >= 10 and recent_accuracy < 90.0:
            state.use_smooth_l1 = True
        else:
            state.use_smooth_l1 = False
        
        bit_accuracy = (100.0 * state.correct_bits) / state.total_bits if state.total_bits > 0 else 0.0
        state.confidence_weight = self.confidence_scheduler.step(bit_accuracy)
        
        state.correct_bits = 0
        state.total_bits = 0
    
    def _log_progress(
        self,
        episode: int,
        plaintexts: List[str],
        decrypted_bits_batch: torch.Tensor,
        ciphertext_batch: torch.Tensor,
        eve_output: torch.Tensor,
        key: torch.Tensor,
    ) -> None:
        """Logs training progress to the console."""

        state = self.state
        cfg = self.config
        
        avg_error = state.get_recent_bob_error(100)
        recent_accuracy = (100 * state.perfect_count / state.total_count) if state.total_count > 0 else 0.0
        eve_accuracy = (100 * state.eve_guess_count / state.total_count) if state.total_count > 0 else 0.0
        
        decrypted_texts = bits_to_text_batch(decrypted_bits_batch.detach())
        eve_texts = bits_to_text_batch(eve_output.detach())
        
        print(f"\nEpisode {episode}/{cfg.training_episodes}")
        print(f"  Avg Bob Error: {avg_error:.6f}")
        print(f"  Perfect (last {cfg.log_interval}): {state.perfect_count} ({recent_accuracy:.1f}%)")
        print(f"  Plateau Count: {state.plateau_count}")
        print(f"  Temperature: Alice={self.alice.temperature.item():.4f}, Bob={self.bob.temperature.item():.4f}")
        print(f"  Last example:")
        print(f"    Original:  '{plaintexts[-1]}'")
        print(f"    Decrypted: '{decrypted_texts[-1]}'")
        
        ciphertext_xored = bits_to_text(xor(ciphertext_batch[-1], key).detach().cpu().numpy())
        ciphertext_readable = bits_to_text(ciphertext_batch[-1].detach().cpu().numpy())
        print(f"    Ciphertext (xor w/ key): '{ciphertext_xored}'")
        print(f"    Encrypted: '{ciphertext_readable}'")
        print(f"    Eve Dec.:  '{eve_texts[-1]}'")
        print(f"    Eve Accuracy: {eve_accuracy:.1f}%")
    
    def _run_test_evaluation(self, key: torch.Tensor, key_batch: torch.Tensor) -> None:
        """Runs test evaluations on standard words and security checks."""
        cfg = self.config
        thresholds = SecurityThresholds()
        
        test_words = [
            "hello world     ", "test coding     ", "neural nets     ",
            "crypto proof    ", "python code     ", "simple test     "
        ]
        
        print(f"\n  Testing standard words:")
        correct, total = evaluate_accuracy(
            self.alice, self.bob, test_words, key_batch, self.device
        )
        
        self.alice.eval()
        self.bob.eval()
        with torch.no_grad():
            for word in test_words:
                pb = torch.tensor(text_to_bits(word), dtype=torch.float32, device=self.device)
                ciph = torch.sign(self.alice(xor(pb, key_batch[0]), single=True))
                dec = bits_to_text(self.bob(xor(ciph, key_batch[0]), single=True))
                match = "YES:" if dec == word else "NO:"
                print(f"    {match} '{word}' -> '{dec}'")
        
        print(f"  Score: {correct}/{total} ({100*correct/total:.0f}%)")
        
        print(f"\n  Security Checks:")
        test_plain = text_to_bits_batch(
            self.message_generator.generate_batch(cfg.batch_size),
            device=self.device
        )
        security_metrics = evaluate_security(self.alice, test_plain, key_batch)
        
        check_names = ["leakage", "diversity", "repetition", "key_sensitivity"]
        for name in check_names:
            val = security_metrics[name]
            status = thresholds.evaluate(val)
            print(f"    {name.title()}: {val:.4f} [{status.value.upper()}]")
    
    def _run_final_test(self, key: torch.Tensor, key_batch: torch.Tensor) -> int:
        """Runs the final test evaluation after training is complete."""
        cfg = self.config
        
        print("=" * 70)
        print("FINAL TEST - Random Strings + Known Words")
        print("=" * 70)
        
        self.alice.eval()
        self.bob.eval()
        
        test_batches = 5
        correct = 0
        
        with torch.no_grad():
            for _ in range(test_batches):
                test_batch = self.message_generator.generate_batch(cfg.batch_size)
                test_bits = text_to_bits_batch(test_batch, device=self.device)
                
                ciph = torch.sign(self.alice(xor(test_bits, key_batch)))
                dec_b = self.bob(xor(ciph, key_batch))
                dec_texts = bits_to_text_batch(dec_b)
                
                error = self.mse_criterion(dec_b, test_bits).item()
                
                for original, decrypted in zip(test_batch, dec_texts):
                    match = "YES:" if original == decrypted else "NO:"
                    print(f"{match} '{original}' -> '{decrypted}' | Error: {error:.6f}")
                    if original == decrypted:
                        correct += 1
            
            real_words_path = os.path.join(cfg.data_dir, "real_words.txt")
            if os.path.exists(real_words_path):
                from NeuroSync.data.loaders import load_word_list
                real_words = load_word_list(real_words_path, cfg.message_length)
                
                for i in range(0, len(real_words), cfg.batch_size):
                    batch_words = real_words[i:i + cfg.batch_size]
                    if len(batch_words) < cfg.batch_size:
                        for w in batch_words:
                            pb = torch.tensor(text_to_bits(w), dtype=torch.float32, device=self.device)
                            ciph = torch.sign(self.alice(xor(pb, key), single=True))
                            dec = bits_to_text(self.bob(xor(ciph, key), single=True))
                            match = "YES:" if w == dec else "NO:"
                            print(f"{match} '{w}' -> '{dec}' | Error: N/A")
                            if w == dec:
                                correct += 1
                        continue
                    
                    test_bits = text_to_bits_batch(batch_words, device=self.device)
                    ciph = torch.sign(self.alice(xor(test_bits, key_batch[:len(batch_words)])))
                    dec_b = self.bob(xor(ciph, key_batch[:len(batch_words)]))
                    dec_texts = bits_to_text_batch(dec_b)
                    
                    error = self.mse_criterion(dec_b, test_bits).item()
                    
                    for original, decrypted in zip(batch_words, dec_texts):
                        match = "YES:" if original == decrypted else "NO:"
                        print(f"{match} '{original}' -> '{decrypted}' | Error: {error:.6f}")
                        if original == decrypted:
                            correct += 1
        
        total_tested = 5 * cfg.batch_size
        if os.path.exists(real_words_path):
            total_tested += len(real_words)
        
        print(f"\n{'=' * 70}")
        print(f"Final Score: {correct}/{total_tested} ({100*correct/total_tested:.1f}%)")
        print(f"{'=' * 70}")
        
        return correct
    
    def train(self, load: bool = False) -> TrainingResult:
        """
        Main training loop for NeuroSync.

        Args:
            load (bool): Whether to load existing checkpoints.

        Returns:
            TrainingResult: The result of the training session.
        """
        cfg = self.config
        state = self.state
        
        print("=" * 70)
        print("NEURAL CRYPTO TRAINING - NeuroSync Library")
        print("=" * 70)
        
        word_list_path = os.path.join(cfg.data_dir, cfg.word_list_file)
        if not os.path.exists(word_list_path):
            self.message_generator.generate_word_list(
                word_list_path,
                num_words=cfg.word_list_size,
            )
            print(f"Generated word list at '{word_list_path}'")
        self.message_generator.load_words(word_list_path)
        self.logger.info(f"Loaded {len(self.message_generator.word_list)} words")
        
        self._init_networks()
        
        if not load or self.key_manager.key is None:
            self.key_manager.generate()
            print(f"Generated {self.key_manager.key_bit_length}-bit key")
        
        if not load or self.eve_key_manager.key is None:
            self.eve_key_manager.generate()
            print(f"Generated {self.eve_key_manager.key_bit_length}-bit Eve key")
        
        self._init_optimizers()
        
        if load:
            self._load_checkpoints(cfg.checkpoint_dir)
        
        print(" Alice, Bob, and Eve initialized")
        print(f"  Message: {cfg.bit_length} bits")
        print(f"  Key: {self.key_manager.key_bit_length} bits")
        print(f"  Hidden: {cfg.hidden_size} units")
        print(f"  Batch Size: {cfg.batch_size}")
        print(f"Word list size: {len(self.message_generator.word_list)}")
        print(f"Training for {cfg.training_episodes} episodes...")
        print("=" * 70)
        
        num_of_batches = cfg.training_episodes // cfg.batch_size
        log_interval_batches = cfg.log_interval // cfg.batch_size
        test_interval_batches = cfg.test_interval // cfg.batch_size
        
        for batch_i in range(num_of_batches):
            if batch_i < 2000:
                plaintexts = self.message_generator.generate_batch(cfg.batch_size // 2)
                plaintexts = plaintexts + plaintexts
                state.adversarial_weight = 0.0
            else:
                plaintexts = self.message_generator.generate_batch(cfg.batch_size)
            
            plain_bits_batch = text_to_bits_batch(plaintexts, device=self.device)
            
            self.key_manager.generate()
            key_batch = self.key_manager.to_tensor(cfg.batch_size)
            key = self.key_manager.to_tensor(1).squeeze(0)
            
            eve_key_batch = self.eve_key_manager.to_tensor(cfg.batch_size)
            
            ciphertext_batch, decrypted_bits_batch, eve_output, loss = self._train_alice_bob_batch(
                plain_bits_batch, key_batch, eve_key_batch
            )
            
            self._update_counters(plaintexts, decrypted_bits_batch, eve_output, plain_bits_batch)
            
            if batch_i % cfg.eve_train_skip == 0:
                self._train_eve_batch(plain_bits_batch, eve_key_batch)
            
            if (batch_i + 1) % log_interval_batches == 0:
                episode = (batch_i + 1) * cfg.batch_size
                recent_accuracy = (100 * state.perfect_count / state.total_count) if state.total_count > 0 else 0.0
                
                self._update_schedulers(episode, recent_accuracy)
                
                self._log_progress(
                    episode, plaintexts, decrypted_bits_batch, 
                    ciphertext_batch, eve_output, key
                )
                
                state.reset_counters()
                
                if (batch_i + 1) % test_interval_batches == 0:
                    self._run_test_evaluation(key, key_batch)
                    self._save_checkpoints(cfg.checkpoint_dir)
        
        print("\n" + "=" * 70)
        print("Saving networks...")
        self._save_checkpoints(cfg.checkpoint_dir)
        print("Saved!")
        
        key = self.key_manager.to_tensor(1).squeeze(0)
        key_batch = self.key_manager.to_tensor(cfg.batch_size)
        self._run_final_test(key, key_batch)
        
        return TrainingResult(
            alice=self.alice,
            bob=self.bob,
            eve=self.eve,
            final_accuracy=state.running_bob_accuracy,
            best_accuracy=state.best_accuracy,
            config=cfg,
        )
