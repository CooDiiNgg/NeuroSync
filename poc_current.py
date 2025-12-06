import re
import torch
import torch.nn as nn
import torch.optim as optim
import string
import os
import numpy as np


MESSAGE_LENGTH = 16
KEY_SIZE = 16
TRAINING_EPISODES = 20000000
BATCH_SIZE = 64

with open("./words.txt", "r") as f:
       word_list = [line.strip() + " " * (MESSAGE_LENGTH - len(line.strip())) for line in f if len(line.strip()) <= MESSAGE_LENGTH]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"Moving model and tensors to 'cuda'.\n")
else:
    print(f"CUDA not available. Using CPU.\n")


def text_to_bits(text):
    """Convert text to binary: each char as 5 bits (0-26 in binary)"""
    text = text.ljust(MESSAGE_LENGTH)[:MESSAGE_LENGTH]
    bits = []
    for c in text:
        if c == ' ':
            val = 52
        elif 'a' <= c <= 'z':
            val = ord(c) - ord('a')
        elif 'A' <= c <= 'Z':
            val = ord(c) - ord('A')
            val += 26
        else:
            val = 52

        for i in range(5, -1, -1):
            bits.append(1.0 if (val >> i) & 1 else -1.0)
    return bits

def text_to_bits_batch(texts):
    """Convert batch of texts to binary tensors"""
    batch_bits = []
    for text in texts:
        bits = text_to_bits(text)
        batch_bits.append(bits)
    return torch.tensor(batch_bits, dtype=torch.float32, device=device)


def bits_to_text(bits):
    """Convert binary back to text"""
    if isinstance(bits, torch.Tensor):
        bits = bits.detach().cpu().numpy()
    
    chars = []
    for i in range(0, len(bits), 6):
        chunk = bits[i:i+6]

        val = 0
        for j, bit in enumerate(chunk):
            if bit > 0:
                val |= (1 << (5 - j))
        val = min(52, val)

        if val == 52:
            chars.append(' ')
        elif val <= 25:
            chars.append(chr(val + ord('a')))
        elif val <= 51:
            chars.append(chr(val - 26 + ord('A')))
    return ''.join(chars)

def bits_to_text_batch(bits):
    """Convert batch of binary tensors back to texts"""
    if isinstance(bits, torch.Tensor):
        bits = bits.detach().cpu().numpy()
    texts = []
    for bit_seq in bits:
        text = bits_to_text(bit_seq)
        texts.append(text)
    return texts


def generate_random_message():
    """Generate random message from word list"""
    return word_list[np.random.randint(0, len(word_list))]

def generate_random_messages(batch_size):
    """Generate batch of random messages from word list"""
    messages = []
    for _ in range(batch_size):
        messages.append(generate_random_message())
    return messages


class StraightThroughSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp(-1.0, 1.0)


def straight_through_sign(input):
    return StraightThroughSign.apply(input)


class ResidualBlock(nn.Module):
    """Residual block - perhaps this will fix the plateauing issue"""
    def __init__(self, size, dropout=0.05):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(size, size)
        self.bn = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = torch.tanh(self.bn(self.fc(x)))
        out = self.dropout(out)
        out = out + residual
        return out


def confidence_loss(input, margin=0.7):
    return torch.mean(torch.clamp(margin - torch.abs(input), min=0.0))

class ImprovedNetwork(nn.Module):
    """Improved multi-layer network with proper architecture"""
    def __init__(self, input_size, hidden_size, output_size, name="net"):
        super(ImprovedNetwork, self).__init__()
        self.name = name
        
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.input_bn = nn.LayerNorm(hidden_size)

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout=0.05) for _ in range(3)
        ])

        self.pre_out = nn.Linear(hidden_size, hidden_size)
        self.pre_out_bn = nn.LayerNorm(hidden_size)

        self.out = nn.Linear(hidden_size, output_size)
        
        self.temperature = nn.Parameter(torch.tensor(0.0))
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Xavier initialization for better gradient flow"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                # nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
                nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def temp(self):
        return torch.nn.functional.softplus(self.temperature) + 0.5
    
    def forward(self, x, single=False):
        """Forward pass with multiple layers and now BatchNorm"""
        temper = self.temp
        if single:
            self.eval()
            with torch.no_grad():
                x = torch.tanh(self.input_bn(self.input_projection(x.unsqueeze(0))))
                for block in self.residual_blocks:
                    x = block(x)
                x = torch.tanh(self.pre_out_bn(self.pre_out(x)))
                x = torch.tanh(self.out(x)/temper)
                x = x.squeeze(0)
            self.train()
        else:
            x = torch.tanh(self.input_bn(self.input_projection(x)))
            for block in self.residual_blocks:
                x = block(x)
            x = torch.tanh(self.pre_out_bn(self.pre_out(x)))
            x = torch.tanh(self.out(x)/temper)
        return x
    
    def save(self, filename):
        """Save model state"""
        torch.save({
            'state_dict': self.state_dict()
        }, filename)
    
    def load(self, filename):
        """Load model state"""
        checkpoint = torch.load(filename, map_location=device)
        self.load_state_dict(checkpoint['state_dict'])


def train(load=False):
    """Train Alice and Bob - FIXED version for 100% accuracy"""
    print("=" * 70)
    print("FIXED NEURAL CRYPTO POC - Optimized for 100% Convergence")
    print("=" * 70)
    
    BIT_LENGTH = MESSAGE_LENGTH * 6
    HIDDEN_SIZE = 512
    
    if load and os.path.exists('key.npy'):
        key_np = np.load('key.npy')
        key = torch.tensor(key_np, dtype=torch.float32, device=device)
        print(f"Loaded {len(key)}-bit key\n")
    else:
        key_np = np.random.choice([-1.0, 1.0], KEY_SIZE * 6)
        np.save('key.npy', key_np)
        key = torch.tensor(key_np, dtype=torch.float32, device=device)
        print(f"Generated {len(key)}-bit key\n")
    
    key_batch = key.unsqueeze(0).repeat(BATCH_SIZE, 1)

    alice = ImprovedNetwork(BIT_LENGTH + len(key), HIDDEN_SIZE, BIT_LENGTH, "Alice").to(device)
    bob = ImprovedNetwork(BIT_LENGTH + len(key), HIDDEN_SIZE, BIT_LENGTH, "Bob").to(device)
    eve = ImprovedNetwork(BIT_LENGTH, HIDDEN_SIZE, BIT_LENGTH, "Eve").to(device)

    if load and os.path.exists('alice_test.pth') and os.path.exists('bob_test.pth'):
        print("Loading saved networks...")
        alice.load('alice_test.pth')
        bob.load('bob_test.pth')
        if os.path.exists('eve_test.pth'):
            eve.load('eve_test.pth')
        print("Loaded!\n")
    
    print(" Alice, Bob, and Eve initialized")
    print(f"  Message: {BIT_LENGTH} bits")
    print(f"  Key: {len(key)} bits")
    print(f"  Hidden: {HIDDEN_SIZE} units")
    print(f"  Batch Size: {BATCH_SIZE}")
    print("Word list size:", len(word_list))
    
    mse_criterion = nn.MSELoss()
    smooth_l1_criterion = nn.SmoothL1Loss()

    alice_and_bob_params = list(alice.parameters()) + list(bob.parameters())
    
    alice_and_bob_optimizer = optim.AdamW(alice_and_bob_params, lr=0.0005, weight_decay=1e-4, betas=(0.9, 0.999))
    # alice_optimizer = optim.AdamW(alice.parameters(), lr=0.001, weight_decay=1e-4, betas=(0.9, 0.999))
    # bob_optimizer = optim.AdamW(bob.parameters(), lr=0.001, weight_decay=1e-4, betas=(0.9, 0.999))
    eve_optimizer = optim.AdamW(eve.parameters(), lr=0.0003, weight_decay=1e-4, betas=(0.9, 0.999))

    
    alice_and_bob_scheduler = optim.lr_scheduler.StepLR(alice_and_bob_optimizer, step_size=50000, gamma=0.5)
    # alice_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(alice_optimizer, T_0=1000, T_mult=2, eta_min=1e-7)
    # bob_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(bob_optimizer, T_0=1000, T_mult=2, eta_min=1e-7)
    eve_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(eve_optimizer, T_0=1000, T_mult=2, eta_min=1e-7)

    print(f"Training for {TRAINING_EPISODES} episodes...")
    print("=" * 70)
    
    bob_errors = []
    eve_errors = []
    perfect_count = 0
    total_count = 0
    best_accuracy = 0.0
    plateau_count = 0
    use_smooth_l1 = False
    eve_use_smooth_l1 = False
    eve_guess_count = 0

    EVE_TRAIN_SKIP = 2
    ADVERSARIAL_WEIGHT = 0.0
    CONFIDENCE_MAX = 0.3
    CONFIDENCE_WEIGHT = 0.0

    # PHASE_1_EPISODES = 5000
    # PHASE_2_EPISODES = 9000
    # discretization_prob = 0.0

    # prev_ciphertext = None
    # repeating_ciphertext = 0
    total_bits = 0
    correct_bits = 0

    if load and os.path.exists('training_state_test.pth'):
        print("Loading training state...")
        training_state = torch.load('training_state_test.pth', map_location=device)
        alice_and_bob_optimizer.load_state_dict(training_state['alice_and_bob_optimizer'])
        # alice_optimizer.load_state_dict(training_state['alice_optimizer'])
        # bob_optimizer.load_state_dict(training_state['bob_optimizer'])
        if 'eve_optimizer' in training_state:
            eve_optimizer.load_state_dict(training_state['eve_optimizer'])
        if 'best_accuracy' in training_state:
            best_accuracy = training_state['best_accuracy']
        print("Loaded training state!\n")

    num_of_batches = TRAINING_EPISODES // BATCH_SIZE
    # torch.autograd.set_detect_anomaly(True)
    for batch_i in range(0, num_of_batches):
        if batch_i < 2000:
            plaintexts = generate_random_messages(BATCH_SIZE//2)
            plaintexts += plaintexts
            ADVERSARIAL_WEIGHT = 0.0
        else:
            ADVERSARIAL_WEIGHT = 0.0
            plaintexts = generate_random_messages(BATCH_SIZE)
        plain_bits_batch = text_to_bits_batch(plaintexts)

        # if batch_i < PHASE_1_EPISODES:
        #     discretization_prob = 0.0
        # elif batch_i < PHASE_2_EPISODES:
        #     discretization_prob = (batch_i - PHASE_1_EPISODES) / (PHASE_2_EPISODES - PHASE_1_EPISODES)
        # else:
        #     discretization_prob = 1.0
        

        alice.train()
        bob.train()

        eve.eval()

        key_np = np.random.choice([-1.0, 1.0], KEY_SIZE * 6)
        np.save('key.npy', key_np)
        key = torch.tensor(key_np, dtype=torch.float32, device=device)
        key_batch = key.unsqueeze(0).repeat(BATCH_SIZE, 1)

        alice_input = torch.cat([plain_bits_batch, key_batch], dim=1)
        ciphertext_batch_original = alice(alice_input)


        # if prev_ciphertext is not None:
        #     if bits_to_text(ciphertext_batch[-1]) == bits_to_text(prev_ciphertext):
        #         repeating_ciphertext += 1
        #     else:
        #         repeating_ciphertext = 0
        # prev_ciphertext = ciphertext_batch[-1].detach().clone()

        # if np.random.rand() < discretization_prob:
        #     ciphertext_batch = straight_through_sign(ciphertext_batch_original)
        # else:
        #     ciphertext_batch = torch.sign(ciphertext_batch_original).detach()

        ciphertext_batch = straight_through_sign(ciphertext_batch_original)

        bob_input = torch.cat([ciphertext_batch, key_batch], dim=1)
        decrypted_bits_batch = bob(bob_input)

        # with torch.no_grad():
        #     eve_output = eve(ciphertext_batch.detach())

        #     if eve_use_smooth_l1:
        #         eve_loss = smooth_l1_criterion(eve_output, plain_bits_batch)
        #     else:
        #         eve_loss = mse_criterion(eve_output, plain_bits_batch)

        #     eve_errors.append(eve_loss.item())


        eve_output_alice = eve(ciphertext_batch)

        if eve_use_smooth_l1:
            eve_loss_alice = smooth_l1_criterion(eve_output_alice, plain_bits_batch)
        else:
            eve_loss_alice = mse_criterion(eve_output_alice, plain_bits_batch)

        # temporary...
        use_smooth_l1 = False
        if use_smooth_l1:
            loss = smooth_l1_criterion(decrypted_bits_batch, plain_bits_batch)
        else:
            loss = mse_criterion(decrypted_bits_batch, plain_bits_batch)
        bob_errors.append(loss.item())

        # if discretization_prob > 0:
        #     total_loss = loss + CONFIDENCE_WEIGHT * confidence_loss(ciphertext_batch_original) - ADVERSARIAL_WEIGHT * eve_loss_alice
        # else:
        #     total_loss = loss - ADVERSARIAL_WEIGHT * eve_loss_alice

        total_loss = loss + CONFIDENCE_WEIGHT * confidence_loss(ciphertext_batch_original) - ADVERSARIAL_WEIGHT * eve_loss_alice
        # if repeating_ciphertext >= 10:
        #     print(f"Detected {repeating_ciphertext} repeating ciphertexts, applying penalty and resetting Alice's temperature.")
        #     total_loss += 200.0
        #     with torch.no_grad():
        #         alice.temperature.data = torch.tensor(1.0, device=device)
        #     repeating_ciphertext = 0

        # alice_optimizer.zero_grad()
        # bob_optimizer.zero_grad()
        alice_and_bob_optimizer.zero_grad()

        total_loss.backward()

        max_norm = 1.0 if loss.item() < 0.1 else 0.5
        torch.nn.utils.clip_grad_norm_(bob.parameters(), max_norm)
        torch.nn.utils.clip_grad_norm_(alice.parameters(), max_norm)

        # alice_optimizer.step()
        # bob_optimizer.step()
        # alice_scheduler.step()
        # bob_scheduler.step()
        alice_and_bob_optimizer.step()
        alice_and_bob_scheduler.step()

        with torch.no_grad():
            decrypted_texts = bits_to_text_batch(decrypted_bits_batch.detach())
            eve_texts = bits_to_text_batch(eve_output_alice.detach())
            for pt, dt in zip(plaintexts, decrypted_texts):
                total_count += 1
                if pt == dt:
                    perfect_count += 1
            for pt, et in zip(plaintexts, eve_texts):
                if pt == et:
                    eve_guess_count += 1
            
            bit_matches = (torch.sign(decrypted_bits_batch) == torch.sign(plain_bits_batch)).float()
            correct_bits += bit_matches.sum().item()
            total_bits += bit_matches.numel()

        if batch_i % EVE_TRAIN_SKIP == 0:
            eve.train()
            alice.eval()
            bob.eval()

            with torch.no_grad():
                alice_input_eve = torch.cat([plain_bits_batch, key_batch], dim=1)
                ciphertext_batch_eve = alice(alice_input_eve) 
                ciphertext_batch_eve = straight_through_sign(ciphertext_batch_eve)
            
            eve_output = eve(ciphertext_batch_eve)
            if eve_use_smooth_l1:
                eve_loss = smooth_l1_criterion(eve_output, plain_bits_batch)
            else:
                eve_loss = mse_criterion(eve_output, plain_bits_batch)
            eve_errors.append(eve_loss.item())
            eve_optimizer.zero_grad()
            eve_loss.backward()
            torch.nn.utils.clip_grad_norm_(eve.parameters(), 0.5)
            eve_optimizer.step()
            eve_scheduler.step()
        
        if (batch_i + 1) % (2500 // BATCH_SIZE) == 0:
            avg_error = np.mean(bob_errors[-100:]) if len(bob_errors) >= 100 else np.mean(bob_errors)
            recent_accuracy = (100 * perfect_count / total_count) if total_count > 0 else 0.0
            episode = (batch_i + 1) * BATCH_SIZE

            eve_accuracy = (100 * eve_guess_count / total_count )if total_count > 0 else 0.0
            eve_guess_count = 0
            # TODO: need to fix this static switch logic a bit later
            if eve_accuracy > 80.0:
                ADVERSARIAL_WEIGHT = min(2.0, ADVERSARIAL_WEIGHT * 1.2)
                eve_use_smooth_l1 = False
            else:
                if eve_accuracy < 20.0 and recent_accuracy > 85.0:
                    ADVERSARIAL_WEIGHT = max(0.5, ADVERSARIAL_WEIGHT * 0.8)
                eve_use_smooth_l1 = True

            if recent_accuracy > best_accuracy:
                best_accuracy = recent_accuracy
                plateau_count = 0
                use_smooth_l1 = False
            else:
                plateau_count += 1
                if plateau_count >= 10 and recent_accuracy < 90.0:
                    use_smooth_l1 = True
            
            bit_accuracy = (100.0 * correct_bits) / total_bits if total_bits > 0 else 0.0
            correct_bits = 0
            total_bits = 0
            if bit_accuracy < 90.0:
                CONFIDENCE_WEIGHT = 0.0
            elif bit_accuracy < 97.0:
                CONFIDENCE_WEIGHT = ((bit_accuracy - 90.0) / 7.0) * CONFIDENCE_MAX
            else:
                CONFIDENCE_WEIGHT = CONFIDENCE_MAX

            print(f"\nEpisode {episode}/{TRAINING_EPISODES}")
            print(f"  Avg Bob Error: {avg_error:.6f}")
            print(f"  Perfect (last 2500): {perfect_count} ({recent_accuracy:.1f}%)")
            print(f"  Plateau Count: {plateau_count}")
            print(f"  Temperature: Alice={alice.temperature.item():.4f}, Bob={bob.temperature.item():.4f}")
            print(f"  Last example:")
            print(f"    Original:  '{plaintexts[-1]}'")
            print(f"    Decrypted: '{decrypted_texts[-1]}'")
            ciphertext_readable = ciphertext_batch[-1].detach().cpu().numpy()
            ciphertext_readable = bits_to_text(ciphertext_readable)
            print(f"    Encrypted: '{ciphertext_readable}'")
            print(f"    Eve Dec.:  '{eve_texts[-1]}'")
            print(f"    Eve Accuracy: {eve_accuracy:.1f}%")

            perfect_count = 0
            total_count = 0
            
            if (batch_i + 1) % (10000 // BATCH_SIZE) == 0:
                alice.eval()
                bob.eval()
                print(f"\n  Testing standard words:")
                test_words = ["hello world     ", "test coding     ", "neural nets     ", 
                              "crypto proof    ", "python code     ", "simple test     "]
                correct = 0
                with torch.no_grad():
                    for word in test_words:
                        pb = text_to_bits(word)
                        pb = torch.tensor(pb, dtype=torch.float32, device=device)
                        ai = torch.cat([pb, key])
                        ciph = alice(ai, single=True)
                        ciph = torch.sign(ciph)
                        bi = torch.cat([ciph, key])
                        dec_b = bob(bi, single=True)
                        dec = bits_to_text(dec_b)
                        match = "YES:" if dec == word else "NO:"
                        print(f"    {match} '{word}' → '{dec}'")
                        if dec == word:
                            correct += 1
                print(f"  Score: {correct}/{len(test_words)} ({100*correct/len(test_words):.0f}%)")
                
                torch.save({
                    'alice_and_bob_optimizer': alice_and_bob_optimizer.state_dict(),
                    # 'alice_optimizer': alice_optimizer.state_dict(),
                    # 'bob_optimizer': bob_optimizer.state_dict(),
                    'eve_optimizer': eve_optimizer.state_dict(),
                    'best_accuracy': best_accuracy
                }, 'training_state_test.pth')

                if correct == len(test_words) and recent_accuracy >= 99.8:
                    print(f"\n Perfect performance achieved! Stopping early at episode {episode + 1}")
                    break
    
    print("\n" + "=" * 70)
    print("Saving networks...")
    alice.save('alice_test.pth')
    bob.save('bob_test.pth')
    eve.save('eve_test.pth')

    torch.save({
        'alice_and_bob_optimizer': alice_and_bob_optimizer.state_dict(),
        # 'alice_optimizer': alice_optimizer.state_dict(),
        # 'bob_optimizer': bob_optimizer.state_dict(),
        'eve_optimizer': eve_optimizer.state_dict(),
        'best_accuracy': best_accuracy
    }, 'training_state_test.pth')
    
    print("Saved!\n")
    
    print("=" * 70)
    print("FINAL TEST - Random Strings + Known Words")
    print("=" * 70)
    
    alice.eval()
    bob.eval()
    
    test_batches = 5
    correct = 0

    criterion = nn.MSELoss()    
    with torch.no_grad():
        for _ in range(test_batches):
            test_batch = generate_random_messages(BATCH_SIZE)
            test_bits = text_to_bits_batch(test_batch)

            ai = torch.cat([test_bits, key_batch], dim=1)
            ciph = alice(ai)
            ciph = torch.sign(ciph)
            bi = torch.cat([ciph, key_batch], dim=1)
            dec_b = bob(bi)
            dec_texts = bits_to_text_batch(dec_b)
            
            error = criterion(dec_b, test_bits).item()
            
            for original, decrypted in zip(test_batch, dec_texts):
                match = "YES:" if original == decrypted else "NO:"
                print(f"{match} '{original}' → '{decrypted}' | Error: {error:.6f}")
                if original == decrypted:
                    correct += 1
        # now some real wods to test:
        with open("./real_words.txt", "r") as f:
            real_words = [line.strip() + " " * (MESSAGE_LENGTH - len(line.strip())) for line in f if len(line.strip()) <= MESSAGE_LENGTH]
        for i in range(0, len(real_words), BATCH_SIZE):
            batch_words = real_words[i:i+BATCH_SIZE]
            if len(batch_words) < BATCH_SIZE:
                for w in batch_words:
                    test_bits_single = text_to_bits(w)
                    test_bits_single = torch.tensor(test_bits_single, dtype=torch.float32, device=device)
                    ai = torch.cat([test_bits_single, key])
                    ciph = alice(ai, single=True)
                    ciph = torch.sign(ciph)
                    bi = torch.cat([ciph, key])
                    dec_b = bob(bi, single=True)
                    dec = bits_to_text(dec_b)
                    match = "YES:" if w == dec else "NO:"
                    print(f"{match} '{w}' → '{dec}' | Error: N/A")
                    if w == dec:
                        correct += 1
                continue
            test_bits = text_to_bits_batch(batch_words)

            ai = torch.cat([test_bits, key_batch], dim=1)
            ciph = alice(ai)
            ciph = torch.sign(ciph)
            bi = torch.cat([ciph, key_batch], dim=1)
            dec_b = bob(bi)
            dec_texts = bits_to_text_batch(dec_b)
            
            error = criterion(dec_b, test_bits).item()
            
            for original, decrypted in zip(batch_words, dec_texts):
                match = "YES:" if original == decrypted else "NO:"
                print(f"{match} '{original}' → '{decrypted}' | Error: {error:.6f}")
                if original == decrypted:
                    correct += 1
    
    print(f"\n{'=' * 70}")
    print(f"Final Score: {correct}/{5*BATCH_SIZE + len(real_words)} ({100*correct/(5*BATCH_SIZE):.1f}%)")
    print(f"{'=' * 70}")


def test_saved():
    """Test saved networks"""
    if not os.path.exists('alice_test.pth') or not os.path.exists('bob_test.pth'):
        print("No saved networks found. Train first.")
        return
    
    print("Loading networks...")
    key_np = np.load('key.npy')
    key = torch.tensor(key_np, dtype=torch.float32, device=device)
    key_batch = key.unsqueeze(0).repeat(BATCH_SIZE, 1)
    
    BIT_LENGTH = MESSAGE_LENGTH * 6
    HIDDEN_SIZE = 512
    alice = ImprovedNetwork(BIT_LENGTH + len(key), HIDDEN_SIZE, BIT_LENGTH, "Alice").to(device)
    bob = ImprovedNetwork(BIT_LENGTH + len(key), HIDDEN_SIZE, BIT_LENGTH, "Bob").to(device)
    alice.load('alice_test.pth')
    bob.load('bob_test.pth')
    
    alice.eval()
    bob.eval()
    
    print("Loaded\n")
    print("=" * 70)
    print("TESTING")
    print("=" * 70)
    
    
    with open("./real_words.txt", "r") as f:
        test_words = [line.strip() + " " * (MESSAGE_LENGTH - len(line.strip())) for line in f if len(line.strip()) <= MESSAGE_LENGTH]
    test_words += [word_list[np.random.randint(0, len(word_list))] for _ in range(50)]
    test_words = ["abcdabcdABCDABCD"]
    criterion = nn.MSELoss()
    correct = 0
    
    with torch.no_grad():
        for w in test_words:
            test_bits = text_to_bits(w)
            test_bits = torch.tensor(test_bits, dtype=torch.float32, device=device)
            ai = torch.cat([test_bits, key])
            ciph = alice(ai, single=True)
            ciph = torch.sign(ciph)
            bi = torch.cat([ciph, key])
            dec_b = bob(bi, single=True)
            dec = bits_to_text(dec_b)
            error = criterion(dec_b, test_bits).item()
            match = "YES:" if w == dec else "NO:"
            print(f"{match} '{w}' → '{dec}' | Error: {error:.6f}")
            if w == dec:
                correct += 1
    
    print(f"\n{'=' * 70}")
    print(f"Score: {correct}/{len(test_words)} ({100*correct/len(test_words):.0f}%)")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Testing saved networks...")
        test_saved()
    elif len(sys.argv) > 1 and sys.argv[1] == "load":
        epoch = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        print("Retraining from saved networks...")
        for e in range(epoch):
            print(f"\n{'=' * 70}")
            print(f"--- Epoch {e + 1}/{epoch} ---")
            print(f"{'=' * 70}\n")
            train(load=True)
    else:
        print("Starting FIXED neural crypto training...")
        print("Optimized parameters to push from 97% to 100%!")
        print("=" * 70)
        train()
        epoch = int(sys.argv[1]) if len(sys.argv) > 1 else 0
        for e in range(epoch):
            train(load=True)
