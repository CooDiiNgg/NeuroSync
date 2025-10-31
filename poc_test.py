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

class ImprovedNetwork(nn.Module):
    """Improved multi-layer network with proper architecture"""
    def __init__(self, input_size, hidden_size, output_size, name="net"):
        super(ImprovedNetwork, self).__init__()
        self.name = name
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size, momentum=0.1)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size, momentum=0.1)

        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size, momentum=0.1)

        self.fc4 = nn.Linear(hidden_size, output_size)
        
        self.dropout = nn.Dropout(0.07)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Xavier initialization for better gradient flow"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='tanh', mode='fan_in')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, single=False):
        """Forward pass with multiple layers and now BatchNorm"""
        if single:
            self.eval()
            with torch.no_grad():
                x = torch.tanh(self.bn1(self.fc1(x.unsqueeze(0)))).squeeze(0)
                x = self.dropout(x)
                x = torch.tanh(self.bn2(self.fc2(x.unsqueeze(0)))).squeeze(0)
                x = self.dropout(x)
                x = torch.tanh(self.bn3(self.fc3(x.unsqueeze(0)))).squeeze(0)
                x = self.dropout(x)
                x = torch.tanh(self.fc4(x))
            self.train()
        else:
            x = torch.tanh(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = torch.tanh(self.bn2(self.fc2(x)))
            x = self.dropout(x)
            x = torch.tanh(self.bn3(self.fc3(x)))
            x = self.dropout(x)
            x = torch.tanh(self.fc4(x))
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

    if load and os.path.exists('alice_test.pth') and os.path.exists('bob_test.pth'):
        print("Loading saved networks...")
        alice.load('alice_test.pth')
        bob.load('bob_test.pth')
        print("Loaded!\n")
    
    print(" Alice and Bob initialized")
    print(f"  Message: {BIT_LENGTH} bits")
    print(f"  Key: {len(key)} bits")
    print(f"  Hidden: {HIDDEN_SIZE} units")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Architecture: Input -> {HIDDEN_SIZE} -> {HIDDEN_SIZE} -> {HIDDEN_SIZE} -> Output\n")
    print("Word list size:", len(word_list))
    
    criterion = nn.MSELoss()

    bob_optimizer = optim.Adam(bob.parameters(), lr=0.001, weight_decay=1e-5)
    alice_optimizer = optim.Adam(alice.parameters(), lr=0.001, weight_decay=1e-5)
    

    bob_scheduler = optim.lr_scheduler.ReduceLROnPlateau(bob_optimizer, mode='min', factor=0.5, patience=30, min_lr=1e-9)
    alice_scheduler = optim.lr_scheduler.ReduceLROnPlateau(alice_optimizer, mode='min', factor=0.5, patience=30, min_lr=1e-9)

    print(f"Training for {TRAINING_EPISODES} episodes...")
    print("=" * 70)
    
    bob_errors = []
    perfect_count = 0
    total_count = 0

    if load and os.path.exists('training_state_test.pth'):
        print("Loading training state...")
        training_state = torch.load('training_state_test.pth', map_location=device)
        bob_optimizer.load_state_dict(training_state['bob_optimizer'])
        alice_optimizer.load_state_dict(training_state['alice_optimizer'])
        bob_scheduler.load_state_dict(training_state['bob_scheduler'])
        alice_scheduler.load_state_dict(training_state['alice_scheduler'])
        print("Loaded training state!\n")

    num_of_batches = TRAINING_EPISODES // BATCH_SIZE

    for batch_i in range(0, num_of_batches):
        plaintexts = generate_random_messages(BATCH_SIZE)
        plain_bits_batch = text_to_bits_batch(plaintexts)
        
        alice.train()
        bob.train()

        alice_input = torch.cat([plain_bits_batch, key_batch], dim=1)
        ciphertext_batch = alice(alice_input)

        bob_input = torch.cat([ciphertext_batch, key_batch], dim=1)
        decrypted_bits_batch = bob(bob_input)

        loss = criterion(decrypted_bits_batch, plain_bits_batch)
        bob_errors.append(loss.item())

        with torch.no_grad():
            decrypted_texts = bits_to_text_batch(decrypted_bits_batch)
            for pt, dt in zip(plaintexts, decrypted_texts):
                total_count += 1
                if pt == dt:
                    perfect_count += 1

        bob_optimizer.zero_grad()
        alice_optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(bob.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(alice.parameters(), 1.0)

        bob_optimizer.step()
        alice_optimizer.step()
        
        if batch_i % 100 == 0 and batch_i > 0:
            avg_error = np.mean(bob_errors[-100:]) if len(bob_errors) >= 100 else np.mean(bob_errors)
            bob_scheduler.step(avg_error)
            alice_scheduler.step(avg_error)
        
        if (batch_i + 1) % (2500 // BATCH_SIZE) == 0:
            avg_error = np.mean(bob_errors[-100:]) if len(bob_errors) >= 100 else np.mean(bob_errors)
            recent_accuracy = 100 * perfect_count / total_count
            episode = (batch_i + 1) * BATCH_SIZE

            print(f"\nEpisode {episode}/{TRAINING_EPISODES}")
            print(f"  Avg Bob Error: {avg_error:.6f}")
            print(f"  Perfect (last 2500): {perfect_count} ({recent_accuracy:.1f}%)")
            print(f"  Current LR: {bob_optimizer.param_groups[0]['lr']:.6f}")
            print(f"  Last example:")
            print(f"    Original:  '{plaintexts[-1]}'")
            print(f"    Decrypted: '{decrypted_texts[-1]}'")
            ciphertext_readable = bits_to_text(ciphertext_batch[-1])
            print(f"    Encrypted: '{ciphertext_readable}'")

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
                        bi = torch.cat([ciph, key])
                        dec_b = bob(bi, single=True)
                        dec = bits_to_text(dec_b)
                        match = "YES:" if dec == word else "NO:"
                        print(f"    {match} '{word}' → '{dec}'")
                        if dec == word:
                            correct += 1
                print(f"  Score: {correct}/{len(test_words)} ({100*correct/len(test_words):.0f}%)")
                
                torch.save({
                    'bob_optimizer': bob_optimizer.state_dict(),
                    'alice_optimizer': alice_optimizer.state_dict(),
                    'bob_scheduler': bob_scheduler.state_dict(),
                    'alice_scheduler': alice_scheduler.state_dict()
                }, 'training_state_test.pth')

                if correct == len(test_words) and recent_accuracy >= 99.5:
                    print(f"\n Perfect performance achieved! Stopping early at episode {episode + 1}")
                    break
    
    print("\n" + "=" * 70)
    print("Saving networks...")
    alice.save('alice_test.pth')
    bob.save('bob_test.pth')

    torch.save({
        'bob_optimizer': bob_optimizer.state_dict(),
        'alice_optimizer': alice_optimizer.state_dict(),
        'bob_scheduler': bob_scheduler.state_dict(),
        'alice_scheduler': alice_scheduler.state_dict()
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
            bi = torch.cat([ciph, key_batch], dim=1)
            dec_b = bob(bi)
            dec_texts = bits_to_text_batch(dec_b)
            
            error = criterion(dec_b, test_bits).item()
            
            for original, decrypted in zip(test_batch, dec_texts):
                match = "YES:" if original == decrypted else "NO:"
                print(f"{match} '{original}' → '{decrypted}' | Error: {error:.6f}")
                if original == decrypted:
                    correct += 1
    
    print(f"\n{'=' * 70}")
    print(f"Final Score: {correct}/{5*BATCH_SIZE} ({100*correct/5*BATCH_SIZE:.1f}%)")
    print(f"{'=' * 70}")


# def test_saved():
#     """Test saved networks"""
#     if not os.path.exists('alice_test.pth') or not os.path.exists('bob_test.pth'):
#         print("No saved networks found. Train first.")
#         return
    
#     print("Loading networks...")
#     key_np = np.load('key.npy')
#     key = torch.tensor(key_np, dtype=torch.float32, device=device)
    
#     BIT_LENGTH = MESSAGE_LENGTH * 5
#     HIDDEN_SIZE = 256
#     alice = ImprovedNetwork(BIT_LENGTH + len(key), HIDDEN_SIZE, BIT_LENGTH, "Alice").to(device)
#     bob = ImprovedNetwork(BIT_LENGTH + len(key), HIDDEN_SIZE, BIT_LENGTH, "Bob").to(device)
#     alice.load('alice_test.pth')
#     bob.load('bob_test.pth')
    
#     alice.eval()
#     bob.eval()
    
#     print("Loaded\n")
#     print("=" * 70)
#     print("TESTING")
#     print("=" * 70)
    
#     test_words = [string.ascii_lowercase[i:i+MESSAGE_LENGTH] 
#                   for i in range(0, 26 - MESSAGE_LENGTH + 1)]
#     test_words += ["hello world     ", "neural network  ", "deep learning   "]
    
#     criterion = nn.MSELoss()
#     correct = 0
    
#     with torch.no_grad():
#         for word in test_words:
#             pb = text_to_bits(word)
#             ai = torch.cat([pb, key])
#             ciph = alice(ai)
#             bi = torch.cat([ciph, key])
#             dec_b = bob(bi)
#             dec = bits_to_text(dec_b)
            
#             error = criterion(dec_b, pb).item()
#             match = "YES:" if dec == word else "NO:"
#             print(f"{match} '{word}' → '{dec}' | Error: {error:.6f}")
#             if dec == word:
#                 correct += 1
    
#     print(f"\n{'=' * 70}")
#     print(f"Score: {correct}/{len(test_words)} ({100*correct/len(test_words):.0f}%)")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # test_saved()
        print("Testing saved networks is currently disabled.")
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
