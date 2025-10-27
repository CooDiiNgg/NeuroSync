import re
import torch
import torch.nn as nn
import torch.optim as optim
import string
import os
import numpy as np


MESSAGE_LENGTH = 16
KEY_SIZE = 16
TRAINING_EPISODES = 1000000

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
    text = text.lower().ljust(MESSAGE_LENGTH)[:MESSAGE_LENGTH]
    bits = []
    for c in text:
        if c == ' ':
            val = 26
        elif 'a' <= c <= 'z':
            val = ord(c) - ord('a')
        else:
            val = 26

        for i in range(4, -1, -1):
            bits.append(1.0 if (val >> i) & 1 else 0.0)
    return torch.tensor(bits, dtype=torch.float32, device=device)


def bits_to_text(bits):
    """Convert binary back to text"""
    if isinstance(bits, torch.Tensor):
        bits = bits.detach().cpu().numpy()
    
    chars = []
    for i in range(0, len(bits), 5):
        chunk = bits[i:i+5]

        val = 0
        for j, bit in enumerate(chunk):
            if bit > 0:
                val |= (1 << (4 - j))
        val = min(26, val)
        
        if val == 26:
            chars.append(' ')
        else:
            chars.append(chr(val + ord('a')))
    return ''.join(chars)


def generate_random_message():
    """Generate random message from word list"""
    return word_list[np.random.randint(0, len(word_list))]


class ImprovedNetwork(nn.Module):
    """Improved multi-layer network with proper architecture"""
    def __init__(self, input_size, hidden_size, output_size, name="net"):
        super(ImprovedNetwork, self).__init__()
        self.name = name
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        self.dropout = nn.Dropout(0.1)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Xavier initialization for better gradient flow"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass with multiple layers"""
        x = torch.tanh(self.fc1(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        # trying to go raw...
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
    """Train Alice and Bob - now improved"""
    print("=" * 70)
    print("IMPROVED NEURAL CRYPTO POC - Improved logic")
    print("=" * 70)
    
    BIT_LENGTH = MESSAGE_LENGTH * 5
    HIDDEN_SIZE = 256
    
    if load and os.path.exists('key.npy'):
        key_np = np.load('key.npy')
        key = torch.tensor(key_np, dtype=torch.float32, device=device)
        print(f"Loaded {len(key)}-bit key\n")
    else:
        key_np = np.random.choice([0.0, 1.0], KEY_SIZE * 5)
        np.save('key.npy', key_np)
        key = torch.tensor(key_np, dtype=torch.float32, device=device)
        print(f"Generated {len(key)}-bit key\n")
    
    alice = ImprovedNetwork(BIT_LENGTH + len(key), HIDDEN_SIZE, BIT_LENGTH, "Alice").to(device)
    bob = ImprovedNetwork(BIT_LENGTH + len(key), HIDDEN_SIZE, BIT_LENGTH, "Bob").to(device)

    if load and os.path.exists('alice_improved.pth') and os.path.exists('bob_improved.pth'):
        print("Loading saved networks...")
        alice.load('alice_improved.pth')
        bob.load('bob_improved.pth')
        print("Loaded!\n")
    
    print(" Alice and Bob initialized")
    print(f"  Message: {BIT_LENGTH} bits")
    print(f"  Key: {len(key)} bits")
    print(f"  Hidden: {HIDDEN_SIZE} units")
    print(f"  Architecture: Input → {HIDDEN_SIZE} → {HIDDEN_SIZE} → Output\n")
    
    criterion = nn.BCEWithLogitsLoss()
    bob_optimizer = optim.Adam(bob.parameters(), lr=0.001, weight_decay=1e-5)
    alice_optimizer = optim.Adam(alice.parameters(), lr=0.001, weight_decay=1e-5)
    
    bob_scheduler = optim.lr_scheduler.ReduceLROnPlateau(bob_optimizer, mode='min', factor=0.5, patience=5000)
    alice_scheduler = optim.lr_scheduler.ReduceLROnPlateau(alice_optimizer, mode='min', factor=0.5, patience=5000)

    print(f"Training for {TRAINING_EPISODES} episodes...")
    print("=" * 70)
    
    bob_errors = []
    perfect_count = 0
    
    for episode in range(TRAINING_EPISODES):
        plaintext = generate_random_message()
        plain_bits = text_to_bits(plaintext)
        
        alice.train()
        bob.train()

        alice_input = torch.cat([plain_bits, key])
        ciphertext_old = alice(alice_input)

        ciphertext = torch.sigmoid(ciphertext_old)

        bob_input = torch.cat([ciphertext, key])
        decrypted_bits = bob(bob_input)

        loss = criterion(decrypted_bits, plain_bits)
        bob_errors.append(loss.item())

        bob_optimizer.zero_grad()
        alice_optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(bob.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(alice.parameters(), 1.0)

        bob_optimizer.step()
        alice_optimizer.step()
        
        if episode % 1000 == 0:
            avg_error = np.mean(bob_errors[-1000:]) if len(bob_errors) >= 1000 else np.mean(bob_errors)
            bob_scheduler.step(avg_error)
            alice_scheduler.step(avg_error)
        
        with torch.no_grad():
            decrypted_text = bits_to_text(decrypted_bits)
            if plaintext == decrypted_text:
                perfect_count += 1
        
        if (episode + 1) % 2500 == 0:
            avg_error = np.mean(bob_errors[-2500:])
            recent_perfect = perfect_count
            perfect_count = 0
            
            print(f"\nEpisode {episode + 1}/{TRAINING_EPISODES}")
            print(f"  Avg Bob Error: {avg_error:.6f}")
            print(f"  Perfect (last 2500): {recent_perfect} ({100*recent_perfect/2500:.1f}%)")
            print(f"  Last example:")
            print(f"    Original:  '{plaintext}'")
            print(f"    Decrypted: '{decrypted_text}'")
            ciphertext_readable = bits_to_text(ciphertext_old)
            print(f"    Encrypted: '{ciphertext_readable}'")
            
            if (episode + 1) % 10000 == 0:
                alice.eval()
                bob.eval()
                print(f"\n  Testing standard words:")
                test_words = ["hello world     ", "test coding     ", "neural nets     ", 
                              "crypto proof    ", "python code     ", "simple test     "]
                correct = 0
                with torch.no_grad():
                    for word in test_words:
                        pb = text_to_bits(word)
                        ai = torch.cat([pb, key])
                        ciph = alice(ai)
                        ciph_bob = torch.sigmoid(ciph)
                        bi = torch.cat([ciph_bob, key])
                        dec_b = bob(bi)
                        dec = bits_to_text(dec_b)
                        match = "YES:" if dec == word else "NO:"
                        print(f"    {match} '{word}' → '{dec}'")
                        if dec == word:
                            correct += 1
                print(f"  Score: {correct}/{len(test_words)} ({100*correct/len(test_words):.0f}%)")
                
                if correct == len(test_words) and recent_perfect > 2480:
                    print(f"\n Perfect performance achieved! Stopping early at episode {episode + 1}")
                    break
    
    print("\n" + "=" * 70)
    print("Saving networks...")
    alice.save('alice_improved.pth')
    bob.save('bob_improved.pth')
    print("Saved!\n")
    
    print("=" * 70)
    print("FINAL TEST - Random Strings + Known Words")
    print("=" * 70)
    
    alice.eval()
    bob.eval()
    
    test_words = [''.join([string.ascii_lowercase[np.random.randint(0, 26)] 
                           for _ in range(MESSAGE_LENGTH)]) for _ in range(50)]
    test_words += ["hello world     ", "test coding     ", "neural nets     ", 
                   "crypto proof    ", "python code     ", "simple test     ",
                   "machine learning", "deep neural nets",
                   "alice and bob   ", "encryption works"]
    correct = 0

    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for word in test_words:
            pb = text_to_bits(word)
            ai = torch.cat([pb, key])
            ciph = alice(ai)
            ciph_bob = torch.sigmoid(ciph)
            bi = torch.cat([ciph_bob, key])
            dec_b = bob(bi)
            dec = bits_to_text(dec_b)
            
            error = criterion(dec_b, pb).item()
            match = "YES:" if dec == word else "NO:"
            
            if dec != word or error > 0.01:
                print(f"{match} '{word}' → '{dec}' | Error: {error:.4f}")
            if dec == word:
                correct += 1
    
    print(f"\n{'=' * 70}")
    print(f"Final Score: {correct}/{len(test_words)} ({100*correct/len(test_words):.1f}%)")
    print(f"{'=' * 70}")


def test_saved():
    """Test saved networks"""
    if not os.path.exists('alice_improved.pth') or not os.path.exists('bob_improved.pth'):
        print("No saved networks found. Train first.")
        return
    
    print("Loading networks...")
    key_np = np.load('key.npy')
    key = torch.tensor(key_np, dtype=torch.float32, device=device)
    
    BIT_LENGTH = MESSAGE_LENGTH * 5
    HIDDEN_SIZE = 256
    alice = ImprovedNetwork(BIT_LENGTH + len(key), HIDDEN_SIZE, BIT_LENGTH, "Alice").to(device)
    bob = ImprovedNetwork(BIT_LENGTH + len(key), HIDDEN_SIZE, BIT_LENGTH, "Bob").to(device)
    alice.load('alice_improved.pth')
    bob.load('bob_improved.pth')
    
    alice.eval()
    bob.eval()
    
    print("Loaded\n")
    print("=" * 70)
    print("TESTING")
    print("=" * 70)
    
    test_words = [string.ascii_lowercase[i:i+MESSAGE_LENGTH] 
                  for i in range(0, 26 - MESSAGE_LENGTH + 1)]
    test_words += ["hello world     ", "neural network  ", "deep learning   "]
    
    criterion = nn.BCEWithLogitsLoss()
    correct = 0
    
    with torch.no_grad():
        for word in test_words:
            pb = text_to_bits(word)
            ai = torch.cat([pb, key])
            ciph = alice(ai)
            ciph_bob = torch.sigmoid(ciph)
            bi = torch.cat([ciph_bob, key])
            dec_b = bob(bi)
            dec = bits_to_text(dec_b)
            
            error = criterion(dec_b, pb).item()
            match = "YES:" if dec == word else "NO:"
            print(f"{match} '{word}' → '{dec}' | Error: {error:.6f}")
            if dec == word:
                correct += 1
    
    print(f"\n{'=' * 70}")
    print(f"Score: {correct}/{len(test_words)} ({100*correct/len(test_words):.0f}%)")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
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
        print("Starting improved neural crypto training...")
        print("Hopefully this improved version will finally work?")
        print("=" * 70)
        train()
        epoch = int(sys.argv[1]) if len(sys.argv) > 1 else 0
        for e in range(epoch):
            train(load=True)
