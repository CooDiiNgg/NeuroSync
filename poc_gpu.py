import re
import torch
import torch.nn as nn
import torch.optim as optim
import string
import os
import numpy as np

word_list = []
with open("./words.txt", "r") as f:
        word_list = [line.strip() + " " * (MESSAGE_LENGTH - len(line.strip())) 
                     for line in f if len(line.strip()) <= MESSAGE_LENGTH]
MESSAGE_LENGTH = 16
KEY_SIZE = 16
TRAINING_EPISODES = 5000000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"Moving model and tensors to 'cuda'.\n")
else:
    print(f"CUDA not available. Using CPU.\n")


def detect_and_escape_local_minimum(network, errors, window_size=10000, threshold=0.0005, perturbation_factor=0.1):
    """
    Detect if Bob is stuck in a local minimum and help it escape.
    """
    if len(errors) < window_size * 3:
        return False
    
    recent_avg = np.mean(errors[-window_size:])
    mid_avg = np.mean(errors[-2*window_size:-window_size])
    old_avg = np.mean(errors[-3*window_size:-2*window_size])
    
    recent_improvement = mid_avg - recent_avg
    previous_improvement = old_avg - mid_avg

    if (0 <= recent_improvement < threshold and
        previous_improvement > recent_improvement and
        recent_avg > 0.001 and
        np.std(errors[-window_size:]) < threshold * 50):
        
        print(f"\n[!] Bob appears stuck in local minimum. Recent avg error: {recent_avg:.6f}")
        print(f"    Previous improvements: {previous_improvement:.6f} → {recent_improvement:.6f}")
        print(f"    Applying gentle perturbation to help escape local minimum...")

        with torch.no_grad():
            for param in network.parameters():
                noise = torch.randn_like(param) * perturbation_factor * torch.std(param)
                param.add_(noise)
        
        return True
    
    return False


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
            bits.append(1.0 if (val >> i) & 1 else -1.0)
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


class SimpleNetwork(nn.Module):
    """Ultra-simple network: just one layer implemented as nn.Module"""
    def __init__(self, input_size, output_size, name="net"):
        super(SimpleNetwork, self).__init__()
        self.name = name
        
        # Define weight and bias as parameters
        self.w = nn.Parameter(torch.randn(input_size, output_size) * 0.1)
        self.b = nn.Parameter(torch.zeros(output_size))
        
    def forward(self, x):
        """Forward pass with tanh"""
        z = x @ self.w + self.b
        out = torch.tanh(z)
        return out
    
    def save(self, filename):
        """Save model state"""
        torch.save({
            'w': self.w.cpu(),
            'b': self.b.cpu()
        }, filename)
    
    def load(self, filename):
        """Load model state"""
        checkpoint = torch.load(filename, map_location=device, weights_only=True)
        self.w.data = checkpoint['w'].to(device)
        self.b.data = checkpoint['b'].to(device)


def train(load=False, prev_epochs=0):
    """Train Alice and Bob end-to-end with clear objective"""
    print("=" * 70)
    print("SIMPLIFIED NEURAL CRYPTO POC - Binary Representation (PyTorch)")
    print("=" * 70)
    
    BIT_LENGTH = MESSAGE_LENGTH * 5
    
    if load and os.path.exists('key.npy'):
        key_np = np.load('key.npy')
        key = torch.tensor(key_np, dtype=torch.float32, device=device)
        print(f"Loaded {len(key)}-bit key\n")
    else:
        key_np = np.random.choice([-1.0, 1.0], KEY_SIZE * 5)
        np.save('key.npy', key_np)
        key = torch.tensor(key_np, dtype=torch.float32, device=device)
        print(f"Generated {len(key)}-bit key\n")
    
    alice = SimpleNetwork(BIT_LENGTH + len(key), BIT_LENGTH, "Alice").to(device)
    bob = SimpleNetwork(BIT_LENGTH + len(key), BIT_LENGTH, "Bob").to(device)

    if load and os.path.exists('alice.pth') and os.path.exists('bob.pth'):
        print("Loading saved networks...")
        alice.load('alice.pth')
        bob.load('bob.pth')
        print("Loaded!\n")
    
    print(" Alice and Bob initialized")
    print(f"  Message: {BIT_LENGTH} bits")
    print(f"  Key: {len(key)} bits")
    print(f"  Total: {BIT_LENGTH + len(key)} bits input\n")
    
    criterion = nn.MSELoss()
    
    bob_start_learning_rate = 0.02
    alice_start_learning_rate = 0.01
    min_bob_lr = 0.0002
    min_alice_lr = 0.0001

    if prev_epochs > 0:
        bob_start_learning_rate *= (0.9 ** prev_epochs)
        alice_start_learning_rate *= (0.9 ** prev_epochs)
        print(f"Adjusted learning rates for previous {prev_epochs} epochs.")
        print(f"  Bob LR start: {bob_start_learning_rate:.6f}, Alice LR start: {alice_start_learning_rate:.6f}\n")
    
    bob_optimizer = optim.SGD(bob.parameters(), lr=bob_start_learning_rate)
    alice_optimizer = optim.SGD(alice.parameters(), lr=alice_start_learning_rate)
    
    print(f"Training for {TRAINING_EPISODES} episodes...")
    print("=" * 70)
    
    bob_errors = []
    perfect_count = 0

    for episode in range(TRAINING_EPISODES):

        progress = episode / float(TRAINING_EPISODES)
        bob_lr = bob_start_learning_rate * (1.0 - progress) + min_bob_lr * progress
        alice_lr = alice_start_learning_rate * (1.0 - progress) + min_alice_lr * progress
        
        for param_group in bob_optimizer.param_groups:
            param_group['lr'] = bob_lr
        for param_group in alice_optimizer.param_groups:
            param_group['lr'] = alice_lr


        plaintext = generate_random_message()
        plain_bits = text_to_bits(plaintext)
        

        alice_input = torch.cat([plain_bits, key])
        ciphertext = alice(alice_input)
        

        bob_input = torch.cat([ciphertext, key])
        decrypted_bits = bob(bob_input)
        
        bob_error = criterion(decrypted_bits, plain_bits)
        bob_errors.append(bob_error.item())

        bob_optimizer.zero_grad()
        bob_error.backward()
        bob_optimizer.step()

        alice_input = torch.cat([plain_bits, key])
        ciphertext_new = alice(alice_input)
        alice_loss = criterion(ciphertext_new, ciphertext.detach())
        
        alice_optimizer.zero_grad()
        alice_loss.backward()
        alice_optimizer.step()
        
        decrypted_text = bits_to_text(decrypted_bits)
        if plaintext == decrypted_text:
            perfect_count += 1
        
        if (episode + 1) % 2500 == 0:
            avg_error = np.mean(bob_errors[-2500:])
            recent_perfect = perfect_count
            perfect_count = 0
            
            print(f"\nEpisode {episode + 1}/{TRAINING_EPISODES}")
            print(f"  Avg Bob Error: {avg_error:.4f}")
            print(f"  Perfect (last 2500): {recent_perfect}")
            print(f"  Last example:")
            print(f"    Original:  '{plaintext}'")
            print(f"    Decrypted: '{decrypted_text}'")
            print(f"    Ciphertext bits_to_text: '{bits_to_text(ciphertext)}'")
            
            if (episode + 1) >= 25000:
                perturbation_applied = detect_and_escape_local_minimum(bob, bob_errors)
                if perturbation_applied:
                    print(f"    Applied perturbation to Bob's weights to escape local minimum.")
            
            if (episode + 1) % 10000 == 0:
                print(f"\n  Testing all training words:")
                test_words = ["hello world     ", "test coding     ", "neural nets     ", 
                              "crypto proof    ", "python code     ", "simple test     "]
                correct = 0
                with torch.no_grad():
                    for word in test_words:
                        pb = text_to_bits(word)
                        ai = torch.cat([pb, key])
                        ciph = alice(ai)
                        bi = torch.cat([ciph, key])
                        dec_b = bob(bi)
                        dec = bits_to_text(dec_b)
                        match = "YES:" if dec == word else "NO:"
                        print(f"    {match} '{word}' → '{dec}'")
                        if dec == word:
                            correct += 1
                print(f"  Score: {correct}/{len(test_words)}")
    
    print("\n" + "=" * 70)
    print("Saving networks...")
    alice.save('alice.pth')
    bob.save('bob.pth')
    print("Saved!\n")
    
    print("=" * 70)
    print("FINAL TEST")
    print("=" * 70)
    
    test_words = [''.join([string.ascii_lowercase[np.random.randint(0, 26)] 
                           for _ in range(MESSAGE_LENGTH)]) for _ in range(100)]
    test_words += ["hello world     ", "test coding     ", "neural nets     ", 
                   "crypto proof    ", "python code     ", "simple test     "]
    correct = 0
    
    with torch.no_grad():
        for word in test_words:
            pb = text_to_bits(word)
            ai = torch.cat([pb, key])
            ciph = alice(ai)
            bi = torch.cat([ciph, key])
            dec_b = bob(bi)
            dec = bits_to_text(dec_b)
            
            error = criterion(dec_b, pb).item()
            match = "YES:" if dec == word else "NO:"
            
            print(f"{match} '{word}' → '{dec}' | Error: {error:.4f}")
            if dec == word:
                correct += 1
    
    print(f"\n{'=' * 70}")
    print(f"Final Score: {correct}/{len(test_words)} ({100*correct/len(test_words):.0f}%)")
    print(f"{'=' * 70}")


def test_saved():
    """Test saved networks"""
    if not os.path.exists('alice.pth') or not os.path.exists('bob.pth'):
        print("No saved networks found. Train first.")
        return
    
    print("Loading networks...")
    key_np = np.load('key.npy')
    key = torch.tensor(key_np, dtype=torch.float32, device=device)
    
    BIT_LENGTH = MESSAGE_LENGTH * 5
    alice = SimpleNetwork(BIT_LENGTH + len(key), BIT_LENGTH, "Alice").to(device)
    bob = SimpleNetwork(BIT_LENGTH + len(key), BIT_LENGTH, "Bob").to(device)
    alice.load('alice.pth')
    bob.load('bob.pth')
    
    print("Loaded\n")
    print("=" * 70)
    print("TESTING")
    print("=" * 70)
    
    test_words = [string.ascii_lowercase[i:i+MESSAGE_LENGTH] 
                  for i in range(0, 26 - MESSAGE_LENGTH + 1)]
    
    with torch.no_grad():
        for word in test_words:
            pb = text_to_bits(word)
            ai = torch.cat([pb, key])
            ciph = alice(ai)
            bi = torch.cat([ciph, key])
            dec_b = bob(bi)
            dec = bits_to_text(dec_b)

            match = "YES:" if dec == word else "NO:"
            print(f"{match} '{word}' → '{dec}'")


if __name__ == "__main__":
    import sys
    epoch = 1
    prev_epochs = 0

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_saved()
    elif len(sys.argv) > 1 and sys.argv[1] == "load":
        if len(sys.argv) > 2 and re.match(r"^\d+$", sys.argv[2]):
            epoch = int(sys.argv[2])
            if len(sys.argv) > 3 and re.match(r"^\d+$", sys.argv[3]):
                prev_epochs = int(sys.argv[3])
        print("Now trying a different method by retraining from saved networks...")
        for e in range(epoch):
            print(f"--- Epoch {e + 1}/{epoch} ---")
            train(load=True, prev_epochs=prev_epochs)
            prev_epochs += 1
    else:
        if len(sys.argv) > 1 and re.match(r"^\d+$", sys.argv[1]):
            epoch = int(sys.argv[1])
        print("Hope for the best mates...")
        print("=" * 70)
        train()
        epoch -= 1
        prev_epochs = 1
        for e in range(epoch):
            print(f"--- Epoch {e + 1}/{epoch} ---")
            train(load=True, prev_epochs=prev_epochs)
            prev_epochs += 1
        print("=" * 70)
