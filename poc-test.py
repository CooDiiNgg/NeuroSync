import re
import numpy as np
import string
import os


MESSAGE_LENGTH = 16
KEY_SIZE = 16
TRAINING_EPISODES = 5000000

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

        error_mask = np.abs(network.last_error) > np.mean(recent_avg)
        noise_w = np.random.randn(*network.w.shape) * perturbation_factor * np.std(network.w)
        noise_b = np.random.randn(*network.b.shape) * perturbation_factor * np.std(network.b)
        
        noise_w = noise_w * error_mask
        noise_b = noise_b * error_mask
        
        network.w += noise_w
        network.b += noise_b
        
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
    return np.array(bits, dtype=np.float32)

def bits_to_text(bits):
    """Convert binary back to text"""
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
    # messages = [''.join([string.ascii_lowercase[np.random.randint(0, 26)] for _ in range(MESSAGE_LENGTH)]) for _ in range(100)]
    # # now some random actual random real words from some random word list i found online... ;)
    # if np.random.rand() < 0.5:
    #     with open("./words.txt", "r") as f:
    #         word_list = [line.strip() for line in f if len(line.strip()) == MESSAGE_LENGTH]
    #     messages = [word_list[np.random.randint(0, len(word_list))] for _ in range(100)]
    # return np.random.choice(messages)
    # actually i need repetition so that it doesnt just learn on noise and is actually consistent (need to get a better wordlist though)
    with open("./words.txt", "r") as f:
        word_list = [line.strip() + " " * (MESSAGE_LENGTH - len(line.strip())) for line in f if len(line.strip()) <= MESSAGE_LENGTH]
    return word_list[np.random.randint(0, len(word_list))]

class SimpleNetwork:
    """Ultra-simple network: just one layer - later maybe more"""
    def __init__(self, input_size, output_size, name="net"):
        self.name = name

        self.w = np.random.randn(input_size, output_size) * 0.1
        self.b = np.zeros(output_size)
        self.last_error = np.zeros(output_size)
        
    def forward(self, x):
        """Forward pass with tanh"""
        self.x = x
        self.z = x @ self.w + self.b
        self.out = np.tanh(self.z)
        return self.out
    
    def update(self, target, learning_rate=0.08):
        """Simple supervised update - for now maybe one layer - late prolly more"""
        error = target - self.out
        self.last_error = error
        
        d_out = error * (1 - self.out ** 2)
        d_w = np.outer(self.x, d_out)
        d_b = d_out
        
        self.w += learning_rate * d_w
        self.b += learning_rate * d_b
        
        return np.mean(error ** 2)
    
    def save(self, filename):
        np.savez(filename, w=self.w, b=self.b)
    
    def load(self, filename):
        data = np.load(filename)
        self.w = data['w']
        self.b = data['b']

def train(load=False, prev_epochs=0):
    """Train Alice and Bob end-to-end with clear objective"""
    print("=" * 70)
    print("SIMPLIFIED NEURAL CRYPTO POC - Binary Representation")
    print("=" * 70)
    
    BIT_LENGTH = MESSAGE_LENGTH * 5
    if load and os.path.exists('key.npy'):
        key = np.load('key.npy')
        print(f"Loaded {len(key)}-bit key\n")
    else:
        key = np.random.choice([-1.0, 1.0], KEY_SIZE * 5)
        np.save('key.npy', key)
        print(f"Generated {len(key)}-bit key\n")
    
    alice = SimpleNetwork(BIT_LENGTH + len(key), BIT_LENGTH, "Alice")
    bob = SimpleNetwork(BIT_LENGTH + len(key), BIT_LENGTH, "Bob")

    if load and os.path.exists('alice.npz') and os.path.exists('bob.npz'):
        print("Loading saved networks...")
        alice.load('alice.npz')
        bob.load('bob.npz')
        print("Loaded!\n")
    
    print(" Alice and Bob initialized")
    print(f"  Message: {BIT_LENGTH} bits")
    print(f"  Key: {len(key)} bits")
    print(f"  Total: {BIT_LENGTH + len(key)} bits input\n")
    
    print(f"Training for {TRAINING_EPISODES} episodes...")
    print("=" * 70)
    
    bob_errors = []
    perfect_count = 0

    bob_start_learning_rate = 0.02
    alice_start_learning_rate = 0.01
    min_bob_lr = 0.0002
    min_alice_lr = 0.0001

    if prev_epochs > 0:
        bob_start_learning_rate *= (0.9 ** prev_epochs)
        alice_start_learning_rate *= (0.9 ** prev_epochs)
        print(f"Adjusted learning rates for previous {prev_epochs} epochs.")
        print(f"  Bob LR start: {bob_start_learning_rate:.6f}, Alice LR start: {alice_start_learning_rate:.6f}\n")

    for episode in range(TRAINING_EPISODES):

        progress = episode / float(TRAINING_EPISODES)
        bob_lr = bob_start_learning_rate * (1.0 - progress) + min_bob_lr * progress
        alice_lr = alice_start_learning_rate * (1.0 - progress) + min_alice_lr * progress

        plaintext = generate_random_message()
        plain_bits = text_to_bits(plaintext)
        
        alice_input = np.concatenate([plain_bits, key])
        ciphertext = alice.forward(alice_input)
        
        bob_input = np.concatenate([ciphertext, key])
        decrypted_bits = bob.forward(bob_input)
        
        bob_error = np.mean((plain_bits - decrypted_bits) ** 2)
        bob_errors.append(bob_error)

        bob.update(plain_bits, learning_rate=bob_lr)

        feedback = ciphertext
        # if np.random.rand() < 0.001:  # 0.1% chance to simulate Eves attack
        #     # change it so that it has a bad reward and makes the cipher more secure
        #     feedback = np.random.choice([-1.0, 1.0], len(ciphertext))
        #     print("Eves attack simulated!")
        #     print(f"Original ciphertext bits_to_text: '{bits_to_text(ciphertext)}'")
        #     print(f"Altered ciphertext bits_to_text:  '{bits_to_text(feedback)}'\n")

        alice.update(feedback, learning_rate=alice_lr)
        
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
                test_words = ["hello world     ", "test coding     ", "neural nets     ", "crypto proof    ", "python code     ", "simple test     "]
                correct = 0
                for word in test_words:
                    pb = text_to_bits(word)
                    ai = np.concatenate([pb, key])
                    ciph = alice.forward(ai)
                    bi = np.concatenate([ciph, key])
                    dec_b = bob.forward(bi)
                    dec = bits_to_text(dec_b)
                    match = "YES:" if dec == word else "NO:"
                    print(f"    {match} '{word}' → '{dec}'")
                    if dec == word:
                        correct += 1
                print(f"  Score: {correct}/{len(test_words)}")
    
    print("\n" + "=" * 70)
    print("Saving networks...")
    alice.save('alice.npz')
    bob.save('bob.npz')
    print("Saved!\n")
    
    print("=" * 70)
    print("FINAL TEST")
    print("=" * 70)
    
    test_words = [''.join([string.ascii_lowercase[np.random.randint(0, 26)] for _ in range(MESSAGE_LENGTH)]) for _ in range(100)]
    test_words += ["hello world     ", "test coding     ", "neural nets     ", "crypto proof    ", "python code     ", "simple test     "]
    correct = 0
    
    for word in test_words:
        pb = text_to_bits(word)
        ai = np.concatenate([pb, key])
        ciph = alice.forward(ai)
        bi = np.concatenate([ciph, key])
        dec_b = bob.forward(bi)
        dec = bits_to_text(dec_b)
        
        error = np.mean((pb - dec_b) ** 2)
        match = "YES:" if dec == word else "NO:"
        
        print(f"{match} '{word}' → '{dec}' | Error: {error:.4f}")
        if dec == word:
            correct += 1
    
    print(f"\n{'=' * 70}")
    print(f"Final Score: {correct}/{len(test_words)} ({100*correct/len(test_words):.0f}%)")
    print(f"{'=' * 70}")

def test_saved():
    """Test saved networks"""
    if not os.path.exists('alice.npz') or not os.path.exists('bob.npz'):
        print("No saved networks found. Train first.")
        return
    
    print("Loading networks...")
    key = np.load('key.npy')
    
    BIT_LENGTH = MESSAGE_LENGTH * 5
    alice = SimpleNetwork(BIT_LENGTH + len(key), BIT_LENGTH, "Alice")
    bob = SimpleNetwork(BIT_LENGTH + len(key), BIT_LENGTH, "Bob")
    alice.load('alice.npz')
    bob.load('bob.npz')
    
    print("Loaded\n")
    print("=" * 70)
    print("TESTING")
    print("=" * 70)
    
    test_words = [string.ascii_lowercase[i:i+MESSAGE_LENGTH] for i in range(0, 26 - MESSAGE_LENGTH + 1)]
    
    for word in test_words:
        pb = text_to_bits(word)
        ai = np.concatenate([pb, key])
        ciph = alice.forward(ai)
        bi = np.concatenate([ciph, key])
        dec_b = bob.forward(bi)
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