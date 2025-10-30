import string
import random

def generate_random_wordlist(word_count=80000000, word_length=16, output_file='wordlist_temp.txt'):
    characters = string.ascii_letters
    with open(output_file, 'w') as f:
        for _ in range(word_count):
            word = ''.join(random.choices(characters, k=word_length))
            f.write(word + '\n')

if __name__ == "__main__":
    generate_random_wordlist()