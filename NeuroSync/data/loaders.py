"""
Data loading utilities for NeuroSync.
"""

from typing import List, Optional
import random
import string

from NeuroSync.encoding.constants import MESSAGE_LENGTH

class WordListLoader:
    """Loads and manages word lists for training"""

    def __init__(self, message_length: int = MESSAGE_LENGTH):
        self.message_length = message_length
        self._words: List[str] = []

    def load(self, filepath: str, pad_char: str = '=') -> List[str]:
        """
        Loads words from file and pads them to the specified message length.
        
        Args:
            filepath: Path to word list file
            pad_char: Character to use for padding
        
        Returns:
            List of padded words
        """
        with open(filepath, 'r', encoding='utf-8') as file:
            self._words = [
                line.strip().ljust(self.message_length, pad_char)
                for line in file
                if len(line.strip()) <= self.message_length
            ]
        return self._words
    
    @property
    def words(self) -> List[str]:
        return self._words
    
    def __len__(self) -> int:
        return len(self._words)


def load_word_list(
    filepath: str,
    message_length: int = MESSAGE_LENGTH,
    pad_char: str = '='
) -> List[str]:
    """
    Convenience function to load a word list.
    
    Args:
        filepath: Path to word list file
        message_length: Target message length
        pad_char: Padding character
    
    Returns:
        List of padded words
    """
    loader = WordListLoader(message_length)
    return loader.load(filepath, pad_char)

def create_word_list(
        filepath: str,
        message_length: int = MESSAGE_LENGTH,
        num_words: int = 1_000_000
) -> List[str]:
    """
    Creates a word list file with random alphanumeric words and loads it.

    Args:
        filepath: Path to save the word list file
        message_length: Length of each word
        num_words: Number of words to generate
    
    Returns:
        List of generated words
    """
    for _ in range(num_words):
        word = ''.join(random.choices(string.ascii_letters + string.digits, k=message_length))
        with open(filepath, 'a', encoding='utf-8') as file:
            file.write(word + '\n')
    loader = WordListLoader(message_length)
    return loader.load(filepath)