import numpy as np
from typing import List, Optional

from NeuroSync.data.loaders import WordListLoader
from NeuroSync.encoding.constants import MESSAGE_LENGTH

class MessageGenerator:
    def __init__(
            self,
            word_list: Optional[List[str]] = None,
            message_length: int = MESSAGE_LENGTH
    ):
        self.message_length = message_length
        self.word_list = word_list or []
    
    def load_words(self, filepath: str) -> None:
        loader = WordListLoader(self.message_length)
        self.word_list = loader.load(filepath)

    def generate(self) -> str:
        if not self.word_list:
            raise ValueError("Word list is empty. Load words before generating messages.")
        return self.word_list[np.random.randint(0, len(self.word_list))]
    
    def generate_batch(self, batch_size: int) -> List[str]:
        if not self.word_list:
            raise ValueError("Word list is empty. Load words before generating messages.")
        return [self.generate() for _ in range(batch_size)]


def generate_random_messages(
        word_list: List[str],
        batch_size: int,
        message_length: int = MESSAGE_LENGTH
) -> List[str]:
    generator = MessageGenerator(word_list, message_length)
    return generator.generate_batch(batch_size)