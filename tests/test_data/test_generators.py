"""Tests for NeuroSync.data.generators."""
import os
import pytest
from NeuroSync.data.generators import MessageGenerator, generate_random_messages


class TestMessageGenerator:
    def test_generate_empty_raises(self):
        with pytest.raises(ValueError):
            MessageGenerator().generate()

    def test_generate_batch_empty_raises(self):
        with pytest.raises(ValueError):
            MessageGenerator().generate_batch(5)

    def test_generate(self):
        assert MessageGenerator(word_list=["word"]).generate() == "word"

    def test_generate_batch(self):
        assert len(MessageGenerator(word_list=["a", "b"]).generate_batch(10)) == 10

    def test_load_words(self, small_word_file):
        gen = MessageGenerator()
        gen.load_words(small_word_file)
        assert len(gen.word_list) == 7

    def test_create_word_list(self, tmp_dir):
        gen = MessageGenerator()
        gen.create_word_list(os.path.join(tmp_dir, "gen.txt"), num_words=20)
        assert len(gen.word_list) == 20


class TestGenerateRandomMessages:
    def test_basic(self):
        words = ["a" * 16, "b" * 16]
        msgs = generate_random_messages(words, batch_size=5)
        assert len(msgs) == 5
