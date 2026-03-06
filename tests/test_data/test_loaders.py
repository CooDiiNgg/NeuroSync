"""Tests for NeuroSync.data.loaders."""
import os
from NeuroSync.data.loaders import WordListLoader, load_word_list, create_word_list


class TestWordListLoader:
    def test_load(self, small_word_file):
        loader = WordListLoader(16)
        words = loader.load(small_word_file)
        assert len(words) == 7
        assert all(len(w) == 16 for w in words)

    def test_padding(self, small_word_file):
        words = WordListLoader(16).load(small_word_file)
        assert words[0] == "hello==========="

    def test_len(self, small_word_file):
        loader = WordListLoader(16)
        loader.load(small_word_file)
        assert len(loader) == 7

    def test_filter_long(self, tmp_dir):
        fp = os.path.join(tmp_dir, "long.txt")
        with open(fp, "w") as f:
            f.write("short\n" + "a" * 100 + "\n" + "ok\n")
        assert len(WordListLoader(16).load(fp)) == 2


class TestCreateWordList:
    def test_create(self, tmp_dir):
        fp = os.path.join(tmp_dir, "gen.txt")
        words = create_word_list(fp, message_length=16, num_words=50)
        assert len(words) == 50
        assert all(len(w) == 16 for w in words)
