"""
test_tokenizer.py
=================
Tests for miniGPT/tokenizer.py -- the CharTokenizer class.

Covers: vocabulary creation, encode, decode, one_hot, save/load, edge cases.
Every test calls real tokenizer methods on real data -- no trivial assertions.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'miniGPT'))
from tokenizer import CharTokenizer


# =============================================================================
#  Vocabulary
# =============================================================================

class TestVocabulary(unittest.TestCase):

    def test_unique_chars_only(self):
        """Vocab contains each unique character exactly once."""
        tok = CharTokenizer("aaabbbccc")
        self.assertEqual(tok.size, 3)
        self.assertEqual(len(tok.vocab), 3)

    def test_vocab_is_sorted(self):
        """Vocab should be in sorted order (deterministic across runs)."""
        tok = CharTokenizer("dcba")
        self.assertEqual(tok.vocab, sorted(set("dcba")))

    def test_size_matches_vocab(self):
        """tok.size must always equal len(tok.vocab)."""
        for text in ["a", "hello", "Hello, World! 123"]:
            tok = CharTokenizer(text)
            self.assertEqual(tok.size, len(tok.vocab))

    def test_ch2idx_covers_full_vocab(self):
        """Every vocab character must appear in ch2idx."""
        tok = CharTokenizer("hello world")
        for ch in tok.vocab:
            self.assertIn(ch, tok.ch2idx)

    def test_idx2ch_covers_all_indices(self):
        """Every index 0..size-1 must appear in idx2ch."""
        tok = CharTokenizer("hello world")
        for i in range(tok.size):
            self.assertIn(i, tok.idx2ch)

    def test_ch2idx_idx2ch_are_inverses(self):
        """ch2idx and idx2ch must be exact inverses of each other."""
        tok = CharTokenizer("the quick brown fox")
        for ch in tok.vocab:
            self.assertEqual(tok.idx2ch[tok.ch2idx[ch]], ch)

    def test_all_chars_in_text_covered(self):
        """Every character in the source text must be in vocab."""
        text = "Hello, World!"
        tok = CharTokenizer(text)
        for ch in text:
            self.assertIn(ch, tok.vocab)


# =============================================================================
#  Encoding
# =============================================================================

class TestEncoding(unittest.TestCase):

    def setUp(self):
        self.text = "hello world"
        self.tok  = CharTokenizer(self.text)

    def test_encode_length_matches_input(self):
        """encode() returns one index per character."""
        encoded = self.tok.encode("hello")
        self.assertEqual(len(encoded), 5)

    def test_encode_returns_ints(self):
        """All encoded values must be Python ints."""
        for idx in self.tok.encode("hello"):
            self.assertIsInstance(idx, int)

    def test_encode_indices_in_range(self):
        """All encoded indices must be in [0, vocab_size)."""
        for idx in self.tok.encode(self.text):
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, self.tok.size)

    def test_encode_empty_string(self):
        """encode('') must return []."""
        self.assertEqual(self.tok.encode(""), [])

    def test_encode_unknown_chars_skipped(self):
        """Characters not in vocab are silently skipped."""
        tok = CharTokenizer("abc")
        result = tok.encode("axbxc")   # x not in vocab
        self.assertEqual(len(result), 3)

    def test_encode_is_deterministic(self):
        """Same input always produces same output."""
        self.assertEqual(
            self.tok.encode("hello"),
            self.tok.encode("hello"),
        )

    def test_encode_single_chars_match_ch2idx(self):
        """Single character encode must match ch2idx directly."""
        for ch in self.tok.vocab:
            self.assertEqual(self.tok.encode(ch), [self.tok.ch2idx[ch]])


# =============================================================================
#  Decoding
# =============================================================================

class TestDecoding(unittest.TestCase):

    def setUp(self):
        self.text = "hello world"
        self.tok  = CharTokenizer(self.text)

    def test_decode_empty(self):
        """decode([]) must return ''."""
        self.assertEqual(self.tok.decode([]), "")

    def test_decode_unknown_index_returns_question_mark(self):
        """Unknown indices are replaced with '?'."""
        result = self.tok.decode([9999])
        self.assertEqual(result, "?")

    def test_roundtrip_full_text(self):
        """encode then decode must recover the original text exactly."""
        encoded = self.tok.encode(self.text)
        decoded = self.tok.decode(encoded)
        self.assertEqual(decoded, self.text)

    def test_roundtrip_single_characters(self):
        """Roundtrip works for every individual vocab character."""
        for ch in self.tok.vocab:
            self.assertEqual(self.tok.decode(self.tok.encode(ch)), ch)

    def test_roundtrip_complex_text(self):
        """Roundtrip on text with punctuation, digits, whitespace."""
        text = "Hello, World! 123\n\t?"
        tok = CharTokenizer(text)
        self.assertEqual(tok.decode(tok.encode(text)), text)


# =============================================================================
#  One-hot encoding
# =============================================================================

class TestOneHot(unittest.TestCase):

    def setUp(self):
        self.tok = CharTokenizer("abcde")

    def test_length_equals_vocab_size(self):
        """one_hot vector length must equal vocab_size."""
        for i in range(self.tok.size):
            self.assertEqual(len(self.tok.one_hot(i)), self.tok.size)

    def test_exactly_one_hot(self):
        """Sum of a one_hot vector must be exactly 1.0."""
        for i in range(self.tok.size):
            self.assertEqual(sum(self.tok.one_hot(i)), 1.0)

    def test_hot_position_is_correct(self):
        """The 1.0 must be at position idx."""
        for i in range(self.tok.size):
            vec = self.tok.one_hot(i)
            self.assertEqual(vec[i], 1.0)

    def test_all_other_positions_zero(self):
        """All positions other than idx must be 0.0."""
        for i in range(self.tok.size):
            vec = self.tok.one_hot(i)
            for j, val in enumerate(vec):
                if j != i:
                    self.assertEqual(val, 0.0)

    def test_values_are_floats(self):
        """one_hot values must be floats, not ints."""
        vec = self.tok.one_hot(0)
        for val in vec:
            self.assertIsInstance(val, float)


# =============================================================================
#  Save / Load
# =============================================================================

class TestSaveLoad(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".json"
        ).name

    def tearDown(self):
        if os.path.exists(self.tmp):
            os.remove(self.tmp)

    def test_save_creates_valid_json(self):
        """save() must write a valid JSON file."""
        tok = CharTokenizer("hello world")
        tok.save(self.tmp)
        with open(self.tmp) as f:
            data = json.load(f)
        self.assertIn("vocab", data)

    def test_load_restores_vocab(self):
        """load() must restore the exact same vocab."""
        tok = CharTokenizer("hello world")
        tok.save(self.tmp)
        tok2 = CharTokenizer.load(self.tmp)
        self.assertEqual(tok.vocab, tok2.vocab)

    def test_load_restores_mappings(self):
        """load() must restore ch2idx and idx2ch correctly."""
        tok = CharTokenizer("hello world")
        tok.save(self.tmp)
        tok2 = CharTokenizer.load(self.tmp)
        self.assertEqual(tok.ch2idx, tok2.ch2idx)
        self.assertEqual(tok.idx2ch, tok2.idx2ch)

    def test_loaded_tokenizer_encodes_same(self):
        """A loaded tokenizer must encode text identically to original."""
        tok = CharTokenizer("hello world")
        tok.save(self.tmp)
        tok2 = CharTokenizer.load(self.tmp)
        self.assertEqual(tok.encode("hello"), tok2.encode("hello"))

    def test_loaded_tokenizer_decodes_same(self):
        """A loaded tokenizer must decode indices identically to original."""
        tok = CharTokenizer("hello world")
        encoded = tok.encode("hello")
        tok.save(self.tmp)
        tok2 = CharTokenizer.load(self.tmp)
        self.assertEqual(tok.decode(encoded), tok2.decode(encoded))

    def test_len_dunder(self):
        """len(tok) must equal tok.size."""
        tok = CharTokenizer("hello world")
        self.assertEqual(len(tok), tok.size)

    def test_repr_contains_size(self):
        """repr(tok) must mention the vocab size."""
        tok = CharTokenizer("hello world")
        self.assertIn(str(tok.size), repr(tok))


if __name__ == "__main__":
    unittest.main()