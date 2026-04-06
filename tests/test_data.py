"""
test_data.py
============
Tests for miniGPT/data.py -- text loading, cleaning, and sample building.

Covers: simplify_text, load_text, make_samples, make_index_arrays.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'miniGPT'))
from data import load_text, make_index_arrays, make_samples, simplify_text
from tokenizer import CharTokenizer


# =============================================================================
#  simplify_text
# =============================================================================

class TestSimplifyText(unittest.TestCase):

    def test_lowercases(self):
        """All output should be lowercase."""
        self.assertEqual(simplify_text("HELLO"), "hello")

    def test_keeps_letters(self):
        """Letters a-z are preserved."""
        result = simplify_text("hello")
        self.assertEqual(result, "hello")

    def test_removes_digits(self):
        """Digits are stripped."""
        result = simplify_text("abc123def")
        self.assertNotIn("1", result)
        self.assertNotIn("2", result)
        self.assertNotIn("3", result)

    def test_keeps_basic_punctuation(self):
        """Comma, period, !, ?, apostrophe, hyphen are kept."""
        result = simplify_text("hello, world. why? yes! can't-do")
        for ch in ",.?!'-":
            self.assertIn(ch, result)

    def test_removes_special_chars(self):
        """@ # $ % ^ & * are stripped."""
        result = simplify_text("a@b#c$d")
        for ch in "@#$%^&*":
            self.assertNotIn(ch, result)

    def test_collapses_whitespace(self):
        """Multiple spaces and newlines collapse to single space."""
        result = simplify_text("a   b\n\n\nc")
        self.assertNotIn("  ", result)    # no double space
        self.assertNotIn("\n\n", result)  # no double newline

    def test_empty_string(self):
        """Empty input returns empty output."""
        self.assertEqual(simplify_text(""), "")

    def test_only_special_chars_returns_empty(self):
        """Input with no valid chars returns empty string."""
        self.assertEqual(simplify_text("@#$%"), "")

    def test_strips_leading_trailing_whitespace(self):
        """Output has no leading or trailing whitespace."""
        result = simplify_text("  hello  ")
        self.assertEqual(result, result.strip())


# =============================================================================
#  load_text
# =============================================================================

class TestLoadText(unittest.TestCase):

    def _write(self, content: str) -> str:
        """Write content to a temp file and return its path."""
        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        self.addCleanup(os.remove, path)
        return path

    def test_loads_content(self):
        """Loaded text must match file contents."""
        path = self._write("hello world")
        self.assertEqual(load_text(path), "hello world")

    def test_max_chars_limits_output(self):
        """max_chars cap must be respected."""
        path = self._write("a" * 1000)
        result = load_text(path, max_chars=100)
        self.assertLessEqual(len(result), 100)

    def test_max_chars_larger_than_file(self):
        """max_chars larger than file returns whole file."""
        content = "hello"
        path = self._write(content)
        self.assertEqual(load_text(path, max_chars=10_000), content)

    def test_empty_file(self):
        """Empty file returns empty string."""
        path = self._write("")
        self.assertEqual(load_text(path), "")

    def test_multiline(self):
        """Newlines are preserved."""
        content = "line1\nline2\nline3"
        path = self._write(content)
        self.assertEqual(load_text(path), content)

    def test_missing_file_raises(self):
        """Non-existent file raises FileNotFoundError or similar."""
        with self.assertRaises(Exception):
            load_text("/nonexistent/path/file.txt")


# =============================================================================
#  make_samples
# =============================================================================

class TestMakeSamples(unittest.TestCase):

    def setUp(self):
        self.text = "abcdefghijklmnopqrstuvwxyz " * 5
        self.tok  = CharTokenizer(self.text)
        self.enc  = self.tok.encode(self.text)

    def test_returns_nonempty_list(self):
        """Returns at least one sample for valid input."""
        samples = make_samples(self.enc, context_size=4, tokenizer=self.tok)
        self.assertGreater(len(samples), 0)

    def test_each_sample_is_two_tuple(self):
        """Each sample is (feature_vector, label_index)."""
        samples = make_samples(self.enc, context_size=4, tokenizer=self.tok)
        for sample in samples:
            self.assertEqual(len(sample), 2)

    def test_feature_vector_length(self):
        """Feature vector length == context_size * vocab_size."""
        context = 4
        samples = make_samples(self.enc, context_size=context, tokenizer=self.tok)
        feat, _ = samples[0]
        self.assertEqual(len(feat), context * self.tok.size)

    def test_label_in_range(self):
        """Label index must be in [0, vocab_size)."""
        samples = make_samples(self.enc, context_size=4, tokenizer=self.tok)
        for _, label in samples:
            self.assertGreaterEqual(label, 0)
            self.assertLess(label, self.tok.size)

    def test_feature_is_one_hot_per_char(self):
        """Each context_size-char block of the feature must sum to 1."""
        context = 1
        samples = make_samples(self.enc, context_size=context, tokenizer=self.tok)
        feat, _ = samples[0]
        self.assertEqual(sum(feat), 1.0)

    def test_max_samples_cap(self):
        """max_samples limits the returned list length."""
        samples = make_samples(
            self.enc, context_size=4, tokenizer=self.tok, max_samples=5
        )
        self.assertLessEqual(len(samples), 5)

    def test_corpus_shorter_than_context_raises(self):
        """Raises ValueError when corpus is shorter than context_size."""
        with self.assertRaises(ValueError):
            make_samples([0, 1], context_size=10, tokenizer=self.tok)

    def test_empty_corpus_raises(self):
        """Raises ValueError for empty corpus."""
        with self.assertRaises(ValueError):
            make_samples([], context_size=4, tokenizer=self.tok)


# =============================================================================
#  make_index_arrays
# =============================================================================

class TestMakeIndexArrays(unittest.TestCase):

    def setUp(self):
        self.text = "hello world " * 20
        self.tok  = CharTokenizer(self.text)
        self.enc  = self.tok.encode(self.text)

    def test_returns_two_arrays(self):
        """Returns (X_idx, Y_idx) tuple."""
        result = make_index_arrays(self.enc, context_size=4)
        self.assertEqual(len(result), 2)

    def test_shapes_match(self):
        """X and Y must have the same shape."""
        X, Y = make_index_arrays(self.enc, context_size=4, max_samples=50)
        self.assertEqual(X.shape, Y.shape)

    def test_first_dim_is_context_size(self):
        """First dimension must equal context_size (T, N) layout."""
        context = 8
        X, Y = make_index_arrays(self.enc, context_size=context, max_samples=50)
        self.assertEqual(X.shape[0], context)

    def test_second_dim_capped_by_max_samples(self):
        """Second dimension (N) must not exceed max_samples."""
        X, Y = make_index_arrays(self.enc, context_size=4, max_samples=30)
        self.assertLessEqual(X.shape[1], 30)

    def test_values_in_range(self):
        """All index values must be non-negative."""
        X, Y = make_index_arrays(self.enc, context_size=4, max_samples=50)
        self.assertTrue((X >= 0).all())
        self.assertTrue((Y >= 0).all())

    def test_y_is_x_shifted_by_one(self):
        """Y[t] should be X[t+1] for t < T-1 (shift-by-one relationship)."""
        X, Y = make_index_arrays(self.enc, context_size=4, max_samples=10)
        # For each column (sample), Y[:-1, n] == X[1:, n]
        import numpy as np
        for n in range(X.shape[1]):
            self.assertTrue(
                np.array_equal(Y[:-1, n], X[1:, n]),
                msg=f"Shift-by-one failed at sample {n}"
            )

    def test_different_context_sizes_work(self):
        """make_index_arrays works for multiple context sizes."""
        for ctx in [1, 4, 8, 16]:
            X, Y = make_index_arrays(self.enc, context_size=ctx, max_samples=20)
            self.assertEqual(X.shape[0], ctx)


if __name__ == "__main__":
    unittest.main()