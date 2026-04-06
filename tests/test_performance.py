"""
test_performance.py
===================
Speed and scalability tests for miniGPT.

Tests measure wall-clock time for key operations and assert they
complete within reasonable bounds on a standard laptop CPU.
No GPU required -- all operations here are pure Python/NumPy.
"""

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'miniGPT'))
from data import make_index_arrays, make_samples
from tokenizer import CharTokenizer


def _time(fn) -> float:
    """Return wall-clock seconds for fn()."""
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


# =============================================================================
#  Tokenizer speed
# =============================================================================

class TestTokenizerSpeed(unittest.TestCase):

    def test_construction_is_fast(self):
        """Building a tokenizer from 100k chars must take under 0.5s."""
        text = "hello world the quick brown fox " * 3000   # ~96k chars
        elapsed = _time(lambda: CharTokenizer(text))
        self.assertLess(elapsed, 0.5,
                        msg=f"Tokenizer construction took {elapsed:.3f}s")

    def test_encode_10k_chars_is_fast(self):
        """Encoding 10k characters must take under 0.1s."""
        text = "hello world " * 1000
        tok  = CharTokenizer(text)
        elapsed = _time(lambda: tok.encode(text))
        self.assertLess(elapsed, 0.1,
                        msg=f"encode(10k chars) took {elapsed:.3f}s")

    def test_encode_100k_chars_under_one_second(self):
        """Encoding 100k characters must take under 1s."""
        text = "hello world " * 10_000
        tok  = CharTokenizer(text)
        elapsed = _time(lambda: tok.encode(text))
        self.assertLess(elapsed, 1.0,
                        msg=f"encode(100k chars) took {elapsed:.3f}s")

    def test_decode_10k_chars_is_fast(self):
        """Decoding 10k indices must take under 0.1s."""
        text    = "hello world " * 1000
        tok     = CharTokenizer(text)
        encoded = tok.encode(text)
        elapsed = _time(lambda: tok.decode(encoded))
        self.assertLess(elapsed, 0.1,
                        msg=f"decode(10k indices) took {elapsed:.3f}s")

    def test_roundtrip_is_fast(self):
        """Full encode + decode roundtrip on 10k chars under 0.2s."""
        text = "hello world " * 1000
        tok  = CharTokenizer(text)
        elapsed = _time(lambda: tok.decode(tok.encode(text)))
        self.assertLess(elapsed, 0.2,
                        msg=f"roundtrip took {elapsed:.3f}s")

    def test_1000_one_hot_calls_under_one_second(self):
        """1000 one_hot() calls must complete under 1s."""
        tok = CharTokenizer("hello world")
        elapsed = _time(lambda: [tok.one_hot(i % tok.size) for i in range(1000)])
        self.assertLess(elapsed, 1.0,
                        msg=f"1000 one_hot calls took {elapsed:.3f}s")


# =============================================================================
#  make_samples speed
# =============================================================================

class TestMakeSamplesSpeed(unittest.TestCase):

    def test_10k_samples_small_vocab_under_5s(self):
        """make_samples(10k) on simple text must complete under 5s."""
        text    = "hello world " * 2000
        tok     = CharTokenizer(text)
        encoded = tok.encode(text)
        elapsed = _time(lambda: make_samples(
            encoded, context_size=8, tokenizer=tok, max_samples=10_000
        ))
        self.assertLess(elapsed, 5.0,
                        msg=f"make_samples(10k) took {elapsed:.3f}s")

    def test_1k_samples_context16_is_fast(self):
        """make_samples(1k, ctx=16) must complete under 1s."""
        text    = "hello world " * 2000
        tok     = CharTokenizer(text)
        encoded = tok.encode(text)
        elapsed = _time(lambda: make_samples(
            encoded, context_size=16, tokenizer=tok, max_samples=1000
        ))
        self.assertLess(elapsed, 1.0,
                        msg=f"make_samples(1k, ctx=16) took {elapsed:.3f}s")


# =============================================================================
#  make_index_arrays speed  (vectorised -- should be very fast)
# =============================================================================

class TestMakeIndexArraysSpeed(unittest.TestCase):

    def test_50k_samples_context32_under_1s(self):
        """make_index_arrays(50k, ctx=32) must complete under 1s (numpy strides)."""
        text    = "hello world the quick brown fox " * 5000
        tok     = CharTokenizer(text)
        encoded = tok.encode(text)
        elapsed = _time(lambda: make_index_arrays(
            encoded, context_size=32, max_samples=50_000
        ))
        self.assertLess(elapsed, 1.0,
                        msg=f"make_index_arrays(50k) took {elapsed:.3f}s")

    def test_10k_samples_faster_than_make_samples(self):
        """make_index_arrays must be faster than make_samples for same N."""
        text    = "hello world the quick brown fox " * 1000
        tok     = CharTokenizer(text)
        encoded = tok.encode(text)

        t_idx = _time(lambda: make_index_arrays(
            encoded, context_size=8, max_samples=5000
        ))
        t_smp = _time(lambda: make_samples(
            encoded, context_size=8, tokenizer=tok, max_samples=5000
        ))
        self.assertLess(t_idx, t_smp,
                        msg=(f"make_index_arrays ({t_idx:.3f}s) was NOT faster "
                             f"than make_samples ({t_smp:.3f}s)"))


# =============================================================================
#  Memory / scalability
# =============================================================================

class TestScalability(unittest.TestCase):

    def test_tokenizer_sizes_scale_linearly(self):
        """Encoding time should scale roughly linearly with text length."""
        base_text = "hello world " * 100
        tok       = CharTokenizer(base_text)

        t1 = _time(lambda: tok.encode(base_text * 1))
        t10 = _time(lambda: tok.encode(base_text * 10))
        # 10x more text shouldn't take more than 100x as long
        self.assertLess(t10, t1 * 100,
                        msg="Encoding does not scale linearly")

    def test_large_roundtrip_is_correct(self):
        """Roundtrip encode/decode must be lossless at 50k chars."""
        text = "hello world the quick brown fox jumps " * 1400   # ~50k
        tok  = CharTokenizer(text)
        self.assertEqual(tok.decode(tok.encode(text)), text)

    def test_index_arrays_different_context_sizes(self):
        """make_index_arrays must work correctly at multiple context sizes."""
        text    = "hello world " * 500
        tok     = CharTokenizer(text)
        encoded = tok.encode(text)
        for ctx in [1, 4, 8, 16, 32]:
            X, Y = make_index_arrays(encoded, context_size=ctx, max_samples=100)
            self.assertEqual(X.shape[0], ctx,
                             msg=f"Wrong first dim for ctx={ctx}")
            self.assertEqual(X.shape, Y.shape,
                             msg=f"X and Y shapes differ for ctx={ctx}")


if __name__ == "__main__":
    unittest.main()