"""
test_integration.py
===================
End-to-end pipeline tests for miniGPT.

Tests complete workflows: raw text -> tokenize -> encode -> train -> generate
-> save -> load -> generate again.  All assertions verify real outputs, not
intermediate constants.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'miniGPT'))

from data import load_text, make_index_arrays, make_samples, simplify_text
from tokenizer import CharTokenizer

try:
    from MiniGPT.model import MiniGPT
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False

_TEXT = "the quick brown fox jumps over the lazy dog " * 40


# =============================================================================
#  Tokenization pipeline
# =============================================================================

class TestTokenizationPipeline(unittest.TestCase):

    def test_text_encode_decode_roundtrip(self):
        """Full text -> encode -> decode must be lossless."""
        tok     = CharTokenizer(_TEXT)
        decoded = tok.decode(tok.encode(_TEXT))
        self.assertEqual(decoded, _TEXT)

    def test_simplify_then_tokenize_smaller_vocab(self):
        """Simplified text must produce a smaller vocab than raw text."""
        raw_text = "Hello, World! 123 @#$% <html>"
        tok_raw    = CharTokenizer(raw_text)
        tok_simple = CharTokenizer(simplify_text(raw_text))
        self.assertLess(tok_simple.size, tok_raw.size)

    def test_onehot_decodes_back_to_character(self):
        """one_hot(ch2idx[c]) must have its 1 at position ch2idx[c]."""
        tok = CharTokenizer(_TEXT)
        for ch in tok.vocab[:5]:   # spot-check first 5
            idx = tok.ch2idx[ch]
            vec = tok.one_hot(idx)
            self.assertEqual(vec[idx], 1.0)
            self.assertEqual(sum(vec), 1.0)

    def test_tokenizer_persists_across_save_load(self):
        """Saved and loaded tokenizer must produce identical encodings."""
        tok  = CharTokenizer(_TEXT)
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        self.addCleanup(os.remove, path)
        tok.save(path)
        tok2 = CharTokenizer.load(path)
        self.assertEqual(tok.encode("the quick"), tok2.encode("the quick"))


# =============================================================================
#  Data pipeline
# =============================================================================

class TestDataPipeline(unittest.TestCase):

    def setUp(self):
        fd, self.tmp = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w") as f:
            f.write(_TEXT)
        self.addCleanup(os.remove, self.tmp)

    def test_load_tokenize_encode_sample_chain(self):
        """Full pipeline load -> tokenize -> encode -> make_samples."""
        text    = load_text(self.tmp, max_chars=500)
        tok     = CharTokenizer(text)
        encoded = tok.encode(text)
        samples = make_samples(encoded, context_size=4, tokenizer=tok)
        # All three steps must produce non-empty results
        self.assertGreater(len(text),    0)
        self.assertGreater(len(encoded), 0)
        self.assertGreater(len(samples), 0)

    def test_index_arrays_match_samples_content(self):
        """make_index_arrays output should agree with make_samples labels."""
        tok     = CharTokenizer(_TEXT)
        encoded = tok.encode(_TEXT)

        # make_index_arrays
        X, Y = make_index_arrays(encoded, context_size=4, max_samples=10)

        # make_samples on the same data
        samples = make_samples(encoded, context_size=4, tokenizer=tok,
                               max_samples=10)

        # The last Y value of a sample == the label in make_samples
        for n in range(min(X.shape[1], len(samples))):
            idx_label   = int(Y[-1, n])
            samp_label  = samples[n][1]
            self.assertEqual(idx_label, samp_label,
                             msg=f"Label mismatch at sample {n}")

    def test_max_chars_limits_loaded_text(self):
        """load_text with max_chars must not return more chars than the cap."""
        text = load_text(self.tmp, max_chars=100)
        self.assertLessEqual(len(text), 100)


# =============================================================================
#  Model pipeline
# =============================================================================

@unittest.skipUnless(HAS_MODEL, "miniGPT/model.py not importable")
class TestModelPipeline(unittest.TestCase):

    def setUp(self):
        fd, self.tmp_txt = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w") as f:
            f.write(_TEXT)

        self.model = MiniGPT(context_size=4, embed_dim=32, num_blocks=1, num_heads=2)
        self.model.train(self.tmp_txt, epochs=2, max_samples=50, log_every=0)

        fd, self.weights = tempfile.mkstemp(suffix=".json")
        os.close(fd)

    def tearDown(self):
        import glob
        base = self.weights.replace(".json", "")
        for p in glob.glob(base + "*.json"):
            if os.path.exists(p):
                os.remove(p)
        if os.path.exists(self.tmp_txt):
            os.remove(self.tmp_txt)

    def test_train_generate_save_load_generate(self):
        """Full pipeline: train -> generate -> save -> load -> generate."""
        text1 = self.model.generate(length=30)
        self.model.save(self.weights)
        loaded = MiniGPT.load(self.weights)
        text2  = loaded.generate(length=30)
        # Both must be non-empty strings
        self.assertIsInstance(text1, str)
        self.assertIsInstance(text2, str)
        self.assertGreater(len(text1), 0)
        self.assertGreater(len(text2), 0)

    def test_loaded_model_has_same_vocab(self):
        """After save/load the tokenizer vocab must be identical."""
        self.model.save(self.weights)
        loaded = MiniGPT.load(self.weights)
        self.assertEqual(self.model.tokenizer.vocab, loaded.tokenizer.vocab)

    def test_resume_continues_adam_t(self):
        """Re-training the model must increment adam_t, not reset it."""
        t_before = self.model.nn._adam_t
        self.model.train(self.tmp_txt, epochs=1, max_samples=50, log_every=0)
        t_after  = self.model.nn._adam_t
        self.assertGreater(t_after, t_before)

    def test_resume_does_not_reset_nn(self):
        """Re-training must not replace self.nn with a new instance."""
        nn_before = self.model.nn
        self.model.train(self.tmp_txt, epochs=1, max_samples=50, log_every=0)
        self.assertIs(self.model.nn, nn_before)


# =============================================================================
#  Reproducibility
# =============================================================================

class TestReproducibility(unittest.TestCase):

    def test_tokenizer_is_deterministic(self):
        """Two tokenizers built from identical text must be identical."""
        tok1 = CharTokenizer(_TEXT)
        tok2 = CharTokenizer(_TEXT)
        self.assertEqual(tok1.vocab,   tok2.vocab)
        self.assertEqual(tok1.ch2idx,  tok2.ch2idx)

    def test_encoding_is_deterministic(self):
        """encode() called twice on the same input must give the same output."""
        tok = CharTokenizer(_TEXT)
        self.assertEqual(tok.encode("the quick"), tok.encode("the quick"))

    def test_samples_with_same_seed_are_equal(self):
        """make_samples with the same random seed produces same samples."""
        import random
        tok     = CharTokenizer(_TEXT)
        encoded = tok.encode(_TEXT)
        random.seed(42)
        s1 = make_samples(encoded, context_size=4, tokenizer=tok, max_samples=20)
        random.seed(42)
        s2 = make_samples(encoded, context_size=4, tokenizer=tok, max_samples=20)
        self.assertEqual(s1, s2)


if __name__ == "__main__":
    unittest.main()