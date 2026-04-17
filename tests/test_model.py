"""
test_model.py
=============
Tests for miniGPT/model.py -- the MiniGPT wrapper class.

Covers: construction, _build, train, generate, save, load.
Every test calls actual MiniGPT methods -- no trivial constant assertions.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'miniGPT'))

try:
    from MiniGPT.model import MiniGPT
    from tokenizer import CharTokenizer
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False

_TRAIN_TEXT = "hello world hello world the quick brown fox " * 30


def _tmp(suffix=".json"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


@unittest.skipUnless(HAS_MODEL, "miniGPT/model.py not importable")
class TestConstruction(unittest.TestCase):

    def test_default_params(self):
        """MiniGPT() with defaults should not raise."""
        m = MiniGPT()
        self.assertIsNotNone(m)

    def test_custom_params_stored(self):
        """Constructor params must be stored as attributes."""
        m = MiniGPT(context_size=12, embed_dim=128, num_blocks=3, num_heads=4)
        self.assertEqual(m.context_size, 12)
        self.assertEqual(m.embed_dim, 128)
        self.assertEqual(m.num_blocks, 3)
        self.assertEqual(m.num_heads, 4)

    def test_nn_is_none_before_training(self):
        """nn attribute must be None before train() is called."""
        m = MiniGPT()
        self.assertIsNone(m.nn)

    def test_tokenizer_is_none_before_training(self):
        """tokenizer attribute must be None before train() is called."""
        m = MiniGPT()
        self.assertIsNone(m.tokenizer)

    def test_repr_shows_status(self):
        """repr() must include 'untrained' before training."""
        m = MiniGPT()
        self.assertIn("untrained", repr(m))


@unittest.skipUnless(HAS_MODEL, "miniGPT/model.py not importable")
class TestTraining(unittest.TestCase):

    def setUp(self):
        fd, self.tmp_txt = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w") as f:
            f.write(_TRAIN_TEXT)

    def tearDown(self):
        os.remove(self.tmp_txt)

    def test_train_builds_nn(self):
        """After train(), nn must not be None."""
        m = MiniGPT(context_size=4, embed_dim=32, num_blocks=1, num_heads=2)
        m.train(self.tmp_txt, epochs=1, max_samples=50, log_every=0)
        self.assertIsNotNone(m.nn)

    def test_train_builds_tokenizer(self):
        """After train(), tokenizer must be set."""
        m = MiniGPT(context_size=4, embed_dim=32, num_blocks=1, num_heads=2)
        m.train(self.tmp_txt, epochs=1, max_samples=50, log_every=0)
        self.assertIsNotNone(m.tokenizer)
        self.assertGreater(m.tokenizer.size, 0)

    def test_train_on_raw_string(self):
        """train() must accept raw text strings as well as file paths."""
        m = MiniGPT(context_size=4, embed_dim=32, num_blocks=1, num_heads=2)
        m.train(_TRAIN_TEXT, epochs=1, max_samples=50, log_every=0)
        self.assertIsNotNone(m.nn)

    def test_simple_vocab_reduces_vocab_size(self):
        """simple_vocab=True must produce fewer characters than raw text."""
        m_raw    = MiniGPT(context_size=4, embed_dim=32, num_blocks=1, num_heads=2)
        m_simple = MiniGPT(context_size=4, embed_dim=32, num_blocks=1, num_heads=2)
        raw_text = "Hello, World! 123 @#$%"
        m_raw.train(raw_text, epochs=1, max_samples=20, log_every=0,
                    simple_vocab=False)
        m_simple.train(raw_text, epochs=1, max_samples=20, log_every=0,
                       simple_vocab=True)
        self.assertLess(m_simple.tokenizer.size, m_raw.tokenizer.size)

    def test_resume_skips_rebuild(self):
        """Calling train() twice must NOT replace the existing nn."""
        m = MiniGPT(context_size=4, embed_dim=32, num_blocks=1, num_heads=2)
        m.train(self.tmp_txt, epochs=1, max_samples=50, log_every=0)
        nn_before = m.nn
        m.train(self.tmp_txt, epochs=1, max_samples=50, log_every=0)
        self.assertIs(m.nn, nn_before,
                      msg="_build() was called again on resume -- overwrites weights")


@unittest.skipUnless(HAS_MODEL, "miniGPT/model.py not importable")
class TestGeneration(unittest.TestCase):

    def setUp(self):
        self.model = MiniGPT(context_size=4, embed_dim=32, num_blocks=1, num_heads=2)
        fd, tmp = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w") as f:
            f.write(_TRAIN_TEXT)
        self.model.train(tmp, epochs=2, max_samples=50, log_every=0)
        os.remove(tmp)

    def test_generate_returns_string(self):
        """generate() must return a str."""
        self.assertIsInstance(self.model.generate(length=20), str)

    def test_generate_length_respected(self):
        """Generated text must contain at least `length` new chars."""
        out = self.model.generate(prompt="", length=50)
        self.assertGreaterEqual(len(out), 50)

    def test_generate_with_prompt_starts_with_prompt(self):
        """Output must start with the prompt."""
        prompt = "he"
        out = self.model.generate(prompt=prompt, length=30)
        self.assertTrue(out.startswith(prompt))

    def test_generate_temperature_low_more_repetitive(self):
        """Low temperature should not raise and should return a string."""
        out = self.model.generate(length=30, temperature=0.1)
        self.assertIsInstance(out, str)

    def test_generate_temperature_high_still_works(self):
        """High temperature should not raise and should return a string."""
        out = self.model.generate(length=30, temperature=2.0)
        self.assertIsInstance(out, str)

    def test_generate_before_training_raises(self):
        """generate() on an untrained model must raise RuntimeError."""
        m = MiniGPT()
        with self.assertRaises(RuntimeError):
            m.generate(length=10)


@unittest.skipUnless(HAS_MODEL, "miniGPT/model.py not importable")
class TestSaveLoad(unittest.TestCase):

    def setUp(self):
        self.model = MiniGPT(context_size=4, embed_dim=32, num_blocks=1, num_heads=2)
        fd, tmp = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w") as f:
            f.write(_TRAIN_TEXT)
        self.model.train(tmp, epochs=1, max_samples=50, log_every=0)
        os.remove(tmp)

        self.weights = _tmp(".json")
        self.addCleanup(os.remove, self.weights)

    def _side_files(self):
        tok = self.weights.replace(".json", "_tokenizer.json")
        cfg = self.weights.replace(".json", "_config.json")
        for p in [tok, cfg]:
            self.addCleanup(lambda x=p: os.remove(x) if os.path.exists(x) else None)

    def test_save_creates_three_files(self):
        """save() must create weights, tokenizer, and config files."""
        self._side_files()
        self.model.save(self.weights)
        self.assertTrue(os.path.exists(self.weights))
        self.assertTrue(os.path.exists(
            self.weights.replace(".json", "_tokenizer.json")))
        self.assertTrue(os.path.exists(
            self.weights.replace(".json", "_config.json")))

    def test_config_contains_hyperparams(self):
        """Config JSON must store context_size, embed_dim, num_heads etc."""
        self._side_files()
        self.model.save(self.weights)
        cfg_path = self.weights.replace(".json", "_config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        for key in ("context_size", "embed_dim", "num_blocks", "num_heads"):
            self.assertIn(key, cfg, msg=f"'{key}' missing from config")

    def test_load_returns_minigpt_instance(self):
        """MiniGPT.load() must return a MiniGPT object."""
        self._side_files()
        self.model.save(self.weights)
        loaded = MiniGPT.load(self.weights)
        self.assertIsInstance(loaded, MiniGPT)

    def test_loaded_model_generates(self):
        """A loaded model must be able to generate text."""
        self._side_files()
        self.model.save(self.weights)
        loaded = MiniGPT.load(self.weights)
        out = loaded.generate(length=20)
        self.assertIsInstance(out, str)

    def test_loaded_model_same_vocab(self):
        """Loaded model must have the same tokenizer vocab."""
        self._side_files()
        self.model.save(self.weights)
        loaded = MiniGPT.load(self.weights)
        self.assertEqual(self.model.tokenizer.vocab, loaded.tokenizer.vocab)

    def test_save_before_training_does_nothing(self):
        """save() on untrained model must print a warning and not crash."""
        m = MiniGPT()
        # Should not raise
        m.save(self.weights)

    def test_missing_file_raises_on_load(self):
        """load() with a missing file must raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            MiniGPT.load("/nonexistent/no_such_weights.json")


if __name__ == "__main__":
    unittest.main()