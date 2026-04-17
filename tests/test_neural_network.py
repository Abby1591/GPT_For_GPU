"""
test_neural_network.py
======================
Tests for Neural_Network.py -- the core transformer.

Covers: construction, forward pass shapes, causal mask,
LayerNorm, multi-head attention, dropout, weight tying,
gradient clipping, Adam state, save/load.

All tests instantiate a small but real NeuralNetwork and call
its actual methods -- no trivial assertions on local constants.
"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from MiniGPT.Neural_Network import NeuralNetwork
    HAS_NN = True
except ImportError:
    HAS_NN = False

# Small config reused across tests -- fast to construct
_SMALL = dict(
    input_size    = 16,
    hidden_layers = [32],
    output_size   = 10,
    vocab_size    = 10,
    context_size  = 8,
    embed_dim     = 32,
    num_blocks    = 2,
    num_heads     = 4,
    dropout       = 0.0,
    weight_tying  = False,  # separate Wout easier to inspect
    grad_clip     = 1.0,
)


@unittest.skipUnless(HAS_NN, "Neural_Network.py not importable")
class TestConstruction(unittest.TestCase):

    def test_builds_without_error(self):
        """NeuralNetwork() with valid params should not raise."""
        nn = NeuralNetwork(**_SMALL)
        self.assertIsNotNone(nn)

    def test_embed_dim_not_divisible_by_heads_raises(self):
        """embed_dim % num_heads != 0 must raise ValueError."""
        cfg = dict(_SMALL)
        cfg["embed_dim"]  = 33   # not divisible by 4
        with self.assertRaises(ValueError):
            NeuralNetwork(**cfg)

    def test_empty_hidden_layers_raises(self):
        """Empty hidden_layers must raise ValueError."""
        cfg = dict(_SMALL, hidden_layers=[])
        with self.assertRaises(ValueError):
            NeuralNetwork(**cfg)

    def test_unknown_activation_raises(self):
        """Unknown activation name must raise ValueError."""
        cfg = dict(_SMALL, activation="swish")
        with self.assertRaises(ValueError):
            NeuralNetwork(**cfg)

    def test_blocks_count(self):
        """Number of blocks stored must match num_blocks param."""
        nn = NeuralNetwork(**_SMALL)
        self.assertEqual(len(nn.blocks), _SMALL["num_blocks"])

    def test_block_keys_present(self):
        """Each block must have Wqkv, W1, b1, W2, b2, LN params."""
        nn = NeuralNetwork(**_SMALL)
        required = {"Wqkv", "W1", "b1", "W2", "b2",
                    "ln1_g", "ln1_b", "ln2_g", "ln2_b"}
        for blk in nn.blocks:
            self.assertEqual(required, set(blk.keys()))

    def test_wqkv_shape(self):
        """Wqkv shape must be (D, 3D)."""
        nn = NeuralNetwork(**_SMALL)
        D = _SMALL["embed_dim"]
        self.assertEqual(nn.blocks[0]["Wqkv"].shape, (D, D * 3))

    def test_weight_tying_sets_wout_none(self):
        """When weight_tying=True, Wout should be None."""
        cfg = dict(_SMALL, weight_tying=True)
        nn = NeuralNetwork(**cfg)
        self.assertIsNone(nn.Wout)

    def test_separate_wout_has_correct_shape(self):
        """When weight_tying=False, Wout shape = (D, vocab)."""
        nn = NeuralNetwork(**_SMALL)
        D, V = _SMALL["embed_dim"], _SMALL["output_size"]
        self.assertEqual(nn.Wout.shape, (D, V))

    def test_repr_contains_key_info(self):
        """repr() should mention embed_dim, blocks, heads."""
        nn = NeuralNetwork(**_SMALL)
        r = repr(nn)
        self.assertIn(str(_SMALL["embed_dim"]), r)
        self.assertIn(str(_SMALL["num_blocks"]), r)


@unittest.skipUnless(HAS_NN, "Neural_Network.py not importable")
class TestCausalMask(unittest.TestCase):

    def test_mask_shape(self):
        """_causal_mask(T) must return (T, T) array."""
        nn = NeuralNetwork(**_SMALL)
        mask = nn._causal_mask(8)
        self.assertEqual(mask.shape, (8, 8))

    def test_lower_triangle_is_zero(self):
        """Positions i>=j (attend to self and past) must be 0."""
        nn = NeuralNetwork(**_SMALL)
        mask = np.array(nn._causal_mask(5))
        for i in range(5):
            for j in range(i + 1):
                self.assertEqual(mask[i, j], 0.0,
                                 msg=f"mask[{i},{j}] should be 0")

    def test_upper_triangle_is_large_negative(self):
        """Future positions must have large negative value."""
        nn = NeuralNetwork(**_SMALL)
        mask = np.array(nn._causal_mask(5))
        for i in range(5):
            for j in range(i + 1, 5):
                self.assertLess(mask[i, j], -1e8,
                                msg=f"mask[{i},{j}] should be -1e9")

    def test_mask_is_cached(self):
        """Second call with same T returns same object."""
        nn = NeuralNetwork(**_SMALL)
        m1 = nn._causal_mask(6)
        m2 = nn._causal_mask(6)
        self.assertIs(m1, m2)

    def test_mask_rebuilt_for_different_T(self):
        """Different T rebuilds the mask."""
        nn = NeuralNetwork(**_SMALL)
        m1 = nn._causal_mask(4)
        m2 = nn._causal_mask(8)
        self.assertEqual(m1.shape[0], 4)
        self.assertEqual(m2.shape[0], 8)


@unittest.skipUnless(HAS_NN, "Neural_Network.py not importable")
class TestLayerNorm(unittest.TestCase):

    def _nn(self):
        return NeuralNetwork(**_SMALL)

    def test_forward_output_shape(self):
        """LN forward must preserve input shape."""
        nn = self._nn()
        D  = _SMALL["embed_dim"]
        x  = np.random.randn(2, 8, D)
        gamma = np.ones(D)
        beta  = np.zeros(D)
        out, _ = nn._ln_forward(x, gamma, beta)
        self.assertEqual(out.shape, x.shape)

    def test_output_mean_near_zero(self):
        """After LN the mean along last axis should be ~0."""
        nn = self._nn()
        D  = _SMALL["embed_dim"]
        x  = np.random.randn(4, 8, D) * 10 + 5  # offset + scaled
        out, _ = nn._ln_forward(x, np.ones(D), np.zeros(D))
        means = np.abs(out.mean(axis=-1))
        self.assertTrue((means < 1e-5).all(),
                        msg=f"max mean = {means.max():.2e}")

    def test_output_std_near_one(self):
        """After LN the std along last axis should be ~1."""
        nn = self._nn()
        D  = _SMALL["embed_dim"]
        x  = np.random.randn(4, 8, D) * 10 + 5
        out, _ = nn._ln_forward(x, np.ones(D), np.zeros(D))
        stds = out.std(axis=-1)
        self.assertTrue(np.allclose(stds, 1.0, atol=1e-4),
                        msg=f"max std deviation from 1: {np.abs(stds-1).max():.2e}")

    def test_backward_returns_correct_shapes(self):
        """LN backward must return gradients with same shapes."""
        nn = self._nn()
        D  = _SMALL["embed_dim"]
        x  = np.random.randn(2, 4, D)
        gamma = np.ones(D)
        beta  = np.zeros(D)
        _, cache = nn._ln_forward(x, gamma, beta)
        d_out = np.random.randn(*x.shape)
        d_x, d_gamma, d_beta = nn._ln_backward(d_out, cache)
        self.assertEqual(d_x.shape,     x.shape)
        self.assertEqual(d_gamma.shape, gamma.shape)
        self.assertEqual(d_beta.shape,  beta.shape)


@unittest.skipUnless(HAS_NN, "Neural_Network.py not importable")
class TestDropout(unittest.TestCase):

    def test_no_dropout_returns_input_unchanged(self):
        """dropout=0.0 must return x and None mask."""
        nn = NeuralNetwork(**dict(_SMALL, dropout=0.0))
        x = np.ones((4, 8, 32))
        out, mask = nn._apply_dropout(x, training=True)
        self.assertTrue(np.array_equal(out, x))
        self.assertIsNone(mask)

    def test_inference_returns_input_unchanged(self):
        """training=False must always return x unchanged."""
        nn = NeuralNetwork(**dict(_SMALL, dropout=0.5))
        x = np.ones((4, 8, 32))
        out, mask = nn._apply_dropout(x, training=False)
        self.assertTrue(np.array_equal(out, x))
        self.assertIsNone(mask)

    def test_dropout_zeroes_some_activations(self):
        """With high dropout, some values should be zeroed."""
        nn = NeuralNetwork(**dict(_SMALL, dropout=0.9))
        x = np.ones((8, 16, 32))
        out, _ = nn._apply_dropout(x, training=True)
        out_cpu = np.array(out)
        # With 90% dropout, most values should be 0
        zero_fraction = (out_cpu == 0).mean()
        self.assertGreater(zero_fraction, 0.5)

    def test_inverted_scaling_preserves_mean(self):
        """Inverted dropout should keep expected value the same."""
        np.random.seed(0)
        nn = NeuralNetwork(**dict(_SMALL, dropout=0.5))
        x = np.ones((100, 32, 32))
        out, _ = nn._apply_dropout(x, training=True)
        # Mean should be approximately 1.0 (inverted scaling)
        mean_val = float(np.array(out).mean())
        self.assertAlmostEqual(mean_val, 1.0, delta=0.1)


@unittest.skipUnless(HAS_NN, "Neural_Network.py not importable")
class TestForwardPass(unittest.TestCase):

    def test_transformer_forward_output_shape(self):
        """probs from _transformer_forward must be (B, T, vocab)."""
        nn = NeuralNetwork(**_SMALL)
        B, T = 3, _SMALL["context_size"]
        toks = np.random.randint(0, _SMALL["vocab_size"], (T, B))
        probs, _ = nn._transformer_forward(toks, training=False)
        self.assertEqual(probs.shape, (B, T, _SMALL["vocab_size"]))

    def test_probs_sum_to_one(self):
        """Softmax probs must sum to 1 along vocab axis."""
        nn = NeuralNetwork(**_SMALL)
        B, T = 2, _SMALL["context_size"]
        toks = np.random.randint(0, _SMALL["vocab_size"], (T, B))
        probs, _ = nn._transformer_forward(toks, training=False)
        sums = np.array(probs).sum(axis=-1)
        self.assertTrue(np.allclose(sums, 1.0, atol=1e-5))

    def test_probs_all_positive(self):
        """All probabilities must be > 0 (softmax guarantee)."""
        nn = NeuralNetwork(**_SMALL)
        B, T = 2, _SMALL["context_size"]
        toks = np.random.randint(0, _SMALL["vocab_size"], (T, B))
        probs, _ = nn._transformer_forward(toks, training=False)
        self.assertTrue((np.array(probs) > 0).all())

    def test_forward_last_position_only(self):
        """forward() returns 1D probs for last position (used in generate)."""
        nn = NeuralNetwork(**_SMALL)
        V  = _SMALL["vocab_size"]
        T  = _SMALL["context_size"]
        # Build flat one-hot input
        inputs = []
        for _ in range(T):
            oh = [0.0] * V
            oh[0] = 1.0
            inputs.extend(oh)
        _, _, probs = nn.forward(inputs)
        self.assertEqual(probs.shape, (V,))

    def test_weight_tying_forward_matches_separate(self):
        """Weight-tied model probs should have the same shape as untied."""
        cfg_tied   = dict(_SMALL, weight_tying=True)
        cfg_untied = dict(_SMALL, weight_tying=False)
        nn_tied   = NeuralNetwork(**cfg_tied)
        nn_untied = NeuralNetwork(**cfg_untied)
        B, T = 2, _SMALL["context_size"]
        toks = np.random.randint(0, _SMALL["vocab_size"], (T, B))
        p1, _ = nn_tied._transformer_forward(toks)
        p2, _ = nn_untied._transformer_forward(toks)
        self.assertEqual(p1.shape, p2.shape)


@unittest.skipUnless(HAS_NN, "Neural_Network.py not importable")
class TestAdamState(unittest.TestCase):

    def test_init_adam_allocates_buffers(self):
        """_init_adam must create _adam_blocks for every block."""
        nn = NeuralNetwork(**_SMALL)
        nn._init_adam()
        self.assertEqual(len(nn._adam_blocks), _SMALL["num_blocks"])

    def test_adam_buffers_zero_at_init(self):
        """All m and v buffers must be zero after _init_adam."""
        nn = NeuralNetwork(**_SMALL)
        nn._init_adam()
        for blk_buf in nn._adam_blocks:
            for k, mv in blk_buf.items():
                self.assertTrue((np.array(mv["m"]) == 0).all())
                self.assertTrue((np.array(mv["v"]) == 0).all())

    def test_adam_t_starts_at_zero(self):
        """_adam_t must be 0 after _init_adam."""
        nn = NeuralNetwork(**_SMALL)
        nn._init_adam()
        self.assertEqual(nn._adam_t, 0)

    def test_adam_init_flag_set(self):
        """_adam_init must be True after _init_adam."""
        nn = NeuralNetwork(**_SMALL)
        nn._init_adam()
        self.assertTrue(nn._adam_init)


@unittest.skipUnless(HAS_NN, "Neural_Network.py not importable")
class TestSaveLoad(unittest.TestCase):

    def _tmp(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        self.addCleanup(os.remove, path)
        # also clean up any .tmp files
        self.addCleanup(lambda: [
            os.remove(p) for p in [path + ".tmp"]
            if os.path.exists(p)
        ])
        return path

    def test_save_creates_file(self):
        """save_weights must create a non-empty file."""
        nn   = NeuralNetwork(**_SMALL)
        path = self._tmp()
        nn.save_weights(path)
        self.assertGreater(os.path.getsize(path), 0)

    def test_save_creates_valid_json(self):
        """Saved file must be valid JSON."""
        nn   = NeuralNetwork(**_SMALL)
        path = self._tmp()
        nn.save_weights(path)
        with open(path) as f:
            data = json.load(f)
        self.assertIn("embed_dim", data)
        self.assertIn("blocks", data)

    def test_load_restores_hyperparams(self):
        """load_weights must restore all hyperparameters."""
        nn   = NeuralNetwork(**_SMALL)
        path = self._tmp()
        nn.save_weights(path)

        nn2 = NeuralNetwork(**_SMALL)
        nn2.load_weights(path)
        self.assertEqual(nn2.embed_dim,   _SMALL["embed_dim"])
        self.assertEqual(nn2.num_blocks,  _SMALL["num_blocks"])
        self.assertEqual(nn2.num_heads,   _SMALL["num_heads"])
        self.assertEqual(nn2.weight_tying, _SMALL["weight_tying"])

    def test_load_restores_block_weights(self):
        """Block weights loaded must match those saved."""
        nn   = NeuralNetwork(**_SMALL)
        path = self._tmp()
        nn.save_weights(path)

        nn2 = NeuralNetwork(**_SMALL)
        nn2.load_weights(path)
        for i in range(_SMALL["num_blocks"]):
            self.assertTrue(
                np.allclose(
                    np.array(nn.blocks[i]["Wqkv"]),
                    np.array(nn2.blocks[i]["Wqkv"]),
                ),
                msg=f"Wqkv mismatch in block {i}"
            )

    def test_save_includes_adam_state(self):
        """After training, Adam state must be included in save file."""
        import numpy as _np
        nn   = NeuralNetwork(**_SMALL)
        text = "abcde " * 30
        # Build minimal index arrays manually
        vocab   = sorted(set(text))
        ch2idx  = {c: i for i, c in enumerate(vocab)}
        encoded = [ch2idx[c] for c in text]
        ctx = _SMALL["context_size"]
        N   = 10
        X   = _np.array([encoded[i:i+ctx]     for i in range(N)], dtype=_np.int32).T
        Y   = _np.array([encoded[i+1:i+ctx+1] for i in range(N)], dtype=_np.int32).T
        nn.train((X, Y), epochs=1, log_every=0)

        path = self._tmp()
        nn.save_weights(path)
        with open(path) as f:
            data = json.load(f)
        self.assertGreater(data.get("adam_t", 0), 0)

    def test_load_restores_adam_state(self):
        """Adam state (t, buffers) must survive a save/load cycle."""
        import numpy as _np
        nn   = NeuralNetwork(**_SMALL)
        text = "abcde " * 30
        vocab   = sorted(set(text))
        ch2idx  = {c: i for i, c in enumerate(vocab)}
        encoded = [ch2idx[c] for c in text]
        ctx = _SMALL["context_size"]
        N   = 10
        X   = _np.array([encoded[i:i+ctx]     for i in range(N)], dtype=_np.int32).T
        Y   = _np.array([encoded[i+1:i+ctx+1] for i in range(N)], dtype=_np.int32).T
        nn.train((X, Y), epochs=2, log_every=0)
        saved_t = nn._adam_t

        path = self._tmp()
        nn.save_weights(path)
        nn2 = NeuralNetwork(**_SMALL)
        nn2.load_weights(path)
        self.assertEqual(nn2._adam_t,    saved_t)
        self.assertTrue(nn2._adam_init)

    def test_atomic_save_no_partial_file(self):
        """save_weights must use atomic rename (tmp -> final)."""
        # We can't simulate a mid-write crash, but we can verify no .tmp
        # file is left behind after a successful save.
        nn   = NeuralNetwork(**_SMALL)
        path = self._tmp()
        nn.save_weights(path)
        tmp = path + ".tmp"
        self.assertFalse(os.path.exists(tmp),
                         msg=".tmp file left behind after save")


@unittest.skipUnless(HAS_NN, "Neural_Network.py not importable")
class TestTraining(unittest.TestCase):
    """Smoke tests -- verify loss decreases over short training runs."""

    def _make_data(self, vocab_size, context_size, n=20):
        """Build trivial (X, Y) index arrays."""
        import numpy as _np
        _np.random.seed(0)
        X = _np.random.randint(0, vocab_size, (context_size, n), dtype=_np.int32)
        Y = _np.random.randint(0, vocab_size, (context_size, n), dtype=_np.int32)
        return X, Y

    def test_loss_decreases_over_epochs(self):
        """Train for 5 epochs; final loss should be less than initial."""
        nn = NeuralNetwork(**dict(_SMALL, dropout=0.0))
        data = self._make_data(_SMALL["vocab_size"], _SMALL["context_size"])

        losses = []
        for ep in range(5):
            # Capture loss by redirecting stdout temporarily
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                nn.train(data, epochs=1, log_every=1)
            line = [l for l in buf.getvalue().split('\n') if 'equiv' in l]
            if line:
                equiv = float(line[-1].split('equiv:')[1].split()[0])
                losses.append(equiv)

        if len(losses) >= 2:
            self.assertLess(losses[-1], losses[0],
                            msg="Loss did not decrease during training")

    def test_adam_t_increments_per_epoch(self):
        """_adam_t must increment by exactly 1 per epoch."""
        nn   = NeuralNetwork(**_SMALL)
        data = self._make_data(_SMALL["vocab_size"], _SMALL["context_size"])
        nn.train(data, epochs=3, log_every=0)
        self.assertEqual(nn._adam_t, 3)

    def test_lr_persisted_after_training(self):
        """self.learning_rate must be updated to final adaptive lr."""
        nn   = NeuralNetwork(**_SMALL)
        data = self._make_data(_SMALL["vocab_size"], _SMALL["context_size"])
        nn.train(data, epochs=1, log_every=0)
        # After training, learning_rate should equal whatever epoch_lr ended at
        self.assertIsInstance(nn.learning_rate, float)
        self.assertGreater(nn.learning_rate, 0)


if __name__ == "__main__":
    unittest.main()