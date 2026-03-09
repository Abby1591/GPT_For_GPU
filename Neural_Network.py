"""
Neural_Network.py  — Mini-Transformer edition
==============================================
Full transformer architecture with backprop through every layer:

1. EMBEDDINGS + POSITIONAL ENCODING
2. SELF-ATTENTION  (Wq, Wk, Wv)
3. FEED-FORWARD BLOCK  (W1/b1 → ReLU → W2/b2)
4. RESIDUAL CONNECTIONS
5. STACKED TRANSFORMER BLOCKS  (num_blocks = 2)
6. ADAM OPTIMIZER  (every parameter updated)

GPU SETUP:
    pip install cupy-cuda12x   # CUDA 12 (Colab T4)
    pip install cupy-cuda11x   # CUDA 11
"""

from __future__ import annotations
import json, os, random
from typing import List, Literal, Tuple

# ── Backend ───────────────────────────────────────────────────────────────────
try:
    import cupy as np
    np.cuda.Device(0).use()
    _DEVICE = "gpu"
    print(f"✓ GPU detected — training on: {np.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
except Exception:
    import numpy as np
    _DEVICE = "cpu"
    print("✗ CuPy not found — training on CPU (numpy)")

ActivationName = Literal["sigmoid", "tanh", "relu", "leaky_relu"]
Sample         = Tuple[List[float], int]

# ── Activations ───────────────────────────────────────────────────────────────
def _relu(x):            return np.maximum(0.0, x)
def _relu_d(x):          return (x > 0).astype(float)
def _tanh(x):            return np.tanh(x)
def _tanh_d(x):          return 1.0 - np.tanh(x) ** 2
def _sigmoid(x):         return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
def _sigmoid_d(x):       s = _sigmoid(x); return s * (1.0 - s)
def _leaky_relu(x, a=0.01):   return np.where(x > 0, x, a * x)
def _leaky_relu_d(x, a=0.01): return np.where(x > 0, 1.0, a)

_ACTIVATIONS = {
    "relu":       (_relu,       _relu_d),
    "tanh":       (_tanh,       _tanh_d),
    "sigmoid":    (_sigmoid,    _sigmoid_d),
    "leaky_relu": (_leaky_relu, _leaky_relu_d),
}


class NeuralNetwork:
    """
    Mini-transformer language model with full backprop through attention,
    feed-forward, and embeddings.

    Architecture per forward pass:
        tokens (context_size,)
        → embedding lookup + positional encoding   → (context_size, embed_dim)
        → transformer_block × num_blocks           → (context_size, embed_dim)
        → last-token slice                         → (embed_dim,)
        → Wout linear                              → (output_size,)
        → softmax                                  → probabilities

    All parameters are updated by Adam every batch.
    """

    def __init__(
        self,
        input_size:    int,
        hidden_layers: List[int],
        output_size:   int,
        activation:    str   = "relu",
        learning_rate: float = 0.001,
        batch_size:    int   = 512,
        use_embedding: bool  = True,
        vocab_size:    int   = 0,
        context_size:  int   = 0,
        embed_dim:     int   = 64,
    ) -> None:
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'.")
        if not hidden_layers:
            raise ValueError("hidden_layers must have at least one entry.")

        self.input_size    = input_size
        self.hidden_layers = hidden_layers
        self.output_size   = output_size
        self.activation    = activation
        self.learning_rate = learning_rate
        self.batch_size    = batch_size
        self.use_embedding = use_embedding and vocab_size > 0
        self.vocab_size    = vocab_size
        self.context_size  = context_size
        self.embed_dim     = embed_dim
        self.device        = _DEVICE
        self.num_blocks    = 2

        self._act_fn, self._act_d = _ACTIVATIONS[activation]

        # ── Embedding table ───────────────────────────────────────────────────
        if self.use_embedding:
            self.embedding     = np.random.randn(vocab_size, embed_dim) * 0.01
            self.pos_embedding = np.random.randn(context_size, embed_dim) * 0.01
        else:
            self.embedding     = None
            self.pos_embedding = None

        # ── Transformer weights (one set shared across blocks) ────────────────
        # Attention
        self.Wq = np.random.randn(embed_dim, embed_dim) * 0.02
        self.Wk = np.random.randn(embed_dim, embed_dim) * 0.02
        self.Wv = np.random.randn(embed_dim, embed_dim) * 0.02
        # Feed-forward
        self.W1 = np.random.randn(embed_dim, embed_dim * 4) * 0.02
        self.b1 = np.zeros(embed_dim * 4)
        self.W2 = np.random.randn(embed_dim * 4, embed_dim) * 0.02
        self.b2 = np.zeros(embed_dim)
        # Output projection
        self.Wout = np.random.randn(embed_dim, output_size) * 0.02
        self.bout = np.zeros(output_size)

        # ── Legacy dense weights (kept so save/load stays backward-compatible) ─
        actual_input = context_size * embed_dim if self.use_embedding else input_size
        layer_sizes  = [actual_input] + hidden_layers + [output_size]
        self.weights = []
        self.biases  = []
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.weights.append(np.random.randn(fan_out, fan_in) * scale)
            self.biases.append(np.zeros((fan_out, 1)))

        # ── Adam moment buffers ───────────────────────────────────────────────
        self._adam_init = False

    # ── Adam ──────────────────────────────────────────────────────────────────

    def _init_adam(self):
        """Initialise Adam first-moment (m) and second-moment (v) buffers."""
        z = np.zeros_like
        self._mWq = z(self.Wq);  self._vWq = z(self.Wq)
        self._mWk = z(self.Wk);  self._vWk = z(self.Wk)
        self._mWv = z(self.Wv);  self._vWv = z(self.Wv)
        self._mW1 = z(self.W1);  self._vW1 = z(self.W1)
        self._mb1 = z(self.b1);  self._vb1 = z(self.b1)
        self._mW2 = z(self.W2);  self._vW2 = z(self.W2)
        self._mb2 = z(self.b2);  self._vb2 = z(self.b2)
        self._mWout = z(self.Wout); self._vWout = z(self.Wout)
        self._mbout = z(self.bout); self._vbout = z(self.bout)
        if self.use_embedding:
            self._me  = z(self.embedding);      self._ve  = z(self.embedding)
            self._mpe = z(self.pos_embedding);  self._vpe = z(self.pos_embedding)
        self._adam_t    = 0
        self._adam_init = True

    def _adam_update(self, param, grad, m, v, lr=None,
                     beta1=0.9, beta2=0.999, eps=1e-8):
        """Apply one Adam gradient update step."""
        if lr is None:
            lr = self.learning_rate
        t    = self._adam_t
        m[:] = beta1 * m + (1 - beta1) * grad
        v[:] = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)
        return param, m, v

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _softmax_rows(self, x):
        """Row-wise softmax: x shape (batch, vocab)."""
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def _softmax(self, x):
        """Column-wise softmax: x shape (vocab, batch) — legacy compat."""
        e = np.exp(x - x.max(axis=0, keepdims=True))
        return e / e.sum(axis=0, keepdims=True)

    # ── Single transformer block (forward + returns cache for backprop) ───────

    def _block_forward(self, x):
        """
        One transformer block: attention → residual → feed-forward → residual.

        x shape: (seq_len, embed_dim)
        Returns x_out and intermediate values needed for backprop.
        """
        # Self-attention
        Q = x @ self.Wq          # (T, D)
        K = x @ self.Wk
        V = x @ self.Wv
        scale   = np.sqrt(float(self.embed_dim))
        scores  = Q @ K.T / scale                     # (T, T)
        exp_s   = np.exp(scores - scores.max(axis=1, keepdims=True))
        A       = exp_s / exp_s.sum(axis=1, keepdims=True)   # (T, T)
        attn_out = A @ V                               # (T, D)
        x_attn   = x + attn_out                       # residual

        # Feed-forward
        h      = np.maximum(0.0, x_attn @ self.W1 + self.b1)   # (T, 4D)
        ff_out = h @ self.W2 + self.b2                           # (T, D)
        x_out  = x_attn + ff_out                                 # residual

        cache = (x, Q, K, V, A, attn_out, x_attn, h, ff_out)
        return x_out, cache

    def _block_backward(self, d_out, cache):
        """
        Backprop through one transformer block.

        d_out shape: (T, D)
        Returns gradients for Wq, Wk, Wv, W1, b1, W2, b2 and d_x (gradient for input).
        """
        x, Q, K, V, A, attn_out, x_attn, h, ff_out = cache
        T = x.shape[0]

        # Feed-forward backward
        d_ff  = d_out                     # gradient flows through residual
        d_x_attn = d_out                  # residual branch

        dW2   = h.T @ d_ff
        db2   = d_ff.sum(axis=0)
        d_h   = d_ff @ self.W2.T
        d_h   = d_h * (h > 0)            # ReLU derivative
        dW1   = x_attn.T @ d_h
        db1   = d_h.sum(axis=0)
        d_x_attn = d_x_attn + d_h @ self.W1.T

        # Attention backward
        d_attn_out = d_x_attn
        d_x_res    = d_x_attn            # residual branch back to input

        # d_A = d_attn_out @ V.T,  d_V = A.T @ d_attn_out
        d_A = d_attn_out @ V.T           # (T, T)
        dV  = A.T @ d_attn_out           # (T, D)

        # Softmax backward through A
        dS  = A * (d_A - (d_A * A).sum(axis=1, keepdims=True))  # (T, T)
        dS /= np.sqrt(float(self.embed_dim))

        dQ  = dS   @ K                   # (T, D)
        dK  = dS.T @ Q                   # (T, D)

        dWq = x.T @ dQ
        dWk = x.T @ dK
        dWv = x.T @ dV
        d_x = dQ @ self.Wq.T + dK @ self.Wk.T + dV @ self.Wv.T + d_x_res

        return d_x, dWq, dWk, dWv, dW1, db1, dW2, db2

    # ── Full forward pass ─────────────────────────────────────────────────────

    def _transformer_forward(self, token_idx_batch):
        """
        Full batched transformer forward pass.

        token_idx_batch: (context_size, batch_size) int array
        Returns probs (batch_size, vocab) and per-sample caches for backprop.
        """
        bs = token_idx_batch.shape[1]
        all_probs  = []
        all_caches = []  # list of (x_init, block_caches) per sample

        # Process each sample individually (simple; vectorised later if needed)
        for s in range(bs):
            toks = token_idx_batch[:, s]     # (T,) int

            # Embedding + positional
            x = self.embedding[toks] + self.pos_embedding   # (T, D)

            block_caches = []
            for _ in range(self.num_blocks):
                x, cache = self._block_forward(x)
                block_caches.append(cache)

            # Output: use last token
            logits = x[-1] @ self.Wout + self.bout   # (vocab,)
            e = np.exp(logits - logits.max())
            probs = e / e.sum()
            all_probs.append(probs)
            all_caches.append((toks, x, block_caches))

        # Stack probs: (vocab, batch) for cross-entropy compatibility
        probs_batch = np.stack(all_probs, axis=1)   # (vocab, bs)
        return probs_batch, all_caches

    # ── Legacy _forward_batch (kept for predict / compat) ─────────────────────

    def _forward_batch(self, X, token_idx_batch=None):
        """Legacy interface — uses transformer path when embeddings active."""
        if self.use_embedding and token_idx_batch is not None:
            probs, _ = self._transformer_forward(token_idx_batch)
            return None, None, probs
        # Fallback dense path (no-embedding mode)
        a = X
        activations = [a]; zs = []
        for i in range(len(self.hidden_layers)):
            z = self.weights[i] @ a + self.biases[i]
            a = self._act_fn(z)
            zs.append(z); activations.append(a)
        z_out = self.weights[-1] @ a + self.biases[-1]
        probs = self._softmax(z_out)
        zs.append(z_out); activations.append(probs)
        return zs, activations, probs

    # ── Single-sample forward (for predict) ──────────────────────────────────

    def forward(self, inputs):
        """Single-sample forward pass used by predict()."""
        if self.use_embedding:
            arr  = np.array(inputs, dtype=float).reshape(self.context_size, self.vocab_size)
            toks = np.array(arr.argmax(axis=1), dtype=int).reshape(self.context_size, 1)
            probs, _ = self._transformer_forward(toks)
            if _DEVICE == "gpu":
                probs = np.asnumpy(probs)
            return None, None, probs[:, 0]
        X = np.array(inputs, dtype=float).reshape(-1, 1)
        zs, activations, probs = self._forward_batch(X)
        return zs, activations, probs[:, 0]

    # ── Train ─────────────────────────────────────────────────────────────────

    def train(self, data: List[Sample], epochs: int, log_every: int = 1) -> None:
        """
        Train with mini-batch Adam gradient descent.

        Backpropagates through the full transformer:
        softmax → Wout → transformer blocks (attention + ff) → embeddings.
        """
        if not self._adam_init:
            self._init_adam()

        n        = len(data)
        ctx_size = self.context_size
        vs       = self.vocab_size

        print("  Loading dataset onto device...")
        X_idx = np.zeros((ctx_size, n), dtype=int)
        Y_all = np.zeros((self.output_size, n), dtype=float)
        for j, (feat, label) in enumerate(data):
            oh = np.array(feat).reshape(ctx_size, vs)
            X_idx[:, j] = oh.argmax(axis=1)
            Y_all[label, j] = 1.0

        print(f"  {_DEVICE.upper()} ready — {n:,} samples | "
              f"batch={self.batch_size} | optimizer=Adam | "
              f"embed={'ON' if self.use_embedding else 'OFF'} | "
              f"transformer=ON (blocks={self.num_blocks})\n")

        for epoch in range(epochs):
            idx    = list(range(n)); random.shuffle(idx)
            idx_np = np.array(idx)
            X_shuf = X_idx[:, idx_np]
            Y_shuf = Y_all[:, idx_np]

            total_loss  = 0.0
            self._adam_t += 1

            # Accumulate gradients for transformer params over all batches
            dWq_acc   = np.zeros_like(self.Wq)
            dWk_acc   = np.zeros_like(self.Wk)
            dWv_acc   = np.zeros_like(self.Wv)
            dW1_acc   = np.zeros_like(self.W1)
            db1_acc   = np.zeros_like(self.b1)
            dW2_acc   = np.zeros_like(self.W2)
            db2_acc   = np.zeros_like(self.b2)
            dWout_acc = np.zeros_like(self.Wout)
            dbout_acc = np.zeros_like(self.bout)
            if self.use_embedding:
                de_acc  = np.zeros_like(self.embedding)
                dpe_acc = np.zeros_like(self.pos_embedding)

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                bs  = end - start
                Xb  = X_shuf[:, start:end]
                Yb  = Y_shuf[:, start:end]

                probs, caches = self._transformer_forward(Xb)
                total_loss   += float(-np.sum(Yb * np.log(probs + 1e-9)))

                # Cross-entropy gradient: (probs - targets) / bs
                delta = (probs - Yb) / bs   # (vocab, bs)

                for s in range(bs):
                    toks_s, x_last, block_caches = caches[s]
                    d_s = delta[:, s]                  # (vocab,)

                    # Output layer backward
                    dWout_acc += np.outer(x_last[-1], d_s)
                    dbout_acc += d_s
                    d_x_last   = self.Wout @ d_s       # (D,)

                    # Gradient flows only through last token position
                    d_x = np.zeros((self.context_size, self.embed_dim))
                    d_x[-1] = d_x_last

                    # Backprop through transformer blocks (reversed)
                    for cache in reversed(block_caches):
                        d_x, dWq, dWk, dWv, dW1, db1, dW2, db2 = \
                            self._block_backward(d_x, cache)
                        dWq_acc += dWq;  dWk_acc += dWk;  dWv_acc += dWv
                        dW1_acc += dW1;  db1_acc += db1
                        dW2_acc += dW2;  db2_acc += db2

                    # Embedding + positional encoding backward
                    if self.use_embedding:
                        dpe_acc += d_x
                        if _DEVICE == "gpu":
                            import numpy as _np
                            d_x_cpu   = np.asnumpy(d_x)
                            toks_cpu  = np.asnumpy(toks_s)
                            de_cpu    = np.asnumpy(de_acc)
                            _np.add.at(de_cpu, toks_cpu, d_x_cpu)
                            de_acc    = np.array(de_cpu)
                        else:
                            import numpy as _np
                            _np.add.at(de_acc, toks_s, d_x)

            # Adam updates for all transformer parameters
            self.Wq,   self._mWq,  self._vWq  = self._adam_update(self.Wq,   dWq_acc,   self._mWq,  self._vWq)
            self.Wk,   self._mWk,  self._vWk  = self._adam_update(self.Wk,   dWk_acc,   self._mWk,  self._vWk)
            self.Wv,   self._mWv,  self._vWv  = self._adam_update(self.Wv,   dWv_acc,   self._mWv,  self._vWv)
            self.W1,   self._mW1,  self._vW1  = self._adam_update(self.W1,   dW1_acc,   self._mW1,  self._vW1)
            self.b1,   self._mb1,  self._vb1  = self._adam_update(self.b1,   db1_acc,   self._mb1,  self._vb1)
            self.W2,   self._mW2,  self._vW2  = self._adam_update(self.W2,   dW2_acc,   self._mW2,  self._vW2)
            self.b2,   self._mb2,  self._vb2  = self._adam_update(self.b2,   db2_acc,   self._mb2,  self._vb2)
            self.Wout, self._mWout,self._vWout = self._adam_update(self.Wout, dWout_acc, self._mWout,self._vWout)
            self.bout, self._mbout,self._vbout = self._adam_update(self.bout, dbout_acc, self._mbout,self._vbout)
            if self.use_embedding:
                self.embedding,     self._me,  self._ve  = self._adam_update(self.embedding,     de_acc,  self._me,  self._ve)
                self.pos_embedding, self._mpe, self._vpe = self._adam_update(self.pos_embedding, dpe_acc, self._mpe, self._vpe)

            if log_every and epoch % log_every == 0:
                print(f"Epoch {epoch:>6} | Loss: {total_loss:.2f}")

        print("Training complete.")

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, inputs) -> Tuple[int, float, "np.ndarray"]:
        """
        Predict the most likely class for one sample.

        :param inputs: Flat one-hot feature vector (same format as training data).
        :type inputs: list[float]
        :return: `(predicted_class, confidence, all_probs)
        :rtype: tuple[int, float, np.ndarray]
        """
        _, _, probs = self.forward(inputs)
        if _DEVICE == "gpu":
            probs = np.asnumpy(probs)
        predicted_class = int(probs.argmax())
        return predicted_class, float(probs[predicted_class]), probs

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> None:
        """Print architecture, device, optimizer, and embedding info."""
        attn_params = self.Wq.size + self.Wk.size + self.Wv.size
        ff_params   = self.W1.size + self.b1.size + self.W2.size + self.b2.size
        out_params  = self.Wout.size + self.bout.size
        emb_params  = self.embedding.size + self.pos_embedding.size if self.use_embedding else 0
        total       = (attn_params + ff_params + out_params + emb_params) * self.num_blocks
        width       = 52
        device      = "GPU (CuPy)" if _DEVICE == "gpu" else "CPU (NumPy)"

        print("╔" + "═" * width + "╗")
        print("║" + " Mini-Transformer Summary".center(width) + "║")
        print("╠" + "═" * width + "╣")
        print(f"║  {'Device':<16} │ {device:<{width - 22}}║")
        print(f"║  {'Optimizer':<16} │ {'Adam':<{width - 22}}║")
        print(f"║  {'Batch size':<16} │ {self.batch_size:<{width - 22}}║")
        if self.use_embedding:
            edim = f"{self.vocab_size} chars × {self.embed_dim}d  context={self.context_size}"
            print(f"║  {'Embedding':<16} │ {edim:<{width - 22}}║")
        print(f"║  {'Blocks':<16} │ {self.num_blocks:<{width - 22}}║")
        print(f"║  {'Attention':<16} │ Wq/Wk/Wv {self.embed_dim}×{self.embed_dim} ({attn_params:,} params each block)║")
        print(f"║  {'Feed-forward':<16} │ {self.embed_dim}→{self.embed_dim*4}→{self.embed_dim} ({ff_params:,} params each block)║")
        print(f"║  {'Output':<16} │ {self.embed_dim}→{self.output_size} ({out_params:,} params)║")
        print("╠" + "═" * width + "╣")
        print(f"║  Total parameters: {total:,}{'':<{width - 23 - len(f'{total:,}')}}║")
        print("╚" + "═" * width + "╝")

    # ── Save ──────────────────────────────────────────────────────────────────

    def save_weights(self, filename: str = "weights.json") -> None:
        """
        Save full model state including embeddings and transformer weights to JSON.

        :param filename: Output path. Default `"weights.json".
        :type filename: str
        """
        to_list = (lambda w: np.asnumpy(w).tolist()) if _DEVICE == "gpu" else (lambda w: w.tolist())
        data = {
            "input_size":    self.input_size,
            "hidden_layers": self.hidden_layers,
            "output_size":   self.output_size,
            "activation":    self.activation,
            "learning_rate": self.learning_rate,
            "batch_size":    self.batch_size,
            "use_embedding": self.use_embedding,
            "vocab_size":    self.vocab_size,
            "context_size":  self.context_size,
            "embed_dim":     self.embed_dim,
            "num_blocks":    self.num_blocks,
            "device":        _DEVICE,
            "weights":       [to_list(w) for w in self.weights],
            "biases":        [to_list(b) for b in self.biases],
            "embedding":     to_list(self.embedding)     if self.use_embedding else None,
            "pos_embedding": to_list(self.pos_embedding) if self.use_embedding else None,
            "Wq":  to_list(self.Wq),  "Wk": to_list(self.Wk), "Wv": to_list(self.Wv),
            "W1":  to_list(self.W1),  "b1": to_list(self.b1),
            "W2":  to_list(self.W2),  "b2": to_list(self.b2),
            "Wout":to_list(self.Wout),"bout":to_list(self.bout),
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Weights saved to '{filename}'.")

    # ── Load ──────────────────────────────────────────────────────────────────

    def load_weights(self, filename: str = "weights.json") -> None:
        """
        Restore model from JSON including transformer weights.

        :param filename: Path to load. Default `"weights.json".
        :type filename: str
        """
        if not os.path.exists(filename):
            print(f"No weights file found at '{filename}'."); return
        with open(filename) as f:
            data = json.load(f)

        self.input_size    = data["input_size"]
        self.hidden_layers = data["hidden_layers"]
        self.output_size   = data["output_size"]
        self.activation    = data["activation"]
        self.learning_rate = data["learning_rate"]
        self.batch_size    = data.get("batch_size", 512)
        self.use_embedding = data.get("use_embedding", False)
        self.vocab_size    = data.get("vocab_size", 0)
        self.context_size  = data.get("context_size", 0)
        self.embed_dim     = data.get("embed_dim", 64)
        self.num_blocks    = data.get("num_blocks", 2)

        self._act_fn, self._act_d = _ACTIVATIONS[self.activation]
        self.weights   = [np.array(w) for w in data["weights"]]
        self.biases    = [np.array(b) for b in data["biases"]]
        self.embedding     = np.array(data["embedding"])     if data.get("embedding")     else None
        self.pos_embedding = np.array(data["pos_embedding"]) if data.get("pos_embedding") else None
        self.Wq   = np.array(data["Wq"])
        self.Wk   = np.array(data["Wk"])
        self.Wv   = np.array(data["Wv"])
        self.W1   = np.array(data["W1"])
        self.b1   = np.array(data["b1"])
        self.W2   = np.array(data["W2"])
        self.b2   = np.array(data["b2"])
        self.Wout = np.array(data["Wout"])
        self.bout = np.array(data["bout"])
        self._adam_init = False
        print(f"Weights loaded from '{filename}'.")

    def __repr__(self):
        layers = " → ".join([str(self.input_size)] +
                            [str(h) for h in self.hidden_layers] +
                            [str(self.output_size)])
        return (f"NeuralNetwork({layers}, act='{self.activation}', "
                f"lr={self.learning_rate}, embed={'ON' if self.use_embedding else 'OFF'}, "
                f"transformer=ON, device='{self.device}')")