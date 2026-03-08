"""
Neural_Network.py  — Adam optimizer + Embedding edition
========================================================
Two key upgrades that dramatically improve language model training:

1. EMBEDDINGS
   Instead of one-hot vectors (135-dim sparse), each character is mapped
   to a dense learned vector (32-dim). This compresses the input from
   1620 floats down to 384, and the embeddings themselves learn semantic
   relationships between characters.

2. ADAM OPTIMIZER
   Replaces plain SGD. Adam tracks the momentum and variance of gradients
   and adapts the step size per-parameter. This eliminates the plateau
   problem — loss drops much faster and more consistently.

Expected improvement: loss should drop to ~50,000 range within 100 epochs
instead of barely moving with plain SGD.

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
    Feedforward neural network with character embeddings and Adam optimizer.

    **What's new vs the previous version:**

    *Embeddings* — each input token index is mapped to a dense learned
    vector of size ``embed_dim`` instead of a sparse one-hot vector.
    This shrinks the input dramatically and lets the network learn that
    some characters are semantically similar.

    *Adam optimizer* — replaces plain SGD. Adam adapts the learning rate
    individually for each weight based on the history of its gradients.
    This eliminates plateaus and converges much faster.

    :param input_size: Number of input features **OR** vocabulary size
        when ``use_embedding=True``. When using embeddings this should
        be set to ``vocab_size * context_size`` (the old flat one-hot
        size) — the network handles the reshaping internally.
    :type input_size: int

    :param hidden_layers: Neuron counts per hidden layer.
    :type hidden_layers: list[int]

    :param output_size: Number of output classes (vocab size for LM).
    :type output_size: int

    :param activation: ``"relu"`` (default), ``"tanh"``, ``"sigmoid"``,
        ``"leaky_relu"``.
    :type activation: str

    :param learning_rate: Adam base learning rate. Default ``0.001``.
        Much smaller values work better with Adam than with SGD.
    :type learning_rate: float

    :param batch_size: Samples per gradient update. Default ``256``.
    :type batch_size: int

    :param use_embedding: If ``True``, replaces one-hot input with a
        learned embedding table. Requires ``vocab_size``,
        ``context_size``, and ``embed_dim`` to be set.
        Default ``True``.
    :type use_embedding: bool

    :param vocab_size: Number of unique tokens (characters). Required
        when ``use_embedding=True``.
    :type vocab_size: int

    :param context_size: Number of context tokens per sample. Required
        when ``use_embedding=True``.
    :type context_size: int

    :param embed_dim: Size of each character embedding vector.
        Default ``32``. Larger = more expressive but slower.
    :type embed_dim: int

    **Example — language model with embeddings:**

    .. code-block:: python

        nn = NeuralNetwork(
            input_size    = vocab_size * context_size,  # kept for compat
            hidden_layers = [512, 256, 128],
            output_size   = vocab_size,
            activation    = "relu",
            learning_rate = 0.001,
            batch_size    = 256,
            use_embedding = True,
            vocab_size    = vocab_size,
            context_size  = context_size,
            embed_dim     = 32,
        )
        nn.train(samples, epochs=100)
    """

    def __init__(
        self,
        input_size:    int,
        hidden_layers: List[int],
        output_size:   int,
        activation:    str   = "relu",
        learning_rate: float = 0.001,
        batch_size: int = 512,
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

        self._act_fn, self._act_d = _ACTIVATIONS[activation]

        # ── Embedding table ───────────────────────────────────────────────────
        # Shape: (vocab_size, embed_dim)
        # Each row is the learned vector for one character.
        if self.use_embedding:
            self.embedding = np.random.randn(vocab_size, embed_dim) * 0.01
            actual_input   = context_size * embed_dim   # dense input size
        else:
            self.embedding = None
            actual_input   = input_size

        # ── Weight matrices ───────────────────────────────────────────────────
        layer_sizes  = [actual_input] + hidden_layers + [output_size]
        self.weights = []
        self.biases  = []
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.weights.append(np.random.randn(fan_out, fan_in) * scale)
            self.biases.append(np.zeros((fan_out, 1)))

        # ── Adam moment buffers (initialised lazily on first train call) ──────
        self._adam_init = False

    def _init_adam(self):
        """Initialise Adam first-moment (m) and second-moment (v) buffers."""
        self._mw = [np.zeros_like(w) for w in self.weights]
        self._vw = [np.zeros_like(w) for w in self.weights]
        self._mb = [np.zeros_like(b) for b in self.biases]
        self._vb = [np.zeros_like(b) for b in self.biases]
        if self.use_embedding:
            self._me = np.zeros_like(self.embedding)
            self._ve = np.zeros_like(self.embedding)
        self._adam_t    = 0
        self._adam_init = True

    # ── Softmax ───────────────────────────────────────────────────────────────

    def _softmax(self, x):
        """Numerically stable column-wise softmax for batches."""
        e = np.exp(x - x.max(axis=0, keepdims=True))
        return e / e.sum(axis=0, keepdims=True)

    # ── Embed + flatten ───────────────────────────────────────────────────────

    def _embed(self, token_indices_batch):
        """
        Look up embeddings for a batch of token-index sequences and flatten.

        :param token_indices_batch: Shape ``(context_size, batch_size)``
            integer array of token indices.
        :type token_indices_batch: np.ndarray
        :return: Flattened embedding matrix,
            shape ``(context_size * embed_dim, batch_size)``.
        :rtype: np.ndarray
        """
        # embedding[idx] for each position → (context, batch, embed_dim)
        # then reshape to (context*embed_dim, batch)
        cs, bs = token_indices_batch.shape
        out    = np.zeros((cs * self.embed_dim, bs), dtype=float)
        for t in range(cs):
            rows = token_indices_batch[t]          # (bs,)  — token indices
            out[t * self.embed_dim:(t + 1) * self.embed_dim, :] = \
                self.embedding[rows].T             # (embed_dim, bs)
        return out

    # ── Forward (batched) ─────────────────────────────────────────────────────

    def _forward_batch(self, X, token_idx_batch=None):
        """
        Batched forward pass.

        :param X: Input matrix ``(input_size, batch_size)`` — used when
            ``use_embedding=False``.
        :param token_idx_batch: Token index matrix
            ``(context_size, batch_size)`` — used when
            ``use_embedding=True``.
        :return: ``(zs, activations, probs)``
        :rtype: tuple
        """
        if self.use_embedding and token_idx_batch is not None:
            a = self._embed(token_idx_batch)
        else:
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
        """
        Single-sample forward pass used by :meth:`predict`.

        Accepts the flat one-hot vector format for backwards compatibility.

        :param inputs: Flat feature vector, length ``input_size``.
        :type inputs: list[float]
        :return: ``(zs, activations, probs)`` with batch dim squeezed.
        :rtype: tuple
        """
        if self.use_embedding:
            # Decode token indices from flat one-hot input
            arr  = np.array(inputs, dtype=float).reshape(self.context_size, self.vocab_size)
            toks = np.array(arr.argmax(axis=1), dtype=int).reshape(self.context_size, 1)
            zs, activations, probs = self._forward_batch(None, toks)
        else:
            X = np.array(inputs, dtype=float).reshape(-1, 1)
            zs, activations, probs = self._forward_batch(X)

        return ([z[:, 0] for z in zs],
                [a[:, 0] for a in activations],
                probs[:, 0])

    # ── Adam update ───────────────────────────────────────────────────────────

    def _adam_update(self, param, grad, m, v, lr=None,
                     beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Apply one Adam gradient update step.

        Adam maintains running estimates of the first moment (mean) and
        second moment (variance) of gradients and uses them to scale the
        effective learning rate per parameter.

        :param param: Weight or bias array to update in-place.
        :param grad: Gradient array, same shape as ``param``.
        :param m: First-moment buffer (updated in-place).
        :param v: Second-moment buffer (updated in-place).
        :param lr: Learning rate override. Uses ``self.learning_rate`` if None.
        :param beta1: Decay for first moment. Default ``0.9``.
        :param beta2: Decay for second moment. Default ``0.999``.
        :param eps: Numerical stability constant. Default ``1e-8``.
        :return: Updated ``(param, m, v)``.
        """
        if lr is None:
            lr = self.learning_rate
        t      = self._adam_t
        m[:]   = beta1 * m + (1 - beta1) * grad
        v[:]   = beta2 * v + (1 - beta2) * grad ** 2
        m_hat  = m / (1 - beta1 ** t)
        v_hat  = v / (1 - beta2 ** t)
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)
        return param, m, v

    # ── Train ─────────────────────────────────────────────────────────────────

    def train(self, data: List[Sample], epochs: int, log_every: int = 1) -> None:
        """
        Train with mini-batch Adam gradient descent and optional embeddings.

        :param data: List of ``([features], label)`` pairs. Same format
            as before — no changes needed in calling code.
        :type data: list[tuple[list[float], int]]
        :param epochs: Full passes over the dataset.
        :type epochs: int
        :param log_every: Print loss every N epochs. Default ``1``.
        :type log_every: int

        **Example:**

        .. code-block:: python

            nn.train(samples, epochs=100, log_every=10)
        """
        if not self._adam_init:
            self._init_adam()

        n          = len(data)
        num_hidden = len(self.hidden_layers)

        print("  Loading dataset onto device...")
        if self.use_embedding:
            # Store token indices instead of one-hot vectors — much smaller
            # Shape: (context_size, n)
            ctx_size = self.context_size
            vs       = self.vocab_size
            X_idx    = np.zeros((ctx_size, n), dtype=int)
            for j, (feat, _) in enumerate(data):
                oh = np.array(feat).reshape(ctx_size, vs)
                X_idx[:, j] = oh.argmax(axis=1)
        else:
            X_all = np.array([s[0] for s in data], dtype=float).T

        Y_all = np.zeros((self.output_size, n), dtype=float)
        for j, (_, label) in enumerate(data):
            Y_all[label, j] = 1.0

        print(f"  {_DEVICE.upper()} ready — {n:,} samples | "
              f"batch={self.batch_size} | optimizer=Adam | "
              f"embed={'ON' if self.use_embedding else 'OFF'}\n")

        for epoch in range(epochs):
            idx    = list(range(n)); random.shuffle(idx)
            idx_np = np.array(idx)
            Y_shuf = Y_all[:, idx_np]

            if self.use_embedding:
                X_shuf = X_idx[:, idx_np]
            else:
                X_shuf = X_all[:, idx_np]

            total_loss = 0.0
            self._adam_t += 1

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                Yb  = Y_shuf[:, start:end]
                bs  = end - start

                if self.use_embedding:
                    Xb_idx = X_shuf[:, start:end]
                    zs, activations, probs = self._forward_batch(None, Xb_idx)
                else:
                    Xb = X_shuf[:, start:end]
                    zs, activations, probs = self._forward_batch(Xb)

                total_loss += float(-np.sum(Yb * np.log(probs + 1e-9)))

                # Backprop
                delta = (probs - Yb) / bs
                weight_grads = [None] * len(self.weights)
                bias_grads   = [None] * len(self.biases)

                weight_grads[-1] = delta @ activations[-2].T
                bias_grads[-1]   = delta.sum(axis=1, keepdims=True)

                for i in range(num_hidden - 1, -1, -1):
                    delta           = (self.weights[i + 1].T @ delta) * self._act_d(zs[i])
                    weight_grads[i] = delta @ activations[i].T
                    bias_grads[i]   = delta.sum(axis=1, keepdims=True)

                # Embedding gradient
                if self.use_embedding:
                    # delta here is the gradient at the first hidden layer input
                    # Propagate back through weights[0] to get embed-space grad
                    d_embed = (self.weights[0].T @ delta)  # (embed_flat, bs)
                    d_embed = d_embed.reshape(self.context_size, self.embed_dim, bs)
                    if _DEVICE == "gpu":
                        d_embed_cpu = np.asnumpy(d_embed)
                        toks_cpu    = np.asnumpy(Xb_idx)
                    else:
                        d_embed_cpu = d_embed
                        toks_cpu    = Xb_idx

                    import numpy as _np
                    grad_e = _np.zeros((_np if _DEVICE == "cpu" else np).asnumpy(self.embedding).shape
                                       if _DEVICE == "gpu" else self.embedding.shape)
                    for t in range(self.context_size):
                        _np.add.at(grad_e, toks_cpu[t], d_embed_cpu[t].T)

                    if _DEVICE == "gpu":
                        grad_e_gpu = np.array(grad_e)
                    else:
                        grad_e_gpu = grad_e

                    self.embedding, self._me, self._ve = self._adam_update(
                        self.embedding, grad_e_gpu / bs, self._me, self._ve)

                # Adam updates for weights and biases
                for i in range(len(self.weights)):
                    self.weights[i], self._mw[i], self._vw[i] = self._adam_update(
                        self.weights[i], weight_grads[i], self._mw[i], self._vw[i])
                    self.biases[i], self._mb[i], self._vb[i] = self._adam_update(
                        self.biases[i], bias_grads[i], self._mb[i], self._vb[i])

            if log_every and epoch % log_every == 0:
                print(f"Epoch {epoch:>6} | Loss: {total_loss:.2f}")

        print("Training complete.")

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, inputs) -> Tuple[int, float, "np.ndarray"]:
        """
        Predict the most likely class for one sample.

        :param inputs: Flat one-hot feature vector (same format as training data).
        :type inputs: list[float]
        :return: ``(predicted_class, confidence, all_probs)``
        :rtype: tuple[int, float, np.ndarray]

        **Example:**

        .. code-block:: python

            cls, conf, probs = nn.predict(features)
        """
        _, _, probs = self.forward(inputs)
        if _DEVICE == "gpu":
            probs = np.asnumpy(probs)
        predicted_class = int(probs.argmax())
        return predicted_class, float(probs[predicted_class]), probs

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> None:
        """Print architecture, device, optimizer, and embedding info."""
        total  = sum(w.size + b.size for w, b in zip(self.weights, self.biases))
        if self.use_embedding:
            total += self.embedding.size
        width  = 52
        device = "GPU (CuPy)" if _DEVICE == "gpu" else "CPU (NumPy)"

        print("╔" + "═" * width + "╗")
        print("║" + " Neural Network Summary".center(width) + "║")
        print("╠" + "═" * width + "╣")
        print(f"║  {'Device':<16} │ {device:<{width - 22}}║")
        print(f"║  {'Optimizer':<16} │ {'Adam':<{width - 22}}║")
        print(f"║  {'Batch size':<16} │ {self.batch_size:<{width - 22}}║")
        if self.use_embedding:
            edim = f"{self.vocab_size} chars × {self.embed_dim}d  →  {self.context_size * self.embed_dim} input"
            print(f"║  {'Embedding':<16} │ {edim:<{width - 22}}║")
        else:
            print(f"║  {'Input':<16} │ features: {self.input_size:<{width - 30}}║")
        for i, size in enumerate(self.hidden_layers):
            p    = self.weights[i].size + self.biases[i].size
            line = f"  {'Hidden ' + str(i+1):<16} │ {size:>4} neurons │ {self.activation} ({p:,} params)"
            print(f"║{line:<{width}}║")
        op   = self.weights[-1].size + self.biases[-1].size
        line = f"  {'Output':<16} │ {self.output_size:>4} classes │ softmax ({op:,} params)"
        print(f"║{line:<{width}}║")
        print("╠" + "═" * width + "╣")
        print(f"║  Total parameters: {total:,}{'':<{width - 23 - len(f'{total:,}')}}║")
        print("╚" + "═" * width + "╝")

    # ── Save ──────────────────────────────────────────────────────────────────

    def save_weights(self, filename: str = "weights.json") -> None:
        """
        Save full model state including embeddings to JSON.

        :param filename: Output path. Default ``"weights.json"``.
        :type filename: str
        """
        to_list = (lambda w: np.asnumpy(w).tolist()) if _DEVICE == "gpu" else (lambda w: w.tolist())
        data = {
            "input_size": self.input_size, "hidden_layers": self.hidden_layers,
            "output_size": self.output_size, "activation": self.activation,
            "learning_rate": self.learning_rate, "batch_size": self.batch_size,
            "use_embedding": self.use_embedding, "vocab_size": self.vocab_size,
            "context_size": self.context_size, "embed_dim": self.embed_dim,
            "device": _DEVICE,
            "weights":   [to_list(w) for w in self.weights],
            "biases":    [to_list(b) for b in self.biases],
            "embedding": to_list(self.embedding) if self.use_embedding else None,
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Weights saved to '{filename}'.")

    # ── Load ──────────────────────────────────────────────────────────────────

    def load_weights(self, filename: str = "weights.json") -> None:
        """
        Restore model from JSON including embeddings.

        :param filename: Path to load. Default ``"weights.json"``.
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
        self.batch_size    = data.get("batch_size", 256)
        self.use_embedding = data.get("use_embedding", False)
        self.vocab_size    = data.get("vocab_size", 0)
        self.context_size  = data.get("context_size", 0)
        self.embed_dim     = data.get("embed_dim", 32)

        self._act_fn, self._act_d = _ACTIVATIONS[self.activation]
        self.weights   = [np.array(w) for w in data["weights"]]
        self.biases    = [np.array(b) for b in data["biases"]]
        self.embedding = np.array(data["embedding"]) if data.get("embedding") else None
        self._adam_init = False
        print(f"Weights loaded from '{filename}'.")

    def __repr__(self):
        layers = " → ".join([str(self.input_size)] +
                            [str(h) for h in self.hidden_layers] +
                            [str(self.output_size)])
        return (f"NeuralNetwork({layers}, act='{self.activation}', "
                f"lr={self.learning_rate}, embed={'ON' if self.use_embedding else 'OFF'}, "
                f"device='{self.device}')")