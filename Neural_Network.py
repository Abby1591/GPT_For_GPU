"""
Neural_Network.py  — Mini-Transformer edition (causal, all-positions training)
===============================================================================
Key upgrades vs previous version:
  1. CAUSAL MASK — each token can only attend to previous tokens (real GPT behaviour)
  2. ALL-POSITIONS TRAINING — every token position predicts the next one,
     giving T x more gradient signal per sample instead of just the last token
  3. PER-BLOCK INDEPENDENT WEIGHTS — each block learns different representations
  4. FULLY VECTORISED — no Python loops over samples

GPU SETUP:
    pip install cupy-cuda12x
"""

from __future__ import annotations
import json, os, random
from typing import List, Literal, Tuple

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

def _relu(x):                 return np.maximum(0.0, x)
def _relu_d(x):               return (x > 0).astype(float)
def _tanh(x):                 return np.tanh(x)
def _tanh_d(x):               return 1.0 - np.tanh(x) ** 2
def _sigmoid(x):              return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
def _sigmoid_d(x):            s = _sigmoid(x); return s * (1.0 - s)
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
    Mini-GPT transformer with causal masking and all-positions training.

    Forward pass:
        tokens (B, T)
        -> embedding + positional encoding   (B, T, D)
        -> transformer block x num_blocks    (B, T, D)   [causal mask applied]
        -> linear projection at ALL positions (B, T, vocab)
        -> softmax

    Training:
        Each sample is a sequence of T+1 tokens.
        Positions 0..T-1 are inputs, positions 1..T are targets.
        Loss computed over all T positions simultaneously.
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

        # ── Embeddings ────────────────────────────────────────────────────────
        if self.use_embedding:
            self.embedding     = np.random.randn(vocab_size, embed_dim) * 0.01
            self.pos_embedding = np.random.randn(context_size, embed_dim) * 0.01
        else:
            self.embedding     = None
            self.pos_embedding = None

        # ── Per-block independent weights ─────────────────────────────────────
        D = embed_dim
        self.blocks = []
        for _ in range(self.num_blocks):
            self.blocks.append({
                "Wq": np.random.randn(D, D) * 0.02,
                "Wk": np.random.randn(D, D) * 0.02,
                "Wv": np.random.randn(D, D) * 0.02,
                "W1": np.random.randn(D, D * 4) * 0.02,
                "b1": np.zeros(D * 4),
                "W2": np.random.randn(D * 4, D) * 0.02,
                "b2": np.zeros(D),
            })

        # ── Output projection ─────────────────────────────────────────────────
        self.Wout = np.random.randn(embed_dim, output_size) * 0.02
        self.bout = np.zeros(output_size)

        # ── Legacy dense weights (save/load compat) ───────────────────────────
        actual_input = context_size * embed_dim if self.use_embedding else input_size
        layer_sizes  = [actual_input] + hidden_layers + [output_size]
        self.weights = []
        self.biases  = []
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.weights.append(np.random.randn(fan_out, fan_in) * scale)
            self.biases.append(np.zeros((fan_out, 1)))

        self._adam_init = False

    # ── Adam ──────────────────────────────────────────────────────────────────

    def _init_adam(self):
        z = np.zeros_like
        self._adam_blocks = []
        for blk in self.blocks:
            self._adam_blocks.append({k: {"m": z(v), "v": z(v)} for k, v in blk.items()})
        self._mWout = z(self.Wout); self._vWout = z(self.Wout)
        self._mbout = z(self.bout); self._vbout = z(self.bout)
        if self.use_embedding:
            self._me  = z(self.embedding);     self._ve  = z(self.embedding)
            self._mpe = z(self.pos_embedding); self._vpe = z(self.pos_embedding)
        self._adam_t    = 0
        self._adam_init = True

    def _adam_update(self, param, grad, m, v, lr=None,
                     beta1=0.9, beta2=0.999, eps=1e-8):
        if lr is None:
            lr = self.learning_rate
        t    = self._adam_t
        m[:] = beta1 * m + (1 - beta1) * grad
        v[:] = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)
        return param, m, v

    # ── Causal mask ───────────────────────────────────────────────────────────

    def _causal_mask(self, T):
        """Upper-triangular mask: position i cannot attend to j > i."""
        mask = np.triu(np.ones((T, T)), k=1) * -1e9
        return mask   # (T, T), added to scores before softmax

    # ── Transformer block ─────────────────────────────────────────────────────

    def _block_forward(self, x, blk):
        """
        Causal transformer block. x: (B, T, D) -> (B, T, D)
        Each position only attends to itself and earlier positions.
        """
        B, T, D = x.shape
        scale = np.sqrt(float(D))
        Wq, Wk, Wv = blk["Wq"], blk["Wk"], blk["Wv"]
        W1, b1, W2, b2 = blk["W1"], blk["b1"], blk["W2"], blk["b2"]

        Q = x @ Wq                                             # (B, T, D)
        K = x @ Wk
        V = x @ Wv

        scores  = np.matmul(Q, K.transpose(0, 2, 1)) / scale  # (B, T, T)
        scores += self._causal_mask(T)                         # mask future
        scores -= scores.max(axis=2, keepdims=True)
        exp_s   = np.exp(scores)
        A       = exp_s / exp_s.sum(axis=2, keepdims=True)

        attn_out = np.matmul(A, V)                             # (B, T, D)
        x_attn   = x + attn_out                                # residual

        h      = np.maximum(0.0, x_attn @ W1 + b1)            # (B, T, 4D)
        ff_out = h @ W2 + b2                                   # (B, T, D)
        x_out  = x_attn + ff_out                               # residual

        cache = (x, Q, K, V, A, attn_out, x_attn, h, ff_out)
        return x_out, cache

    def _block_backward(self, d_out, cache, blk):
        """Vectorised backprop through causal block. d_out: (B, T, D)"""
        x, Q, K, V, A, attn_out, x_attn, h, ff_out = cache
        B, T, D = x.shape
        Wq, Wk, Wv = blk["Wq"], blk["Wk"], blk["Wv"]
        W1, W2 = blk["W1"], blk["W2"]

        # Feed-forward backward
        d_x_attn = d_out.copy()
        dW2  = np.einsum('bti,btj->ij', h, d_out)
        db2  = d_out.sum(axis=(0, 1))
        d_h  = d_out @ W2.T
        d_h *= (h > 0)
        dW1  = np.einsum('bti,btj->ij', x_attn, d_h)
        db1  = d_h.sum(axis=(0, 1))
        d_x_attn += d_h @ W1.T

        # Attention backward (causal mask is baked into A so no extra step needed)
        d_x_res = d_x_attn.copy()
        d_A = np.matmul(d_x_attn, V.transpose(0, 2, 1))
        dV  = np.matmul(A.transpose(0, 2, 1), d_x_attn)
        dS  = A * (d_A - (d_A * A).sum(axis=2, keepdims=True))
        dS /= np.sqrt(float(D))
        dQ  = np.matmul(dS, K)
        dK  = np.matmul(dS.transpose(0, 2, 1), Q)
        dWq = np.einsum('bti,btj->ij', x, dQ)
        dWk = np.einsum('bti,btj->ij', x, dK)
        dWv = np.einsum('bti,btj->ij', x, dV)
        d_x = (np.matmul(dQ, Wq.T) +
               np.matmul(dK, Wk.T) +
               np.matmul(dV, Wv.T) +
               d_x_res)

        grads = {"Wq": dWq, "Wk": dWk, "Wv": dWv,
                 "W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return d_x, grads

    # ── Forward pass ──────────────────────────────────────────────────────────

    def _transformer_forward(self, token_idx_batch):
        """
        Full forward pass over all positions.
        token_idx_batch: (T, B) int
        Returns probs (B, T, vocab) and cache.
        """
        toks = token_idx_batch.T                               # (B, T)
        x    = self.embedding[toks] + self.pos_embedding       # (B, T, D)

        block_caches = []
        for blk in self.blocks:
            x, cache = self._block_forward(x, blk)
            block_caches.append(cache)

        # Project ALL positions: (B, T, D) -> (B, T, vocab)
        logits  = x @ self.Wout + self.bout
        logits -= logits.max(axis=2, keepdims=True)
        e       = np.exp(logits)
        probs   = e / e.sum(axis=2, keepdims=True)             # (B, T, vocab)

        return probs, (toks, x, block_caches)

    def forward(self, inputs):
        """Single-sample forward for generation — returns last-position probs."""
        if self.use_embedding:
            arr  = np.array(inputs, dtype=float).reshape(self.context_size, self.vocab_size)
            toks = np.array(arr.argmax(axis=1), dtype=int).reshape(self.context_size, 1)
            probs, _ = self._transformer_forward(toks)         # (1, T, vocab)
            if _DEVICE == "gpu":
                probs = np.asnumpy(probs)
            return None, None, probs[0, -1, :]                 # last position
        X = np.array(inputs, dtype=float).reshape(-1, 1)
        a = X; zs = []; activations = [a]
        for i in range(len(self.hidden_layers)):
            z = self.weights[i] @ a + self.biases[i]
            a = self._act_fn(z)
            zs.append(z); activations.append(a)
        z_out = self.weights[-1] @ a + self.biases[-1]
        e     = np.exp(z_out - z_out.max(axis=0, keepdims=True))
        p     = e / e.sum(axis=0, keepdims=True)
        return zs, activations, p[:, 0]

    # ── Train ─────────────────────────────────────────────────────────────────

    def train(self, data: List[Sample], epochs: int, log_every: int = 1) -> None:
        """
        All-positions causal training with adaptive LR.
        """
        if not self._adam_init:
            self._init_adam()

        import math
        import numpy as _np   # always CPU numpy for index ops
        n        = len(data)
        ctx_size = self.context_size
        vs       = self.vocab_size
        D        = self.embed_dim
        lr_max   = self.learning_rate
        lr_min   = lr_max / 10.0

        print("  Loading dataset onto device...")

        # ── Fast path: data is already index arrays (from make_index_arrays) ──
        if isinstance(data, tuple) and len(data) == 2 and hasattr(data[0], 'shape'):
            X_idx_cpu, Y_idx_cpu = data          # (T, N) numpy int arrays
            n = X_idx_cpu.shape[1]
        else:
            # Legacy path: decode one-hot samples back to indices
            X_idx_cpu = _np.zeros((ctx_size, n), dtype=_np.int32)
            Y_idx_cpu = _np.zeros((ctx_size, n), dtype=_np.int32)
            for j, (feat, label) in enumerate(data):
                oh   = _np.array(feat).reshape(ctx_size, vs)
                toks = oh.argmax(axis=1)
                X_idx_cpu[:, j]   = toks
                Y_idx_cpu[:-1, j] = toks[1:]
                Y_idx_cpu[-1,  j] = label

        X_idx = np.array(X_idx_cpu)   # move to GPU if CuPy
        Y_idx = np.array(Y_idx_cpu)

        print(f"  {_DEVICE.upper()} ready — {n:,} samples | "
              f"batch={self.batch_size} | optimizer=Adam+adaptive | "
              f"lr={lr_max:.5f} (bounce×0.5, plateau×0.6, min={lr_min:.5f}) | "
              f"embed={'ON' if self.use_embedding else 'OFF'} | "
              f"transformer=ON (blocks={self.num_blocks}, independent) | "
              f"causal=ON | all-positions=ON\n")

        # ── Adaptive LR state ─────────────────────────────────────────────────
        epoch_lr          = lr_max
        best_loss         = float("inf")
        plateau_count     = 0
        plateau_patience  = 5    # epochs without improvement before reducing
        plateau_factor    = 0.6  # lr multiplier on plateau
        bounce_factor     = 0.5  # lr multiplier when loss goes UP
        prev_loss         = float("inf")
        lr_change_msg     = ""

        for epoch in range(epochs):
            # ── Warmup for first 5 epochs ─────────────────────────────────────
            if epoch < 5:
                epoch_lr = lr_max * (epoch + 1) / 5
                lr_change_msg = ""

            idx    = list(range(n)); random.shuffle(idx)
            idx_np = np.array(idx)
            X_shuf = X_idx[:, idx_np]
            Y_shuf = Y_idx[:, idx_np]

            total_loss = 0.0
            self._adam_t += 1

            blk_grad_acc = [
                {k: np.zeros_like(v) for k, v in blk.items()}
                for blk in self.blocks
            ]
            dWout_acc = np.zeros_like(self.Wout)
            dbout_acc = np.zeros_like(self.bout)
            if self.use_embedding:
                de_acc  = np.zeros_like(self.embedding)
                dpe_acc = np.zeros_like(self.pos_embedding)

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                bs  = end - start
                Xb  = X_shuf[:, start:end]     # (T, B)
                Yb  = Y_shuf[:, start:end]     # (T, B)

                probs, (toks, x_out, block_caches) = self._transformer_forward(Xb)
                # probs: (B, T, vocab),  toks/Yb: (B,T) / (T,B)

                T    = ctx_size
                Yb_T = Yb.T                    # (B, T) target indices

                # ── Loss: gather prob at correct index (no one-hot needed) ────
                b_idx = np.arange(bs)[:, None]
                t_idx = np.arange(T)[None, :]
                correct_probs = probs[b_idx, t_idx, Yb_T]        # (B, T)
                total_loss += float(-np.sum(np.log(correct_probs + 1e-9)))

                # ── Gradient: softmax grad = probs - one_hot, but computed ────
                # directly without building the full one-hot matrix
                delta = probs.copy()                              # (B, T, vocab)
                delta[b_idx, t_idx, Yb_T] -= 1.0
                delta /= (bs * T)

                dWout_acc += np.einsum('bti,btj->ij', x_out, delta)
                dbout_acc += delta.sum(axis=(0, 1))
                d_x = delta @ self.Wout.T                         # (B, T, D)

                for i, (cache, blk) in enumerate(zip(reversed(block_caches), reversed(self.blocks))):
                    d_x, grads = self._block_backward(d_x, cache, blk)
                    acc = blk_grad_acc[self.num_blocks - 1 - i]
                    for k in grads:
                        acc[k] += grads[k]

                if self.use_embedding:
                    dpe_acc += d_x.sum(axis=0)
                    if _DEVICE == "gpu":
                        # Stay on GPU — use cupyx.scatter_add instead of CPU roundtrip
                        try:
                            import cupyx
                            cupyx.scatter_add(de_acc, toks.reshape(-1), d_x.reshape(-1, D))
                        except Exception:
                            import numpy as _np
                            d_x_cpu  = np.asnumpy(d_x)
                            toks_cpu = np.asnumpy(toks)
                            de_cpu   = np.asnumpy(de_acc)
                            _np.add.at(de_cpu, toks_cpu.reshape(-1), d_x_cpu.reshape(-1, D))
                            de_acc = np.array(de_cpu)
                    else:
                        import numpy as _np
                        _np.add.at(de_acc, toks.reshape(-1), d_x.reshape(-1, D))

            # ── Adam updates with cosine lr ───────────────────────────────────
            for blk, acc, adam_buf in zip(self.blocks, blk_grad_acc, self._adam_blocks):
                for k in blk:
                    blk[k], adam_buf[k]["m"], adam_buf[k]["v"] = self._adam_update(
                        blk[k], acc[k], adam_buf[k]["m"], adam_buf[k]["v"], lr=epoch_lr
                    )
            self.Wout, self._mWout, self._vWout = self._adam_update(self.Wout, dWout_acc, self._mWout, self._vWout, lr=epoch_lr)
            self.bout, self._mbout, self._vbout = self._adam_update(self.bout, dbout_acc, self._mbout, self._vbout, lr=epoch_lr)
            if self.use_embedding:
                self.embedding,     self._me,  self._ve  = self._adam_update(self.embedding,     de_acc,  self._me,  self._ve,  lr=epoch_lr)
                self.pos_embedding, self._mpe, self._vpe = self._adam_update(self.pos_embedding, dpe_acc, self._mpe, self._vpe, lr=epoch_lr)

            # ── Adaptive LR: update for next epoch ────────────────────────────
            if epoch >= 5:
                lr_change_msg = ""
                if total_loss > prev_loss:
                    # Loss went UP — bounce detected, reduce immediately
                    new_lr = max(epoch_lr * bounce_factor, lr_min)
                    if new_lr < epoch_lr:
                        lr_change_msg = f"  ↓ bounce  {epoch_lr:.6f}→{new_lr:.6f}"
                    epoch_lr = new_lr
                    plateau_count = 0
                else:
                    if total_loss < best_loss:
                        best_loss     = total_loss
                        plateau_count = 0
                    else:
                        plateau_count += 1
                    if plateau_count >= plateau_patience:
                        new_lr = max(epoch_lr * plateau_factor, lr_min)
                        if new_lr < epoch_lr:
                            lr_change_msg = f"  ↓ plateau {epoch_lr:.6f}→{new_lr:.6f}"
                        epoch_lr      = new_lr
                        plateau_count = 0
                prev_loss = total_loss

            if log_every and epoch % log_every == 0:
                print(f"Epoch {epoch:>6} | Loss: {total_loss:.2f}  lr={epoch_lr:.6f}{lr_change_msg}")

        print("Training complete.")

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, inputs) -> Tuple[int, float, "np.ndarray"]:
        _, _, probs = self.forward(inputs)
        if _DEVICE == "gpu":
            probs = np.asnumpy(probs)
        predicted_class = int(probs.argmax())
        return predicted_class, float(probs[predicted_class]), probs

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> None:
        blk    = self.blocks[0]
        attn_p = blk["Wq"].size + blk["Wk"].size + blk["Wv"].size
        ff_p   = blk["W1"].size + blk["b1"].size + blk["W2"].size + blk["b2"].size
        out_p  = self.Wout.size + self.bout.size
        emb_p  = self.embedding.size + self.pos_embedding.size if self.use_embedding else 0
        total  = (attn_p + ff_p) * self.num_blocks + out_p + emb_p
        width  = 52
        device = "GPU (CuPy)" if _DEVICE == "gpu" else "CPU (NumPy)"
        D      = self.embed_dim

        print("╔" + "═" * width + "╗")
        print("║" + " Mini-Transformer Summary".center(width) + "║")
        print("╠" + "═" * width + "╣")
        print(f"║  {'Device':<16} │ {device:<{width-22}}║")
        print(f"║  {'Optimizer':<16} │ {'Adam + adaptive LR':<{width-22}}║")
        print(f"║  {'Batch size':<16} │ {self.batch_size:<{width-22}}║")
        if self.use_embedding:
            edim = f"{self.vocab_size} chars × {D}d  context={self.context_size}"
            print(f"║  {'Embedding':<16} │ {edim:<{width-22}}║")
        print(f"║  {'Blocks':<16} │ {str(self.num_blocks) + ' (independent)':<{width-22}}║")
        print(f"║  {'Attention':<16} │ {'causal, Wq/Wk/Wv ' + str(D) + 'x' + str(D) + ' (' + str(attn_p) + ' params/block)':<{width-22}}║")
        print(f"║  {'Feed-forward':<16} │ {str(D) + '->' + str(D*4) + '->' + str(D) + ' (' + str(ff_p) + ' params/block)':<{width-22}}║")
        print(f"║  {'Output':<16} │ {'all positions -> ' + str(self.output_size):<{width-22}}║")
        print(f"║  {'Vectorised':<16} │ {'ON (batch matmul)':<{width-22}}║")
        print("╠" + "═" * width + "╣")
        print(f"║  Total parameters: {total:,}{'':<{width-23-len(f'{total:,}')}}║")
        print("╚" + "═" * width + "╝")

    # ── Save ──────────────────────────────────────────────────────────────────

    def save_weights(self, filename: str = "weights.json") -> None:
        to_list = (lambda w: np.asnumpy(w).tolist()) if _DEVICE == "gpu" else (lambda w: w.tolist())
        data = {
            "input_size": self.input_size, "hidden_layers": self.hidden_layers,
            "output_size": self.output_size, "activation": self.activation,
            "learning_rate": self.learning_rate, "batch_size": self.batch_size,
            "use_embedding": self.use_embedding, "vocab_size": self.vocab_size,
            "context_size": self.context_size, "embed_dim": self.embed_dim,
            "num_blocks": self.num_blocks, "device": _DEVICE,
            "weights": [to_list(w) for w in self.weights],
            "biases":  [to_list(b) for b in self.biases],
            "embedding":     to_list(self.embedding)     if self.use_embedding else None,
            "pos_embedding": to_list(self.pos_embedding) if self.use_embedding else None,
            "Wout": to_list(self.Wout), "bout": to_list(self.bout),
            "blocks": [{k: to_list(v) for k, v in blk.items()} for blk in self.blocks],
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Weights saved to '{filename}'.")

    # ── Load ──────────────────────────────────────────────────────────────────

    def load_weights(self, filename: str = "weights.json") -> None:
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
        self.Wout = np.array(data["Wout"]); self.bout = np.array(data["bout"])
        if "blocks" in data:
            self.blocks = [{k: np.array(v) for k, v in blk.items()} for blk in data["blocks"]]
        else:
            # Backwards compat: old single-weight format
            self.blocks = []
            for _ in range(self.num_blocks):
                self.blocks.append({
                    "Wq": np.array(data["Wq"]), "Wk": np.array(data["Wk"]),
                    "Wv": np.array(data["Wv"]), "W1": np.array(data["W1"]),
                    "b1": np.array(data["b1"]), "W2": np.array(data["W2"]),
                    "b2": np.array(data["b2"]),
                })
        self._adam_init = False
        print(f"Weights loaded from '{filename}'.")

    def __repr__(self):
        return (f"NeuralNetwork(embed={self.embed_dim}, blocks={self.num_blocks}, "
                f"causal=True, all_positions=True, "
                f"lr={self.learning_rate}, device='{self.device}')")