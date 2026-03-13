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
    try:
        import cupyx as _cupyx
        _scatter_add = _cupyx.scatter_add
    except Exception:
        _scatter_add = None
except Exception:
    import numpy as np
    _DEVICE = "cpu"
    _scatter_add = None
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
        batch_size:    int   = 1024,
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

        # ── Precomputed constants ─────────────────────────────────────────────
        self._scale = 1.0 / (embed_dim ** 0.5)   # attention scale, avoid recomputing

        # ── Per-block independent weights ─────────────────────────────────────
        D = embed_dim
        self.blocks = []
        for _ in range(self.num_blocks):
            self.blocks.append({
                "Wqkv": np.random.randn(D, D * 3) * 0.02,   # fused Q,K,V projection
                "W1":   np.random.randn(D, D * 4) * 0.02,
                "b1":   np.zeros(D * 4),
                "W2":   np.random.randn(D * 4, D) * 0.02,
                "b2":   np.zeros(D),
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
        """Upper-triangular mask — cached per context size."""
        if not hasattr(self, '_mask_cache') or self._mask_cache.shape[0] != T:
            self._mask_cache = np.triu(np.ones((T, T)), k=1) * -1e9
        return self._mask_cache

    # ── Transformer block ─────────────────────────────────────────────────────

    def _block_forward(self, x, blk):
        """
        Causal transformer block. x: (B, T, D) -> (B, T, D)
        Each position only attends to itself and earlier positions.
        """
        B, T, D = x.shape
        W1, b1, W2, b2 = blk["W1"], blk["b1"], blk["W2"], blk["b2"]

        # Fuse Q,K,V into one matmul then split — 1 kernel instead of 3
        QKV = x.reshape(B * T, D) @ blk["Wqkv"]                  # (BT, 3D)
        QKV = QKV.reshape(B, T, 3, D)
        Q, K, V = QKV[:, :, 0, :], QKV[:, :, 1, :], QKV[:, :, 2, :]

        scores  = np.matmul(Q, K.transpose(0, 2, 1)) * self._scale  # (B, T, T)
        scores += self._causal_mask(T)
        scores -= scores.max(axis=2, keepdims=True)
        exp_s   = np.exp(scores)
        A       = exp_s / exp_s.sum(axis=2, keepdims=True)

        attn_out = np.matmul(A, V)                                 # (B, T, D)
        x_attn   = x + attn_out

        h      = np.maximum(0.0, x_attn @ W1 + b1)
        ff_out = h @ W2 + b2
        x_out  = x_attn + ff_out

        cache = (x, Q, K, V, A, attn_out, h, ff_out)              # x_attn removed
        return x_out, cache

    def _block_backward(self, d_out, cache, blk):
        """Vectorised backprop through causal block. d_out: (B, T, D)"""
        x, Q, K, V, A, attn_out, h, ff_out = cache              # no x_attn in cache
        B, T, D = x.shape
        W1, W2 = blk["W1"], blk["W2"]
        BT = B * T

        x_attn = x + attn_out                                    # recompute, saves memory

        # Feed-forward backward
        # d_out flows through residual AND ff — accumulate into d_x_res directly
        dW2  = h.reshape(BT, -1).T @ d_out.reshape(BT, -1)
        db2  = d_out.sum(axis=(0, 1))
        d_h  = d_out @ W2.T
        d_h *= (h > 0)
        dW1  = x_attn.reshape(BT, -1).T @ d_h.reshape(BT, -1)
        db1  = d_h.sum(axis=(0, 1))
        d_x_attn = d_out + d_h @ W1.T                            # no copy needed

        # Attention backward
        d_A = np.matmul(d_x_attn, V.transpose(0, 2, 1))
        dV  = np.matmul(A.transpose(0, 2, 1), d_x_attn)
        dS  = A * (d_A - (d_A * A).sum(axis=2, keepdims=True))
        dS  *= self._scale
        dQ  = np.matmul(dS, K)
        dK  = np.matmul(dS.transpose(0, 2, 1), Q)
        d_x_res = d_x_attn                                       # no copy — reuse directly

        # Fused Wqkv backward — concatenate dQ,dK,dV and do one matmul each way
        x_r    = x.reshape(BT, D)
        dQKV_r = np.concatenate([dQ.reshape(BT, D),
                                  dK.reshape(BT, D),
                                  dV.reshape(BT, D)], axis=1)      # (BT, 3D)
        dWqkv  = x_r.T @ dQKV_r                                    # (D, 3D)
        Wqkv   = blk["Wqkv"]
        d_x    = (np.matmul(dQ, Wqkv[:, :D].T) +
                  np.matmul(dK, Wqkv[:, D:2*D].T) +
                  np.matmul(dV, Wqkv[:, 2*D:].T) +
                  d_x_res)

        grads = {"Wqkv": dWqkv, "W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
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

    def train(self, data: List[Sample], epochs: int, log_every: int = 1,
              save_every: int = 0, save_path: str = "checkpoint.json") -> None:
        """
        All-positions causal training with adaptive LR.
        save_every: checkpoint every N epochs (0 = disabled)
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
        lr_min   = lr_max / 5.0

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
              f"lr={lr_max:.5f} (bounce×0.7, plateau×0.7, min={lr_min:.5f}) | "
              f"embed={'ON' if self.use_embedding else 'OFF'} | "
              f"transformer=ON (blocks={self.num_blocks}, independent) | "
              f"causal=ON | all-positions=ON\n")

        # Skip warmup and use correct starting lr if resuming with Adam state
        _resuming = self._adam_init and self._adam_t > 0
        epoch_lr  = lr_max if not _resuming else self.learning_rate
        best_loss         = float("inf")
        plateau_count     = 0
        plateau_patience  = 5    # epochs without improvement before reducing
        plateau_factor    = 0.7  # lr multiplier on plateau
        bounce_factor     = 0.7  # lr multiplier when loss goes UP
        bounce_count      = 0    # consecutive up-epochs
        bounce_patience   = 2    # need 2 bad epochs in a row before reducing
        prev_loss         = float("inf")
        lr_change_msg     = ""

        # ── Pre-allocate gradient buffers once — zero in-place each epoch ───────
        blk_grad_acc = [{k: np.zeros_like(v) for k, v in blk.items()}
                        for blk in self.blocks]
        dWout_acc = np.zeros_like(self.Wout)
        dbout_acc = np.zeros_like(self.bout)
        de_acc    = np.zeros_like(self.embedding)     if self.use_embedding else None
        dpe_acc   = np.zeros_like(self.pos_embedding) if self.use_embedding else None

        # Hoist constants that never change across epochs/batches
        T          = ctx_size
        t_idx      = np.arange(T)[None, :]            # (1, T)
        b_idx_full = np.arange(self.batch_size)[:, None]  # (B, 1) for full batches

        for epoch in range(epochs):
            # ── Warmup for first 5 epochs (fresh training only) ───────────────
            if not _resuming and epoch < 5:
                epoch_lr = lr_max * (epoch + 1) / 5
                lr_change_msg = ""

            idx_np = np.random.permutation(n)
            X_shuf = X_idx[:, idx_np]
            Y_shuf = Y_idx[:, idx_np]

            total_loss = 0.0
            self._adam_t += 1

            # Zero grad buffers in-place — no allocation
            for acc in blk_grad_acc:
                for a in acc.values(): a[...] = 0.0
            dWout_acc[...] = 0.0
            dbout_acc[...] = 0.0
            if self.use_embedding:
                de_acc[...]  = 0.0
                dpe_acc[...] = 0.0

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                bs  = end - start
                Xb  = X_shuf[:, start:end]     # (T, B)
                Yb  = Y_shuf[:, start:end]     # (T, B)

                probs, (toks, x_out, block_caches) = self._transformer_forward(Xb)

                Yb_T  = Yb.T
                b_idx = b_idx_full if bs == self.batch_size else np.arange(bs)[:, None]

                # ── Loss: gather prob at correct index (no one-hot needed) ────
                correct_probs = probs[b_idx, t_idx, Yb_T]
                total_loss += float(-np.sum(np.log(correct_probs + 1e-9)))
                del correct_probs

                # ── Gradient: reuse probs buffer in-place as delta ────────────
                probs[b_idx, t_idx, Yb_T] -= 1.0
                probs *= 1.0 / (bs * T)
                delta  = probs                                    # alias, no copy

                dWout_acc += x_out.reshape(bs * T, -1).T @ delta.reshape(bs * T, -1)
                dbout_acc += delta.sum(axis=(0, 1))
                d_x = delta @ self.Wout.T
                del probs, x_out                                  # free immediately

                for i, (cache, blk) in enumerate(zip(reversed(block_caches), reversed(self.blocks))):
                    d_x, grads = self._block_backward(d_x, cache, blk)
                    del cache                                      # free as we go
                    acc = blk_grad_acc[self.num_blocks - 1 - i]
                    for k in grads:
                        acc[k] += grads[k]
                del block_caches

                if self.use_embedding:
                    dpe_acc += d_x.sum(axis=0)
                    if _DEVICE == "gpu" and _scatter_add is not None:
                        _scatter_add(de_acc, toks.reshape(-1), d_x.reshape(-1, D))
                    elif _DEVICE == "gpu":
                        import numpy as _np
                        d_x_cpu  = np.asnumpy(d_x)
                        toks_cpu = np.asnumpy(toks)
                        de_cpu   = np.asnumpy(de_acc)
                        _np.add.at(de_cpu, toks_cpu.reshape(-1), d_x_cpu.reshape(-1, D))
                        de_acc = np.array(de_cpu)
                    else:
                        import numpy as _np
                        _np.add.at(de_acc, toks.reshape(-1), d_x.reshape(-1, D))

            # ── Adam updates — precompute bias correction once per epoch ────────
            t      = self._adam_t
            bc1    = 1.0 - 0.9   ** t    # 1 - beta1^t
            bc2    = 1.0 - 0.999 ** t    # 1 - beta2^t
            lr_eff = epoch_lr * (bc2 ** 0.5) / bc1   # fused corrected lr

            def _adam_step(param, grad, m, v):
                m *= 0.9;  m += 0.1   * grad
                v *= 0.999; v += 0.001 * grad * grad
                param -= lr_eff * m / (np.sqrt(v) + 1e-8)

            for blk, acc, adam_buf in zip(self.blocks, blk_grad_acc, self._adam_blocks):
                for k in blk:
                    _adam_step(blk[k], acc[k], adam_buf[k]["m"], adam_buf[k]["v"])
            _adam_step(self.Wout, dWout_acc, self._mWout, self._vWout)
            _adam_step(self.bout, dbout_acc, self._mbout, self._vbout)
            if self.use_embedding:
                _adam_step(self.embedding,     de_acc,  self._me,  self._ve)
                _adam_step(self.pos_embedding, dpe_acc, self._mpe, self._vpe)

            # ── Adaptive LR: update for next epoch ────────────────────────────
            if epoch >= 5:
                lr_change_msg = ""
                if total_loss > prev_loss:
                    # Loss went UP — only reduce after bounce_patience consecutive bad epochs
                    bounce_count += 1
                    plateau_count = 0
                    if bounce_count >= bounce_patience:
                        new_lr = max(epoch_lr * bounce_factor, lr_min)
                        if new_lr < epoch_lr:
                            lr_change_msg = f"  ↓ bounce×{bounce_count}  {epoch_lr:.6f}→{new_lr:.6f}"
                        epoch_lr     = new_lr
                        bounce_count = 0
                else:
                    bounce_count = 0
                    if total_loss < best_loss:
                        best_loss     = total_loss
                        plateau_count = 0
                    else:
                        plateau_count += 1
                    if plateau_count >= plateau_patience:
                        new_lr = max(epoch_lr * plateau_factor, lr_min)
                        if new_lr < epoch_lr:
                            lr_change_msg = f"  ↓ plateau  {epoch_lr:.6f}→{new_lr:.6f}"
                        epoch_lr      = new_lr
                        plateau_count = 0
                prev_loss = total_loss

            if log_every and epoch % log_every == 0:
                equiv = total_loss / (n * ctx_size)
                print(f"Epoch {epoch:>6} | Loss: {total_loss:.2f}  equiv: {equiv:.4f}  lr={epoch_lr:.6f}{lr_change_msg}", flush=True)

            if save_every and epoch > 0 and epoch % save_every == 0:
                self.save_weights(save_path)
                print(f"  ✓ checkpoint saved → {save_path}", flush=True)

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
        attn_p = blk["Wqkv"].size
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
        print(f"║  {'Attention':<16} │ {'causal, fused Wqkv ' + str(D) + 'x' + str(D*3) + ' (' + str(attn_p) + ' params/block)':<{width-22}}║")
        print(f"║  {'Feed-forward':<16} │ {str(D) + '->' + str(D*4) + '->' + str(D) + ' (' + str(ff_p) + ' params/block)':<{width-22}}║")
        print(f"║  {'Output':<16} │ {'all positions -> ' + str(self.output_size):<{width-22}}║")
        print(f"║  {'Vectorised':<16} │ {'ON (batch matmul)':<{width-22}}║")
        print("╠" + "═" * width + "╣")
        print(f"║  Total parameters: {total:,}{'':<{width-23-len(f'{total:,}')}}║")
        print("╚" + "═" * width + "╝")

    # ── Save ──────────────────────────────────────────────────────────────────

    def save_weights(self, filename: str = "weights.json") -> None:
        to_list = (lambda w: np.asnumpy(w).tolist()) if _DEVICE == "gpu" else (lambda w: w.tolist())
        # Save Adam buffers so resume continues smoothly
        adam_blocks = []
        if self._adam_init:
            for buf in self._adam_blocks:
                adam_blocks.append({k: {"m": to_list(v["m"]), "v": to_list(v["v"])}
                                    for k, v in buf.items()})
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
            # Adam state
            "adam_t":       self._adam_t if self._adam_init else 0,
            "adam_mWout":   to_list(self._mWout)  if self._adam_init else None,
            "adam_vWout":   to_list(self._vWout)  if self._adam_init else None,
            "adam_mbout":   to_list(self._mbout)  if self._adam_init else None,
            "adam_vbout":   to_list(self._vbout)  if self._adam_init else None,
            "adam_me":      to_list(self._me)     if (self._adam_init and self.use_embedding) else None,
            "adam_ve":      to_list(self._ve)     if (self._adam_init and self.use_embedding) else None,
            "adam_mpe":     to_list(self._mpe)    if (self._adam_init and self.use_embedding) else None,
            "adam_vpe":     to_list(self._vpe)    if (self._adam_init and self.use_embedding) else None,
            "adam_blocks":  adam_blocks,
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
            self.blocks = []
            for blk in data["blocks"]:
                b = {k: np.array(v) for k, v in blk.items()}
                # Upgrade old Wq/Wk/Wv to fused Wqkv on load
                if "Wq" in b and "Wqkv" not in b:
                    import numpy as _np
                    Wq = np.asnumpy(b.pop("Wq")) if _DEVICE == "gpu" else b.pop("Wq")
                    Wk = np.asnumpy(b.pop("Wk")) if _DEVICE == "gpu" else b.pop("Wk")
                    Wv = np.asnumpy(b.pop("Wv")) if _DEVICE == "gpu" else b.pop("Wv")
                    b["Wqkv"] = np.array(_np.concatenate([Wq, Wk, Wv], axis=1))
                self.blocks.append(b)
        else:
            # Very old single-weight format
            self.blocks = []
            for _ in range(self.num_blocks):
                import numpy as _np
                Wq = _np.array(data["Wq"]); Wk = _np.array(data["Wk"]); Wv = _np.array(data["Wv"])
                self.blocks.append({
                    "Wqkv": np.array(_np.concatenate([Wq, Wk, Wv], axis=1)),
                    "W1": np.array(data["W1"]), "b1": np.array(data["b1"]),
                    "W2": np.array(data["W2"]), "b2": np.array(data["b2"]),
                })
        self._adam_init = False
        self._scale = 1.0 / (self.embed_dim ** 0.5)
        # Restore Adam state if present (enables smooth resume)
        if data.get("adam_t") and data["adam_t"] > 0:
            self._init_adam()   # allocate buffers with correct shapes
            self._adam_t = data["adam_t"]
            self._mWout  = np.array(data["adam_mWout"])
            self._vWout  = np.array(data["adam_vWout"])
            self._mbout  = np.array(data["adam_mbout"])
            self._vbout  = np.array(data["adam_vbout"])
            if self.use_embedding and data.get("adam_me") is not None:
                self._me  = np.array(data["adam_me"])
                self._ve  = np.array(data["adam_ve"])
                self._mpe = np.array(data["adam_mpe"])
                self._vpe = np.array(data["adam_vpe"])
            for i, buf in enumerate(data.get("adam_blocks", [])):
                for k, mv in buf.items():
                    self._adam_blocks[i][k]["m"] = np.array(mv["m"])
                    self._adam_blocks[i][k]["v"] = np.array(mv["v"])
            print(f"  Adam state restored (t={self._adam_t})")
        print(f"Weights loaded from '{filename}'.")

    def __repr__(self):
        return (f"NeuralNetwork(embed={self.embed_dim}, blocks={self.num_blocks}, "
                f"causal=True, all_positions=True, "
                f"lr={self.learning_rate}, device='{self.device}')")