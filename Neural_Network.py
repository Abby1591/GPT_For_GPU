"""
Neural_Network.py  —  Mini-Transformer (causal, all-positions training)
========================================================================

Architecture
------------
tokens (B, T)
  -> token embedding + positional encoding        (B, T, D)
  -> N x transformer block (independent weights)  (B, T, D)   <- causal mask
  -> linear projection at ALL positions           (B, T, vocab)
  -> softmax -> probabilities

Training signal
---------------
Every token position predicts the next token simultaneously.
A context window of length T produces T (input, target) pairs per sample,
giving Tx more gradient signal than predicting only the final position.

GPU setup
---------
    pip install cupy-cuda12x        # CUDA 12.x  (Colab T4)
    pip install cupy-cuda11x        # CUDA 11.x  (older GPUs)
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, List, Literal, Tuple

# ---- GPU / CPU backend selection --------------------------------------------
# We import CuPy as `np` so the rest of the code is device-agnostic.
# If CuPy is unavailable we fall back to NumPy silently.
try:
    import cupy as np
    np.cuda.Device(0).use()
    _DEVICE = "gpu"
    print(
        f"GPU detected -- training on: "
        f"{np.cuda.runtime.getDeviceProperties(0)['name'].decode()}"
    )
    # cupyx.scatter_add is the GPU equivalent of numpy.add.at used to
    # accumulate embedding gradients without leaving the GPU.
    try:
        from cupyx import scatter_add as _scatter_add
    except Exception:
        _scatter_add = None

except Exception:
    import numpy as np
    _DEVICE = "cpu"
    _scatter_add = None
    print("CuPy not found -- falling back to CPU (NumPy)")

# CPU numpy is always needed for index operations that CuPy does not support.
import numpy as _np_cpu


# ---- Type aliases ------------------------------------------------------------
ActivationName = Literal["sigmoid", "tanh", "relu", "leaky_relu"]
Sample         = Tuple[List[float], int]


# ---- Activation functions and their derivatives -----------------------------
# Stored as (forward, derivative) pairs looked up by name.
# All functions accept and return arrays (CuPy or NumPy).

def _relu(x):                  return np.maximum(0.0, x)
def _relu_d(x):                return (x > 0).astype(float)

def _tanh(x):                  return np.tanh(x)
def _tanh_d(x):                return 1.0 - np.tanh(x) ** 2

def _sigmoid(x):               return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
def _sigmoid_d(x):             s = _sigmoid(x); return s * (1.0 - s)

def _leaky_relu(x, a=0.01):    return np.where(x > 0, x, a * x)
def _leaky_relu_d(x, a=0.01):  return np.where(x > 0, 1.0, a)

_ACTIVATIONS: Dict[str, tuple] = {
    "relu":       (_relu,       _relu_d),
    "tanh":       (_tanh,       _tanh_d),
    "sigmoid":    (_sigmoid,    _sigmoid_d),
    "leaky_relu": (_leaky_relu, _leaky_relu_d),
}


# ---- Main class --------------------------------------------------------------

class NeuralNetwork:
    """
    Mini-GPT character-level transformer.

    Parameters
    ----------
    input_size : int
        Flat input size (context_size x vocab_size when using embeddings).
    hidden_layers : list[int]
        Legacy dense hidden-layer widths (kept for save/load compatibility only).
    output_size : int
        Vocabulary size -- number of classes in the softmax output.
    activation : str
        Activation for legacy dense layers. One of relu/tanh/sigmoid/leaky_relu.
    learning_rate : float
        Peak learning rate for Adam. The adaptive scheduler may reduce this.
    batch_size : int
        Samples processed per gradient step. Larger = faster on GPU, more VRAM.
    use_embedding : bool
        If True, use a learned token + positional embedding layer.
    vocab_size : int
        Number of unique tokens (characters). Required when use_embedding=True.
    context_size : int
        Sequence length T -- how many previous tokens the model sees at once.
    embed_dim : int
        Width D of the embedding and all transformer hidden states.
    num_blocks : int
        Number of transformer blocks stacked. Each block has independent weights
        (no weight sharing). Default 2.
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
        num_blocks:    int   = 2,
    ) -> None:
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Choose from: {list(_ACTIVATIONS)}"
            )
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
        self.num_blocks    = num_blocks
        self.device        = _DEVICE
        self._act_fn, self._act_d = _ACTIVATIONS[activation]

        # ---- Token + positional embeddings ----------------------------------
        # Token embedding: maps each vocab index to a D-dimensional vector.
        #   Shape (vocab_size, D) -- one row per character.
        # Positional embedding: adds position-specific info to each slot.
        #   Shape (context_size, D) -- one row per position 0..T-1.
        # Both are learned parameters updated by Adam like any other weight.
        if self.use_embedding:
            self.embedding     = np.random.randn(vocab_size, embed_dim) * 0.01
            self.pos_embedding = np.random.randn(context_size, embed_dim) * 0.01
        else:
            self.embedding     = None
            self.pos_embedding = None

        # ---- Attention scale factor -----------------------------------------
        # Scaled dot-product attention divides Q.K^T by sqrt(D) before softmax.
        # Without this, dot products grow large as D increases, pushing softmax
        # into saturation and killing gradients. Precomputed to avoid recomputing
        # every forward pass.
        self._scale = 1.0 / (embed_dim ** 0.5)

        # ---- Transformer blocks ---------------------------------------------
        # Each block:
        #   Wqkv -- fused Q, K, V projection  (D, 3D)
        #   W1,b1 -- feed-forward layer 1     (D, 4D) + bias (4D,)
        #   W2,b2 -- feed-forward layer 2     (4D, D) + bias (D,)
        #
        # WHY FUSED Wqkv?
        #   Separate Wq, Wk, Wv = 3 matmul kernel launches.
        #   One (D, 3D) matrix = 1 launch -- ~3x faster on GPU where kernel
        #   launch overhead dominates for these matrix sizes.
        #
        # WHY 4D IN FEED-FORWARD?
        #   GPT-2 convention: D -> 4D -> D bottleneck lets the network combine
        #   features in a high-dimensional space before projecting back.
        #
        # INITIALISATION: 0.02 std (GPT-2 convention, keeps activations scaled).
        D = embed_dim
        self.blocks: List[Dict] = []
        for _ in range(self.num_blocks):
            self.blocks.append({
                "Wqkv": np.random.randn(D, D * 3) * 0.02,
                "W1":   np.random.randn(D, D * 4) * 0.02,
                "b1":   np.zeros(D * 4),
                "W2":   np.random.randn(D * 4, D) * 0.02,
                "b2":   np.zeros(D),
            })

        # ---- Output projection ----------------------------------------------
        # Maps final hidden states (B, T, D) -> logits (B, T, vocab_size).
        # Applied at ALL token positions so every position trains simultaneously.
        self.Wout = np.random.randn(embed_dim, output_size) * 0.02
        self.bout = np.zeros(output_size)

        # ---- Legacy dense weights (backward compatibility ONLY) -------------
        # Pre-transformer versions trained a plain MLP. These arrays are kept
        # so old weight files can still be loaded. They are NOT used in the
        # transformer forward/backward pass.
        actual_input = context_size * embed_dim if self.use_embedding else input_size
        layer_sizes  = [actual_input] + hidden_layers + [output_size]
        self.weights = []
        self.biases  = []
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            # Xavier/Glorot init: keeps variance ~1 across layers
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.weights.append(np.random.randn(fan_out, fan_in) * scale)
            self.biases.append(np.zeros((fan_out, 1)))

        # Adam not initialised until first train() or load_weights() call
        self._adam_init = False

    # ---- Adam optimiser -----------------------------------------------------

    def _init_adam(self) -> None:
        """
        Allocate zeroed momentum buffers for every learnable parameter.

        Adam keeps two running averages per parameter:
          m -- first moment  (exponential moving average of gradient)
          v -- second moment (exponential moving average of gradient^2)

        These adapt the effective lr per-parameter: large-gradient parameters
        get smaller steps; noisy/small-gradient ones get larger steps.

        Calling this resets _adam_t to 0. When resuming, load_weights() calls
        this first for allocation, then overwrites the buffers from the file.
        """
        z = np.zeros_like
        self._adam_blocks = []
        for blk in self.blocks:
            self._adam_blocks.append(
                {k: {"m": z(v), "v": z(v)} for k, v in blk.items()}
            )
        self._mWout = z(self.Wout);  self._vWout = z(self.Wout)
        self._mbout = z(self.bout);  self._vbout = z(self.bout)
        if self.use_embedding:
            self._me  = z(self.embedding);      self._ve  = z(self.embedding)
            self._mpe = z(self.pos_embedding);  self._vpe = z(self.pos_embedding)
        self._adam_t    = 0     # global step counter (for bias correction)
        self._adam_init = True

    # ---- Causal mask ---------------------------------------------------------

    def _causal_mask(self, T: int):
        """
        Upper-triangular mask: prevents each position from attending to future
        positions. Shape (T, T). Entry [i,j] = -1e9 if j > i, else 0.

        Adding to attention scores before softmax zeroes out future weights --
        the "causal" property required for autoregressive generation.
        Cached per T: rebuilt only when context size changes (never in practice).
        """
        if not hasattr(self, "_mask_cache") or self._mask_cache.shape[0] != T:
            # triu(k=1) gives the strict upper triangle (diagonal excluded)
            self._mask_cache = np.triu(np.ones((T, T)), k=1) * -1e9
        return self._mask_cache

    # ---- Transformer block: forward -----------------------------------------

    def _block_forward(self, x, blk):
        """
        One causal transformer block. Input/output shape: (B, T, D).

        Steps
        -----
        1. Fused QKV projection     x @ Wqkv -> split into Q, K, V
        2. Scaled dot-product attn  softmax(QK^T/sqrt(D) + mask) * V
        3. Residual add             x = x + attn_out
        4. Feed-forward             h = ReLU(x @ W1 + b1); ff = h @ W2 + b2
        5. Residual add             x = x + ff

        WHY RESIDUALS?
        Without them gradients must flow through every layer in full, causing
        vanishing gradients in deep networks. Residuals provide a direct
        gradient highway from the loss back to each block's input.

        NOTE: x_attn is NOT stored in cache -- it is recomputed cheaply in
        backward (x + attn_out) to save VRAM.
        """
        B, T, D = x.shape
        W1, b1, W2, b2 = blk["W1"], blk["b1"], blk["W2"], blk["b2"]

        # Fused QKV: one matmul instead of three kernel launches
        QKV = x.reshape(B * T, D) @ blk["Wqkv"]          # (BT, 3D)
        QKV = QKV.reshape(B, T, 3, D)
        Q   = QKV[:, :, 0, :]                             # (B, T, D)
        K   = QKV[:, :, 1, :]
        V   = QKV[:, :, 2, :]

        # Scaled dot-product attention with numerical stability (subtract max)
        scores  = np.matmul(Q, K.transpose(0, 2, 1)) * self._scale  # (B, T, T)
        scores += self._causal_mask(T)
        scores -= scores.max(axis=2, keepdims=True)
        exp_s   = np.exp(scores)
        A       = exp_s / exp_s.sum(axis=2, keepdims=True)          # (B, T, T)

        attn_out = np.matmul(A, V)                                   # (B, T, D)
        x_attn   = x + attn_out                                      # residual

        # Feed-forward: D -> 4D (ReLU) -> D
        h      = np.maximum(0.0, x_attn @ W1 + b1)  # (B, T, 4D)
        ff_out = h @ W2 + b2                          # (B, T, D)
        x_out  = x_attn + ff_out                      # residual

        cache = (x, Q, K, V, A, attn_out, h, ff_out)  # x_attn omitted
        return x_out, cache

    # ---- Transformer block: backward ----------------------------------------

    def _block_backward(self, d_out, cache, blk):
        """
        Backprop through one causal transformer block.

        d_out : upstream gradient  (B, T, D)
        Returns (d_x, grads_dict).

        Key tricks
        ----------
        * x_attn recomputed from cache instead of stored -- saves VRAM.
        * Fused Wqkv backward: concatenate dQ, dK, dV into (BT, 3D), then
          one matmul each direction instead of three.
        * Softmax backward: dS = A * (d_A - sum(d_A*A)) -- standard
          Jacobian-vector product, avoids building the full Jacobian.
        """
        x, Q, K, V, A, attn_out, h, ff_out = cache
        B, T, D = x.shape
        W1, W2  = blk["W1"], blk["W2"]
        BT      = B * T

        x_attn = x + attn_out   # recompute rather than store

        # Feed-forward backward
        # d_out flows through residual (direct pass) AND ff branch (W2->ReLU->W1)
        dW2      = h.reshape(BT, -1).T @ d_out.reshape(BT, -1)    # (4D, D)
        db2      = d_out.sum(axis=(0, 1))                           # (D,)
        d_h      = d_out @ W2.T                                     # (B, T, 4D)
        d_h     *= (h > 0)                                          # ReLU derivative
        dW1      = x_attn.reshape(BT, -1).T @ d_h.reshape(BT, -1)  # (D, 4D)
        db1      = d_h.sum(axis=(0, 1))                             # (4D,)
        # Gradient into x_attn: residual branch (d_out) + FF branch
        d_x_attn = d_out + d_h @ W1.T                              # (B, T, D)

        # Attention backward
        d_A = np.matmul(d_x_attn, V.transpose(0, 2, 1))            # (B, T, T)
        dV  = np.matmul(A.transpose(0, 2, 1), d_x_attn)            # (B, T, D)
        # Softmax Jacobian-vector product
        dS  = A * (d_A - (d_A * A).sum(axis=2, keepdims=True))     # (B, T, T)
        dS *= self._scale
        dQ  = np.matmul(dS, K)                                      # (B, T, D)
        dK  = np.matmul(dS.transpose(0, 2, 1), Q)                   # (B, T, D)

        # Fused Wqkv backward: concat dQ,dK,dV -> one matmul each direction
        x_r    = x.reshape(BT, D)
        dQKV_r = np.concatenate(
            [dQ.reshape(BT, D), dK.reshape(BT, D), dV.reshape(BT, D)], axis=1
        )                                                            # (BT, 3D)
        dWqkv  = x_r.T @ dQKV_r                                     # (D, 3D)

        Wqkv = blk["Wqkv"]
        d_x  = (
            np.matmul(dQ, Wqkv[:, :D].T)      +
            np.matmul(dK, Wqkv[:, D:2*D].T)   +
            np.matmul(dV, Wqkv[:, 2*D:].T)    +
            d_x_attn                            # residual connection gradient
        )

        grads = {"Wqkv": dWqkv, "W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return d_x, grads

    # ---- Full forward pass --------------------------------------------------

    def _transformer_forward(self, token_idx_batch):
        """
        Complete forward pass from token indices to probabilities.

        token_idx_batch : int array (T, B)
        Returns probs (B, T, vocab) and cache tuple for backward.
        """
        toks = token_idx_batch.T                               # (B, T)

        # Token embedding lookup + positional embedding.
        # pos_embedding (T, D) broadcasts over batch dimension automatically.
        x = self.embedding[toks] + self.pos_embedding          # (B, T, D)

        block_caches = []
        for blk in self.blocks:
            x, cache = self._block_forward(x, blk)
            block_caches.append(cache)

        # Project ALL T positions to vocab logits at once
        logits  = x @ self.Wout + self.bout                    # (B, T, vocab)
        logits -= logits.max(axis=2, keepdims=True)            # stable softmax
        e       = np.exp(logits)
        probs   = e / e.sum(axis=2, keepdims=True)             # (B, T, vocab)

        return probs, (toks, x, block_caches)

    def forward(self, inputs):
        """Single-sample forward pass for text generation (last position only)."""
        if self.use_embedding:
            arr  = np.array(inputs, dtype=float).reshape(
                self.context_size, self.vocab_size
            )
            toks = np.array(arr.argmax(axis=1), dtype=int).reshape(
                self.context_size, 1
            )
            probs, _ = self._transformer_forward(toks)
            if _DEVICE == "gpu":
                probs = np.asnumpy(probs)
            return None, None, probs[0, -1, :]

        # Legacy MLP path
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

    # ---- Training loop ------------------------------------------------------

    def train(
        self,
        data,
        epochs:     int,
        log_every:  int = 1,
        save_every: int = 0,
        save_path:  str = "checkpoint.json",
    ) -> None:
        """
        Train with batched gradient descent, Adam, and adaptive LR.

        Parameters
        ----------
        data : tuple (X_idx, Y_idx) or legacy list of (features, label)
            Preferred: index arrays from make_index_arrays() -- avoids
            allocating one-hot floats that are decoded back to indices anyway.
        epochs : int
            Full passes over the training data.
        log_every : int
            Print loss every N epochs (0 = silent).
        save_every : int
            Checkpoint every N epochs (0 = disabled). Uses ATOMIC WRITES --
            interrupted saves never corrupt the target file.

        Adaptive LR schedule
        --------------------
        Warmup (fresh only):
            Epochs 0-4 ramp lr from lr/5 -> lr_max. Skipped on resume.
        Bounce:
            Loss up for bounce_patience=2 consecutive epochs -> lr *= 0.7.
        Plateau:
            No new best loss for plateau_patience=5 epochs -> lr *= 0.7.
            Floored at lr_max / 5.

        Adam notes
        ----------
        Bias correction is computed ONCE per epoch and baked into lr_eff:
            lr_eff = epoch_lr * sqrt(1-beta2^t) / (1-beta1^t)
        This avoids recomputing two power operations per parameter tensor.

        Gradient buffers are allocated ONCE before the loop and zeroed
        in-place each epoch (buf[...] = 0.0), avoiding thousands of GPU
        memory allocations that would each force a CUDA sync.
        """
        if not self._adam_init:
            self._init_adam()

        ctx_size = self.context_size
        vs       = self.vocab_size
        D        = self.embed_dim
        lr_max   = self.learning_rate
        lr_min   = lr_max / 5.0

        print("  Loading dataset onto device...")

        # Fast path: index arrays from make_index_arrays() (no one-hot floats)
        if isinstance(data, tuple) and len(data) == 2 and hasattr(data[0], "shape"):
            X_idx_cpu, Y_idx_cpu = data
            n = X_idx_cpu.shape[1]
        else:
            # Legacy: decode one-hot -> indices
            n = len(data)
            X_idx_cpu = _np_cpu.zeros((ctx_size, n), dtype=_np_cpu.int32)
            Y_idx_cpu = _np_cpu.zeros((ctx_size, n), dtype=_np_cpu.int32)
            for j, (feat, label) in enumerate(data):
                oh   = _np_cpu.array(feat).reshape(ctx_size, vs)
                toks = oh.argmax(axis=1)
                X_idx_cpu[:, j]    = toks
                Y_idx_cpu[:-1, j]  = toks[1:]
                Y_idx_cpu[-1,  j]  = label

        X_idx = np.array(X_idx_cpu)   # move to GPU if CuPy
        Y_idx = np.array(Y_idx_cpu)

        print(
            f"  {_DEVICE.upper()} ready -- {n:,} samples | "
            f"batch={self.batch_size} | optimizer=Adam+adaptive | "
            f"lr={lr_max:.5f} (bounce*0.7, plateau*0.7, min={lr_min:.5f}) | "
            f"embed={'ON' if self.use_embedding else 'OFF'} | "
            f"transformer=ON (blocks={self.num_blocks}, independent) | "
            f"causal=ON | all-positions=ON\n"
        )

        # Resume detection: if Adam state was restored (_adam_t > 0), skip
        # warmup and start at the saved learning rate immediately.
        _resuming = self._adam_init and self._adam_t > 0
        epoch_lr  = lr_max if not _resuming else self.learning_rate

        # Adaptive LR state
        best_loss        = float("inf")
        plateau_count    = 0
        plateau_patience = 5
        plateau_factor   = 0.7
        bounce_factor    = 0.7
        bounce_count     = 0
        bounce_patience  = 2
        prev_loss        = float("inf")
        lr_change_msg    = ""

        # Pre-allocate gradient accumulation buffers (zeroed in-place each epoch)
        blk_grad_acc = [
            {k: np.zeros_like(v) for k, v in blk.items()}
            for blk in self.blocks
        ]
        dWout_acc = np.zeros_like(self.Wout)
        dbout_acc = np.zeros_like(self.bout)
        de_acc    = np.zeros_like(self.embedding)     if self.use_embedding else None
        dpe_acc   = np.zeros_like(self.pos_embedding) if self.use_embedding else None

        # Hoisted loop constants (never change -- computing inside loop wastes cycles)
        T          = ctx_size
        t_idx      = np.arange(T)[None, :]                # (1, T)
        b_idx_full = np.arange(self.batch_size)[:, None]  # (B, 1)

        for epoch in range(epochs):

            # Warmup: linearly ramp lr from lr/5 -> lr_max over first 5 epochs.
            # Prevents large updates before Adam momentum has warmed up.
            # Skipped when resuming (Adam already has good momentum estimates).
            if not _resuming and epoch < 5:
                epoch_lr      = lr_max * (epoch + 1) / 5
                lr_change_msg = ""

            # GPU-side shuffle: stays on the device, avoids CPU<->GPU round-trip
            idx_np = np.random.permutation(n)
            X_shuf = X_idx[:, idx_np]
            Y_shuf = Y_idx[:, idx_np]

            total_loss   = 0.0
            self._adam_t += 1

            # Zero grad buffers in-place (no allocations)
            for acc in blk_grad_acc:
                for a in acc.values():
                    a[...] = 0.0
            dWout_acc[...] = 0.0
            dbout_acc[...] = 0.0
            if self.use_embedding:
                de_acc[...]  = 0.0
                dpe_acc[...] = 0.0

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                bs  = end - start
                Xb  = X_shuf[:, start:end]    # (T, B)
                Yb  = Y_shuf[:, start:end]    # (T, B)

                probs, (toks, x_out, block_caches) = self._transformer_forward(Xb)

                Yb_T  = Yb.T
                b_idx = (b_idx_full if bs == self.batch_size
                         else np.arange(bs)[:, None])

                # Cross-entropy loss: gather p at correct token, take -log
                # No one-hot allocation needed -- index directly into probs.
                correct_probs = probs[b_idx, t_idx, Yb_T]
                total_loss   += float(-np.sum(np.log(correct_probs + 1e-9)))
                del correct_probs

                # Softmax gradient simplification: delta = (p - one_hot(y)) / (B*T)
                # Computed in-place on probs buffer -- no extra allocation.
                probs[b_idx, t_idx, Yb_T] -= 1.0
                probs *= 1.0 / (bs * T)
                delta  = probs    # alias, not a copy

                # Output layer gradients
                dWout_acc += x_out.reshape(bs * T, -1).T @ delta.reshape(bs * T, -1)
                dbout_acc += delta.sum(axis=(0, 1))
                d_x        = delta @ self.Wout.T
                del probs, x_out  # free VRAM immediately

                # Backprop through transformer blocks in reverse order
                for i, (cache, blk) in enumerate(
                    zip(reversed(block_caches), reversed(self.blocks))
                ):
                    d_x, grads = self._block_backward(d_x, cache, blk)
                    del cache    # free as we go to keep VRAM low
                    acc = blk_grad_acc[self.num_blocks - 1 - i]
                    for k in grads:
                        acc[k] += grads[k]
                del block_caches

                # Embedding gradients
                # pos_embedding: sum d_x over batch (same position, all samples)
                # token embedding: scatter-add because multiple tokens in a batch
                #   may map to the same vocab row (must accumulate, not assign).
                if self.use_embedding:
                    dpe_acc += d_x.sum(axis=0)

                    if _DEVICE == "gpu" and _scatter_add is not None:
                        # Stays entirely on GPU
                        _scatter_add(de_acc, toks.reshape(-1), d_x.reshape(-1, D))
                    elif _DEVICE == "gpu":
                        # Fallback: move to CPU, add.at, move back
                        d_x_cpu  = np.asnumpy(d_x)
                        toks_cpu = np.asnumpy(toks)
                        de_cpu   = np.asnumpy(de_acc)
                        _np_cpu.add.at(de_cpu, toks_cpu.reshape(-1), d_x_cpu.reshape(-1, D))
                        de_acc   = np.array(de_cpu)
                    else:
                        _np_cpu.add.at(de_acc, toks.reshape(-1), d_x.reshape(-1, D))

            # Adam updates
            # Fused bias correction: lr_eff = epoch_lr * sqrt(1-b2^t) / (1-b1^t)
            # Computed ONCE per epoch, not once per parameter tensor.
            t      = self._adam_t
            bc1    = 1.0 - 0.9   ** t
            bc2    = 1.0 - 0.999 ** t
            lr_eff = epoch_lr * (bc2 ** 0.5) / bc1

            def _adam_step(param, grad, m, v):
                """In-place Adam update. No allocations."""
                m *= 0.9;   m += 0.1   * grad
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

            # Adaptive LR scheduler
            if epoch >= 5:
                lr_change_msg = ""
                if total_loss > prev_loss:
                    bounce_count  += 1
                    plateau_count  = 0
                    if bounce_count >= bounce_patience:
                        new_lr = max(epoch_lr * bounce_factor, lr_min)
                        if new_lr < epoch_lr:
                            lr_change_msg = (
                                f"  down bounce*{bounce_count}  "
                                f"{epoch_lr:.6f}->{new_lr:.6f}"
                            )
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
                            lr_change_msg = (
                                f"  down plateau  "
                                f"{epoch_lr:.6f}->{new_lr:.6f}"
                            )
                        epoch_lr      = new_lr
                        plateau_count = 0
            prev_loss = total_loss

            if log_every and epoch % log_every == 0:
                # equiv = average cross-entropy per (sample, position) pair.
                # Comparable across different context lengths / dataset sizes.
                equiv = total_loss / (n * ctx_size)
                print(
                    f"Epoch {epoch:>6} | Loss: {total_loss:.2f}  "
                    f"equiv: {equiv:.4f}  lr={epoch_lr:.6f}{lr_change_msg}",
                    flush=True,
                )

            if save_every and epoch > 0 and epoch % save_every == 0:
                self.save_weights(save_path)
                print(f"  checkpoint saved -> {save_path}", flush=True)

        # Persist the final adaptive lr so the next resume starts exactly
        # where this run left off instead of jumping back to the original lr.
        self.learning_rate = epoch_lr
        print(f"Training complete. (final lr={epoch_lr:.6f})")

    # ---- Predict ------------------------------------------------------------

    def predict(self, inputs) -> Tuple[int, float, "np.ndarray"]:
        """Run forward pass. Returns (predicted_class, confidence, probs)."""
        _, _, probs = self.forward(inputs)
        if _DEVICE == "gpu":
            probs = np.asnumpy(probs)
        predicted_class = int(probs.argmax())
        return predicted_class, float(probs[predicted_class]), probs

    # ---- Summary ------------------------------------------------------------

    def summary(self) -> None:
        """Print a formatted table of model architecture and parameter counts."""
        blk    = self.blocks[0]
        attn_p = blk["Wqkv"].size
        ff_p   = blk["W1"].size + blk["b1"].size + blk["W2"].size + blk["b2"].size
        out_p  = self.Wout.size + self.bout.size
        emb_p  = (self.embedding.size + self.pos_embedding.size
                  if self.use_embedding else 0)
        total  = (attn_p + ff_p) * self.num_blocks + out_p + emb_p
        width  = 52
        device = "GPU (CuPy)" if _DEVICE == "gpu" else "CPU (NumPy)"
        D      = self.embed_dim

        print("+" + "=" * width + "+")
        print("|" + " Mini-Transformer Summary".center(width) + "|")
        print("+" + "=" * width + "+")
        print(f"|  {'Device':<16} | {device:<{width-22}}|")
        print(f"|  {'Optimizer':<16} | {'Adam + adaptive LR':<{width-22}}|")
        print(f"|  {'Batch size':<16} | {self.batch_size:<{width-22}}|")
        if self.use_embedding:
            edim = f"{self.vocab_size} chars x {D}d  context={self.context_size}"
            print(f"|  {'Embedding':<16} | {edim:<{width-22}}|")
        print(f"|  {'Blocks':<16} | {str(self.num_blocks) + ' (independent)':<{width-22}}|")
        attn_str = f"causal, fused Wqkv {D}x{D*3} ({attn_p} params/block)"
        ff_str   = f"{D}->{D*4}->{D} ({ff_p} params/block)"
        print(f"|  {'Attention':<16} | {attn_str:<{width-22}}|")
        print(f"|  {'Feed-forward':<16} | {ff_str:<{width-22}}|")
        print(f"|  {'Output':<16} | {'all positions -> ' + str(self.output_size):<{width-22}}|")
        print(f"|  {'Vectorised':<16} | {'ON (batch matmul)':<{width-22}}|")
        print("+" + "=" * width + "+")
        pad = width - 23 - len(f"{total:,}")
        print(f"|  Total parameters: {total:,}{'':<{pad}}|")
        print("+" + "=" * width + "+")

    # ---- Save ---------------------------------------------------------------

    def save_weights(self, filename: str = "weights.json") -> None:
        """
        Save all weights, biases, and Adam state to a JSON file.

        ATOMIC WRITE: saves to a temp file first, then renames over the target.
        A Colab disconnect or OOM mid-save leaves the previous checkpoint intact
        rather than producing a corrupt file.

        No indentation (indent=None): files are ~4x smaller (~50MB vs 220MB).
        Loads identically -- just less human-readable.
        """
        to_list = (
            (lambda w: np.asnumpy(w).tolist()) if _DEVICE == "gpu"
            else (lambda w: w.tolist())
        )

        adam_blocks = []
        if self._adam_init:
            for buf in self._adam_blocks:
                adam_blocks.append(
                    {k: {"m": to_list(v["m"]), "v": to_list(v["v"])}
                     for k, v in buf.items()}
                )

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
            "Wout":          to_list(self.Wout),
            "bout":          to_list(self.bout),
            "blocks":        [{k: to_list(v) for k, v in blk.items()}
                              for blk in self.blocks],
            "adam_t":        self._adam_t if self._adam_init else 0,
            "adam_mWout":    to_list(self._mWout) if self._adam_init else None,
            "adam_vWout":    to_list(self._vWout) if self._adam_init else None,
            "adam_mbout":    to_list(self._mbout) if self._adam_init else None,
            "adam_vbout":    to_list(self._vbout) if self._adam_init else None,
            "adam_me":       to_list(self._me)  if (self._adam_init and self.use_embedding) else None,
            "adam_ve":       to_list(self._ve)  if (self._adam_init and self.use_embedding) else None,
            "adam_mpe":      to_list(self._mpe) if (self._adam_init and self.use_embedding) else None,
            "adam_vpe":      to_list(self._vpe) if (self._adam_init and self.use_embedding) else None,
            "adam_blocks":   adam_blocks,
        }

        # Atomic write: temp file -> rename (os.replace is atomic on Linux + Windows)
        dir_name = os.path.dirname(os.path.abspath(filename))
        tmp_fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(data, f)           # no indent = 4x smaller
            os.replace(tmp_path, filename)   # atomic rename
        except Exception:
            os.unlink(tmp_path)              # clean up on failure
            raise

        print(f"Weights saved to '{filename}'.")

    # ---- Load ---------------------------------------------------------------

    def load_weights(self, filename: str = "weights.json") -> None:
        """
        Load weights and (optionally) Adam state from a JSON file.

        Handles three legacy formats:
          1. Current: fused Wqkv per block
          2. Old: separate Wq/Wk/Wv per block -> merged on load
          3. Very old: single shared Wq/Wk/Wv -> replicated per block

        Adam state (adam_t > 0) is restored fully so that resuming training
        skips warmup and continues with correct momentum estimates.
        """
        if not os.path.exists(filename):
            print(f"No weights file found at '{filename}'.")
            return

        with open(filename) as f:
            data = json.load(f)

        self.input_size    = data["input_size"]
        self.hidden_layers = data["hidden_layers"]
        self.output_size   = data["output_size"]
        self.activation    = data["activation"]
        self.learning_rate = data["learning_rate"]
        self.batch_size    = data.get("batch_size",    1024)
        self.use_embedding = data.get("use_embedding", False)
        self.vocab_size    = data.get("vocab_size",    0)
        self.context_size  = data.get("context_size",  0)
        self.embed_dim     = data.get("embed_dim",     64)
        self.num_blocks    = data.get("num_blocks",    2)
        self._act_fn, self._act_d = _ACTIVATIONS[self.activation]
        self._scale = 1.0 / (self.embed_dim ** 0.5)

        self.weights = [np.array(w) for w in data["weights"]]
        self.biases  = [np.array(b) for b in data["biases"]]
        self.embedding     = np.array(data["embedding"])     if data.get("embedding")     else None
        self.pos_embedding = np.array(data["pos_embedding"]) if data.get("pos_embedding") else None
        self.Wout = np.array(data["Wout"])
        self.bout = np.array(data["bout"])

        if "blocks" in data:
            self.blocks = []
            for blk in data["blocks"]:
                b = {k: np.array(v) for k, v in blk.items()}
                if "Wq" in b and "Wqkv" not in b:
                    import numpy as _nl
                    def _cpu(a): return np.asnumpy(a) if _DEVICE == "gpu" else a
                    Wq = _cpu(b.pop("Wq"))
                    Wk = _cpu(b.pop("Wk"))
                    Wv = _cpu(b.pop("Wv"))
                    b["Wqkv"] = np.array(_nl.concatenate([Wq, Wk, Wv], axis=1))
                self.blocks.append(b)
        else:
            import numpy as _nl
            self.blocks = []
            for _ in range(self.num_blocks):
                Wq = _nl.array(data["Wq"])
                Wk = _nl.array(data["Wk"])
                Wv = _nl.array(data["Wv"])
                self.blocks.append({
                    "Wqkv": np.array(_nl.concatenate([Wq, Wk, Wv], axis=1)),
                    "W1":   np.array(data["W1"]),
                    "b1":   np.array(data["b1"]),
                    "W2":   np.array(data["W2"]),
                    "b2":   np.array(data["b2"]),
                })

        self._adam_init = False
        if data.get("adam_t") and data["adam_t"] > 0:
            self._init_adam()
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

    def __repr__(self) -> str:
        return (
            f"NeuralNetwork(embed={self.embed_dim}, blocks={self.num_blocks}, "
            f"causal=True, all_positions=True, "
            f"lr={self.learning_rate}, device='{self.device}')"
        )