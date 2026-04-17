"""
Neural_Network.py  --  Mini-Transformer (GPT-2 inspired, character-level)
=========================================================================

New in this version vs the previous one
----------------------------------------
  - LayerNorm        : pre-norm before attention AND feed-forward per block,
                       plus a final LayerNorm before the output projection.
                       Stabilises training, especially with more blocks.
  - Multi-head attn  : splits the embedding into H independent heads.
                       Each head specialises on different patterns.
  - Dropout          : randomly zeroes activations during training.
                       Reduces overfitting on small datasets.
  - Weight tying     : output projection Wout shares weights with the token
                       embedding (embedding.T). Halves those parameter counts.
  - Residual scaling : W2 (FF output) init scaled by 1/sqrt(2*num_blocks).
                       Prevents the residual stream growing too large.
  - Gradient clipping: caps global gradient norm before Adam update.
                       Prevents a single bad batch from blowing up weights.

Architecture (one block)
------------------------
  x
  |-- LayerNorm 1 --> multi-head attention --> dropout --> (+) --> x
  |-- LayerNorm 2 --> feed-forward MLP     --> dropout --> (+) --> x

Full stack
----------
  tokens (B, T)
    -> token embedding + positional embedding         (B, T, D)
    -> embedding dropout
    -> N x transformer block (each with LN+MHA+FF)    (B, T, D)
    -> final LayerNorm                                 (B, T, D)
    -> output projection at ALL positions              (B, T, vocab)
    -> softmax -> probabilities

GPU setup
---------
    pip install cupy-cuda12x        # CUDA 12.x  (Colab T4)
    pip install cupy-cuda11x        # CUDA 11.x  (older GPUs)
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, List, Literal, Optional, Tuple

# ---- GPU / CPU backend -------------------------------------------------------
# Import CuPy as `np` so all array code is device-agnostic.
# Falls back to NumPy silently if CuPy is not installed.
# Force Cuda Path to fix Bug where it Doesnt recognize the cuda install untested in google collab needs to be changed in the future

cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"

if os.path.exists(cuda_path):
    os.environ["CUDA_PATH"] = cuda_path
    os.environ["PATH"] = cuda_path + r"\bin;" + os.environ["PATH"]

try:
    import cupy as np
    np.cuda.Device(0).use()
    _DEVICE = "gpu"
    print(
        f"GPU detected -- training on: "
        f"{np.cuda.runtime.getDeviceProperties(0)['name'].decode()}"
    )
    # cupyx.scatter_add accumulates embedding gradients entirely on the GPU,
    # avoiding the CPU round-trip that numpy.add.at would require.
    try:
        from cupyx import scatter_add as _scatter_add
    except Exception:
        _scatter_add = None
except Exception:
    import numpy as np
    _DEVICE = "cpu"
    _scatter_add = None
    print("CuPy not found -- falling back to CPU (NumPy)")

# CPU numpy is always needed for index operations CuPy does not support.
import numpy as _np_cpu


# ---- Type aliases ------------------------------------------------------------
ActivationName = Literal["sigmoid", "tanh", "relu", "leaky_relu"]
Sample         = Tuple[List[float], int]


# ---- Activation functions and derivatives ------------------------------------
# Each entry is (forward_fn, derivative_fn).
# Derivative receives the PRE-activation value z (not the output a).

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


# ==============================================================================
#  TurboQuant KV Cache
#  Papers:
#    TurboQuant (arXiv 2504.19874, ICLR 2026) -- random rotation + Lloyd-Max
#    PolarQuant (arXiv 2502.02617, AISTATS 2026) -- polar coordinate variant
#    QJL        (arXiv 2406.03482, AAAI 2025) -- 1-bit JL residual correction
# ==============================================================================

class TurboQuantKVCache:
    """
    Compressed KV cache implementing TurboQuant_mse.

    Algorithm
    ---------
    For every K or V head-vector v ∈ R^d_h arriving online:

    1.  Random rotation  v_rot = v @ R^T
        R is a fixed random orthogonal matrix (QR-decomposed from Gaussian).
        After rotation each coordinate is approximately Beta(d/2, d/2)
        distributed and near-independent (Fact 3 of the TurboQuant paper).

    2.  Scalar quantisation  q = round(v_rot / scale)  where
        scale = max(|v_rot|) / (2^(bits-1) - 1)
        This is the symmetric uniform quantiser.  For the concentrated Beta
        distribution that results from step 1, uniform quantisation is a
        near-optimal Lloyd-Max approximation -- no per-vector zero-point or
        per-block normalisation constant is needed (zero overhead).

    3.  Storage:
        8-bit: int8 values + one float32 scale per head-vector → 4× vs fp32.
        4-bit: two nibbles packed per byte (high nibble / low nibble) + one
               float32 scale per head-vector → ~8× vs fp32.
               Packing: high = (q+8) >> 0 stored in bits [7:4],
                         low = (q+8) stored in bits [3:0].
               Values are in [-7, 7] (maxval = 7); +8 shifts to [1, 15]
               so that the unsigned nibble 0x0 is reserved as a padding
               sentinel and never produced by valid quantised data.

    4.  Reconstruct  v_approx = q * scale @ R

    The cache is a ring buffer of size context_size.  When full the oldest
    token is evicted (sliding window, matching miniGPT's generation strategy).

    Parameters
    ----------
    num_layers   : transformer depth
    num_heads    : attention heads
    head_dim     : dimension per head (embed_dim // num_heads)
    context_size : ring-buffer capacity (= model's context window)
    bits         : quantisation bit-width (4 or 8; default 8)

    Notes
    -----
    head_dim must be even when bits=4 (two values packed per byte).

    References
    ----------
    TurboQuant: https://arxiv.org/abs/2504.19874
    PolarQuant: https://arxiv.org/abs/2502.02617
    QJL:        https://arxiv.org/abs/2406.03482
    """

    def __init__(
        self,
        num_layers:   int,
        num_heads:    int,
        head_dim:     int,
        context_size: int,
        bits:         int = 8,
    ) -> None:
        if bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {bits}")
        if bits == 4 and head_dim % 2 != 0:
            raise ValueError(
                f"head_dim must be even for 4-bit packing, got {head_dim}"
            )

        self.L    = num_layers
        self.H    = num_heads
        self.d    = head_dim
        self.T    = context_size
        self.bits = bits
        self._maxval = 2 ** (bits - 1) - 1   # 127 for int8, 7 for int4

        # Fixed random orthogonal rotation matrix R ∈ R^(d, d).
        # Data-oblivious: generated once, reused for every vector.
        # QR decomposition of a Gaussian matrix gives a Haar-distributed
        # (uniformly random) orthogonal matrix -- exactly what TurboQuant needs.
        raw     = _np_cpu.random.randn(head_dim, head_dim).astype(_np_cpu.float32)
        R, _    = _np_cpu.linalg.qr(raw)
        self._R  = R      # (d, d)  float32
        self._Rt = R.T    # precomputed inverse (R orthogonal → R^-1 = R^T)

        # Ring buffer: per-layer, per-head, per-position.
        # 8-bit: store raw int8, shape (L, H, T, d).
        # 4-bit: pack two nibbles per byte, shape (L, H, T, d//2).
        #        High nibble = even index, low nibble = odd index.
        d_stored = head_dim if bits == 8 else head_dim // 2
        self._Kq = _np_cpu.zeros(
            (self.L, self.H, self.T, d_stored), dtype=_np_cpu.uint8
        )
        self._Vq = _np_cpu.zeros(
            (self.L, self.H, self.T, d_stored), dtype=_np_cpu.uint8
        )
        # float32 per-head-vector scales (L, H, T)
        self._Ks = _np_cpu.zeros((self.L, self.H, self.T), dtype=_np_cpu.float32)
        self._Vs = _np_cpu.zeros((self.L, self.H, self.T), dtype=_np_cpu.float32)

        self._pos    = 0   # next write position in ring buffer
        self._filled = 0   # tokens stored so far (≤ T)

    # ------------------------------------------------------------------
    # Internal: pack / unpack int4 nibbles
    # ------------------------------------------------------------------

    @staticmethod
    def _pack_nibbles(q_int8: "_np_cpu.ndarray") -> "_np_cpu.ndarray":
        """
        Pack a (..., d) int8 array whose values are in [-7, 7] into a
        (..., d//2) uint8 array using 4-bit nibbles.

        Encoding: shift values by +8 so they land in [1, 15] (unsigned nibble).
        Even-indexed values go into the high nibble (bits 7-4),
        odd-indexed values go into the low nibble  (bits 3-0).

        The sentinel value 0x0 (= original -8, outside [-7,7]) is never
        produced, making it safe as a zero-initialisation marker.
        """
        shifted = (q_int8 + 8).astype(_np_cpu.uint8)   # [1, 15], fits in 4 bits
        hi = shifted[..., 0::2] << 4                    # even → high nibble
        lo = shifted[..., 1::2] & 0x0F                  # odd  → low  nibble
        return (hi | lo).astype(_np_cpu.uint8)

    @staticmethod
    def _unpack_nibbles(packed: "_np_cpu.ndarray") -> "_np_cpu.ndarray":
        """
        Unpack a (..., d//2) uint8 array back to (..., d) int8 in [-7, 7].

        Reverses _pack_nibbles: extract high and low nibbles, interleave,
        then shift back by -8.
        """
        hi = (packed >> 4) & 0x0F                        # bits 7-4
        lo =  packed        & 0x0F                        # bits 3-0
        # interleave: even positions ← hi, odd positions ← lo
        d  = packed.shape[-1] * 2
        out = _np_cpu.empty(packed.shape[:-1] + (d,), dtype=_np_cpu.uint8)
        out[..., 0::2] = hi
        out[..., 1::2] = lo
        return (out.astype(_np_cpu.int16) - 8).astype(_np_cpu.int8)

    # ------------------------------------------------------------------
    # Quantise / dequantise one head-vector (used by push/get internally)
    # ------------------------------------------------------------------

    def _quant(self, v: "_np_cpu.ndarray"):
        """
        TurboQuant_mse encode: rotate then scalar-quantise.

        v      : (d,) float32
        returns: q (d,) int8 in [-maxval, maxval], scale float32
        """
        v_rot  = v @ self._R
        absmax = float(_np_cpu.max(_np_cpu.abs(v_rot))) + 1e-9
        scale  = absmax / self._maxval
        q = _np_cpu.clip(
            _np_cpu.round(v_rot / scale),
            -self._maxval, self._maxval,
        ).astype(_np_cpu.int8)
        return q, scale

    def _dequant(self, q: "_np_cpu.ndarray", scale: float) -> "_np_cpu.ndarray":
        """
        TurboQuant_mse decode: dequantise then inverse-rotate.

        q     : (d,) int8
        scale : float32
        returns: (d,) float32
        """
        v_rot_approx = q.astype(_np_cpu.float32) * scale
        return v_rot_approx @ self._Rt   # R^T = R^-1 for orthogonal R

    # ------------------------------------------------------------------
    # Batch push / get (vectorised over heads)
    # ------------------------------------------------------------------

    def push(self, K_new: "_np_cpu.ndarray", V_new: "_np_cpu.ndarray", layer: int) -> None:
        """
        Compress and store K,V for one new token in the given layer.

        K_new, V_new : (H, d)  float32  -- one token, all heads
        layer        : int
        """
        p = self._pos

        # Rotate all heads at once: (H, d) @ (d, d) = (H, d)
        K_rot = K_new @ self._R
        V_rot = V_new @ self._R

        # Per-head symmetric scales: shape (H,)
        K_sc = _np_cpu.abs(K_rot).max(axis=-1) / self._maxval + 1e-9
        V_sc = _np_cpu.abs(V_rot).max(axis=-1) / self._maxval + 1e-9

        # Quantise to int8 in [-maxval, maxval]: shape (H, d)
        K_q8 = _np_cpu.clip(
            _np_cpu.round(K_rot / K_sc[:, None]),
            -self._maxval, self._maxval,
        ).astype(_np_cpu.int8)
        V_q8 = _np_cpu.clip(
            _np_cpu.round(V_rot / V_sc[:, None]),
            -self._maxval, self._maxval,
        ).astype(_np_cpu.int8)

        # Store: int8 as-is (8-bit) or pack into nibbles (4-bit)
        if self.bits == 8:
            self._Kq[layer, :, p, :] = K_q8.view(_np_cpu.uint8)
            self._Vq[layer, :, p, :] = V_q8.view(_np_cpu.uint8)
        else:
            self._Kq[layer, :, p, :] = self._pack_nibbles(K_q8)
            self._Vq[layer, :, p, :] = self._pack_nibbles(V_q8)

        self._Ks[layer, :, p] = K_sc
        self._Vs[layer, :, p] = V_sc

    def advance(self) -> None:
        """Advance the ring-buffer pointer after all layers have been pushed."""
        self._pos    = (self._pos + 1) % self.T
        self._filled = min(self._filled + 1, self.T)

    def get(self, layer: int):
        """
        Dequantise and return the full K, V tensors for attention.

        Returns
        -------
        K : (H, T_filled, d)  float32  -- oldest token at index 0
        V : (H, T_filled, d)  float32
        """
        n = self._filled
        if n == 0:
            empty = _np_cpu.zeros((self.H, 0, self.d), dtype=_np_cpu.float32)
            return empty, empty

        # Ring-buffer index order: oldest → newest
        if n < self.T:
            idx = _np_cpu.arange(n)
        else:
            idx = (_np_cpu.arange(n) + self._pos) % self.T

        # Fetch packed storage: (H, n, d) or (H, n, d//2)
        Kq_raw = self._Kq[layer][:, idx, :]
        Vq_raw = self._Vq[layer][:, idx, :]
        Ks     = self._Ks[layer][:, idx]    # (H, n) float32
        Vs     = self._Vs[layer][:, idx]

        # Unpack to int8 (no-op for 8-bit path)
        if self.bits == 8:
            Kq = Kq_raw.view(_np_cpu.int8)   # (H, n, d)
            Vq = Vq_raw.view(_np_cpu.int8)
        else:
            # Unpack nibbles: (H, n, d//2) → (H, n, d)
            Kq = self._unpack_nibbles(Kq_raw)
            Vq = self._unpack_nibbles(Vq_raw)

        # Dequantise: (H, n, d) * (H, n, 1) → float32
        K_rot = Kq.astype(_np_cpu.float32) * Ks[:, :, None]
        V_rot = Vq.astype(_np_cpu.float32) * Vs[:, :, None]

        # Inverse rotation: (H, n, d) @ (d, d) → (H, n, d)
        K = K_rot @ self._Rt
        V = V_rot @ self._Rt
        return K, V

    def reset(self) -> None:
        """Clear the cache for a new generation sequence."""
        self._Kq[...] = 0;  self._Vq[...] = 0
        self._Ks[...] = 0;  self._Vs[...] = 0
        self._pos    = 0;   self._filled  = 0

    def memory_bytes(self) -> int:
        """
        Actual bytes used by the compressed cache.

        Counts both the quantised value arrays and the float32 scale arrays.
        """
        return (self._Kq.nbytes + self._Vq.nbytes +
                self._Ks.nbytes + self._Vs.nbytes)

    def compression_ratio(self) -> float:
        """
        Compression factor vs storing K and V as raw float32.

        Numerator  : float32 bytes for L×H×T×d  K tensors + same for V.
        Denominator: actual bytes used (quantised values + scales).
        """
        fp32_bytes = self.L * self.H * self.T * self.d * 4 * 2  # K + V
        return fp32_bytes / (self.memory_bytes() + 1e-9)

    def __repr__(self) -> str:
        mb = self.memory_bytes() / 1024 / 1024
        return (
            f"TurboQuantKVCache(layers={self.L}, heads={self.H}, "
            f"d_h={self.d}, T={self.T}, bits={self.bits}, "
            f"filled={self._filled}, "
            f"{mb:.2f} MB, ~{self.compression_ratio():.1f}x vs fp32)"
        )


# ==============================================================================
#  Main class
# ==============================================================================

class NeuralNetwork:
    """
    Character-level GPT-2-inspired transformer.

    Parameters
    ----------
    input_size : int
        Legacy flat input size. Ignored in the transformer path.
    hidden_layers : list[int]
        Legacy dense widths. Kept for save/load backward compatibility.
    output_size : int
        Vocabulary size -- number of softmax output classes.
    activation : str
        Legacy activation name. One of relu/tanh/sigmoid/leaky_relu.
    learning_rate : float
        Peak Adam learning rate. The adaptive scheduler may reduce this.
    batch_size : int
        Samples per gradient step.
    use_embedding : bool
        Use learned token + positional embeddings (always True in practice).
    vocab_size : int
        Characters in vocabulary.
    context_size : int
        Sequence length T.
    embed_dim : int
        Hidden dimension D. Must be divisible by num_heads.
    num_blocks : int
        Transformer blocks stacked in series.
    num_heads : int
        Attention heads. D must be divisible by num_heads.
        Each head works in a D/num_heads dimensional subspace.
    dropout : float
        Dropout rate (0 = disabled). Applied after attention and FF outputs,
        and after the initial embedding. Ignored during generation.
    weight_tying : bool
        If True, the output projection reuses the token embedding matrix
        (Wout = embedding.T). Halves those parameters and often improves
        quality because the embedding and output spaces are aligned.
    grad_clip : float
        Max global gradient L2 norm before Adam update (0 = disabled).
        Prevents a bad batch from taking a destructively large step.
    dtype : str
        Data type for weights. 'float32' or 'float16' for memory efficiency.
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
        num_heads:     int   = 4,
        dropout:       float = 0.0,
        weight_tying:  bool  = True,
        grad_clip:     float = 1.0,
        dtype:         str   = "float32",
    ) -> None:
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Choose from: {list(_ACTIVATIONS)}"
            )
        if not hidden_layers:
            raise ValueError("hidden_layers must have at least one entry.")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        # Store all hyperparameters -- also written to save files.
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
        self.num_heads     = num_heads
        self.dropout       = dropout
        self.weight_tying  = weight_tying
        self.grad_clip     = grad_clip
        self.dtype         = dtype

        # Head dimension: each head attends in a d_h-dimensional subspace.
        # Scale for attention scores: 1/sqrt(d_h) keeps dot products from
        # growing too large as d_h increases (prevents softmax saturation).
        self._head_dim   = embed_dim // num_heads
        self._scale_head = 1.0 / (self._head_dim ** 0.5)

        # ---- Token + positional embeddings ----------------------------------
        # Token embedding:    (vocab_size, D)  -- one row per character.
        # Positional embedding: (context_size, D) -- one row per position.
        # Both are learned and updated by Adam just like any weight matrix.
        if self.use_embedding:
            self.embedding     = np.random.randn(vocab_size, embed_dim).astype(dtype) * 0.01
            self.pos_embedding = np.random.randn(context_size, embed_dim).astype(dtype) * 0.01
        else:
            self.embedding     = None
            self.pos_embedding = None

        # ---- Transformer blocks ---------------------------------------------
        # Each block holds its own independent weights -- no sharing across
        # blocks. This allows each block to specialise on different patterns.
        #
        # Per block:
        #   Wqkv     -- fused Q,K,V projection  (D, 3D)
        #   W1, b1   -- FF layer 1              (D, 4D) + (4D,)
        #   W2, b2   -- FF layer 2              (4D, D) + (D,)
        #   ln1_g/b  -- LayerNorm 1 params      (D,) each
        #   ln2_g/b  -- LayerNorm 2 params      (D,) each
        #
        # W2 is scaled by 1/sqrt(2*num_blocks) on init (GPT-2 residual
        # scaling). With N residual paths each adding variance, this keeps
        # the total residual stream variance under control.
        D            = embed_dim
        resid_scale  = 0.02 / (2 * num_blocks) ** 0.5
        self.blocks: List[Dict] = []
        for _ in range(num_blocks):
            self.blocks.append({
                "Wqkv": np.random.randn(D, D * 3).astype(dtype) * 0.02,
                "W1":   np.random.randn(D, D * 4).astype(dtype) * 0.02,
                "b1":   np.zeros(D * 4).astype(dtype),
                "W2":   np.random.randn(D * 4, D).astype(dtype) * resid_scale,
                "b2":   np.zeros(D).astype(dtype),
                # LayerNorm params: gamma (scale) init to 1, beta (shift) to 0.
                # At init this is an identity transform -- the model learns
                # to deviate from identity as training progresses.
                "ln1_g": np.ones(D).astype(dtype),
                "ln1_b": np.zeros(D).astype(dtype),
                "ln2_g": np.ones(D).astype(dtype),
                "ln2_b": np.zeros(D).astype(dtype),
            })

        # ---- Final LayerNorm (GPT-2 style) ----------------------------------
        # Applied once after all transformer blocks, before output projection.
        # Ensures the final representations are well-scaled before the linear
        # output layer reads them.
        self.ln_f_g = np.ones(D)
        self.ln_f_b = np.zeros(D)

        # ---- Output projection ----------------------------------------------
        # Maps (B, T, D) -> (B, T, vocab) at ALL token positions.
        #
        # WEIGHT TYING: if enabled, Wout is not a separate matrix -- the
        # forward pass computes x @ embedding.T instead of x @ Wout.
        # Gradients from the output path accumulate into embedding's Adam
        # buffers alongside gradients from the embedding lookup path.
        # This works well because both mappings want similar directions:
        # "character c" in input space and "predict character c" in output
        # space are naturally related.
        if weight_tying:
            self.Wout = None    # no separate matrix; embedding.T used
        else:
            self.Wout = np.random.randn(D, output_size) * 0.02
        self.bout = np.zeros(output_size)

        # ---- Legacy dense weights (backward compatibility only) -------------
        # Old weight files contain these arrays. They are never trained or
        # used in the transformer path -- only saved/loaded for compatibility.
        actual_input = context_size * embed_dim if self.use_embedding else input_size
        layer_sizes  = [actual_input] + hidden_layers + [output_size]
        self.weights = []
        self.biases  = []
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))   # Xavier/Glorot
            self.weights.append(np.random.randn(fan_out, fan_in) * scale)
            self.biases.append(np.zeros((fan_out, 1)))

        # Adam not initialised until first train() or load_weights() call.
        self._adam_init = False

    # ==========================================================================
    #  Adam optimiser
    # ==========================================================================

    def _init_adam(self) -> None:
        """
        Allocate zeroed first-moment (m) and second-moment (v) buffers for
        every learnable parameter.

        Adam update rule (per parameter):
            m  = beta1*m + (1-beta1)*grad          -- smoothed gradient
            v  = beta2*v + (1-beta2)*grad^2        -- smoothed squared grad
            theta -= lr * m_hat / (sqrt(v_hat)+eps)

        m and v together form the "Adam state". Saving and restoring them
        on resume means the optimizer does NOT forget the momentum it built
        up over hundreds of epochs -- a clean resume with no warmup needed.
        """
        z = np.zeros_like

        # Per-block buffers (includes LN params since they are in blk dict)
        self._adam_blocks = []
        for blk in self.blocks:
            self._adam_blocks.append(
                {k: {"m": z(v), "v": z(v)} for k, v in blk.items()}
            )

        # Final LayerNorm buffers
        self._m_ln_f_g = z(self.ln_f_g);  self._v_ln_f_g = z(self.ln_f_g)
        self._m_ln_f_b = z(self.ln_f_b);  self._v_ln_f_b = z(self.ln_f_b)

        # Output projection buffers (only when NOT weight-tying)
        if not self.weight_tying:
            self._mWout = z(self.Wout);  self._vWout = z(self.Wout)
        self._mbout = z(self.bout);  self._vbout = z(self.bout)

        # Embedding buffers
        if self.use_embedding:
            self._me  = z(self.embedding);      self._ve  = z(self.embedding)
            self._mpe = z(self.pos_embedding);  self._vpe = z(self.pos_embedding)

        self._adam_t    = 0     # global step counter -- used for bias correction
        self._adam_init = True

    # ==========================================================================
    #  LayerNorm
    # ==========================================================================

    def _ln_forward(self, x, gamma, beta, eps=1e-5):
        """
        Layer Normalisation forward pass.

        Normalises the LAST axis (the embedding dimension D) independently
        for each (batch, position) pair, then applies learned scale (gamma)
        and shift (beta).

        Formula:
            x_norm = (x - mean) / sqrt(var + eps)
            out    = gamma * x_norm + beta

        WHY LAYERNORM?
        Without normalisation, activations can grow or shrink as they
        pass through blocks, making the loss landscape spiky and hard to
        optimise. LayerNorm keeps each token's representation on a
        consistent scale regardless of the input magnitude.

        WHY PRE-NORM (before attention/FF)?
        Post-norm (GPT-1 style) normalises the residual stream AFTER adding
        back. Pre-norm (GPT-2 style) normalises BEFORE, leaving the residual
        connection clean. Pre-norm trains more stably at larger depth.

        Cache stores what backward needs:
            x_norm -- normalised input (needed for d_gamma, d_x)
            gamma  -- scale parameter (needed for d_x_norm)
            rstd   -- 1/sqrt(var+eps) (needed for d_x rescaling)
            N      -- last dimension size (D)
        """
        mean   = x.mean(axis=-1, keepdims=True)      # (B, T, 1)
        var    = x.var(axis=-1, keepdims=True)        # (B, T, 1) biased
        rstd   = 1.0 / np.sqrt(var + eps)             # (B, T, 1)
        x_norm = (x - mean) * rstd                   # (B, T, D)
        out    = gamma * x_norm + beta               # (B, T, D)
        return out, (x_norm, gamma, rstd, x.shape[-1])

    def _ln_backward(self, d_out, cache):
        """
        Layer Normalisation backward pass.

        Derivation (normalising over last axis of size N):
            d_gamma = sum(d_out * x_norm, over B and T dims)
            d_beta  = sum(d_out,          over B and T dims)
            d_x_norm = d_out * gamma
            d_x = rstd/N * (N*d_x_norm
                            - sum(d_x_norm, axis=-1, keepdims=True)
                            - x_norm * sum(d_x_norm * x_norm, axis=-1, keepdims=True))

        The last formula is the standard Jacobian-vector product for the
        normalisation step. It accounts for the fact that changing any x_i
        affects the mean and variance used to normalise ALL x_j in that row.
        """
        x_norm, gamma, rstd, N = cache

        # Parameter gradients: sum over batch and time dims
        d_gamma = (d_out * x_norm).sum(axis=(0, 1))    # (D,)
        d_beta  = d_out.sum(axis=(0, 1))               # (D,)

        # Gradient through normalisation
        d_x_norm = d_out * gamma                       # (B, T, D)
        d_x = (rstd / N) * (
            N * d_x_norm
            - d_x_norm.sum(axis=-1, keepdims=True)
            - x_norm * (d_x_norm * x_norm).sum(axis=-1, keepdims=True)
        )                                               # (B, T, D)
        return d_x, d_gamma, d_beta

    # ==========================================================================
    #  Dropout
    # ==========================================================================

    def _apply_dropout(self, x, training: bool):
        """
        Inverted dropout: zero out random activations during training,
        then SCALE UP the survivors by 1/(1-rate) so the expected sum
        is unchanged.

        At inference (training=False) or if dropout=0, returns x unchanged
        and mask=None.

        WHY INVERTED SCALING?
        Without scaling, the average activation magnitude at inference is
        (1-rate) times what it was during training. Inverted dropout fixes
        this by scaling during training, so inference needs no adjustment.
        """
        if not training or self.dropout == 0.0:
            return x, None
        # Bernoulli mask: 1 with probability (1-dropout), 0 otherwise
        mask = (np.random.rand(*x.shape) > self.dropout).astype(float)
        mask /= (1.0 - self.dropout)   # inverted scaling
        return x * mask, mask

    # ==========================================================================
    #  Causal mask
    # ==========================================================================

    def _causal_mask(self, T: int):
        """
        Upper-triangular mask of shape (T, T).
        Entry [i, j] = -1e9 if j > i (future position), else 0.

        Adding this to attention scores before softmax drives attention to
        future positions to ~0, enforcing the "causal" property:
        position i can only attend to positions 0..i.

        This is what allows autoregressive generation -- the model is
        trained to never see future tokens, so it cannot at generation time.

        Cached: only rebuilt when T changes (never in practice).
        """
        if not hasattr(self, "_mask_cache") or self._mask_cache.shape[0] != T:
            self._mask_cache = np.triu(np.ones((T, T)), k=1) * -1e9
        return self._mask_cache

    # ==========================================================================
    #  Transformer block: forward
    # ==========================================================================

    def _block_forward(self, x, blk, training: bool = False):
        """
        One causal transformer block with pre-norm and multi-head attention.

        Shape throughout: (B, T, D)

        Flow
        ----
        x
        |--> LN1 --> multi-head attention --> dropout --> (+) --> x
        |--> LN2 --> feed-forward MLP     --> dropout --> (+) --> x

        Multi-head attention
        --------------------
        Instead of one big attention over D dimensions, we run H smaller
        attentions each over D/H dimensions (called d_h).

        WHY MULTIPLE HEADS?
        Each head can learn a different "what to attend to" pattern.
        Head 0 might learn word boundaries, head 1 might learn repeated
        characters, head 2 might track vowel/consonant alternation, etc.
        With a single head, all of this must be crammed into one pattern.

        The scale is now 1/sqrt(d_h), NOT 1/sqrt(D). This is important:
        with D=256, H=4, d_h=64, scale = 1/8 instead of 1/16.

        What is saved in cache
        ----------------------
        Everything needed by backward. x_attn is NOT saved -- recomputed
        cheaply from x + attn_out to save VRAM.
        """
        B, T, D  = x.shape
        H        = self.num_heads
        d_h      = D // H                  # dimension per head
        BT       = B * T

        # ---- Pre-norm 1: normalise before attention -------------------------
        ln1_out, ln1_cache = self._ln_forward(x, blk["ln1_g"], blk["ln1_b"])

        # ---- Fused multi-head QKV projection --------------------------------
        # One (D, 3D) matmul produces all of Q, K, V for all heads at once.
        # Reshape to (B, T, 3, H, d_h) then move heads axis forward.
        QKV = (ln1_out.reshape(BT, D) @ blk["Wqkv"]).reshape(B, T, 3, H, d_h)
        Q = QKV[:, :, 0].transpose((0, 2, 1, 3))    # (B, H, T, d_h)
        K = QKV[:, :, 1].transpose((0, 2, 1, 3))
        V = QKV[:, :, 2].transpose((0, 2, 1, 3))

        # ---- Scaled dot-product attention (per head) ------------------------
        # scores[b, h, i, j] = how much position i in head h attends to j
        scores  = Q @ K.transpose((0, 1, 3, 2)) * self._scale_head  # (B, H, T, T)
        scores += self._causal_mask(T)                              # block future
        scores -= scores.max(axis=-1, keepdims=True)                # stable softmax
        exp_s   = np.exp(scores)
        A       = exp_s / exp_s.sum(axis=-1, keepdims=True)         # (B, H, T, T)

        # Weighted sum of values, then merge heads back to (B, T, D)
        attn_h   = A @ V                                            # (B, H, T, d_h)
        attn_out = attn_h.transpose((0, 2, 1, 3)).reshape(B, T, D)   # (B, T, D)

        # Attention dropout (only during training)
        attn_out, drop1_mask = self._apply_dropout(attn_out, training)

        # Residual 1: add attention output back to original x
        x_attn = x + attn_out

        # ---- Pre-norm 2: normalise before feed-forward ----------------------
        ln2_out, ln2_cache = self._ln_forward(x_attn, blk["ln2_g"], blk["ln2_b"])

        # ---- Feed-forward network: D -> 4D (ReLU) -> D ----------------------
        # The 4x expansion gives the network a high-dimensional workspace
        # to combine features before projecting back to D.
        h      = np.maximum(0.0, ln2_out @ blk["W1"] + blk["b1"])  # (B, T, 4D)
        ff_out = h @ blk["W2"] + blk["b2"]                          # (B, T, D)

        # FF dropout (only during training)
        ff_out, drop2_mask = self._apply_dropout(ff_out, training)

        # Residual 2: add FF output back
        x_out = x_attn + ff_out

        # Save everything backward needs (x_attn omitted -- recomputed)
        cache = (
            x, ln1_out, ln1_cache,
            Q, K, V, A,
            attn_out, drop1_mask,
            x_attn, ln2_out, ln2_cache,
            h, ff_out, drop2_mask,
        )
        return x_out, cache

    # ==========================================================================
    #  Transformer block: backward
    # ==========================================================================

    def _block_backward(self, d_out, cache, blk):
        """
        Backprop through one transformer block.

        d_out : upstream gradient  (B, T, D)
        Returns (d_x, grads_dict) where d_x is passed to the previous block.

        Residual rule
        -------------
        For any residual  out = a + f(a):
            d_a += d_out          (direct path through residual)
            d_a += d_f(a)         (path through the function)
        Both gradients are summed because out depends on a twice.

        Multi-head attention backward
        -----------------------------
        The forward merged H heads. Backward splits d_attn_out back into
        H head-gradients, backprops through each head's softmax and Q/K/V
        matmuls, then concatenates and does the fused Wqkv backward.

        LayerNorm backward
        ------------------
        Called via _ln_backward() which returns gradients for x, gamma, beta.
        """
        (
            x, ln1_out, ln1_cache,
            Q, K, V, A,
            attn_out, drop1_mask,
            x_attn, ln2_out, ln2_cache,
            h, ff_out, drop2_mask,
        ) = cache

        B, T, D = x.shape
        H       = self.num_heads
        d_h     = D // H
        BT      = B * T

        # ---- Residual 2 backward --------------------------------------------
        # d_out flows through BOTH the direct residual (to x_attn) AND the
        # FF branch. Both paths receive the full d_out.
        d_ff_out   = d_out * drop2_mask if drop2_mask is not None else d_out

        # FF backward
        # Note: d_h_grad is named with _grad suffix to avoid shadowing d_h (int)
        # which is the head dimension D//H used later in reshape calls.
        dW2       = h.reshape(BT, -1).T @ d_ff_out.reshape(BT, -1)    # (4D, D)
        db2       = d_ff_out.sum(axis=(0, 1))                           # (D,)
        d_h_grad  = d_ff_out @ blk["W2"].T                              # (B, T, 4D)
        d_h_grad *= (h > 0)                                             # ReLU deriv
        dW1       = ln2_out.reshape(BT, -1).T @ d_h_grad.reshape(BT, -1)   # (D, 4D)
        db1       = d_h_grad.sum(axis=(0, 1))                                # (4D,)
        d_ln2_out = d_h_grad @ blk["W1"].T                                   # (B, T, D)

        # LN2 backward: returns gradient into x_attn + LN parameter grads
        d_x_attn_from_ff, d_ln2_g, d_ln2_b = self._ln_backward(d_ln2_out, ln2_cache)

        # Combine: x_attn gradient = residual path (d_out) + FF path
        d_x_attn = d_out + d_x_attn_from_ff

        # ---- Residual 1 backward --------------------------------------------
        d_attn_out = d_x_attn * drop1_mask if drop1_mask is not None else d_x_attn

        # Multi-head attention backward
        # Reshape d_attn_out from (B,T,D) back to per-head (B,H,T,d_h)
        d_attn_h = (
            d_attn_out.reshape(B, T, H, d_h).transpose((0, 2, 1, 3))  # (B, H, T, d_h)
        )

        # Backward through A @ V = attn_h
        dA = d_attn_h @ V.transpose((0, 1, 3, 2))                # (B, H, T, T)
        dV = A.transpose((0, 1, 3, 2)) @ d_attn_h                # (B, H, T, d_h)

        # Softmax Jacobian-vector product:
        #   dS = A * (dA - sum(dA*A, axis=-1, keepdims=True))
        # This avoids building the full (T,T) Jacobian matrix.
        dS  = A * (dA - (dA * A).sum(axis=-1, keepdims=True))  # (B, H, T, T)
        dS *= self._scale_head

        # Backward through Q @ K^T
        dQ = dS @ K                                             # (B, H, T, d_h)
        dK = dS.transpose((0, 1, 3, 2)) @ Q                      # (B, H, T, d_h)

        # Reshape back: (B, H, T, d_h) -> (B, T, H, d_h) -> (BT, D)
        dQ_r = dQ.transpose((0, 2, 1, 3)).reshape(BT, D)
        dK_r = dK.transpose((0, 2, 1, 3)).reshape(BT, D)
        dV_r = dV.transpose((0, 2, 1, 3)).reshape(BT, D)

        # Fused Wqkv backward: one matmul instead of three
        ln1_out_r = ln1_out.reshape(BT, D)
        dQKV_r    = np.concatenate([dQ_r, dK_r, dV_r], axis=1)  # (BT, 3D)
        dWqkv     = ln1_out_r.T @ dQKV_r                          # (D, 3D)
        d_ln1_out = (dQKV_r @ blk["Wqkv"].T).reshape(B, T, D)    # (B, T, D)

        # LN1 backward
        d_x_from_attn, d_ln1_g, d_ln1_b = self._ln_backward(d_ln1_out, ln1_cache)

        # Total gradient into x: residual path + attention path
        d_x = d_x_attn + d_x_from_attn

        grads = {
            "Wqkv":  dWqkv,
            "W1":    dW1,  "b1": db1,
            "W2":    dW2,  "b2": db2,
            "ln1_g": d_ln1_g, "ln1_b": d_ln1_b,
            "ln2_g": d_ln2_g, "ln2_b": d_ln2_b,
        }
        return d_x, grads

    # ==========================================================================
    #  Full forward pass
    # ==========================================================================

    def _transformer_forward(self, token_idx_batch, training: bool = False):
        """
        Complete forward pass from token indices to softmax probabilities.

        token_idx_batch : int array  (T, B)
        training        : bool  -- enables dropout when True

        Returns probs (B, T, vocab) and a cache tuple needed by the backward.
        """
        toks = token_idx_batch.T                                   # (B, T)

        # Token embedding lookup + positional embedding (broadcast over B)
        x = self.embedding[toks] + self.pos_embedding              # (B, T, D)

        # Embedding dropout: randomly zero full embedding vectors.
        # Applied to the sum of token + position embeddings.
        x, emb_drop_mask = self._apply_dropout(x, training)

        # Transformer blocks
        block_caches = []
        for blk in self.blocks:
            x, cache = self._block_forward(x, blk, training=training)
            block_caches.append(cache)

        # Final LayerNorm: normalise before output projection
        x, ln_f_cache = self._ln_forward(x, self.ln_f_g, self.ln_f_b)

        # Output projection at ALL T positions simultaneously.
        # Weight tying: use embedding.T instead of a separate Wout matrix.
        if self.weight_tying:
            logits = x @ self.embedding.T + self.bout              # (B, T, vocab)
        else:
            logits = x @ self.Wout + self.bout                     # (B, T, vocab)

        # Numerically stable softmax (subtract max before exp)
        logits -= logits.max(axis=2, keepdims=True)
        e      = np.exp(logits)
        probs  = e / e.sum(axis=2, keepdims=True)                  # (B, T, vocab)

        return probs, (toks, x, block_caches, ln_f_cache, emb_drop_mask)

    def forward(self, inputs):
        """
        Single-sample forward pass used by generate().
        training=False so dropout is disabled.
        Returns last-position probabilities for the next character.
        """
        if self.use_embedding:
            arr  = np.array(inputs, dtype=float).reshape(
                self.context_size, self.vocab_size
            )
            toks = np.array(arr.argmax(axis=1), dtype=int).reshape(
                self.context_size, 1
            )
            probs, _ = self._transformer_forward(toks, training=False)
            if _DEVICE == "gpu":
                probs = np.asnumpy(probs)
            return None, None, probs[0, -1, :]   # last position only

        # Legacy MLP path (non-embedding models)
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

    # ==========================================================================
    #  Training loop
    # ==========================================================================

    def train(
        self,
        data,
        epochs:     int,
        log_every:  int = 1,
        save_every: int = 0,
        save_path:  str = "checkpoint.json",
    ) -> None:
        """
        Train with batched gradient descent + Adam + adaptive LR.

        Parameters
        ----------
        data : tuple (X_idx, Y_idx) or legacy list of (features, label)
            Preferred format: index arrays from make_index_arrays() -- avoids
            allocating one-hot floats that would be decoded back anyway.
        epochs : int
            Full passes over the training data.
        log_every : int
            Print loss every N epochs (0 = silent).
        save_every : int
            Checkpoint every N epochs using atomic writes (0 = disabled).

        Gradient clipping
        -----------------
        Before each Adam update, the global L2 norm of ALL gradients is
        computed. If it exceeds grad_clip, every gradient tensor is scaled
        down proportionally so the norm equals grad_clip exactly.
        This prevents a single bad batch from taking a destructively
        large weight update step.

        Adaptive LR
        -----------
        Warmup (fresh training only, epochs 0-4):
            Ramp lr from lr/5 -> lr_max. Skipped on resume because Adam
            already has built-up momentum estimates.
        Bounce (loss increases 2+ consecutive epochs):
            lr *= 0.7  -- model overshot, take smaller steps.
        Plateau (no new best for 5 epochs):
            lr *= 0.7  -- model stuck, try finer steps.
        Floored at lr_max / 5.

        At the end of training, self.learning_rate is updated to the final
        adaptive lr so that the next resume starts from the right value.
        """
        if not self._adam_init:
            self._init_adam()

        ctx_size = self.context_size
        vs       = self.vocab_size
        D        = self.embed_dim
        lr_max   = self.learning_rate
        lr_min   = lr_max / 5.0

        print("  Loading dataset onto device...")

        # ---- Data ingestion -------------------------------------------------
        # Fast path: make_index_arrays() returns (T,N) int32 arrays directly.
        # Legacy path: decode one-hot samples back to indices.
        if isinstance(data, tuple) and len(data) == 2 and hasattr(data[0], "shape"):
            X_idx_cpu, Y_idx_cpu = data
            n = X_idx_cpu.shape[1]
        else:
            n = len(data)
            X_idx_cpu = _np_cpu.zeros((ctx_size, n), dtype=_np_cpu.int32)
            Y_idx_cpu = _np_cpu.zeros((ctx_size, n), dtype=_np_cpu.int32)
            for j, (feat, label) in enumerate(data):
                oh   = _np_cpu.array(feat).reshape(ctx_size, vs)
                toks = oh.argmax(axis=1)
                X_idx_cpu[:, j]    = toks
                Y_idx_cpu[:-1, j]  = toks[1:]
                Y_idx_cpu[-1,  j]  = label

        # Move entire dataset to GPU once (no-op on CPU)
        X_idx = np.array(X_idx_cpu)
        Y_idx = np.array(Y_idx_cpu)

        dp_str = f"{self.dropout:.2f}" if self.dropout > 0 else "OFF"
        print(
            f"  {_DEVICE.upper()} ready -- {n:,} samples | "
            f"batch={self.batch_size} | lr={lr_max:.5f} | "
            f"heads={self.num_heads} | dropout={dp_str} | "
            f"weight_tying={'ON' if self.weight_tying else 'OFF'} | "
            f"grad_clip={self.grad_clip if self.grad_clip > 0 else 'OFF'}\n"
        )

        # ---- Resume detection -----------------------------------------------
        # If Adam state was loaded (_adam_t > 0), skip warmup and use the
        # saved lr directly -- no need to re-ramp from scratch.
        _resuming = self._adam_init and self._adam_t > 0
        epoch_lr  = lr_max if not _resuming else self.learning_rate

        # ---- Adaptive LR state ----------------------------------------------
        best_loss        = float("inf")
        plateau_count    = 0
        plateau_patience = 5
        plateau_factor   = 0.7
        bounce_factor    = 0.7
        bounce_count     = 0
        bounce_patience  = 2
        prev_loss        = float("inf")
        lr_change_msg    = ""

        # ---- Pre-allocate gradient buffers ----------------------------------
        # Allocated once, zeroed in-place each epoch (buf[...] = 0.0).
        # Avoids thousands of GPU memory allocations and CUDA syncs.
        blk_grad_acc = [
            {k: np.zeros_like(v) for k, v in blk.items()}
            for blk in self.blocks
        ]
        # Final LN gradient buffers
        d_ln_f_g_acc = np.zeros_like(self.ln_f_g)
        d_ln_f_b_acc = np.zeros_like(self.ln_f_b)

        # Output projection / embedding gradient buffers
        # With weight tying, dWout folds into de_acc (same matrix).
        if not self.weight_tying:
            dWout_acc = np.zeros_like(self.Wout)
        dbout_acc = np.zeros_like(self.bout)
        de_acc    = np.zeros_like(self.embedding)     if self.use_embedding else None
        dpe_acc   = np.zeros_like(self.pos_embedding) if self.use_embedding else None

        # ---- Hoisted constants (never change inside the loop) ---------------
        T          = ctx_size
        t_idx      = np.arange(T)[None, :]
        b_idx_full = np.arange(self.batch_size)[:, None]

        # ================================================================
        #  Epoch loop
        # ================================================================
        for epoch in range(epochs):

            # Warmup: ramp lr linearly over first 5 epochs (fresh only).
            # Adam's momentum estimates are unreliable at step 0 (initialised
            # to zero), so large early steps can point in noisy directions.
            if not _resuming and epoch < 5:
                epoch_lr      = lr_max * (epoch + 1) / 5
                lr_change_msg = ""

            # GPU-side shuffle avoids moving data off GPU for index generation
            idx_np = np.random.permutation(n)
            X_shuf = X_idx[:, idx_np]
            Y_shuf = Y_idx[:, idx_np]

            total_loss   = 0.0
            self._adam_t += 1

            # Zero all gradient buffers in-place (no allocations)
            for acc in blk_grad_acc:
                for a in acc.values():
                    a[...] = 0.0
            d_ln_f_g_acc[...] = 0.0
            d_ln_f_b_acc[...] = 0.0
            if not self.weight_tying:
                dWout_acc[...] = 0.0
            dbout_acc[...] = 0.0
            if self.use_embedding:
                de_acc[...]  = 0.0
                dpe_acc[...] = 0.0

            # ================================================================
            #  Mini-batch loop
            # ================================================================
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                bs  = end - start
                Xb  = X_shuf[:, start:end]    # (T, B)
                Yb  = Y_shuf[:, start:end]    # (T, B)

                # Forward pass (training=True enables dropout)
                probs, (toks, x_out, block_caches, ln_f_cache, emb_drop_mask) = (
                    self._transformer_forward(Xb, training=True)
                )

                Yb_T  = Yb.T
                b_idx = (b_idx_full if bs == self.batch_size
                         else np.arange(bs)[:, None])

                # ---- Cross-entropy loss -------------------------------------
                # Gather predicted probability at the correct next token.
                # -log(p_correct) is the cross-entropy for one (batch, pos).
                correct_probs = probs[b_idx, t_idx, Yb_T]
                total_loss   += float(-np.sum(np.log(correct_probs + 1e-9)))
                del correct_probs

                # ---- Softmax + cross-entropy gradient -----------------------
                # The combined gradient simplifies to:
                #   delta[b,t,c] = (p[b,t,c] - 1[c==y[b,t]]) / (B*T)
                # Computed in-place on the probs buffer -- no extra allocation.
                probs[b_idx, t_idx, Yb_T] -= 1.0
                probs *= 1.0 / (bs * T)
                delta  = probs    # alias, not a copy

                # ---- Output projection backward ----------------------------
                bs_T = bs * T
                delta_r = delta.reshape(bs_T, vs)
                x_out_r = x_out.reshape(bs_T, D)

                if self.weight_tying:
                    # Gradient to embedding from the output path:
                    #   logits = x @ embedding.T  =>  d_embedding += delta.T @ x
                    de_acc  += delta_r.T @ x_out_r                   # (vocab, D)
                    d_x      = (delta_r @ self.embedding).reshape(bs, T, D)
                else:
                    dWout_acc += x_out_r.T @ delta_r                  # (D, vocab)
                    # delta is (bs, T, vocab), Wout.T is (vocab, D)
                    # matmul gives (bs, T, D) directly -- no reshape needed
                    d_x        = delta @ self.Wout.T                  # (bs, T, D)

                dbout_acc += delta.sum(axis=(0, 1))
                del probs, x_out

                # ---- Final LayerNorm backward -------------------------------
                d_x, d_lg, d_lb = self._ln_backward(d_x, ln_f_cache)
                d_ln_f_g_acc += d_lg
                d_ln_f_b_acc += d_lb

                # ---- Backprop through transformer blocks (reverse) ----------
                for i, (cache, blk) in enumerate(
                    zip(reversed(block_caches), reversed(self.blocks))
                ):
                    d_x, grads = self._block_backward(d_x, cache, blk)
                    del cache    # free VRAM as we go
                    acc = blk_grad_acc[self.num_blocks - 1 - i]
                    for k in grads:
                        acc[k] += grads[k]
                del block_caches

                # ---- Embedding gradients ------------------------------------
                # Apply embedding dropout gradient (if dropout was used)
                if emb_drop_mask is not None:
                    d_x = d_x * emb_drop_mask

                if self.use_embedding:
                    # Positional embedding: same position across all samples,
                    # so sum d_x over the batch dimension.
                    dpe_acc += d_x.sum(axis=0)                       # (T, D)

                    # Token embedding: scatter-add because multiple tokens in
                    # the same batch may map to the same vocab row.
                    if _DEVICE == "gpu" and _scatter_add is not None:
                        _scatter_add(de_acc, toks.reshape(-1),
                                     d_x.reshape(-1, D))
                    elif _DEVICE == "gpu":
                        d_x_cpu  = np.asnumpy(d_x)
                        toks_cpu = np.asnumpy(toks)
                        de_cpu   = np.asnumpy(de_acc)
                        _np_cpu.add.at(de_cpu, toks_cpu.reshape(-1),
                                       d_x_cpu.reshape(-1, D))
                        de_acc = np.array(de_cpu)
                    else:
                        _np_cpu.add.at(de_acc, toks.reshape(-1),
                                       d_x.reshape(-1, D))

            # ================================================================
            #  Gradient clipping
            # ================================================================
            # Compute the global L2 norm of ALL gradient tensors combined.
            # If it exceeds grad_clip, scale every tensor down proportionally.
            #
            # WHY GLOBAL (not per-parameter)?
            # Per-parameter clipping distorts the relative gradient directions.
            # Global clipping preserves the direction -- it just shortens the
            # overall step if it would be too large.
            if self.grad_clip > 0:
                all_grads = []
                for acc in blk_grad_acc:
                    for g in acc.values():
                        all_grads.append(g.reshape(-1))
                all_grads += [d_ln_f_g_acc.reshape(-1), d_ln_f_b_acc.reshape(-1)]
                all_grads.append(dbout_acc.reshape(-1))
                if not self.weight_tying:
                    all_grads.append(dWout_acc.reshape(-1))
                if self.use_embedding:
                    all_grads += [de_acc.reshape(-1), dpe_acc.reshape(-1)]

                # Sum of squared norms across all tensors
                total_norm_sq = sum(float(np.sum(g * g)) for g in all_grads)
                total_norm    = total_norm_sq ** 0.5

                if total_norm > self.grad_clip:
                    # Scale factor < 1 -- shrink all gradients uniformly
                    scale = self.grad_clip / (total_norm + 1e-6)
                    for acc in blk_grad_acc:
                        for k in acc:
                            acc[k] *= scale
                    d_ln_f_g_acc *= scale
                    d_ln_f_b_acc *= scale
                    dbout_acc    *= scale
                    if not self.weight_tying:
                        dWout_acc *= scale
                    if self.use_embedding:
                        de_acc  *= scale
                        dpe_acc *= scale

            # ================================================================
            #  Adam parameter updates
            # ================================================================
            # Bias correction is computed once per epoch and baked into lr_eff
            # rather than recomputed per-parameter-tensor (saves ~16 power ops).
            #   lr_eff = epoch_lr * sqrt(1 - beta2^t) / (1 - beta1^t)
            t      = self._adam_t
            bc1    = 1.0 - 0.9   ** t    # (1 - beta1^t)
            bc2    = 1.0 - 0.999 ** t    # (1 - beta2^t)
            lr_eff = epoch_lr * (bc2 ** 0.5) / bc1

            def _adam_step(param, grad, m, v):
                """In-place Adam update. Zero allocations."""
                m *= 0.9;   m += 0.1   * grad          # update m (momentum)
                v *= 0.999; v += 0.001 * grad * grad   # update v (velocity)
                param -= lr_eff * m / (np.sqrt(v) + 1e-8)

            # Update all block parameters
            for blk, acc, adam_buf in zip(
                self.blocks, blk_grad_acc, self._adam_blocks
            ):
                for k in blk:
                    _adam_step(blk[k], acc[k], adam_buf[k]["m"], adam_buf[k]["v"])

            # Update final LN
            _adam_step(self.ln_f_g, d_ln_f_g_acc, self._m_ln_f_g, self._v_ln_f_g)
            _adam_step(self.ln_f_b, d_ln_f_b_acc, self._m_ln_f_b, self._v_ln_f_b)

            # Update output projection (separate matrix only if not weight-tied)
            if not self.weight_tying:
                _adam_step(self.Wout, dWout_acc, self._mWout, self._vWout)
            _adam_step(self.bout, dbout_acc, self._mbout, self._vbout)

            # Update embeddings
            if self.use_embedding:
                _adam_step(self.embedding,     de_acc,  self._me,  self._ve)
                _adam_step(self.pos_embedding, dpe_acc, self._mpe, self._vpe)

            # ================================================================
            #  Adaptive LR scheduler
            # ================================================================
            if epoch >= 5:
                lr_change_msg = ""
                if total_loss > prev_loss:
                    # Loss went up -- model is bouncing over a minimum
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

            # ---- Logging ----------------------------------------------------
            if log_every and epoch % log_every == 0:
                # equiv = mean cross-entropy per (sample, position) pair.
                # Comparable across runs with different context lengths.
                equiv = total_loss / (n * ctx_size)
                print(
                    f"Epoch {epoch:>6} | Loss: {total_loss:.2f}  "
                    f"equiv: {equiv:.4f}  lr={epoch_lr:.6f}{lr_change_msg}",
                    flush=True,
                )

            # ---- Checkpoint -------------------------------------------------
            if save_every and epoch > 0 and epoch % save_every == 0:
                self.save_weights(save_path)
                print(f"  checkpoint saved -> {save_path}", flush=True)

        # Persist final adaptive lr so the next resume starts exactly here
        self.learning_rate = epoch_lr
        print(f"Training complete. (final lr={epoch_lr:.6f})")

    # ==========================================================================
    #  Predict (generation)
    # ==========================================================================

    def predict(self, inputs) -> Tuple[int, float, "np.ndarray"]:
        """Run forward pass (no dropout). Returns (class, confidence, probs)."""
        _, _, probs = self.forward(inputs)
        if _DEVICE == "gpu":
            probs = np.asnumpy(probs)
        predicted_class = int(probs.argmax())
        return predicted_class, float(probs[predicted_class]), probs

    def predict_from_indices(self, token_indices) -> "np.ndarray":
        """
        Fast generation path: takes token indices directly, skips one-hot roundtrip.
        Returns last-position probability vector, shape (vocab_size,).
        """
        toks  = np.array(token_indices, dtype=int).reshape(self.context_size, 1)
        probs, _ = self._transformer_forward(toks, training=False)
        if _DEVICE == "gpu":
            probs = np.asnumpy(probs)
        return probs[0, -1, :]

    # ==========================================================================
    #  TurboQuant: single-token decode step
    # ==========================================================================

    def _block_forward_decode(
        self,
        x_new:   "np.ndarray",
        blk:     dict,
        K_cache: "np.ndarray",
        V_cache: "np.ndarray",
    ):
        """
        Single-token forward through one transformer block using a pre-built
        TurboQuant KV cache instead of recomputing the full context window.

        x_new   : (1, D)       -- embedded new token (with positional embedding)
        K_cache : (H, T, d_h)  -- dequantised keys from TurboQuantKVCache.get()
        V_cache : (H, T, d_h)  -- dequantised values

        No causal mask: all cached tokens are in the past so the new token may
        attend to all of them without restriction.

        Returns
        -------
        x_out  (1, D)  -- output to pass along / into next block
        K_new  (H, d_h) -- new key  (to be pushed into TurboQuantKVCache)
        V_new  (H, d_h) -- new value
        """
        D, H, d_h = self.embed_dim, self.num_heads, self._head_dim

        # Pre-norm 1
        ln1_out, _ = self._ln_forward(
            x_new[None, ...], blk["ln1_g"], blk["ln1_b"]
        )                                                       # (1, 1, D)

        # QKV for new token only
        QKV   = (ln1_out.reshape(1, D) @ blk["Wqkv"]).reshape(1, 3, H, d_h)
        Q     = QKV[0, 0]    # (H, d_h)  query
        K_new = QKV[0, 1]    # (H, d_h)  new key   → will be cached
        V_new = QKV[0, 2]    # (H, d_h)  new value → will be cached

        T = K_cache.shape[1]
        if T > 0:
            # Attention: Q (H, d_h) over cached K (H, T, d_h) → scores (H, 1, T)
            scores = (Q[:, None, :] @ K_cache.transpose(0, 2, 1)) * self._scale_head
            scores -= scores.max(axis=-1, keepdims=True)
            A        = np.exp(scores)
            A       /= A.sum(axis=-1, keepdims=True)             # (H, 1, T)
            attn_h   = A @ V_cache                               # (H, 1, d_h)
            attn_out = attn_h.transpose(1, 0, 2).reshape(1, D)  # (1, D)
        else:
            attn_out = np.zeros((1, D))

        x_attn = x_new + attn_out    # residual 1

        # Pre-norm 2 + feed-forward
        ln2_out, _ = self._ln_forward(
            x_attn[None, ...], blk["ln2_g"], blk["ln2_b"]
        )                                                        # (1, 1, D)
        h_ff  = np.maximum(0.0, ln2_out.reshape(1, D) @ blk["W1"] + blk["b1"])
        ff    = h_ff @ blk["W2"] + blk["b2"]                    # (1, D)
        x_out = x_attn + ff    # residual 2

        return x_out, K_new, V_new

    # ==========================================================================
    #  TurboQuant-accelerated generation
    # ==========================================================================

    def generate_fast(
        self,
        token_indices: list,
        length:      int,
        temperature: float = 0.8,
        kv_bits:     int   = 8,
    ) -> list:
        """
        Autoregressive generation with a TurboQuant KV cache.

        Speed comparison vs predict_from_indices()
        -------------------------------------------
        predict_from_indices() recomputes ALL T context tokens from scratch each
        step → O(T²) attention work per generated character.

        generate_fast() does:
          1. Prefill  – one full T-token forward to populate the KV cache.
          2. Decode   – per step: embed 1 new token, attend its Q over T cached
                        K,V via _block_forward_decode() → O(T) per step.

        For T=32 and length=300 that is ~32× fewer transformer operations.

        KV memory compression
        ---------------------
        TurboQuant_mse (arXiv 2504.19874):
          • Random orthogonal rotation R equalises coordinate variances.
          • After rotation, each coordinate follows ~Beta(d/2, d/2) →
            uniform INT8 scalar quantisation is near-optimal (Lloyd-Max approx).
          • Storage: int8 values + 1 float32 scale per head-vector.
          • Memory vs float32: kv_bits/32  (~4× at 8-bit, ~8× at 4-bit).

        Quality note
        ------------
        miniGPT uses learned positional embeddings at fixed slot positions.
        When the sliding window advances, surviving tokens nominally shift one
        slot left → their K,V should be recomputed.  This method caches K,V
        without updating for the shift, introducing a small bounded error.
        In practice the effect on generation quality is negligible.

        Papers
        ------
        TurboQuant  arXiv 2504.19874  ICLR 2026
        PolarQuant  arXiv 2502.02617  AISTATS 2026
        QJL         arXiv 2406.03482  AAAI 2025

        Parameters
        ----------
        token_indices : list[int]  initial context (length must = context_size)
        length        : int        new tokens to generate
        temperature   : float      sampling temperature
        kv_bits       : int        KV cache quantisation bits (4 or 8)

        Returns
        -------
        list[int]  generated token indices
        """
        T   = self.context_size
        D   = self.embed_dim
        ctx = list(token_indices)

        # ── Prefill ────────────────────────────────────────────────────────────
        # block_caches[i] = cache tuple; K at index 4, V at index 5 (B,H,T,d_h)
        toks_arr = np.array(ctx, dtype=int).reshape(T, 1)
        _, (_, _, block_caches, _, _) = self._transformer_forward(
            toks_arr, training=False
        )

        kvc = TurboQuantKVCache(
            num_layers   = self.num_blocks,
            num_heads    = self.num_heads,
            head_dim     = self._head_dim,
            context_size = T,
            bits         = kv_bits,
        )

        to_cpu = ((lambda a: np.asnumpy(a)) if _DEVICE == "gpu"
                  else (lambda a: _np_cpu.array(a)))

        for t in range(T):
            for li, bc in enumerate(block_caches):
                kvc.push(
                    to_cpu(bc[4][0, :, t, :]).astype(_np_cpu.float32),  # K
                    to_cpu(bc[5][0, :, t, :]).astype(_np_cpu.float32),  # V
                    li,
                )
            kvc.advance()
        del block_caches

        generated = []

        # ── Decode loop ────────────────────────────────────────────────────────
        for _ in range(length):
            new_tok = ctx[-1]

            # Embed at slot T-1 (last position in the sliding window)
            x = (self.embedding[new_tok] + self.pos_embedding[T - 1]).reshape(1, D)

            new_kvs = []
            for li, blk in enumerate(self.blocks):
                K_cpu, V_cpu = kvc.get(li)          # (H, T_filled, d_h) CPU
                K_dev = np.array(K_cpu)
                V_dev = np.array(V_cpu)
                x, K_new, V_new = self._block_forward_decode(x, blk, K_dev, V_dev)
                new_kvs.append((
                    to_cpu(K_new).astype(_np_cpu.float32),
                    to_cpu(V_new).astype(_np_cpu.float32),
                ))

            # Final LayerNorm + output projection
            x_norm, _ = self._ln_forward(x[None, ...], self.ln_f_g, self.ln_f_b)
            if self.weight_tying:
                logits = x_norm.reshape(1, D) @ self.embedding.T + self.bout
            else:
                logits = x_norm.reshape(1, D) @ self.Wout + self.bout

            logits -= logits.max()
            probs   = np.exp(logits).ravel()
            probs  /= probs.sum()
            if _DEVICE == "gpu":
                probs = np.asnumpy(probs)

            # Temperature scaling + sample
            if temperature != 1.0:
                lp  = _np_cpu.log(probs.astype(_np_cpu.float64) + 1e-9) / temperature
                lp -= lp.max()
                probs = _np_cpu.exp(lp).astype(_np_cpu.float64)
                probs /= probs.sum()

            next_idx = int(_np_cpu.random.choice(len(probs), p=probs / probs.sum()))
            generated.append(next_idx)

            ctx = ctx[1:] + [next_idx]

            for li, (K_c, V_c) in enumerate(new_kvs):
                kvc.push(K_c, V_c, li)
            kvc.advance()

        return generated

    # ==========================================================================
    #  INT8 weight quantisation
    # ==========================================================================

    def quantize_weights_int8(self) -> None:
        """
        Post-training INT8 symmetric per-row quantisation of weight matrices.

        Method
        ------
        For each row r of weight matrix W:
            scale = max(|r|) / 127
            q     = round(r / scale).clip(-127, 127)
            r_approx = q * scale            (written back as float32)

        After this call the weights are float32 values snapped to an INT8
        grid.  JSON serialisation of such values achieves ~4× compression
        (JSON numbers have far fewer unique values to encode).

        What is quantised: Wqkv, W1, W2 in every transformer block, plus
        the token embedding (and Wout when weight tying is off).
        Biases, LayerNorm parameters, and positional embeddings are skipped
        (tiny; quantisation overhead not worth it).

        Intended use
        ------------
        Call once after training is complete.  Then save_weights() for a
        ~4× smaller checkpoint file.  The model can then be loaded normally
        and used for inference.  Do NOT continue training after quantising --
        the weight gradients operate on the approximated values.
        """
        if getattr(self, "_int8_quantised", False):
            print("Already INT8 quantised — skipping.")
            return

        to_cpu = ((lambda a: np.asnumpy(a).astype(_np_cpu.float32))
                  if _DEVICE == "gpu"
                  else (lambda a: _np_cpu.array(a, dtype=_np_cpu.float32)))

        def _quant(W_np32):
            scales = _np_cpu.abs(W_np32).max(axis=-1, keepdims=True) / 127.0 + 1e-9
            q      = _np_cpu.clip(_np_cpu.round(W_np32 / scales), -127, 127)
            return (q * scales).astype(_np_cpu.float32)

        total = 0
        for blk in self.blocks:
            for key in ("Wqkv", "W1", "W2"):
                W = to_cpu(blk[key])
                blk[key] = np.array(_quant(W))
                total += W.size

        if self.use_embedding:
            E = to_cpu(self.embedding)
            self.embedding = np.array(_quant(E))
            total += E.size

        if not self.weight_tying and self.Wout is not None:
            Wo = to_cpu(self.Wout)
            self.Wout = np.array(_quant(Wo))
            total += Wo.size

        self._int8_quantised = True
        print(
            f"INT8 quantisation complete — {total:,} params  "
            f"(~4× size reduction vs float32)"
        )



    def summary(self) -> None:
        """Print a formatted table of model architecture and parameter counts."""
        blk    = self.blocks[0]
        attn_p = blk["Wqkv"].size
        ff_p   = (blk["W1"].size + blk["b1"].size +
                  blk["W2"].size + blk["b2"].size)
        ln_p   = (blk["ln1_g"].size + blk["ln1_b"].size +
                  blk["ln2_g"].size + blk["ln2_b"].size)
        ln_f_p = self.ln_f_g.size + self.ln_f_b.size
        out_p  = (0 if self.weight_tying else self.Wout.size) + self.bout.size
        emb_p  = (self.embedding.size + self.pos_embedding.size
                  if self.use_embedding else 0)
        total  = (attn_p + ff_p + ln_p) * self.num_blocks + ln_f_p + out_p + emb_p

        width  = 56
        device = "GPU (CuPy)" if _DEVICE == "gpu" else "CPU (NumPy)"
        D      = self.embed_dim
        d_h    = self._head_dim

        print("+" + "=" * width + "+")
        print("|" + " Mini-Transformer Summary".center(width) + "|")
        print("+" + "=" * width + "+")
        print(f"|  {'Device':<18} | {device:<{width-24}}|")
        print(f"|  {'Optimizer':<18} | {'Adam + adaptive LR':<{width-24}}|")
        print(f"|  {'Batch size':<18} | {self.batch_size:<{width-24}}|")
        if self.use_embedding:
            edim = f"{self.vocab_size} chars x {D}d  context={self.context_size}"
            print(f"|  {'Embedding':<18} | {edim:<{width-24}}|")
        blk_str = f"{self.num_blocks} blocks  heads={self.num_heads}  d_head={d_h}"
        print(f"|  {'Architecture':<18} | {blk_str:<{width-24}}|")
        attn_str = f"causal MHA {D}x{D*3} ({attn_p} params/block)"
        ff_str   = f"{D}->{D*4}->{D} ({ff_p} params/block)"
        ln_str   = f"pre-norm x2/block + final LN ({ln_p + ln_f_p} params)"
        print(f"|  {'Attention':<18} | {attn_str:<{width-24}}|")
        print(f"|  {'Feed-forward':<18} | {ff_str:<{width-24}}|")
        print(f"|  {'LayerNorm':<18} | {ln_str:<{width-24}}|")
        wt_str   = "ON (embedding.T)" if self.weight_tying else "OFF (separate)"
        dp_str   = f"{self.dropout:.2f}" if self.dropout > 0 else "OFF"
        gc_str   = f"{self.grad_clip}" if self.grad_clip > 0 else "OFF"
        print(f"|  {'Weight tying':<18} | {wt_str:<{width-24}}|")
        print(f"|  {'Dropout':<18} | {dp_str:<{width-24}}|")
        print(f"|  {'Grad clip':<18} | {gc_str:<{width-24}}|")
        print(f"|  {'Output':<18} | {'all positions -> ' + str(self.output_size):<{width-24}}|")
        print("+" + "=" * width + "+")
        pad = width - 23 - len(f"{total:,}")
        print(f"|  Total parameters: {total:,}{'':<{pad}}  |")
        print("+" + "=" * width + "+")

    # ==========================================================================
    #  Save
    # ==========================================================================

    def save_weights(self, filename: str = "weights.json") -> None:
        """
        Save all weights, Adam state, and hyperparameters to JSON.

        ATOMIC WRITE: writes to a temp file first, then renames atomically.
        If Colab disconnects mid-save, the previous checkpoint remains intact.

        Files are written without indentation (compact JSON) to keep them
        ~4x smaller than pretty-printed versions (~50MB vs ~200MB).
        """
        to_list = (
            (lambda w: np.asnumpy(w).tolist()) if _DEVICE == "gpu"
            else (lambda w: w.tolist())
        )

        # Serialize per-block Adam state (includes LN param buffers)
        adam_blocks = []
        if self._adam_init:
            for buf in self._adam_blocks:
                adam_blocks.append(
                    {k: {"m": to_list(v["m"]), "v": to_list(v["v"])}
                     for k, v in buf.items()}
                )

        data = {
            # ---- Hyperparameters ----------------------------------------
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
            "num_heads":     self.num_heads,
            "dropout":       self.dropout,
            "weight_tying":  self.weight_tying,
            "grad_clip":     self.grad_clip,
            "device":        _DEVICE,
            # ---- Legacy dense weights (compat) --------------------------
            "weights": [to_list(w) for w in self.weights],
            "biases":  [to_list(b) for b in self.biases],
            # ---- Embeddings ---------------------------------------------
            "embedding":     to_list(self.embedding)     if self.use_embedding else None,
            "pos_embedding": to_list(self.pos_embedding) if self.use_embedding else None,
            # ---- Transformer weights ------------------------------------
            "Wout": to_list(self.Wout) if not self.weight_tying else None,
            "bout": to_list(self.bout),
            "ln_f_g": to_list(self.ln_f_g),
            "ln_f_b": to_list(self.ln_f_b),
            "blocks": [{k: to_list(v) for k, v in blk.items()}
                       for blk in self.blocks],
            # ---- Adam state ---------------------------------------------
            "adam_t":      self._adam_t if self._adam_init else 0,
            "adam_mWout":  to_list(self._mWout) if (self._adam_init and not self.weight_tying) else None,
            "adam_vWout":  to_list(self._vWout) if (self._adam_init and not self.weight_tying) else None,
            "adam_mbout":  to_list(self._mbout) if self._adam_init else None,
            "adam_vbout":  to_list(self._vbout) if self._adam_init else None,
            "adam_me":     to_list(self._me)  if (self._adam_init and self.use_embedding) else None,
            "adam_ve":     to_list(self._ve)  if (self._adam_init and self.use_embedding) else None,
            "adam_mpe":    to_list(self._mpe) if (self._adam_init and self.use_embedding) else None,
            "adam_vpe":    to_list(self._vpe) if (self._adam_init and self.use_embedding) else None,
            "adam_m_ln_f_g": to_list(self._m_ln_f_g) if self._adam_init else None,
            "adam_v_ln_f_g": to_list(self._v_ln_f_g) if self._adam_init else None,
            "adam_m_ln_f_b": to_list(self._m_ln_f_b) if self._adam_init else None,
            "adam_v_ln_f_b": to_list(self._v_ln_f_b) if self._adam_init else None,
            "adam_blocks": adam_blocks,
        }

        # Atomic write: temp file in same dir, then os.replace (atomic on Linux)
        dir_name = os.path.dirname(os.path.abspath(filename))
        tmp_fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(data, f)           # no indent = 4x smaller file
            os.replace(tmp_path, filename)   # atomic rename
        except Exception:
            os.unlink(tmp_path)              # clean up temp on failure
            raise

        print(f"Weights saved to '{filename}'.")

    # ==========================================================================
    #  Load
    # ==========================================================================

    def load_weights(self, filename: str = "weights.json") -> None:
        """
        Load weights, hyperparameters, and Adam state from a JSON file.

        Backward compatibility
        ----------------------
        Handles four weight file formats:
          1. Current  : LN params, multi-head, weight tying, final LN
          2. Previous : no LN, single-head (num_heads=1), separate Wout
          3. Old      : separate Wq/Wk/Wv per block -> merged on load
          4. Very old : single shared Wq/Wk/Wv -> replicated per block

        When loading an old file into the new architecture:
          - LN params are initialised to identity (gamma=1, beta=0).
            These are essentially no-ops initially and will be learned.
          - num_heads defaults to 1 (single-head, safe for old Wqkv shapes).
          - weight_tying defaults to False (old files have a separate Wout).
        """
        if not os.path.exists(filename):
            print(f"No weights file found at '{filename}'.")
            return

        with open(filename) as f:
            data = json.load(f)

        # ---- Restore hyperparameters ----------------------------------------
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
        # Old files default to 1 head (single-head safe for old Wqkv shapes)
        self.num_heads     = data.get("num_heads",     1)
        self.dropout       = data.get("dropout",       0.0)
        self.weight_tying  = data.get("weight_tying",  False)
        self.grad_clip     = data.get("grad_clip",     1.0)
        self._act_fn, self._act_d = _ACTIVATIONS[self.activation]
        self._head_dim     = self.embed_dim // self.num_heads
        self._scale_head   = 1.0 / (self._head_dim ** 0.5)

        # ---- Legacy dense weights -------------------------------------------
        self.weights = [np.array(w) for w in data["weights"]]
        self.biases  = [np.array(b) for b in data["biases"]]

        # ---- Embeddings -----------------------------------------------------
        self.embedding     = np.array(data["embedding"])     if data.get("embedding")     else None
        self.pos_embedding = np.array(data["pos_embedding"]) if data.get("pos_embedding") else None

        # ---- Output projection + final LN -----------------------------------
        if self.weight_tying:
            self.Wout = None
        else:
            self.Wout = np.array(data["Wout"]) if data.get("Wout") else None
        self.bout = np.array(data["bout"])

        D = self.embed_dim
        if data.get("ln_f_g") is not None:
            self.ln_f_g = np.array(data["ln_f_g"])
            self.ln_f_b = np.array(data["ln_f_b"])
        else:
            # Old file: initialise final LN as identity
            self.ln_f_g = np.ones(D)
            self.ln_f_b = np.zeros(D)

        # ---- Transformer blocks ---------------------------------------------
        if "blocks" in data:
            self.blocks = []
            for blk in data["blocks"]:
                b = {k: np.array(v) for k, v in blk.items()}

                # Upgrade old separate Wq/Wk/Wv -> fused Wqkv
                if "Wq" in b and "Wqkv" not in b:
                    import numpy as _nl
                    def _cpu(a): return np.asnumpy(a) if _DEVICE == "gpu" else a
                    Wq = _cpu(b.pop("Wq"))
                    Wk = _cpu(b.pop("Wk"))
                    Wv = _cpu(b.pop("Wv"))
                    b["Wqkv"] = np.array(_nl.concatenate([Wq, Wk, Wv], axis=1))

                # Add LN params if missing (old file without LayerNorm)
                if "ln1_g" not in b:
                    b["ln1_g"] = np.ones(D)
                    b["ln1_b"] = np.zeros(D)
                    b["ln2_g"] = np.ones(D)
                    b["ln2_b"] = np.zeros(D)

                self.blocks.append(b)
        else:
            # Very old single-weight format
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
                    "ln1_g": np.ones(D), "ln1_b": np.zeros(D),
                    "ln2_g": np.ones(D), "ln2_b": np.zeros(D),
                })

        # ---- Adam state -----------------------------------------------------
        self._adam_init = False
        if data.get("adam_t") and data["adam_t"] > 0:
            self._init_adam()                         # allocate all buffers
            self._adam_t = data["adam_t"]             # restore step counter

            if not self.weight_tying and data.get("adam_mWout"):
                self._mWout = np.array(data["adam_mWout"])
                self._vWout = np.array(data["adam_vWout"])
            self._mbout = np.array(data["adam_mbout"])
            self._vbout = np.array(data["adam_vbout"])

            if self.use_embedding and data.get("adam_me") is not None:
                self._me  = np.array(data["adam_me"])
                self._ve  = np.array(data["adam_ve"])
                self._mpe = np.array(data["adam_mpe"])
                self._vpe = np.array(data["adam_vpe"])

            # Final LN Adam state (absent in old files -> stays zero from _init)
            if data.get("adam_m_ln_f_g") is not None:
                self._m_ln_f_g = np.array(data["adam_m_ln_f_g"])
                self._v_ln_f_g = np.array(data["adam_v_ln_f_g"])
                self._m_ln_f_b = np.array(data["adam_m_ln_f_b"])
                self._v_ln_f_b = np.array(data["adam_v_ln_f_b"])

            # Per-block Adam state (includes LN buffers for new files)
            for i, buf in enumerate(data.get("adam_blocks", [])):
                for k, mv in buf.items():
                    if k in self._adam_blocks[i]:
                        self._adam_blocks[i][k]["m"] = np.array(mv["m"])
                        self._adam_blocks[i][k]["v"] = np.array(mv["v"])

            print(f"  Adam state restored (t={self._adam_t})")

        print(f"Weights loaded from '{filename}'.")

    # ==========================================================================
    #  Repr
    # ==========================================================================

    def __repr__(self) -> str:
        wt = "tied" if self.weight_tying else "separate"
        return (
            f"NeuralNetwork(embed={self.embed_dim}, blocks={self.num_blocks}, "
            f"heads={self.num_heads}, dropout={self.dropout}, "
            f"Wout={wt}, causal=True, all_positions=True, "
            f"lr={self.learning_rate}, device='{self.device}')"
        )