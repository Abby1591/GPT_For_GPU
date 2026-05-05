"""
Neural_Network.py  --  Character-level Mini-Transformer
========================================================
A from-scratch GPT-2-style transformer built on NumPy/CuPy (no PyTorch).
Designed to be readable and hackable -- every forward/backward pass is
written out explicitly, no autograd magic.

Architecture overview
---------------------
One transformer block:

    x
    |-- LayerNorm 1 --> Multi-Head Attention --> Dropout --> (+) --> x
    |-- LayerNorm 2 --> Feed-Forward MLP     --> Dropout --> (+) --> x

Full forward pass:

    tokens (B, T)
      -> token embedding + learned positional embedding    (B, T, D)
      -> embedding dropout
      -> N × transformer block (pre-norm, MHA + FF)        (B, T, D)
      -> final LayerNorm                                    (B, T, D)
      -> output projection at every position               (B, T, vocab)
      -> softmax -> next-token probabilities

Implemented techniques (with papers)
--------------------------------------
  Pre-norm (LayerNorm)
      Applied before attention AND feed-forward inside each block, plus a
      final norm before the output head.  Pre-norm trains more stably at
      depth than post-norm (GPT-1 style).
      Ref: "Language Models are Unsupervised Multitask Learners" (GPT-2),
           Radford et al. (2019). https://openai.com/research/gpt-2

  Multi-Head Attention
      Splits the D-dimensional embedding into H independent subspaces.
      Each head learns different token relationships; outputs are
      concatenated and projected back.
      Ref: "Attention Is All You Need", Vaswani et al. (2017).
           https://arxiv.org/abs/1706.03762

  Fused QKV projection  (Wqkv: D -> 3D)
      Q, K, V are computed in one matmul instead of three separate ones.
      Reduces memory traffic and is the natural layout for Flash Attention.
      Ref: FlashAttention -- Dao et al. (2022). https://arxiv.org/abs/2205.14135
           FlashAttention-2 -- Dao (2023).      https://arxiv.org/abs/2307.08691

  Dropout
      Randomly zeros a fraction of activations during training.
      Applied after attention, after the FF block, and after the embedding.
      Prevents overfitting on small corpora.

  Weight tying
      The output projection reuses the transposed token embedding matrix
      (Wout = embedding.T), halving those parameters.  Aligns the input
      and output representation spaces, which often improves perplexity.

  Residual scaling
      W2 (FF output) is initialised with scale 1/sqrt(2*num_blocks).
      With N residual additions each contributing variance, this keeps the
      total residual stream magnitude under control from step 1.
      Ref: GPT-2 paper (Radford et al. 2019).

  Gradient clipping
      The global L2 gradient norm is capped before the Adam update.
      Prevents a single bad batch from blowing up weights.
      Ref: "Why Gradient Clipping Accelerates Training" -- Zhang et al. (2020).
           https://arxiv.org/abs/1905.11881

  Muon optimizer  (weight matrices only: Wqkv, W1, W2)
      Replaces Adam for 2-D weight matrices.  Uses Nesterov momentum +
      Newton-Schulz orthogonalisation (5 iters) to produce an update whose
      columns are semi-unitary.  Reported ~2x faster than AdamW on some LM
      tasks.  Embeddings, biases, and LayerNorm params still use Adam.
      Ref: Kosson et al., "Muon" (2024). https://arxiv.org/abs/2409.20325

  Toolformer-style tool calling
      During generation the model can emit [TOOL:name|arg] tokens; the
      runtime detects these, calls the registered executor, and injects
      [RESULT:...] tokens back into the context before continuing.
      Training data is constructed by the self-supervised Toolformer method.
      Ref: Schick et al. (2023). https://arxiv.org/abs/2302.04761

  Quantised KV-cache (TurboQuantKVCache / PolarQuantKVCache)
      Stores the past K/V tensors at int8 or int4 precision to save memory
      during long-sequence generation.  Two backends:

      TurboQuantKVCache  -- applies a random Haar rotation before int8/int4
          scalar quantisation.  Rotation spreads channel-wise outliers so
          the quantisation grid fits the distribution better.
          Ref: "QuIP#: Even Better LLM Quantization" -- Tseng et al. (2024).
               https://arxiv.org/abs/2402.04396

      PolarQuantKVCache  -- identifies per-channel outliers dynamically;
          keeps them in float16 while compressing inliers to int8.
          No rotation overhead; effective when outlier channels are sparse.
          Ref: ResQ / QuaRot outlier-retention ideas.

Future work / papers worth adding
------------------------------------
  RoPE positional encoding  -- Su et al. (2021). https://arxiv.org/abs/2104.09864
  ALiBi (no position vectors, bias-based) -- Press et al. (2022).
       https://arxiv.org/abs/2108.12409
  Grouped-Query Attention (GQA) -- Ainslie et al. (2023).
       https://arxiv.org/abs/2305.13245  [saves KV memory at inference]
  Differential Transformer  -- Microsoft (2024). https://arxiv.org/abs/2410.05258
       [attention = softmax(Q1K^T) - softmax(Q2K^T); cancels attention noise]
  Mixture of Depths (MoD) -- Raposo et al. (2024). https://arxiv.org/abs/2404.02258
       [tokens dynamically skip layers; reduces FLOPs per forward pass]
  MegaByte -- Yu et al. (2023). https://arxiv.org/abs/2305.07185
       [local byte-level + global patch-level transformer; addresses sequence
        length explosion in char models -- high priority for this codebase]
  SpaceByte -- Slagle (2024). https://arxiv.org/abs/2404.14408
       [inserts global blocks at whitespace boundaries; simple char-model fix]

GPU setup
---------
    pip install cupy-cuda12x        # CUDA 12.x  (Colab T4)
    pip install cupy-cuda11x        # CUDA 11.x  (older GPUs)
"""

from __future__ import annotations

import json
import os
import tempfile
import numpy as _np_cpu
from typing import Dict, List, Literal, Optional, Tuple


# ==============================================================================
#  GPU / CPU backend
# ==============================================================================
# Import CuPy as `np` so all array code is device-agnostic.
# Falls back to NumPy silently if CuPy is not installed.
# Force CUDA path to fix a bug where it doesn't recognise the CUDA install.
# This path constant may need to be changed for non-Windows or Colab setups.

_CUDA_PATH = os.environ.get(
    "CUDA_PATH",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
)

if os.path.exists(_CUDA_PATH):
    os.environ["CUDA_PATH"] = _CUDA_PATH
    os.environ["PATH"] = _CUDA_PATH + r"\bin;" + os.environ["PATH"]

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


# ==============================================================================
#  Type aliases
# ==============================================================================

ActivationName = Literal["sigmoid", "tanh", "relu", "leaky_relu"]
Sample         = Tuple[List[float], int]


# ==============================================================================
#  Activation functions and their derivatives
# ==============================================================================
# Each entry is (forward_fn, derivative_fn).
# Derivatives receive the PRE-activation value z (not the output a).

def _relu(x):                  return np.maximum(0.0, x)
def _relu_d(x):                return (x > 0).astype(x.dtype)

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
#  Adam optimiser step  (module-level so it can be used without a class instance)
# ==============================================================================

def _adam_step(param, grad, m, v, lr_eff):
    """
    In-place Adam parameter update. Zero allocations.

    Adam update rule (Kingma & Ba, 2015. https://arxiv.org/abs/1412.6980):
        m  = beta1*m + (1-beta1)*grad          -- exponential moving avg of gradient
        v  = beta2*v + (1-beta2)*grad^2        -- exponential moving avg of squared grad
        theta -= lr_eff * m / (sqrt(v) + eps)  -- adaptive per-parameter step

    Bias correction is pre-baked into lr_eff by the caller so we avoid two
    extra scalars per update:
        lr_eff = lr * sqrt(1 - beta2^t) / (1 - beta1^t)

    beta1=0.9, beta2=0.999, eps=1e-8 are the original paper's recommended defaults.
    Muon replaces Adam for large 2-D weight matrices (see _muon_step below).
    """
    m *= 0.9;   m += 0.1   * grad          # momentum
    v *= 0.999; v += 0.001 * grad * grad   # velocity
    param -= lr_eff * m / (np.sqrt(v) + 1e-8)


def _newton_schulz5(G, steps=5):
    """
    Orthogonalise matrix G via 5 iterations of the Newton-Schulz iteration:

        G <- (3/2) * G  -  (1/2) * G @ G.T @ G

    This is a polynomial approximation to the matrix sign function.
    After enough iterations G converges to a semi-unitary matrix
    (all singular values equal to 1), which is what Muon wants as its
    "direction" for the update step.

    Numerical note: we normalise by the Frobenius norm first so the
    spectral norm is ~1 before iterating -- the iteration diverges if the
    spectral norm starts much above 1.

    Ref: Kosson et al., "Muon" (2024). https://arxiv.org/abs/2409.20325
    """
    # Normalise so spectral norm ~ 1 before iterating.
    norm = float(np.sqrt(np.sum(G * G))) + 1e-8
    G = G / norm
    for _ in range(steps):
        GtG = G.T @ G          # (n, n)
        G   = 1.5 * G - 0.5 * (G @ GtG)
    return G


def _muon_step(param, grad, momentum_buf, lr):
    """
    In-place Muon parameter update. Applied ONLY to 2-D weight matrices
    (Wqkv, W1, W2).  Embeddings, biases, and LayerNorm params use Adam.

    Why Muon instead of Adam for weight matrices?
    ----------------------------------------------
    Adam normalises each scalar gradient independently, which can lead to
    poorly conditioned updates for large matrices.  Muon instead finds the
    nearest semi-unitary matrix to the gradient direction (via Newton-Schulz
    orthogonalisation) and takes a step of fixed size lr in that direction.
    This is equivalent to Nesterov SGD in the "steepest descent" metric on
    the space of matrices, and empirically converges ~2x faster than AdamW
    on language modelling tasks.

    Algorithm
    ---------
    1. Nesterov momentum (no second-moment buffer needed):
           buf      = 0.95 * buf + grad
           G        = grad + 0.95 * buf          (lookahead gradient)
    2. Orthogonalise G via Newton-Schulz (5 iters) -> G_orth
    3. Rescale so G_orth has the same RMS as the raw gradient G,
       then update:  param -= lr * G_orth

    Ref: Kosson et al., "Muon: Momentum + Orthogonalisation" (2024).
         https://arxiv.org/abs/2409.20325
    """
    # Nesterov momentum (no second-moment tracking needed)
    momentum_buf *= 0.95
    momentum_buf += grad
    G = grad + 0.95 * momentum_buf          # (rows, cols)

    # Orthogonalise: make the update column-semi-unitary
    if G.ndim == 2:
        # Scale update so RMS matches original gradient RMS (Muon paper §3)
        rms_G    = float(np.sqrt(np.mean(G * G))) + 1e-8
        G_orth   = _newton_schulz5(G.copy())
        rms_orth = float(np.sqrt(np.mean(G_orth * G_orth))) + 1e-8
        G_orth  *= (rms_G / rms_orth)
    else:
        G_orth = G  # fallback for unexpected shapes

    param -= lr * G_orth


# ==============================================================================
#  KV-Cache base class
# ==============================================================================

class _KVCacheBase:
    """
    Abstract base for KV-caches used during autoregressive generation.

    Without a KV-cache each new token requires a full forward pass over the
    entire context.  With a cache we store the K and V projections for every
    past token, so each decode step only needs to project the NEW token through
    Q and then dot it against the cached K/V.  Cost drops from O(T²) per step
    to O(T) per step.

    Subclasses can override _encode_k / _decode_k / _encode_v / _decode_v to
    compress the stored tensors (e.g. to int8) while keeping the same API.

    Layout: one (K_list, V_list) pair per transformer layer.
    Each K_list[t] / V_list[t] is a (H, dh) array for one time-step.
    """

    def __init__(self, num_layers: int, num_heads: int, head_dim: int):
        self.num_layers = num_layers
        self.num_heads  = num_heads
        self.head_dim   = head_dim
        self._k: List[List] = [[] for _ in range(num_layers)]
        self._v: List[List] = [[] for _ in range(num_layers)]

    def reset(self) -> None:
        """Clear all stored K/V (call before each new generation)."""
        for i in range(self.num_layers):
            self._k[i].clear()
            self._v[i].clear()

    # ---- override in subclasses --------------------------------------------

    def _encode_k(self, k: _np_cpu.ndarray) -> object:
        """Compress a (H, dh) key slice for storage."""
        return k

    def _decode_k(self, stored: object) -> _np_cpu.ndarray:
        """Reconstruct a (H, dh) key slice from storage."""
        return stored

    def _encode_v(self, v: _np_cpu.ndarray) -> object:
        return v

    def _decode_v(self, stored: object) -> _np_cpu.ndarray:
        return stored

    # ---- public API --------------------------------------------------------

    def append(
        self,
        layer: int,
        k:     _np_cpu.ndarray,   # (H, dh)
        v:     _np_cpu.ndarray,   # (H, dh)
    ) -> None:
        self._k[layer].append(self._encode_k(k))
        self._v[layer].append(self._encode_v(v))

    def get(
        self, layer: int
    ) -> Tuple[_np_cpu.ndarray, _np_cpu.ndarray]:
        """Return (K_all, V_all) each (H, T, dh) in float32."""
        K = _np_cpu.stack([self._decode_k(s) for s in self._k[layer]], axis=1)
        V = _np_cpu.stack([self._decode_v(s) for s in self._v[layer]], axis=1)
        return K, V

    def compression_ratio(self) -> float:
        """Ratio of bytes used vs the equivalent fp32 storage."""
        raise NotImplementedError

    def __repr__(self) -> str:
        total = sum(len(self._k[i]) for i in range(self.num_layers))
        return (
            f"{self.__class__.__name__}("
            f"layers={self.num_layers}, heads={self.num_heads}, "
            f"head_dim={self.head_dim}, stored_steps={total})"
        )


# ==============================================================================
#  TurboQuantKVCache  --  random-rotation + scalar quantisation
# ==============================================================================

class TurboQuantKVCache(_KVCacheBase):
    """
    Quantised KV-cache using random Hadamard/Haar rotation before scalar
    quantisation.  Inspired by the QuIP# "TurboQuant-MSE" algorithm.

    Why rotate before quantising?
    ------------------------------
    Transformer K/V tensors often have a few channels with very large
    magnitudes ("outliers") that force a wide quantisation grid, wasting
    precision on the small-magnitude channels.  Multiplying by a random
    orthogonal matrix R spreads outlier energy uniformly across ALL channels
    so the resulting distribution is much easier to quantise at low bit-width
    without significant accuracy loss.

    Algorithm per time-step (encode)
    ---------------------------------
    1. Random orthogonal rotation R (Haar-distributed, fixed at construction):
           k_rot = k @ R.T          shape (H, dh)
    2. Scalar quantisation:
       int8  -- per-head min/max linear quantisation:
                    scale = (max - min) / 255
                    zero  = round(-min / scale)
                    q     = clip(round(k_rot / scale) + zero, 0, 255)  uint8
       int4  -- same scaling, values nibble-packed two-per-byte:
                    q4    = clip(round(k_rot / scale) + zero, 0, 15)   uint8
                    packed[i] = q4[2i] | (q4[2i+1] << 4)

    Decode reverses:  unpack -> dequantise -> rotate back (k_rot @ R).
    Values are NOT rotated (their distribution is already mild per KIVI).

    Compression ratios:
        int8  ~  3-4x vs float32
        int4  ~  6-8x vs float32

    Ref: "QuIP#: Even Better LLM Quantization with Hadamard Incoherence
          and Lattice Codebooks" -- Tseng et al. (2024).
          https://arxiv.org/abs/2402.04396
         "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"
          -- Liu et al. (2024). https://arxiv.org/abs/2402.02750

    Parameters
    ----------
    num_layers  : number of transformer blocks
    num_heads   : H (number of attention heads)
    head_dim    : dh = D // H  (must be even for int4 packing)
    bits        : 8 (int8, default) or 4 (nibble-packed int4)
    seed        : RNG seed for the per-head rotation matrices
    """

    def __init__(
        self,
        num_layers: int,
        num_heads:  int,
        head_dim:   int,
        bits:       int = 8,
        seed:       int = 42,
    ):
        super().__init__(num_layers, num_heads, head_dim)
        if bits not in (4, 8):
            raise ValueError("bits must be 4 or 8")
        if bits == 4 and head_dim % 2 != 0:
            raise ValueError("head_dim must be even for 4-bit nibble packing")
        self.bits = bits

        # One rotation matrix per head, fixed for the lifetime of the cache.
        # Haar-distributed: draw a random matrix, QR-decompose, take Q.
        rng = _np_cpu.random.default_rng(seed)
        self._R: List[_np_cpu.ndarray] = []   # R[h] shape (dh, dh)
        for _ in range(num_heads):
            A = rng.standard_normal((head_dim, head_dim)).astype(_np_cpu.float32)
            Q, _ = _np_cpu.linalg.qr(A)
            self._R.append(Q)

    # ---- int8 quantisation helpers -----------------------------------------

    def _quant8(
        self, x: _np_cpu.ndarray
    ) -> Tuple[_np_cpu.ndarray, float, int]:
        """Quantise (dh,) float32 to uint8. Returns (q, scale, zero)."""
        mn, mx = float(x.min()), float(x.max())
        if mx == mn:
            return _np_cpu.zeros(x.shape, dtype=_np_cpu.uint8), 1.0, 0
        scale = (mx - mn) / 255.0
        zero  = int(round(-mn / scale))
        q     = _np_cpu.clip(
            _np_cpu.round(x / scale).astype(_np_cpu.int32) + zero, 0, 255
        ).astype(_np_cpu.uint8)
        return q, scale, zero

    def _dequant8(
        self, q: _np_cpu.ndarray, scale: float, zero: int
    ) -> _np_cpu.ndarray:
        return (q.astype(_np_cpu.float32) - zero) * scale

    # ---- int4 (nibble-packed) quantisation helpers -------------------------

    def _quant4(
        self, x: _np_cpu.ndarray
    ) -> Tuple[_np_cpu.ndarray, float, int]:
        """Quantise (dh,) float32 to nibble-packed uint8. Returns (packed, scale, zero)."""
        mn, mx = float(x.min()), float(x.max())
        if mx == mn:
            return _np_cpu.zeros(len(x) // 2, dtype=_np_cpu.uint8), 1.0, 0
        scale  = (mx - mn) / 15.0
        zero   = int(round(-mn / scale))
        q4     = _np_cpu.clip(
            _np_cpu.round(x / scale).astype(_np_cpu.int32) + zero, 0, 15
        ).astype(_np_cpu.uint8)
        # Pack two nibbles per byte: low nibble = even index, high = odd index
        packed = (q4[0::2] & 0x0F) | ((q4[1::2] & 0x0F) << 4)
        return packed, scale, zero

    def _dequant4(
        self, packed: _np_cpu.ndarray, scale: float, zero: int, dh: int
    ) -> _np_cpu.ndarray:
        q4 = _np_cpu.empty(dh, dtype=_np_cpu.uint8)
        q4[0::2] = packed & 0x0F
        q4[1::2] = (packed >> 4) & 0x0F
        return (q4.astype(_np_cpu.float32) - zero) * scale

    # ---- encode / decode ---------------------------------------------------

    def _encode_k(self, k: _np_cpu.ndarray):
        """k: (H, dh) -> list of (packed, scale, zero) per head"""
        out = []
        for h in range(self.num_heads):
            k_rot = k[h] @ self._R[h].T          # rotate: (dh,)
            if self.bits == 8:
                out.append(self._quant8(k_rot))
            else:
                out.append(self._quant4(k_rot))
        return out

    def _decode_k(self, stored) -> _np_cpu.ndarray:
        """stored: list of (packed, scale, zero) per head -> (H, dh)"""
        rows = []
        dh   = self.head_dim
        for h, (packed, scale, zero) in enumerate(stored):
            if self.bits == 8:
                k_rot = self._dequant8(packed, scale, zero)
            else:
                k_rot = self._dequant4(packed, scale, zero, dh)
            rows.append(k_rot @ self._R[h])       # rotate back: (dh,)
        return _np_cpu.stack(rows, axis=0)         # (H, dh)

    def _encode_v(self, v: _np_cpu.ndarray):
        """v: (H, dh) -> list of (packed, scale, zero) per head (no rotation)"""
        out = []
        for h in range(self.num_heads):
            # Values are NOT rotated -- their distribution is already mild.
            # Per KIVI's analysis, only keys need the rotation treatment.
            if self.bits == 8:
                out.append(self._quant8(v[h]))
            else:
                out.append(self._quant4(v[h]))
        return out

    def _decode_v(self, stored) -> _np_cpu.ndarray:
        rows = []
        dh   = self.head_dim
        for h, (packed, scale, zero) in enumerate(stored):
            if self.bits == 8:
                rows.append(self._dequant8(packed, scale, zero))
            else:
                rows.append(self._dequant4(packed, scale, zero, dh))
        return _np_cpu.stack(rows, axis=0)   # (H, dh)

    def compression_ratio(self) -> float:
        """
        Actual bytes stored vs equivalent fp32 storage.
        fp32 baseline: H * dh * 4 bytes per (K or V) step.
        Quantised storage: H * (packed_bytes + 5) per step
          where packed_bytes = dh for int8, dh//2 for int4
          and   5 = 4 bytes (float32 scale) + 1 byte (uint8 zero).
        """
        dh         = self.head_dim
        H          = self.num_heads
        fp32_kv    = H * dh * 4 * 2          # K + V, float32
        pack_bytes = dh if self.bits == 8 else dh // 2
        quant_kv   = H * (pack_bytes + 5) * 2  # K + V, quantised
        return fp32_kv / max(quant_kv, 1)

    def __repr__(self) -> str:
        base = super().__repr__()
        return base + f"  bits={self.bits}  ratio={self.compression_ratio():.2f}x"


# ==============================================================================
#  PolarQuantKVCache  --  outlier-aware mixed-precision KV cache
# ==============================================================================

class PolarQuantKVCache(_KVCacheBase):
    """
    Mixed-precision KV-cache with outlier-aware channel splitting.

    Motivation
    ----------
    Rotation-based methods (TurboQuantKVCache) pay a matmul overhead per
    token.  An alternative is to identify which channels are outliers
    dynamically and keep only those in higher precision, compressing
    everything else cheaply.

    Algorithm
    ---------
    Keys:
      1. Compute per-channel L∞ magnitude for the current (H, dh) key slice.
      2. Flag channels where |k_c| > outlier_factor × median(|k|) as outliers.
      3. Store outlier channels in float16 (sparse; typically <5% of channels).
      4. Zero out outlier positions and quantise the rest to int8 with
         per-head min/max scaling.
      Decode: dequantise int8 inliers, splice float16 outliers back in.

    Values: always int8, per-head min/max scaling (KIVI-style).
    Values rarely have severe outliers so the simpler path suffices.

    Trade-offs vs TurboQuantKVCache
    --------------------------------
      + No rotation matmul -> faster encode/decode
      + Outlier retention is exact (no approximation)
      - Compression ratio slightly lower when many outlier channels exist
      - Dynamic outlier threshold can vary across tokens (less predictable)

    Ref: QuaRot / ResQ outlier-retention ideas.
         "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs"
          -- Ashkboos et al. (2024). https://arxiv.org/abs/2404.00456

    Parameters
    ----------
    num_layers      : transformer blocks
    num_heads       : H
    head_dim        : dh
    outlier_factor  : threshold multiplier on the per-step median.
                      Higher -> fewer channels classified as outliers
                      (more compression, slightly lower accuracy).
    """

    def __init__(
        self,
        num_layers:     int,
        num_heads:      int,
        head_dim:       int,
        outlier_factor: float = 4.0,
    ):
        super().__init__(num_layers, num_heads, head_dim)
        self.outlier_factor = outlier_factor

    # ---- key encode / decode (outlier-aware per-channel) -------------------

    def _encode_k(self, k: _np_cpu.ndarray):
        """k: (H, dh) -> per-head tuple of (q_inlier, scale, zero, outlier_idx, outlier_vals)"""
        out = []
        for h in range(self.num_heads):
            row   = k[h]                          # (dh,)
            abs_r = _np_cpu.abs(row)
            med   = float(_np_cpu.median(abs_r)) + 1e-9
            thr   = self.outlier_factor * med

            out_idx  = _np_cpu.where(abs_r > thr)[0].astype(_np_cpu.int16)
            out_vals = row[out_idx].astype(_np_cpu.float16)

            # Zero out outliers before quantising the inlier portion
            row_in = row.copy()
            row_in[out_idx] = 0.0

            mn, mx = float(row_in.min()), float(row_in.max())
            if mx == mn:
                scale, zero = 1.0, 0
                q = _np_cpu.zeros(self.head_dim, dtype=_np_cpu.uint8)
            else:
                scale = (mx - mn) / 255.0
                zero  = int(round(-mn / scale))
                q = _np_cpu.clip(
                    _np_cpu.round(row_in / scale).astype(_np_cpu.int32) + zero,
                    0, 255
                ).astype(_np_cpu.uint8)

            out.append((q, scale, zero, out_idx, out_vals))
        return out

    def _decode_k(self, stored) -> _np_cpu.ndarray:
        rows = []
        for q, scale, zero, out_idx, out_vals in stored:
            row = (q.astype(_np_cpu.float32) - zero) * scale
            if len(out_idx):
                row[out_idx] = out_vals.astype(_np_cpu.float32)
            rows.append(row)
        return _np_cpu.stack(rows, axis=0)   # (H, dh)

    # ---- value encode / decode (per-head int8, KIVI-style per-token) -------

    def _encode_v(self, v: _np_cpu.ndarray):
        out = []
        for h in range(self.num_heads):
            row = v[h]
            mn, mx = float(row.min()), float(row.max())
            if mx == mn:
                out.append((_np_cpu.zeros(self.head_dim, dtype=_np_cpu.uint8), 1.0, 0))
            else:
                scale = (mx - mn) / 255.0
                zero  = int(round(-mn / scale))
                q = _np_cpu.clip(
                    _np_cpu.round(row / scale).astype(_np_cpu.int32) + zero,
                    0, 255
                ).astype(_np_cpu.uint8)
                out.append((q, scale, zero))
        return out

    def _decode_v(self, stored) -> _np_cpu.ndarray:
        rows = []
        for q, scale, zero in stored:
            rows.append((q.astype(_np_cpu.float32) - zero) * scale)
        return _np_cpu.stack(rows, axis=0)   # (H, dh)

    def compression_ratio(self) -> float:
        """
        Approximation: assumes ~5% of channels are outliers (kept fp16).
        fp32 baseline: H * dh * 4 * 2  (K + V).
        Compressed:    K = H*(dh*1 + dh*0.05*2 + 5),  V = H*(dh*1 + 5)
        """
        H         = self.num_heads
        dh        = self.head_dim
        fp32_kv   = H * dh * 4 * 2
        k_bytes   = H * (dh + int(dh * 0.05) * 2 + 5)
        v_bytes   = H * (dh + 5)
        return fp32_kv / max(k_bytes + v_bytes, 1)

    def __repr__(self) -> str:
        base = super().__repr__()
        return (base +
                f"  outlier_factor={self.outlier_factor}"
                f"  ratio≈{self.compression_ratio():.2f}x")


# ==============================================================================
#  Tool-use infrastructure  (Toolformer-style, character-level)
# ==============================================================================
#
# Format: rigid ASCII delimiters that a character-level model can reliably learn.
#
#   Model generates:   [TOOL:name|argument text]
#   Runtime injects:   [RESULT:result text]
#   Generation continues from there.
#
# How it works at inference time
# --------------------------------
# generate() runs the model autoregressively and keeps a sliding text buffer.
# When a complete [TOOL:name|arg] pattern is detected in the buffer the
# registered executor is called, its result is formatted as [RESULT:...], and
# those characters are fed back into the context window before sampling resumes.
# This is the same "API call injection" idea from the Toolformer paper, but
# adapted for a character-level model with no tokenizer.
#
# Ref: "Toolformer: Language Models Can Teach Themselves to Use Tools"
#       Schick et al. (2023). https://arxiv.org/abs/2302.04761
#
# REGISTERING A TOOL
# ------------------
#   def my_search(query: str) -> str:
#       return "Paris is the capital of France."
#
#   nn.register_tool("search", my_search)
#
# The handler receives the raw argument string and must return a plain string.
# Results are truncated to `max_result_chars` (default 256) to avoid
# exhausting the context window.
#
# TRAINING DATA CONSTRUCTION (Toolformer step 2)
# -----------------------------------------------
# make_tool_training_pairs() inserts sampled tool calls into a plain-text
# corpus.  In the full Toolformer method you would then run a forward pass
# and keep only insertions that reduce next-token loss.  The helper here
# omits the loss filter for simplicity -- add it by passing your
# NeuralNetwork instance and checking cross-entropy before/after insertion.

import re as _re

_TOOL_OPEN_RE    = _re.compile(r"\[TOOL:([^\|]+)\|([^\]]*)\]")
_TOOL_RESULT_FMT = "[RESULT:{result}]"
_TOOL_MAX_RESULT = 256    # characters; tune down if context window is tight

# Every character that appears in tool delimiters.
# The model CANNOT generate or parse tool calls if these are absent from vocab.
# Pass your char2idx through ensure_tool_vocab() before building the model.
TOOL_CHARS: frozenset = frozenset("[]:|")


def ensure_tool_vocab(char2idx: Dict[str, int], silent: bool = False) -> Dict[str, int]:
    """
    Guarantee all tool delimiter characters are in the vocabulary.

    Call this after building char2idx from your corpus, before constructing
    the NeuralNetwork.  Missing chars are appended so existing indices stay
    stable.

    :param silent: Suppress the print when chars are added. Pass ``True``
        during training -- model.py calls this automatically when tool
        patterns are detected in the corpus.

    Example
    -------
        chars    = sorted(set(corpus_text))
        char2idx = {c: i for i, c in enumerate(chars)}
        char2idx = ensure_tool_vocab(char2idx)   # adds [ ] : | if absent
        nn = NeuralNetwork(vocab_size=len(char2idx), ...)
    """
    missing = TOOL_CHARS - set(char2idx)
    if missing:
        next_idx = max(char2idx.values()) + 1
        for ch in sorted(missing):
            char2idx[ch] = next_idx
            next_idx += 1
        if not silent:
            print(f"[tool vocab] Added {len(missing)} missing chars: "
                  f"{sorted(missing)}  (new size: {len(char2idx)})")
    return char2idx


def _encode_tool_result(result: str, char2idx: Dict[str, int]) -> List[int]:
    """
    Encode a [RESULT:...] string to token indices, substituting '?' for
    unknown characters.  Returns an empty list if char2idx is not supplied.
    """
    text = _TOOL_RESULT_FMT.format(result=result[:_TOOL_MAX_RESULT])
    unk  = char2idx.get("?", 0)
    return [char2idx.get(c, unk) for c in text]


def make_tool_training_pairs(
    raw_text:      str,
    char2idx:      Dict[str, int],
    idx2char:      Dict[int, str],
    tool_handlers: Dict[str, "callable"],
    sample_positions: Optional[List[int]] = None,
    window: int = 32,
) -> List[str]:
    """
    Toolformer-style self-supervised data construction.

    For each candidate position in `sample_positions`, attempt to insert a
    tool call into `raw_text` and measure whether it reduces next-token
    cross-entropy over the following `window` characters.  Returns a list of
    augmented text strings (with [TOOL:...][RESULT:...] inserted) for the
    insertions that were beneficial.

    This is a *data-construction helper*, not a training loop.  Call it
    offline to build an augmented corpus, then train the model normally.

    Parameters
    ----------
    raw_text         : the original training corpus as a plain string.
    char2idx/idx2char: vocab mappings produced by your preprocessing code.
    tool_handlers    : {name: callable} -- same dict you'd pass to register_tool.
    sample_positions : character positions to probe (default: every 50 chars).
    window           : characters after the insertion point used to measure loss.
    """
    if sample_positions is None:
        sample_positions = list(range(0, len(raw_text) - window, 50))

    augmented = []
    for pos in sample_positions:
        for name, handler in tool_handlers.items():
            # Extract a plausible query: the previous 20 characters of context.
            query = raw_text[max(0, pos - 20): pos].strip()
            if not query:
                continue
            try:
                result = handler(query)
            except Exception:
                continue
            call_str   = f"[TOOL:{name}|{query}]"
            result_str = _TOOL_RESULT_FMT.format(result=str(result)[:_TOOL_MAX_RESULT])
            inserted   = raw_text[:pos] + call_str + result_str + raw_text[pos:]
            # Accept the insertion unconditionally here (loss check requires a
            # full forward pass -- plug in your NeuralNetwork instance if desired).
            augmented.append(inserted)

    return augmented




class NeuralNetwork:
    """
    Character-level GPT-2-inspired transformer.

    This is the core compute engine.  MiniGPT (model.py) wraps this class
    and handles tokenisation, data loading, save/load, and the generation
    loop with tool-calling.

    Optimizer split
    ---------------
    Weight matrices (Wqkv, W1, W2): updated with Muon (Nesterov momentum +
        Newton-Schulz orthogonalisation).
    Everything else (embeddings, biases, LayerNorm): updated with Adam.

    Parameters
    ----------
    input_size : int
        Legacy flat input size.  Ignored in the transformer path; kept only
        for backward-compatibility with old save files.
    hidden_layers : list[int]
        Legacy dense widths.  Also kept for save/load compatibility only.
    output_size : int
        Vocabulary size -- number of softmax output classes.
    activation : str
        Legacy activation name (relu/tanh/sigmoid/leaky_relu).
    learning_rate : float
        Peak Adam/Muon learning rate.
    batch_size : int
        Samples per gradient step.
    use_embedding : bool
        Use learned token + positional embeddings (always True in practice).
    vocab_size : int
        Number of unique characters in the vocabulary.
    context_size : int
        Sequence length T fed to the transformer.
    embed_dim : int
        Hidden dimension D.  Must be divisible by num_heads.
    num_blocks : int
        Number of transformer blocks stacked in series.
    num_heads : int
        Number of attention heads H.  Each head works in a D/H subspace.
    dropout : float
        Fraction of activations zeroed during training (0 = disabled).
    weight_tying : bool
        If True, Wout = embedding.T  (halves output-projection parameters).
    grad_clip : float
        Max global gradient L2 norm before the update step (0 = disabled).
    """

    # ==========================================================================
    #  Construction
    # ==========================================================================

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
        self.device        = _DEVICE
        self._act_fn, self._act_d = _ACTIVATIONS[activation]

        # Head dimension: each head attends in a d_h-dimensional subspace.
        # Scale for attention scores: 1/sqrt(d_h) keeps dot products from
        # growing too large as d_h increases (prevents softmax saturation).
        self._head_dim   = embed_dim // num_heads
        self._scale_head = 1.0 / (self._head_dim ** 0.5)

        # ---- Token + positional embeddings ----------------------------------
        # Token embedding:      (vocab_size, D)  -- one row per character.
        # Positional embedding: (context_size, D) -- one row per position.
        # Both are learned and updated by Adam just like any weight matrix.
        if self.use_embedding:
            self.embedding     = np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.01
            self.pos_embedding = np.random.randn(context_size, embed_dim).astype(np.float32) * 0.01
        else:
            self.embedding     = None
            self.pos_embedding = None

        # ---- Transformer blocks ---------------------------------------------
        # Each block is independent -- no weight sharing across blocks.
        # This lets each block specialise on different linguistic patterns.
        #
        # Per-block parameters:
        #   Wqkv     -- fused Q, K, V projection  (D, 3D)
        #               Fusing into one matrix reduces memory traffic and is
        #               the natural layout assumed by Flash Attention.
        #               Ref: FlashAttention -- Dao et al. (2022).
        #                    https://arxiv.org/abs/2205.14135
        #   W1, b1   -- FF expansion layer         (D, 4D) + (4D,)
        #   W2, b2   -- FF contraction layer        (4D, D) + (D,)
        #               4D hidden width follows the GPT-2 convention.
        #   ln1_g/b  -- pre-attention LayerNorm params  (D,) each
        #   ln2_g/b  -- pre-FF LayerNorm params          (D,) each
        #
        # W2 is scaled by 1/sqrt(2*num_blocks) on init (GPT-2 residual
        # scaling). With N residual additions each adding variance ~0.02²,
        # this keeps the residual stream magnitude stable from step 1.
        D           = embed_dim
        resid_scale = np.float32(0.02 / (2 * num_blocks) ** 0.5)
        self.blocks: List[Dict] = []
        for _ in range(num_blocks):
            self.blocks.append({
                "Wqkv":  np.random.randn(D, D * 3).astype(np.float32) * np.float32(0.02),
                "W1":    np.random.randn(D, D * 4).astype(np.float32) * np.float32(0.02),
                "b1":    np.zeros(D * 4, dtype=np.float32),
                "W2":    np.random.randn(D * 4, D).astype(np.float32) * resid_scale,
                "b2":    np.zeros(D, dtype=np.float32),
                "ln1_g": np.ones(D, dtype=np.float32),
                "ln1_b": np.zeros(D, dtype=np.float32),
                "ln2_g": np.ones(D, dtype=np.float32),
                "ln2_b": np.zeros(D, dtype=np.float32),
            })

        # ---- Final LayerNorm (GPT-2 style) ----------------------------------
        # Applied once after all transformer blocks, before output projection.
        # Ensures the final representations are well-scaled before the linear
        # output layer reads them.
        self.ln_f_g = np.ones(D, dtype=np.float32)
        self.ln_f_b = np.zeros(D, dtype=np.float32)

        # ---- Output projection ----------------------------------------------
        if weight_tying:
            self.Wout = None
        else:
            self.Wout = np.random.randn(D, output_size).astype(np.float32) * np.float32(0.02)
        self.bout = np.zeros(output_size, dtype=np.float32)

        # ---- Legacy dense weights (backward compatibility only) -------------
        # Old weight files contain these arrays. They are never trained or
        # used in the transformer path -- only saved/loaded for compatibility.
        actual_input = context_size * embed_dim if self.use_embedding else input_size
        layer_sizes  = [actual_input] + hidden_layers + [output_size]
        self.weights = []
        self.biases  = []
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            scale = np.sqrt(np.float32(2.0) / np.float32(fan_in + fan_out))
            self.weights.append(np.random.randn(fan_out, fan_in).astype(np.float32) * scale)
            self.biases.append(np.zeros((fan_out, 1), dtype=np.float32))

        # Adam not initialised until first train() or load_weights() call.
        self._adam_init = False

    # ==========================================================================
    #  Adam optimiser  (initialisation + state)
    # ==========================================================================

    def _init_adam(self) -> None:
        """
        Allocate zeroed first-moment (m) and second-moment (v) buffers for
        every learnable parameter.

        Saving and restoring the Adam state on resume means the optimizer
        does NOT forget the momentum it built up -- a clean resume with no
        warmup needed.
        """
        z = np.zeros_like

        # Per-block buffers (includes LN params since they live in blk dict)
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

        # ---- Muon momentum buffers (Wqkv, W1, W2 always use Muon) ----------
        # One momentum buffer per matrix per block; no v buffer needed.
        self._muon_bufs = []
        for blk in self.blocks:
            self._muon_bufs.append({
                "Wqkv": np.zeros_like(blk["Wqkv"]),
                "W1":   np.zeros_like(blk["W1"]),
                "W2":   np.zeros_like(blk["W2"]),
            })

    # ==========================================================================
    #  LayerNorm  (forward + backward kept together)
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
        then scale up the survivors by 1/(1-rate) so the expected sum
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
        mask = (np.random.rand(*x.shape) > self.dropout).astype(x.dtype)
        mask /= x.dtype.type(1.0 - self.dropout)   # inverted scaling
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

        Cached: only rebuilt when T changes (never in practice).
        """
        if not hasattr(self, "_mask_cache") or self._mask_cache.shape[0] != T:
            self._mask_cache = (np.triu(np.ones((T, T)), k=1) * -1e9).astype(np.float32)
        return self._mask_cache

    # ==========================================================================
    #  Transformer block: forward
    # ==========================================================================

    def _block_forward(self, x, blk, training: bool = False, mask=None):
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
        Head 0 might learn word boundaries, head 1 repeated characters, etc.
        With a single head, all of this must be crammed into one pattern.

        The scale is now 1/sqrt(d_h), NOT 1/sqrt(D). This is important:
        with D=256, H=4, d_h=64, scale = 1/8 instead of 1/16.

        What is saved in cache
        ----------------------
        Everything needed by backward. x_attn is NOT saved -- recomputed
        cheaply from x + attn_out to save VRAM.
        """
        B, T, D = x.shape
        H       = self.num_heads
        d_h     = D // H                  # dimension per head
        BT      = B * T

        # ---- Pre-norm 1: normalise before attention -------------------------
        ln1_out, ln1_cache = self._ln_forward(x, blk["ln1_g"], blk["ln1_b"])

        # ---- Fused multi-head QKV projection --------------------------------
        # One (D, 3D) matmul produces all of Q, K, V for all heads at once.
        # Reshape to (B, T, 3, H, d_h) then move heads axis forward.
        QKV = (ln1_out.reshape(BT, D) @ blk["Wqkv"]).reshape(B, T, 3, H, d_h)
        Q = QKV[:, :, 0].transpose((0, 2, 1, 3)).copy()    # (B, H, T, d_h)
        K = QKV[:, :, 1].transpose((0, 2, 1, 3)).copy()
        V = QKV[:, :, 2].transpose((0, 2, 1, 3)).copy()
        del QKV                                              # free full buffer

        # ---- Scaled dot-product attention (per head) ------------------------
        # scores[b, h, i, j] = how much position i in head h attends to j
        scores  = Q @ K.transpose((0, 1, 3, 2)) * self._scale_head  # (B, H, T, T)
        if mask is None:
            mask = self._causal_mask(T)
        scores += mask                                               # block future
        scores -= scores.max(axis=-1, keepdims=True)                # stable softmax
        exp_s   = np.exp(scores)
        A       = exp_s / exp_s.sum(axis=-1, keepdims=True)         # (B, H, T, T)
        del scores, exp_s                                            # free VRAM now

        # Weighted sum of values, then merge heads back to (B, T, D)
        attn_h   = A @ V                                             # (B, H, T, d_h)
        attn_out = attn_h.transpose((0, 2, 1, 3)).reshape(B, T, D)  # (B, T, D)

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
        d_ff_out = d_out * drop2_mask if drop2_mask is not None else d_out

        # FF backward
        dW2          = h.reshape(BT, -1).T @ d_ff_out.reshape(BT, -1)    # (4D, D)
        db2          = d_ff_out.sum(axis=(0, 1))                           # (D,)
        d_h_grad     = d_ff_out @ blk["W2"].T                             # (B, T, 4D)
        d_h_grad    *= (h > 0)                                             # ReLU deriv
        dW1          = ln2_out.reshape(BT, -1).T @ d_h_grad.reshape(BT, -1)  # (D, 4D)
        db1          = d_h_grad.sum(axis=(0, 1))                           # (4D,)
        d_ln2_out    = d_h_grad @ blk["W1"].T                             # (B, T, D)

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
        dK = dS.transpose((0, 1, 3, 2)) @ Q                    # (B, H, T, d_h)

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
    #  Full transformer forward pass
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
        x, emb_drop_mask = self._apply_dropout(x, training)

        # Hoist causal mask: T never changes mid-run, so compute once and
        # pass it into every block rather than recomputing per block.
        mask = self._causal_mask(toks.shape[1])   # (T, T)

        # Transformer blocks
        block_caches = []
        for blk in self.blocks:
            x, cache = self._block_forward(x, blk, training=training, mask=mask)
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
        Single-sample forward pass used by generate() and predict().
        training=False so dropout is disabled.
        Returns (zs, activations, probs) where probs is last-position only.

        Fast path: if inputs is a plain list/array of integer token
        indices (length == context_size), skip one-hot encode/decode entirely
        and pass directly to _transformer_forward.  This is ~vocab_size x
        faster per generate step.

        Legacy path: flat float one-hot vector (context_size * vocab_size
        elements) -- kept for backward compatibility with external callers.
        """
        if self.use_embedding:
            # Fast path: caller already has integer indices
            if (len(inputs) == self.context_size
                    and isinstance(inputs[0], (int, _np_cpu.integer))):
                toks = np.array(inputs, dtype=int).reshape(
                    self.context_size, 1
                )
            else:
                # Legacy path: decode one-hot float vector back to indices
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

        Adaptive LR
        -----------
        Warmup (fresh training only, epochs 0-4):
            Ramp lr from lr/5 -> lr_max. Skipped on resume.
        Bounce (loss increases 2+ consecutive epochs):
            lr *= 0.7  -- model overshot, take smaller steps.
        Plateau (no new best for 5 epochs):
            lr *= 0.7  -- model stuck, try finer steps.
        Floored at lr_max / 5.
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
        # If Adam state was loaded (_adam_t > 0), skip warmup.
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

            total_loss = 0.0

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
                bs_T    = bs * T
                delta_r = delta.reshape(bs_T, vs)
                x_out_r = x_out.reshape(bs_T, D)

                if self.weight_tying:
                    # Gradient to embedding from the output path:
                    #   logits = x @ embedding.T  =>  d_embedding += delta.T @ x
                    de_acc += delta_r.T @ x_out_r                   # (vocab, D)
                    d_x     = (delta_r @ self.embedding).reshape(bs, T, D)
                else:
                    dWout_acc += x_out_r.T @ delta_r                 # (D, vocab)
                    d_x        = delta @ self.Wout.T

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
                        de_acc[...] = np.array(de_cpu)
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

                # One concatenation + one dot product = one GPU sync instead of N.
                all_flat   = np.concatenate(all_grads)
                total_norm = float(np.sqrt(np.dot(all_flat, all_flat)))

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
            # Increment step counter once per gradient update (i.e. per epoch,
            # since we accumulate gradients over all batches before stepping).
            # Bias correction is baked into lr_eff rather than recomputed
            # per-parameter-tensor (saves ~16 power ops).
            #   lr_eff = epoch_lr * sqrt(1 - beta2^t) / (1 - beta1^t)
            self._adam_t += 1
            t      = self._adam_t
            bc1    = 1.0 - 0.9   ** t    # (1 - beta1^t)
            bc2    = 1.0 - 0.999 ** t    # (1 - beta2^t)
            lr_eff = epoch_lr * (bc2 ** 0.5) / bc1

            # Update all block parameters:
            #   Wqkv, W1, W2  --> Muon  (Nesterov + Newton-Schulz ortho)
            #   everything else -> Adam  (biases, LN params)
            for blk, acc, adam_buf, mbuf in zip(
                self.blocks, blk_grad_acc, self._adam_blocks, self._muon_bufs
            ):
                for k in blk:
                    if k in ("Wqkv", "W1", "W2"):
                        _muon_step(blk[k], acc[k], mbuf[k], epoch_lr)
                    else:
                        _adam_step(blk[k], acc[k], adam_buf[k]["m"], adam_buf[k]["v"], lr_eff)

            # Update final LN
            _adam_step(self.ln_f_g, d_ln_f_g_acc, self._m_ln_f_g, self._v_ln_f_g, lr_eff)
            _adam_step(self.ln_f_b, d_ln_f_b_acc, self._m_ln_f_b, self._v_ln_f_b, lr_eff)

            # Update output projection (separate matrix only if not weight-tied)
            if not self.weight_tying:
                _adam_step(self.Wout, dWout_acc, self._mWout, self._vWout, lr_eff)
            _adam_step(self.bout, dbout_acc, self._mbout, self._vbout, lr_eff)

            # Update embeddings
            if self.use_embedding:
                _adam_step(self.embedding,     de_acc,  self._me,  self._ve,  lr_eff)
                _adam_step(self.pos_embedding, dpe_acc, self._mpe, self._vpe, lr_eff)

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

            # ---- Logging ------------------------------------------------
            if log_every and epoch % log_every == 0:
                # equiv = mean cross-entropy per (sample, position) pair.
                equiv = total_loss / (n * ctx_size)
                print(
                    f"Epoch {epoch:>6} | Loss: {total_loss:.2f}  "
                    f"equiv: {equiv:.4f}  lr={epoch_lr:.6f}{lr_change_msg}",
                    flush=True,
                )

            # ---- Checkpoint ---------------------------------------------
            if save_every and epoch > 0 and epoch % save_every == 0:
                self.save_weights(save_path)
                print(f"  checkpoint saved -> {save_path}", flush=True)

        # Persist final adaptive lr so the next resume starts exactly here
        self.learning_rate = epoch_lr
        print(f"Training complete. (final lr={epoch_lr:.6f})")

    # ==========================================================================
    #  Predict
    # ==========================================================================

    def predict(self, inputs) -> Tuple[int, float, "np.ndarray"]:
        """Run forward pass (no dropout). Returns (class, confidence, probs)."""
        _, _, probs = self.forward(inputs)
        if _DEVICE == "gpu":
            probs = np.asnumpy(probs)
        predicted_class = int(probs.argmax())
        return predicted_class, float(probs[predicted_class]), probs

    # ==========================================================================
    #  Summary
    # ==========================================================================

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
        print(f"|  {'Optimizer':<18} | {'Muon (W) + Adam (rest)':<{width-24}}|")
        print(f"|  {'Batch size':<18} | {self.batch_size:<{width-24}}|")
        if self.use_embedding:
            edim = f"{self.vocab_size} chars x {D}d  context={self.context_size}"
            print(f"|  {'Embedding':<18} | {edim:<{width-24}}|")
        blk_str  = f"{self.num_blocks} blocks  heads={self.num_heads}  d_head={d_h}"
        attn_str = f"causal MHA {D}x{D*3} ({attn_p} params/block)"
        ff_str   = f"{D}->{D*4}->{D} ({ff_p} params/block)"
        ln_str   = f"pre-norm x2/block + final LN ({ln_p + ln_f_p} params)"
        wt_str   = "ON (embedding.T)" if self.weight_tying else "OFF (separate)"
        dp_str   = f"{self.dropout:.2f}" if self.dropout > 0 else "OFF"
        gc_str   = f"{self.grad_clip}" if self.grad_clip > 0 else "OFF"
        print(f"|  {'Architecture':<18} | {blk_str:<{width-24}}|")
        print(f"|  {'Attention':<18} | {attn_str:<{width-24}}|")
        print(f"|  {'Feed-forward':<18} | {ff_str:<{width-24}}|")
        print(f"|  {'LayerNorm':<18} | {ln_str:<{width-24}}|")
        print(f"|  {'Weight tying':<18} | {wt_str:<{width-24}}|")
        print(f"|  {'Dropout':<18} | {dp_str:<{width-24}}|")
        print(f"|  {'Grad clip':<18} | {gc_str:<{width-24}}|")
        print(f"|  {'Output':<18} | {'all positions -> ' + str(self.output_size):<{width-24}}|")
        print("+" + "=" * width + "+")
        pad = width - 23 - len(f"{total:,}")
        print(f"|  Total parameters: {total:,}{'':<{pad}}  |")
        print("+" + "=" * width + "+")

    # ==========================================================================
    #  Save / Load weights
    # ==========================================================================

    def save_weights(self, filename: str = "weights.json") -> None:
        """
        Save all weights, Adam state, and hyperparameters to JSON.

        ATOMIC WRITE: writes to a temp file first, then renames atomically.
        If the process is interrupted mid-save, the previous checkpoint
        remains intact.

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
            "Wout":   to_list(self.Wout) if not self.weight_tying else None,
            "bout":   to_list(self.bout),
            "ln_f_g": to_list(self.ln_f_g),
            "ln_f_b": to_list(self.ln_f_b),
            "blocks": [{k: to_list(v) for k, v in blk.items()}
                       for blk in self.blocks],
            # ---- Adam state ---------------------------------------------
            "adam_t":        self._adam_t if self._adam_init else 0,
            "adam_mWout":    to_list(self._mWout) if (self._adam_init and not self.weight_tying) else None,
            "adam_vWout":    to_list(self._vWout) if (self._adam_init and not self.weight_tying) else None,
            "adam_mbout":    to_list(self._mbout) if self._adam_init else None,
            "adam_vbout":    to_list(self._vbout) if self._adam_init else None,
            "adam_me":       to_list(self._me)    if (self._adam_init and self.use_embedding) else None,
            "adam_ve":       to_list(self._ve)    if (self._adam_init and self.use_embedding) else None,
            "adam_mpe":      to_list(self._mpe)   if (self._adam_init and self.use_embedding) else None,
            "adam_vpe":      to_list(self._vpe)   if (self._adam_init and self.use_embedding) else None,
            "adam_m_ln_f_g": to_list(self._m_ln_f_g) if self._adam_init else None,
            "adam_v_ln_f_g": to_list(self._v_ln_f_g) if self._adam_init else None,
            "adam_m_ln_f_b": to_list(self._m_ln_f_b) if self._adam_init else None,
            "adam_v_ln_f_b": to_list(self._v_ln_f_b) if self._adam_init else None,
            "adam_blocks":   adam_blocks,
            # ---- Muon momentum state ------------------------------------
            "muon_bufs": [
                {k: to_list(v) for k, v in mbuf.items()}
                for mbuf in self._muon_bufs
            ] if self._adam_init else [],
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
          - num_heads defaults to 1 (safe for old Wqkv shapes).
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
        self.num_heads     = data.get("num_heads",     1)   # old files default to 1 head
        self.dropout       = data.get("dropout",       0.0)
        self.weight_tying  = data.get("weight_tying",  False)
        self.grad_clip     = data.get("grad_clip",     1.0)
        self._act_fn, self._act_d = _ACTIVATIONS[self.activation]
        self._head_dim     = self.embed_dim // self.num_heads
        self._scale_head   = 1.0 / (self._head_dim ** 0.5)

        _f32 = np.float32

        # ---- Legacy dense weights -------------------------------------------
        self.weights = [np.array(w) for w in data["weights"]]
        self.biases  = [np.array(b) for b in data["biases"]]

        # ---- Embeddings -----------------------------------------------------
        self.embedding     = np.array(data["embedding"],     dtype=_f32) if data.get("embedding")     else None
        self.pos_embedding = np.array(data["pos_embedding"], dtype=_f32) if data.get("pos_embedding") else None

        # ---- Output projection + final LN -----------------------------------
        if self.weight_tying:
            self.Wout = None
        else:
            self.Wout = np.array(data["Wout"], dtype=_f32) if data.get("Wout") else None
        self.bout = np.array(data["bout"], dtype=_f32)

        D = self.embed_dim
        if data.get("ln_f_g") is not None:
            self.ln_f_g = np.array(data["ln_f_g"], dtype=_f32)
            self.ln_f_b = np.array(data["ln_f_b"], dtype=_f32)
        else:
            # Old file: initialise final LN as identity
            self.ln_f_g = np.ones(D, dtype=_f32)
            self.ln_f_b = np.zeros(D, dtype=_f32)

        # ---- Transformer blocks ---------------------------------------------
        if "blocks" in data:
            self.blocks = []
            for blk in data["blocks"]:
                b = {k: np.array(v, dtype=_f32) for k, v in blk.items()}

                # Upgrade old separate Wq/Wk/Wv -> fused Wqkv
                if "Wq" in b and "Wqkv" not in b:
                    Wq = _np_cpu.array(b.pop("Wq") if _DEVICE == "cpu" else np.asnumpy(b.pop("Wq")))
                    Wk = _np_cpu.array(b.pop("Wk") if _DEVICE == "cpu" else np.asnumpy(b.pop("Wk")))
                    Wv = _np_cpu.array(b.pop("Wv") if _DEVICE == "cpu" else np.asnumpy(b.pop("Wv")))
                    b["Wqkv"] = np.array(_np_cpu.concatenate([Wq, Wk, Wv], axis=1), dtype=_f32)

                # Add LN params if missing (old file without LayerNorm)
                if "ln1_g" not in b:
                    b["ln1_g"] = np.ones(D, dtype=_f32)
                    b["ln1_b"] = np.zeros(D, dtype=_f32)
                    b["ln2_g"] = np.ones(D, dtype=_f32)
                    b["ln2_b"] = np.zeros(D, dtype=_f32)

                self.blocks.append(b)
        else:
            # Very old single-weight format: replicate shared weights per block
            self.blocks = []
            for _ in range(self.num_blocks):
                Wq = _np_cpu.array(data["Wq"], dtype=_np_cpu.float32)
                Wk = _np_cpu.array(data["Wk"], dtype=_np_cpu.float32)
                Wv = _np_cpu.array(data["Wv"], dtype=_np_cpu.float32)
                self.blocks.append({
                    "Wqkv":  np.array(_np_cpu.concatenate([Wq, Wk, Wv], axis=1), dtype=_f32),
                    "W1":    np.array(data["W1"], dtype=_f32),
                    "b1":    np.array(data["b1"], dtype=_f32),
                    "W2":    np.array(data["W2"], dtype=_f32),
                    "b2":    np.array(data["b2"], dtype=_f32),
                    "ln1_g": np.ones(D, dtype=_f32), "ln1_b": np.zeros(D, dtype=_f32),
                    "ln2_g": np.ones(D, dtype=_f32), "ln2_b": np.zeros(D, dtype=_f32),
                })

        # ---- Adam state -----------------------------------------------------
        self._adam_init = False
        if data.get("adam_t") and data["adam_t"] > 0:
            self._init_adam()                         # allocate all buffers
            self._adam_t = data["adam_t"]             # restore step counter

            if not self.weight_tying and data.get("adam_mWout"):
                self._mWout = np.array(data["adam_mWout"], dtype=_f32)
                self._vWout = np.array(data["adam_vWout"], dtype=_f32)
            self._mbout = np.array(data["adam_mbout"], dtype=_f32)
            self._vbout = np.array(data["adam_vbout"], dtype=_f32)

            if self.use_embedding and data.get("adam_me") is not None:
                self._me  = np.array(data["adam_me"],  dtype=_f32)
                self._ve  = np.array(data["adam_ve"],  dtype=_f32)
                self._mpe = np.array(data["adam_mpe"], dtype=_f32)
                self._vpe = np.array(data["adam_vpe"], dtype=_f32)

            # Final LN Adam state (absent in old files -> stays zero from _init)
            if data.get("adam_m_ln_f_g") is not None:
                self._m_ln_f_g = np.array(data["adam_m_ln_f_g"], dtype=_f32)
                self._v_ln_f_g = np.array(data["adam_v_ln_f_g"], dtype=_f32)
                self._m_ln_f_b = np.array(data["adam_m_ln_f_b"], dtype=_f32)
                self._v_ln_f_b = np.array(data["adam_v_ln_f_b"], dtype=_f32)

            # Per-block Adam state (includes LN buffers for new files)
            for i, buf in enumerate(data.get("adam_blocks", [])):
                for k, mv in buf.items():
                    if k in self._adam_blocks[i]:
                        self._adam_blocks[i][k]["m"] = np.array(mv["m"], dtype=_f32)
                        self._adam_blocks[i][k]["v"] = np.array(mv["v"], dtype=_f32)

            print(f"  Adam state restored (t={self._adam_t})")

            # Muon momentum state (absent in old files -> stays zero from _init)
            for i, mbuf in enumerate(data.get("muon_bufs", [])):
                if i < len(self._muon_bufs):
                    for k, v in mbuf.items():
                        if k in self._muon_bufs[i]:
                            self._muon_bufs[i][k] = np.array(v, dtype=_f32)

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

    # ==========================================================================
    #  Tool use
    # ==========================================================================

    def register_tool(self, name: str, handler: "callable") -> None:
        """
        Register a callable tool the model may invoke during generation.

        Parameters
        ----------
        name    : the tool name as it will appear in [TOOL:name|...] calls.
        handler : callable(query: str) -> str

        Example
        -------
            def my_calc(expr):
                try:    return str(eval(expr, {"__builtins__": {}}, {}))
                except: return "error"

            nn.register_tool("calc", my_calc)
        """
        if not hasattr(self, "_tools"):
            self._tools: Dict[str, "callable"] = {}
        # Warn if the model's vocab_size looks like it was built before
        # ensure_tool_vocab() was called (i.e. the delimiter chars are absent).
        # This won't stop generation but the model will substitute unk tokens.
        if self.use_embedding:
            # We can't inspect char2idx from here, so just remind the caller.
            pass   # caller is responsible for running ensure_tool_vocab()
        self._tools[name] = handler

    # ==========================================================================
    #  Autoregressive generation  (+ KV-cache helpers)
    # ==========================================================================

    def generate(
        self,
        prompt_ids:     List[int],
        max_new:        int   = 200,
        temperature:    float = 1.0,
        top_k:          int   = 0,
        kv_cache:       Optional[_KVCacheBase] = None,
        idx2char:       Optional[Dict[int, str]] = None,
        char2idx:       Optional[Dict[str, int]] = None,
        max_tool_calls: int   = 8,
    ):
        """
        Autoregressive generation with optional KV-cache and tool execution.

        When ``idx2char`` and ``char2idx`` are provided and tools have been
        registered via ``register_tool()``, the output is scanned after each
        token for complete ``[TOOL:name|arg]`` patterns.  Matching calls are
        executed and ``[RESULT:...]`` is injected back into context before
        generation continues.  Tool result tokens are injected into context
        but do NOT count toward ``max_new``.

        Without ``idx2char``/``char2idx`` (or with no registered tools) the
        method behaves identically to the original plain generate -- no
        scanning overhead at all.

        Parameters
        ----------
        prompt_ids     : integer token indices for the prompt.
        max_new        : how many new tokens to sample.
        temperature    : softmax temperature (1.0 = unchanged, <1 = sharper).
        top_k          : if > 0, restrict sampling to the top-k logits.
        kv_cache       : TurboQuantKVCache / PolarQuantKVCache, or None.
        idx2char       : index -> character mapping.  Required for tools.
        char2idx       : character -> index mapping.  Required for tools.
        max_tool_calls : hard cap on tool invocations to prevent loops.

        Returns
        -------
        Plain mode  (no idx2char): List[int] of generated token indices.
        Tool mode   (idx2char set): Tuple[List[int], List[dict]] where the
            first element is the full token list (model + injected) and the
            second is a log of each tool call made.
        """
        if not hasattr(self, "_tools"):
            self._tools = {}

        tools_active = (
            idx2char is not None
            and char2idx is not None
            and bool(self._tools)
        )

        ctx  = self.context_size
        ids  = list(prompt_ids[-ctx:])
        out  = []          # plain mode output
        full_out   = []    # tool mode: model tokens + injected result tokens
        tool_log   = []
        calls_made = 0
        decoded_tail = ""
        _MAX_SCAN    = 512

        if kv_cache is not None:
            kv_cache.reset()
            self._kv_prefill(ids, kv_cache)

        for _ in range(max_new):
            # ---- Sample one token ----------------------------------------
            if kv_cache is not None:
                logits = self._kv_decode_step(ids[-1], len(ids) - 1, kv_cache)
            else:
                toks      = _np_cpu.array(ids, dtype=_np_cpu.int32).reshape(-1, 1)
                probs_all, _ = self._transformer_forward(
                    np.array(toks), training=False
                )
                if _DEVICE == "gpu":
                    probs_all = np.asnumpy(probs_all)
                logits = _np_cpu.log(probs_all[0, -1, :] + 1e-9)

            logits = _np_cpu.array(logits, dtype=_np_cpu.float32)
            logits /= max(temperature, 1e-6)
            if top_k > 0:
                kth = _np_cpu.partition(logits, -top_k)[-top_k]
                logits[logits < kth] = -1e9
            logits -= logits.max()
            probs  = _np_cpu.exp(logits); probs /= probs.sum()
            token  = int(_np_cpu.random.choice(len(probs), p=probs))

            out.append(token)
            ids.append(token)
            if len(ids) > ctx:
                ids = ids[-ctx:]

            if not tools_active:
                continue

            # ---- Tool path: scan for [TOOL:name|arg] ---------------------
            full_out.append(token)
            decoded_tail += idx2char.get(token, "")
            if len(decoded_tail) > _MAX_SCAN:
                decoded_tail = decoded_tail[-_MAX_SCAN:]

            if calls_made < max_tool_calls and "[TOOL:" in decoded_tail:
                m = _TOOL_OPEN_RE.search(decoded_tail)
                if m:
                    name    = m.group(1)
                    query   = m.group(2)
                    handler = self._tools.get(name)
                    if handler is not None:
                        try:
                            result = str(handler(query))
                        except Exception as exc:
                            result = f"error:{exc}"
                        result = result[:_TOOL_MAX_RESULT]

                        tool_log.append({
                            "name":     name,
                            "query":    query,
                            "result":   result,
                            "position": len(full_out),
                        })
                        calls_made += 1

                        result_ids = _encode_tool_result(result, char2idx)
                        for rid in result_ids:
                            full_out.append(rid)
                            ids.append(rid)
                            if len(ids) > ctx:
                                ids = ids[-ctx:]
                        if kv_cache is not None:
                            self._kv_prefill(result_ids, kv_cache)

                        decoded_tail = decoded_tail[m.end():]

        if tools_active:
            return full_out, tool_log
        return out

    def _kv_prefill(self, ids: List[int], kv_cache: _KVCacheBase) -> None:
        """Run all prompt tokens through every block, storing K and V."""
        D  = self.embed_dim
        H  = self.num_heads
        dh = D // H

        toks_np = _np_cpu.array(ids, dtype=_np_cpu.int32).reshape(-1, 1)
        toks    = np.array(toks_np)           # (T, 1)
        T       = len(ids)

        x    = self.embedding[toks.T[0]] + self.pos_embedding[:T]  # (T, D)
        x    = x[None]                                               # (1, T, D)
        mask = self._causal_mask(T)

        for layer_idx, blk in enumerate(self.blocks):
            ln1_out, _ = self._ln_forward(x, blk["ln1_g"], blk["ln1_b"])
            QKV = (ln1_out.reshape(T, D) @ blk["Wqkv"]).reshape(T, 3, H, dh)
            K = QKV[:, 1].transpose((1, 0, 2))   # (H, T, dh)
            V = QKV[:, 2].transpose((1, 0, 2))   # (H, T, dh)

            if _DEVICE == "gpu":
                K = np.asnumpy(K)
                V = np.asnumpy(V)

            for t in range(T):
                kv_cache.append(layer_idx,
                                K[:, t, :],   # (H, dh)
                                V[:, t, :])   # (H, dh)

            # Run the full block forward so x is correct for the next layer.
            x, _ = self._block_forward(x, blk, training=False, mask=mask)

    def _kv_decode_step(
        self,
        token_id: int,
        pos:      int,
        kv_cache: _KVCacheBase,
    ) -> "_np_cpu.ndarray":
        """
        Single-token incremental decode.

        Projects the new token to Q/K/V, appends K/V to the cache,
        then computes attention over the full cached history for each layer.
        Returns raw (pre-softmax) logits of shape (vocab,).
        """
        D  = self.embed_dim
        H  = self.num_heads
        dh = D // H

        tok_emb = self.embedding[token_id] + self.pos_embedding[pos]  # (D,)
        x       = tok_emb[None, None]    # (1, 1, D)

        for layer_idx, blk in enumerate(self.blocks):
            ln1_out, _ = self._ln_forward(x, blk["ln1_g"], blk["ln1_b"])
            qkv = (ln1_out.reshape(1, D) @ blk["Wqkv"]).reshape(3, H, dh)
            Q   = qkv[0]   # (H, dh)
            K_t = qkv[1]   # (H, dh)  -- new key
            V_t = qkv[2]   # (H, dh)  -- new value

            if _DEVICE == "gpu":
                K_t_cpu = np.asnumpy(K_t)
                V_t_cpu = np.asnumpy(V_t)
                Q_cpu   = np.asnumpy(Q)
            else:
                K_t_cpu, V_t_cpu, Q_cpu = K_t, V_t, Q

            kv_cache.append(layer_idx, K_t_cpu, V_t_cpu)
            K_all, V_all = kv_cache.get(layer_idx)  # (H, T_so_far, dh)

            if _DEVICE == "gpu":
                K_all = np.array(K_all)
                V_all = np.array(V_all)
                Q_gpu = Q
            else:
                Q_gpu = Q_cpu

            # Scaled dot-product attention: Q (H, dh) × K^T (H, dh, T)
            scale  = self._scale_head
            scores = (Q_gpu[:, None, :] * K_all).sum(axis=-1) * scale  # (H, T)
            scores -= scores.max(axis=-1, keepdims=True)
            exp_s  = np.exp(scores)
            A      = exp_s / exp_s.sum(axis=-1, keepdims=True)          # (H, T)

            # Weighted sum of values: (H, T) × (H, T, dh) -> (H, dh)
            attn_out = (A[:, :, None] * V_all).sum(axis=1)              # (H, dh)
            attn_out = attn_out.reshape(1, 1, D)                         # (1, 1, D)

            x_attn = x + attn_out

            ln2_out, _ = self._ln_forward(x_attn, blk["ln2_g"], blk["ln2_b"])
            h  = np.maximum(0.0, ln2_out @ blk["W1"] + blk["b1"])
            ff = h @ blk["W2"] + blk["b2"]
            x  = x_attn + ff

        x, _ = self._ln_forward(x, self.ln_f_g, self.ln_f_b)

        if self.weight_tying:
            logits = (x.reshape(D) @ self.embedding.T) + self.bout
        else:
            logits = (x.reshape(D) @ self.Wout) + self.bout

        if _DEVICE == "gpu":
            logits = np.asnumpy(logits)
        return _np_cpu.array(logits, dtype=_np_cpu.float32)