# miniGPT

A character-level GPT transformer built from scratch in pure NumPy/CuPy.
No PyTorch, no TensorFlow — every forward pass, backward pass, and Adam
update is written by hand.

---

## What it is

miniGPT trains a small transformer language model on any plain-text file and
generates new text character by character. It is designed to run on a free
Google Colab T4 GPU and converges to recognisable words in a few hours.

**Current training results** (Wikipedia dataset, 2M chars, simple vocab):

| Epochs | equiv loss | Quality |
|--------|-----------|---------|
| 0      | 3.50      | Random (theoretical max = log 33) |
| 100    | 2.63      | Letter pairs, spaces correct |
| 200    | 2.57      | Real words appearing ("the", "and") |
| 300+   | < 2.50    | Short word sequences |
| target | ~1.50     | Readable sentences |

---

## Files

```
GPT/
├── Neural_Network.py          # The transformer: forward, backward, Adam, save/load
├── miniGPT/
│   ├── model.py               # MiniGPT class: train(), generate(), save(), load()
│   ├── cli.py                 # Command-line interface
│   ├── data.py                # Text loading and dataset building
│   ├── tokenizer.py           # Character tokenizer
│   └── __init__.py
├── data/
│   └── wiki_dataset.txt       # Training corpus
├── weights/                   # Saved model checkpoints
└── README.md
```

---

## Quick start

### Train from scratch

```bash
python miniGPT/cli.py \
  --train wiki_dataset.txt \
  --epochs 100 \
  --samples 10000 \
  --context 16 \
  --embed_dim 256 \
  --lr 0.0005 \
  --max_chars 2000000 \
  --simple_vocab \
  --save_every 10 \
  --save gpt_weights_v1.json
```

### Resume training

```bash
python miniGPT/cli.py \
  --train wiki_dataset.txt \
  --resume gpt_weights_v1.json \
  --epochs 100 \
  --samples 10000 \
  --max_chars 2000000 \
  --simple_vocab \
  --save_every 10 \
  --save gpt_weights_v2.json
```

When resuming, the optimizer state (Adam momentum) is fully restored — no
warmup phase, loss picks up exactly where it left off.

### Generate text

```bash
python miniGPT/cli.py \
  --load gpt_weights_v2.json \
  --prompt "the " \
  --length 300 \
  --temperature 0.6
```

---

## All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--train FILE` | — | Path to .txt dataset |
| `--resume FILE` | — | Resume from checkpoint (use with --train) |
| `--load FILE` | — | Load for generation only |
| `--save FILE` | gpt_weights.json | Output filename |
| `--epochs N` | 100 | Training epochs |
| `--samples N` | 20000 | Max training samples |
| `--context N` | 12 | Context window size (tokens) |
| `--embed_dim N` | 64 | Embedding / hidden dimension |
| `--num_blocks N` | 2 | Transformer blocks |
| `--batch_size N` | 1024 | Samples per gradient step |
| `--lr F` | 0.001 | Peak learning rate |
| `--max_chars N` | 500000 | Characters to read from file |
| `--simple_vocab` | off | Reduce vocab to a-z + basic punctuation |
| `--save_every N` | 0 | Checkpoint every N epochs (0=disabled) |
| `--log_every N` | 1 | Print loss every N epochs |
| `--prompt TEXT` | "" | Generation seed text |
| `--length N` | 300 | Characters to generate |
| `--temperature F` | 0.6 | < 1 = focused, > 1 = creative |

---

## Architecture

```
Input tokens (B, T)
    |
    v
Token embedding (vocab, D)  +  Positional embedding (T, D)
    |                              both learned, added together
    v
Transformer block x N  (independent weights per block)
    |
    |-- Self-attention (causal) ----------------------------
    |     Fused QKV projection: x @ Wqkv  (D -> 3D, split)
    |     Attention scores: Q @ K^T / sqrt(D)
    |     Causal mask: upper triangle = -1e9
    |     Softmax -> attention weights A
    |     Output: A @ V
    |     Residual: x = x + attn_out
    |
    |-- Feed-forward network --------------------------------
    |     Linear: D -> 4D
    |     ReLU activation
    |     Linear: 4D -> D
    |     Residual: x = x + ff_out
    |
    v
Output projection (D -> vocab)  at ALL T positions
    |
    v
Softmax -> probabilities (B, T, vocab)
    |
    v
Cross-entropy loss averaged over all (batch, position) pairs
```

**Key design choices:**

- **Causal mask**: each token can only attend to itself and earlier tokens.
  Required for autoregressive generation (the model can't "cheat" by looking
  at future tokens during training).

- **All-positions training**: every token position predicts the next token,
  giving T× more gradient signal per sample than predicting only the last
  position. Context size 16 means 16 loss signals per sample instead of 1.

- **Independent block weights**: each transformer block has its own Wqkv, W1,
  W2. No weight sharing. This lets each block specialise — early blocks learn
  low-level patterns (character pairs), later blocks learn higher-level
  structure (word boundaries, common sequences).

---

## Optimisations

Every optimisation listed below has a measurable effect on training speed or
convergence quality. They are explained in detail here because understanding
*why* they work is as useful as having them.

### 1. Fused QKV projection

**What:** Instead of three separate weight matrices Wq, Wk, Wv (each D×D),
we use a single fused matrix Wqkv of shape (D, 3D). One matrix multiply
produces all three, which are then split along the last axis.

**Why it's faster:** Every matrix multiply on a GPU launches a CUDA kernel.
Kernel launch has overhead (~5-20 microseconds). Three small launches
(3 × D×D matmuls) are slower than one larger launch (1 × D×3D matmul),
even though the total FLOPs are the same. The larger matmul also makes
better use of the GPU's tensor cores which prefer wide operations.

**Implementation:** `QKV = x.reshape(BT, D) @ Wqkv` then
`Q, K, V = QKV[:,:,0,:], QKV[:,:,1,:], QKV[:,:,2,:]`

The backward pass mirrors this: `dWqkv = x^T @ concat([dQ, dK, dV])`.

---

### 2. All-positions training

**What:** Instead of computing loss only at the last token position, we compute
it at all T positions simultaneously.

**Why it helps:** With context size T=16, each forward pass produces 16 sets
of predictions. Each of those is compared to the correct next token, giving 16
independent gradient signals per sample. This is equivalent to training on
16× more data with no extra computation cost (the forward pass runs over all
positions anyway in a transformer).

**Implementation:** The output projection `x @ Wout + bout` maps (B, T, D) →
(B, T, vocab). Loss is computed as `-sum(log(p[b, t, y[b,t]]))` for all b, t.

---

### 3. Pre-allocated gradient buffers

**What:** All gradient accumulation arrays (`dWout_acc`, `de_acc`, etc.) are
allocated once before the training loop and zeroed in-place at the start of
each epoch with `buf[...] = 0.0`.

**Why it's faster:** On GPU, every `np.zeros_like()` call allocates new VRAM
and forces a CUDA synchronisation point. With batch_size=1024 and 10,000
samples, there are ~10 batches per epoch. Over 100 epochs that's 1,000 epochs
× (number of parameter tensors) potential allocations — hundreds of thousands
of GPU syncs. Zeroing in-place avoids all of them.

---

### 4. Fused Adam bias correction

**What:** Adam's bias correction requires computing `1 - beta1^t` and
`1 - beta2^t` for step t. These are combined into a single effective learning
rate scalar `lr_eff = epoch_lr * sqrt(1 - beta2^t) / (1 - beta1^t)` before
the update loop.

**Why it's faster:** Without this, every call to `_adam_step()` would
recompute `beta1^t` and `beta2^t` — which are the same for every parameter
tensor in the same epoch. We have ~(5 tensors per block × 2 blocks + 4 output
tensors + 2 embedding tensors) = 16 parameter tensors. Fusing saves 15 × 2
power operations per epoch — small, but free.

---

### 5. GPU-side shuffling

**What:** `np.random.permutation(n)` when `np` is CuPy runs entirely on the
GPU.

**Why it matters:** If we used Python's `random.shuffle()` or NumPy on the
CPU, we would need to move the shuffled indices to the GPU for each epoch.
That's a PCIe transfer (~10GB/s) for an array of N int32s. Small overhead,
but zero is better than small.

---

### 6. In-place softmax gradient

**What:** The gradient of cross-entropy + softmax simplifies to
`delta = (p - one_hot(y)) / (B * T)`. We compute this in-place on the
`probs` buffer returned by the forward pass.

**Why it matters:** `probs` is already a (B, T, vocab) array on the GPU.
Computing the gradient in-place means no second allocation of the same size.
For vocab=33, B=1024, T=16 that's 33 × 1024 × 16 × 4 bytes = ~2MB per batch
that doesn't need to be allocated.

---

### 7. Cached causal mask

**What:** The upper-triangular attention mask is computed once and stored in
`self._mask_cache`. It is only rebuilt if the context size T changes.

**Why it matters:** The mask is a (T, T) matrix of zeros and -1e9 values. For
T=16 that's only 256 floats — tiny. But without caching, `np.triu()` is
called every forward pass, every batch. Cached = zero cost per batch.

---

### 8. cupyx.scatter_add for embedding gradients

**What:** Token embedding gradients require accumulating rows into a lookup
table: `de[token_idx] += d_x`. Multiple tokens in a batch can share the same
vocabulary row, so this must be an accumulation, not an assignment.

NumPy's `np.add.at()` does this but runs on CPU. For GPU arrays, it would
require a CPU round-trip. CuPy's `cupyx.scatter_add()` does the same
operation entirely on the GPU.

**Why it matters:** Without this, every batch requires moving the embedding
gradient (vocab_size × embed_dim × 4 bytes = 33 × 256 × 4 = ~33KB) to CPU,
updating it, and moving it back. At 10 batches per epoch × 100 epochs that's
1000 PCIe transfers. With scatter_add, it's zero.

---

### 9. Atomic checkpoint saves

**What:** `save_weights()` writes to a temporary file in the same directory,
then calls `os.replace()` to atomically rename it over the target.

**Why it matters:** JSON serialization of a 220MB model takes several seconds.
If Colab disconnects or runs out of memory halfway through, a naive
`open(filename, 'w')` would leave a truncated, corrupt JSON file — exactly
what happened with v3 and v4 in practice. `os.replace()` is atomic on both
Linux (Colab) and Windows: the target file is either the old version or the
new version, never a partial write.

---

### 10. Compact JSON (no indentation)

**What:** `json.dump(data, f)` with no `indent` argument writes compact JSON.

**Why it matters:** The weight file with `indent=2` was 220MB. Without
indentation the same data is ~50MB — a 4× reduction. This matters for:
- Upload/download time to/from Google Drive
- Colab disk usage (15GB limit shared with OS)
- JSON parse time on load

---

### 11. stride_tricks for dataset building

**What:** `make_index_arrays()` uses `numpy.lib.stride_tricks.as_strided()`
to build all sliding windows simultaneously as a zero-copy view of the corpus
array, rather than copying data in a Python loop.

**Why it's faster:** The old version looped over N sample indices in Python
and copied each T+1-length window into pre-allocated arrays. For 10,000
samples with T=16, that's 10,000 Python iterations and 170,000 individual
array element copies. `as_strided()` does this in one NumPy call with no data
movement at all — it creates a view of the same memory with a different shape
and stride descriptor. The subsequent `[::step][:max_samples]` subsample is
also zero-copy.

---

## Adaptive learning rate

The scheduler has three phases:

**Warmup (fresh training only, epochs 0-4):**
Linearly ramps learning rate from `lr/5` to `lr_max`. Prevents large gradient
steps before Adam's momentum estimates (m and v) have converged. The first few
Adam steps have high variance because the moving averages are initialised at
zero — warmup keeps step sizes small until the estimates stabilise.
Automatically skipped on resume because Adam already has good momentum.

**Bounce reduction:**
If the loss increases for `bounce_patience=2` consecutive epochs, `lr *= 0.7`.
"Bounce" means the model overshot a minimum — reducing lr helps it settle.
Requires 2 consecutive bad epochs to avoid reacting to single noisy batches.

**Plateau reduction:**
If no new best loss is achieved for `plateau_patience=5` epochs, `lr *= 0.7`.
Plateau means the model is stuck in a flat region — a smaller lr can help
escape it by taking more careful steps. Floored at `lr_max / 5`.

---

## Colab workflow

Colab disconnects can corrupt checkpoint files if they happen mid-save.
Atomic saves protect against this. For very long runs, use this pattern:

```python
# Run in a cell — output streams live even if the browser disconnects
import subprocess, sys

proc = subprocess.Popen(
    [sys.executable, "miniGPT/cli.py",
     "--train", "wiki_dataset.txt",
     "--resume", "gpt_weights_v5.json",
     "--epochs", "100",
     "--samples", "10000",
     "--max_chars", "2000000",
     "--simple_vocab",
     "--save_every", "10",
     "--save", "gpt_weights_v6.json"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True, bufsize=1,
)
for line in proc.stdout:
    print(line, end="", flush=True)
```

**Saving to Google Drive** (avoids upload corruption for large files):

```python
from google.colab import drive
drive.mount('/content/drive')

# After training completes:
import shutil
shutil.copy('gpt_weights_v6.json', '/content/drive/MyDrive/gpt_weights_v6.json')
```

---

## Loss reference

| equiv | Interpretation |
|-------|----------------|
| 3.50  | Random (ln 33 ≈ 3.497 for 33-char vocab) |
| 2.97  | Spaces roughly correct, common letter pairs |
| 2.60  | Real words appearing ("the", "and", "in") |
| 2.30  | Short word sequences, punctuation placement |
| 2.00  | Short readable phrases |
| 1.50  | Grammatical sentences |
| 1.00  | Coherent paragraphs |

`equiv = total_loss / (n_samples * context_size)` — average cross-entropy
per (sample, position) pair. Comparable across runs with different batch
sizes, sample counts, or context lengths.

---

## GPU setup

```bash
pip install cupy-cuda12x   # Colab T4 (CUDA 12)
pip install cupy-cuda11x   # older setups (CUDA 11)
```

If CuPy is not installed, the code falls back to NumPy automatically.
Training will be ~10-50x slower on CPU for the current model size.


---

## New features (v2 architecture)

### LayerNorm

Every transformer block now normalises activations **before** attention and **before** the feed-forward network (pre-norm, GPT-2 style). A final LayerNorm is also applied after all blocks before the output projection.

**What LayerNorm does:** For each (batch, position) pair, it computes the mean and variance of the D-dimensional vector, normalises it to zero mean and unit variance, then applies learned scale (gamma) and shift (beta) parameters. Formula: `y = (x - mean) / sqrt(var + eps) * gamma + beta`.

**Why it helps:** Without normalisation, activations can grow or shrink as they pass through blocks — this makes the loss surface spiky and Adam has to fight against constantly changing magnitudes. LayerNorm pins each vector to a consistent scale, so gradients flow cleanly and the model can use a higher learning rate without instability.

**Pre-norm vs post-norm:** GPT-1 used post-norm (normalise after the residual add). GPT-2 switched to pre-norm (normalise before attention/FF, leave the residual clean). Pre-norm trains more stably at larger depth because the residual stream itself stays unmodified — gradients can flow back through the residuals without passing through normalisation layers.

**Parameters:** Each block adds 4 × D parameters (gamma and beta for each of the two LN layers). Final LN adds another 2 × D. For D=256 and 2 blocks: 2560 extra parameters — negligible.

---

### Multi-head attention

Instead of one attention computation over the full D-dimensional space, the embedding is split into H heads each of dimension d_h = D/H. Each head independently computes Q, K, V projections and attention scores, then the results are concatenated.

**Why multiple heads?** A single attention head must learn one pattern for "what attends to what." Multiple heads can specialise: one might track word boundaries, another repeated character patterns, another vowel sequences. The heads operate in parallel over different d_h-dimensional subspaces of the same representation.

**Scale changes:** The attention scale is now `1/sqrt(d_h)` not `1/sqrt(D)`. With D=256, H=4: d_h=64, scale=1/8 instead of 1/16. This correctly reflects that each head's dot products are over 64-dimensional vectors, not 256.

**Implementation:** Still uses the fused Wqkv matrix (one matmul, not three). The output is reshaped from `(B, T, 3D)` into `(B, T, 3, H, d_h)`, then transposed to `(B, H, T, d_h)` per head. After attention, heads are concatenated back to `(B, T, D)`.

**Default:** `--num_heads 4` with D=256 gives d_h=64.

---

### Weight tying

The output projection matrix Wout (shape D × vocab) is replaced by `embedding.T`. Instead of a separate learned matrix, the model reuses the token embedding in the output direction.

**Why it works:** The token embedding maps "character c" → a D-dimensional vector. The output projection maps D-dimensional hidden state → "probability of character c". These are inverse operations that naturally want to agree on what "character c" means in vector space. Sharing the weights makes this explicit and halves the parameters for these two matrices.

**In practice:** Gradients flow from both the embedding lookup path (forward, which tokens were the inputs) and the output projection path (backward, which tokens should be predicted) into the same matrix. Adam handles both gradient streams naturally.

**Enabled by default.** Use `--no_weight_tying` to disable.

---

### Dropout

Randomly zeroes a fraction of activations during training, then scales up the survivors so the expected sum is unchanged ("inverted dropout").

Applied at three points:
- After the initial embedding (embedding dropout)
- After the attention output (before the first residual add)
- After the feed-forward output (before the second residual add)

**Why it helps:** Forces the model to not rely on any single neuron or activation. The model has to distribute information across many neurons because any given one might be zeroed. This improves generalisation — especially useful when the dataset is small relative to model size.

**During generation:** Dropout is completely disabled (`training=False`). The model uses all its weights at full strength.

**Default:** `--dropout 0.0` (disabled). Try `0.1` for small datasets, `0.2` for very small.

---

### Gradient clipping

Before each Adam update, the global L2 norm of all gradient tensors is computed. If it exceeds `grad_clip`, every gradient is scaled down proportionally so the total norm equals `grad_clip`.

**Why global, not per-parameter?** Per-parameter clipping changes the relative direction of the gradient (it clips each parameter independently). Global clipping preserves the direction — it just shortens the step if it would be too large, like a speed limiter rather than a steering correction.

**When it fires:** Rarely during stable training. It is most valuable when a particularly noisy batch produces a large gradient spike that would otherwise overshoot a minimum and cause the loss to jump up (the "bounce" events visible in earlier training logs).

**Default:** `--grad_clip 1.0`. Set to 0 to disable.

---

### Residual output scaling

W2 (the second feed-forward layer) is initialised with standard deviation `0.02 / sqrt(2 * num_blocks)` instead of just `0.02`.

**Why:** Each residual block adds its output to the residual stream. With N blocks, if each block contributes variance σ², the total variance after N blocks is N×σ². By scaling init by `1/sqrt(N)`, each block contributes σ²/N and the total is still σ² regardless of depth. This keeps activations well-scaled as you add more blocks.

---

## Updated CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--num_heads N` | 4 | Attention heads (embed_dim must be divisible by this) |
| `--dropout F` | 0.0 | Dropout rate. 0 = disabled. Try 0.1 for small datasets |
| `--no_weight_tying` | off | Disable weight tying (use separate Wout matrix) |
| `--grad_clip F` | 1.0 | Global gradient norm clip. 0 = disabled |

---

## Recommended settings for a fresh run with new dataset

```bash
python miniGPT/cli.py \
  --train YOUR_DATASET.txt \
  --epochs 100 \
  --samples 10000 \
  --context 32 \
  --embed_dim 256 \
  --num_blocks 4 \
  --num_heads 4 \
  --dropout 0.1 \
  --grad_clip 1.0 \
  --lr 0.0005 \
  --max_chars 2000000 \
  --simple_vocab \
  --save_every 10 \
  --save gpt_v2_b1.json
```

`--no_weight_tying` is intentionally omitted (weight tying is ON by default).