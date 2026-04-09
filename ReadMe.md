# miniGPT

A character-level GPT-2 inspired transformer built from scratch in pure NumPy/CuPy.
No PyTorch. No TensorFlow. Every forward pass, backward pass, and optimiser update
is hand-written. Runs on a free Google Colab T4 GPU.

---

## Project Structure

```
GPT/
├── Neural_Network.py          # Core transformer: forward, backward, Adam, save/load
├── build_dataset.py           # Dataset crawler: Gutenberg, Wikipedia, Wikiquote
├── miniGPT/
│   ├── __init__.py
│   ├── tokenizer.py           # CharTokenizer: encode, decode, one_hot, save/load
│   ├── data.py                # load_text, make_samples, make_index_arrays
│   ├── model.py               # MiniGPT: train, generate, save, load
│   └── cli.py                 # Command-line interface
├── tests/
│   ├── test_tokenizer.py
│   ├── test_data.py
│   ├── test_neural_network.py
│   ├── test_model.py
│   ├── test_build_dataset.py
│   ├── test_integration.py
│   ├── test_performance.py
│   └── run_all_tests.py
└── README.md
```

---

## Architecture

```
Input tokens (B, T)
  -> token embedding (vocab, D)  +  positional embedding (T, D)
  -> embedding dropout
  -> N x Transformer Block
  |    pre-LayerNorm
  |    -> multi-head causal self-attention (H heads, d_h = D/H each)
  |    -> dropout -> residual add
  |    pre-LayerNorm
  |    -> feed-forward: D -> 4D (ReLU) -> D
  |    -> dropout -> residual add
  -> final LayerNorm
  -> output projection at ALL T positions  (B, T, vocab)
  -> softmax -> probabilities
```

### Key design choices

| Feature | What it does |
|---------|-------------|
| Causal mask | Each token only attends to itself and earlier positions. Required for autoregressive generation. |
| All-positions training | Every token position predicts the next token simultaneously. T× more gradient signal per sample. |
| Fused QKV | Q, K, V projected with one (D, 3D) matmul instead of three. ~3× faster on GPU. |
| Multi-head attention | H independent attention patterns in parallel. Each head specialises on different relationships. |
| Pre-LayerNorm | Normalise before attention and FF (GPT-2 style). Trains more stably than post-norm at depth. |
| Weight tying | Output projection reuses embedding.T. Halves those parameters; input and output spaces aligned. |
| Residual scaling | W2 init scaled by 1/sqrt(2*num_blocks). Prevents residual stream growing with depth. |
| Gradient clipping | Global L2 norm capped before Adam update. Prevents bad batches from blowing up weights. |
| Dropout | Applied after attention output, FF output, and initial embedding. Reduces overfitting. |
| Atomic saves | Writes to temp file then renames. Colab disconnects never corrupt checkpoints. |
| Compact JSON | No indentation. ~4× smaller files (~50MB vs ~200MB). |

---

## Quick Start

### Install GPU backend (Colab)
```bash
pip install cupy-cuda12x   # CUDA 12.x (Colab T4)
```

### Build a dataset
```bash
python build_dataset.py
# Output: diverse_dataset.txt (~2.2M chars)
# Sources: Gutenberg books (40%), Wikipedia (35%), Simple Wikipedia (15%), Wikiquote (10%)
# LGBTQ+ articles always guaranteed in output regardless of other source sizes
```

### Train from scratch
```bash
python miniGPT/cli.py \
  --train diverse_dataset.txt \
  --epochs 100 \
  --samples 30000 \
  --context 32 \
  --embed_dim 256 \
  --num_blocks 4 \
  --num_heads 4 \
  --dropout 0.1 \
  --lr 0.0005 \
  --max_chars 2200000 \
  --simple_vocab \
  --save_every 10 \
  --save gpt_v2_b1.json
```

### Resume training
```bash
python miniGPT/cli.py \
  --train diverse_dataset.txt \
  --resume gpt_v2_b1.json \
  --epochs 100 \
  --samples 30000 \
  --max_chars 2200000 \
  --simple_vocab \
  --save_every 10 \
  --save gpt_v2_b2.json
```

When resuming, Adam momentum state is fully restored. No warmup repeat.

### Generate text
```bash
python miniGPT/cli.py \
  --load gpt_v2_b2.json \
  --prompt "the " \
  --length 300 \
  --temperature 0.7
```

### Override learning rate on resume
```bash
# Safe resume (lr from checkpoint):
python miniGPT/cli.py --train data.txt --resume weights.json --epochs 100 ...

# Force higher lr (e.g. to escape plateau):
python miniGPT/cli.py --train data.txt --resume weights.json --force_lr 0.0003 ...

# Force higher lr AND wipe Adam momentum (use when boosting lr by more than ~2x):
python miniGPT/cli.py --train data.txt --resume weights.json --force_lr 0.0003 --reset_adam ...
```

WARNING: using `--force_lr` without `--reset_adam` when boosting lr significantly
causes loss spikes because old momentum was built at the lower lr.

---

## All CLI Flags

### Mode
| Flag | Description |
|------|-------------|
| `--train FILE` | Path to .txt dataset |
| `--resume FILE` | Load checkpoint and continue training |
| `--load FILE` | Load for generation only |
| `--save FILE` | Output path (default: gpt_weights.json) |

### Training
| Flag | Default | Description |
|------|---------|-------------|
| `--epochs N` | 100 | Training epochs |
| `--samples N` | 20000 | Max training samples per epoch |
| `--context N` | 12 | Context window size (tokens) |
| `--embed_dim N` | 64 | Embedding / hidden dimension D |
| `--num_blocks N` | 2 | Transformer blocks |
| `--num_heads N` | 4 | Attention heads (embed_dim must be divisible) |
| `--dropout F` | 0.0 | Dropout rate (0=disabled, try 0.1) |
| `--no_weight_tying` | off | Disable weight tying (separate Wout matrix) |
| `--grad_clip F` | 1.0 | Global gradient norm clip (0=disabled) |
| `--batch_size N` | 1024 | Samples per gradient step |
| `--lr F` | 0.001 | Peak learning rate |
| `--max_chars N` | 500000 | Characters to read from file |
| `--simple_vocab` | off | Reduce to a-z + basic punctuation (~33 chars) |
| `--save_every N` | 0 | Checkpoint every N epochs (0=disabled) |
| `--log_every N` | 1 | Print loss every N epochs |
| `--force_lr F` | — | Hard override lr even when Adam state exists |
| `--reset_adam` | off | Wipe Adam momentum on resume |

### Generation
| Flag | Default | Description |
|------|---------|-------------|
| `--prompt TEXT` | "" | Seed text |
| `--length N` | 300 | Characters to generate |
| `--temperature F` | 0.6 | < 1 = focused, > 1 = creative |

---

## Adaptive Learning Rate

The scheduler has three phases that run automatically during training:

**Warmup** (fresh training only, epochs 0-4):
Linearly ramps lr from lr/5 to lr_max. Prevents large steps before Adam
momentum estimates have warmed up from zero. Automatically skipped on resume.

**Bounce reduction** (loss increases 2 consecutive epochs):
`lr *= 0.7` — model overshot a minimum, smaller steps needed.

**Plateau reduction** (no new best loss for 5 epochs):
`lr *= 0.7` — model stuck in flat region, finer steps may help.

Floor: `lr_max / 5`. The final lr is saved to the checkpoint so the next
resume starts at exactly the right value.

---

## Dataset Builder

`build_dataset.py` crawls four public sources with no API keys required:

| Source | Share | What it provides |
|--------|-------|-----------------|
| Project Gutenberg | 40% | Long-form prose: novels, philosophy, history |
| Wikipedia | 35% | Factual articles across all topics |
| Simple Wikipedia | 15% | Plain English, shorter sentences |
| Wikiquote | 10% | Short quotes from diverse voices |

**LGBTQ+ guarantee:** Articles on Stonewall, Harvey Milk, Marsha P. Johnson,
Sylvia Rivera, transgender history, same-sex marriage and 25 more are fetched
first with no percentage cap — they are always in the dataset.

**Rate limiting:** 5 second delay between requests, exponential backoff (15s,
30s, 60s...) on HTTP 429 responses, 60 second cooldown every 200 requests.

```bash
python build_dataset.py                              # default 2.2M chars
python build_dataset.py --target_chars 5000000       # larger dataset
python build_dataset.py --no_gutenberg               # skip Gutenberg
python build_dataset.py --no_wikipedia               # skip Wikipedia (if banned)
```

---

## Training on Colab

### Recommended workflow
```python
# Mount Drive for large file transfers (avoid browser upload corruption)
from google.colab import drive
drive.mount('/content/drive')

# Copy dataset and weights from Drive
import shutil
shutil.copy('/content/drive/MyDrive/diverse_dataset.txt', '/content/')
shutil.copy('/content/drive/MyDrive/gpt_v2_b1.json', '/content/')

# After training, save back to Drive
shutil.copy('gpt_v2_b2.json', '/content/drive/MyDrive/')
```

### Keep training running if browser disconnects
```python
import subprocess, sys
proc = subprocess.Popen(
    [sys.executable, "miniGPT/cli.py",
     "--train", "diverse_dataset.txt",
     "--resume", "gpt_v2_b1.json",
     "--epochs", "100", "--samples", "30000",
     "--max_chars", "2200000", "--simple_vocab",
     "--save_every", "10", "--save", "gpt_v2_b2.json"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1,
)
for line in proc.stdout:
    print(line, end="", flush=True)
```

### Performance (Colab T4, context=32, embed=256, 30k samples)
- ~55 seconds per epoch
- ~92 minutes per 100 epochs  
- ~2.5 hours free GPU per day

---

## Loss Reference

| equiv | Quality |
|-------|---------|
| 3.50 | Random (theoretical max = ln 33 for 33-char vocab) |
| 2.60 | Spaces and common letter pairs correct |
| 2.30 | Real words appearing ("the", "and", "in") |
| 2.00 | Short word sequences, punctuation correct |
| 1.80 | Short readable phrases (expected target with v2 arch) |
| 1.50 | Grammatical sentences |
| 1.00 | Coherent paragraphs |

`equiv = total_loss / (n_samples * context_size)` -- average cross-entropy
per (sample, position) pair. Comparable across runs with different context
lengths, sample counts, or batch sizes.

**Training history on wiki_dataset.txt:**

| Checkpoint | Epochs total | equiv | Notes |
|-----------|-------------|-------|-------|
| v1-v4 | 400 | ~2.60 | Warmup bug wasted epochs on every resume |
| v5-v11 | 1100 | 2.08 | Bugs fixed, clean resuming |
| v12-v13 | 1300 | 2.04 | Approaching capacity limit of old architecture |
| v2 target | — | ~1.80 | New architecture + diverse dataset |

---

## Tests

```bash
python tests/run_all_tests.py              # all 190 tests
python tests/run_all_tests.py --quick      # tokenizer only (~1s)
python tests/run_all_tests.py --module neural_network
python tests/run_all_tests.py -v           # verbose output
```

| Module | Tests | Covers |
|--------|-------|--------|
| test_tokenizer.py | 31 | Vocabulary, encode, decode, one_hot, save/load |
| test_data.py | 30 | simplify_text, load_text, make_samples, make_index_arrays |
| test_neural_network.py | 42 | Construction, LayerNorm, dropout, causal mask, forward pass, Adam, save/load, training |
| test_model.py | 23 | MiniGPT train, generate, save, load, resume |
| test_build_dataset.py | 37 | Text cleaning, HTTP helpers, data lists, LGBTQ+ guarantee |
| test_integration.py | 14 | Full pipelines end-to-end |
| test_performance.py | 13 | Speed and scalability benchmarks |

---

## Bugs Fixed

| Bug | Symptom | Fix |
|-----|---------|-----|
| `_build()` recreated nn on resume | Wiped weights + Adam state every resume | Check `if self.nn is None` before calling `_build()` |
| Duplicate `_resuming` assignment | Warmup triggered on every resume | Removed the duplicate assignment inside the epoch loop |
| Adam state not saved (v1-v3) | No smooth resume possible | Added full Adam m/v buffer serialisation to save file |
| lr not persisted after training | Resume always jumped back to original lr causing 15-epoch spike | `self.learning_rate = epoch_lr` at end of `train()` |
| `d_h` variable shadowing | `TypeError` crash in backward pass with multi-head attention | Renamed FF gradient variable to `d_h_grad` |
| Wrong reshape in output backward | `ValueError` on any model where vocab != embed_dim | `delta @ Wout.T` directly, no reshape needed |
| Checkpoint corruption on disconnect | Partial JSON files unloadable | Atomic write: temp file + `os.replace()` |
| Wikipedia category API returning 0 | `urllib.parse` not imported at module level | Moved import to top of file |
| `--force_lr` without `--reset_adam` | Loss spike of ~0.5 equiv over 15 epochs | Added `--reset_adam` flag; documented the interaction |

---

## What's Next

The v2 architecture (LayerNorm + multi-head attention + dropout + weight tying)
is ready. The diverse dataset is being built. Expected improvement path:

1. Train v2 from scratch on diverse_dataset.txt (~3 Colab sessions to match v13 quality)
2. Continue training until ~1.80 equiv (readable phrases)
3. Possible future upgrades: cosine LR schedule, BPE tokenizer, larger context