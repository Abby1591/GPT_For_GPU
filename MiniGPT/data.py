"""
data.py
=======
Dataset utilities for miniGPT: loading text files and building
(input, label) training pairs from a tokenized corpus.

Typical usage::

    from data import load_text, make_samples
    from tokenizer import CharTokenizer

    text     = load_text("wiki_dataset.txt", max_chars=500_000)
    tok      = CharTokenizer(text)
    encoded  = tok.encode(text)
    samples  = make_samples(encoded, context_size=8, tokenizer=tok)
    # samples[0] → ([0.0, 1.0, 0.0, ...], 14)
"""

from __future__ import annotations

from typing import List, Tuple

from tokenizer import CharTokenizer

# A single training sample: (flat one-hot feature vector, target class index)
Sample = Tuple[List[float], int]


def simplify_text(text: str) -> str:
    """
    Strip text down to a tiny vocabulary: lowercase a-z, space, and basic
    punctuation ( . , ! ? ' - ).  Everything else is dropped.

    This reduces vocab from ~267 chars to ~36, making the problem
    roughly 6x easier for small models.

    :param text: Raw input text.
    :return: Cleaned text with simplified vocabulary.

    **Example:**

    .. code-block:: python

        text = simplify_text("Hello, World! 123")
        # -> "hello, world! "
    """
    import re
    text = text.lower()
    text = re.sub(r"[^a-z .,!?'\-\n]", "", text)
    # Collapse multiple spaces/newlines into single space
    text = re.sub(r"[ \n]+", " ", text)
    return text.strip()


def load_text(path: str, max_chars: int = 500_000) -> str:
    """
    Read a plain-text file and return its contents as a string.

    Loading is capped at ``max_chars`` to prevent running out of memory
    on large datasets. Increase this value for better model quality at
    the cost of more RAM and longer training time.

    :param path: File-system path to the ``.txt`` dataset.
    :type path: str
    :param max_chars: Maximum number of characters to read.
        Defaults to ``500_000`` (~500 KB), which trains in a few minutes.
        Use ``2_000_000`` for better results if you have time.
    :type max_chars: int
    :return: The loaded text string.
    :rtype: str
    :raises FileNotFoundError: If ``path`` does not exist.

    **Example:**

    .. code-block:: python

        text = load_text("wiki_dataset.txt")
        text = load_text("wiki_dataset.txt", max_chars=1_000_000)
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read(max_chars)
    print(f"Loaded {len(text):,} characters from '{path}'")
    return text


def make_index_arrays(
    encoded:      List[int],
    context_size: int,
    max_samples:  int = 20_000,
):
    """
    Build X_idx (T, N) and Y_idx (T, N) integer arrays directly from the
    encoded corpus — no one-hot vectors, no Python sample list.

    This is 10-50x faster than make_samples() for large datasets because
    it never allocates one-hot floats that train() would immediately decode
    back to indices anyway.

    :return: Tuple of (X_idx, Y_idx) as numpy int32 arrays, shape (T, N).
    """
    import numpy as _np
    n              = len(encoded)
    total_possible = n - context_size - 1
    step           = max(1, total_possible // max_samples)
    indices        = list(range(0, total_possible, step))[:max_samples]
    N              = len(indices)

    arr    = _np.array(encoded, dtype=_np.int32)
    X_idx  = _np.zeros((context_size, N), dtype=_np.int32)
    Y_idx  = _np.zeros((context_size, N), dtype=_np.int32)

    for col, i in enumerate(indices):
        window         = arr[i : i + context_size + 1]   # T+1 tokens
        X_idx[:, col]  = window[:context_size]
        Y_idx[:-1, col] = window[1:context_size]
        Y_idx[-1,  col] = window[context_size]

    return X_idx, Y_idx


def make_samples(
    encoded:      List[int],
    context_size: int,
    tokenizer:    CharTokenizer,
    max_samples:  int = 20_000,
) -> List[Sample]:
    """
    Build supervised ``(features, label)`` training pairs from a token list.

    **How it works:**

    A sliding window of size ``context_size`` moves through the encoded
    corpus one step at a time. For each window position:

    - **Input** — the ``context_size`` characters in the window are each
      one-hot encoded and concatenated into a single flat vector of length
      ``context_size × vocab_size``.
    - **Label** — the integer index of the *next* character immediately
      after the window.

    To keep memory and training time manageable the window advances in
    steps larger than 1 when the corpus is longer than ``max_samples``
    allows, so samples are spread evenly across the full corpus.

    :param encoded: The full corpus as a list of integer token indices,
        as returned by ``CharTokenizer.encode()``.
    :type encoded: list[int]
    :param context_size: Number of preceding characters used as context.
        Larger values give the model more history but increase input
        dimensions and training time.  Good starting values: 6–12.
    :type context_size: int
    :param tokenizer: A fitted :class:`~tokenizer.CharTokenizer` used
        to produce one-hot vectors.
    :type tokenizer: CharTokenizer
    :param max_samples: Maximum number of training samples to create.
        More samples → better coverage but slower training.
        Defaults to ``20_000``.
    :type max_samples: int
    :return: List of ``(feature_vector, label_index)`` pairs ready to
        pass directly to ``NeuralNetwork.train()``.
    :rtype: list[tuple[list[float], int]]

    **Example:**

    .. code-block:: python

        from tokenizer import CharTokenizer
        from data import make_samples

        tok      = CharTokenizer("abcdef" * 100)
        encoded  = tok.encode("abcdef" * 100)
        samples  = make_samples(encoded, context_size=4, tokenizer=tok)

        features, label = samples[0]
        print(len(features))  # 4 × 6 = 24
        print(label)          # index of character at position 4
    """
    samples: List[Sample] = []
    n       = len(encoded)

    if n <= context_size:
        raise ValueError(
            f"Corpus length ({n}) must be greater than context_size ({context_size})."
        )

    # Step size: spread samples evenly across the corpus
    total_possible = n - context_size
    step           = max(1, total_possible // max_samples)

    for i in range(0, total_possible, step):
        context = encoded[i : i + context_size]
        target  = encoded[i + context_size]

        # Concatenate one-hot vectors for each context character
        features: List[float] = []
        for token_idx in context:
            features.extend(tokenizer.one_hot(token_idx))

        samples.append((features, target))

        if len(samples) >= max_samples:
            break

    return samples