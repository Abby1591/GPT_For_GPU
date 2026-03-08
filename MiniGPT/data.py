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
