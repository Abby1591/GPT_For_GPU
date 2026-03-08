"""
tokenizer.py
============
Character-level tokenizer for miniGPT.

Converts raw text into integer indices and back, and provides
one-hot encoding for use as neural network inputs.

Typical usage::

    from tokenizer import CharTokenizer

    tok = CharTokenizer("hello world")
    indices = tok.encode("hello")   # [3, 2, 7, 7, 8]
    text    = tok.decode(indices)   # "hello"
    vec     = tok.one_hot(3)        # [0,0,0,1,0,...]

    tok.save("vocab.json")
    tok2 = CharTokenizer.load("vocab.json")
"""

from __future__ import annotations

import json
from typing import List


class CharTokenizer:
    """
    Maps every unique character in a corpus to an integer index and back.

    **Why character-level?**

    Word-level tokenizers need large vocabularies (50,000+ tokens) and
    struggle with unknown words. A character-level tokenizer only needs
    ~50–100 entries to cover all printable ASCII, making it ideal for a
    lightweight model like miniGPT.

    :param text: The full training corpus. Every unique character found
        in this string becomes a vocabulary entry.
    :type text: str

    **Attributes:**

    - ``vocab`` (*list[str]*) — sorted list of unique characters.
    - ``size`` (*int*) — number of characters in the vocabulary.
    - ``ch2idx`` (*dict[str, int]*) — character → index lookup.
    - ``idx2ch`` (*dict[int, str]*) — index → character lookup.

    **Example:**

    .. code-block:: python

        tok = CharTokenizer("hello world")
        print(tok.size)          # 8  (h,e,l,o,' ',w,r,d)
        print(tok.encode("he"))  # [3, 2]
        print(tok.decode([3,2])) # "he"
    """

    def __init__(self, text: str) -> None:
        chars       = sorted(set(text))
        self.vocab  = chars
        self.size   = len(chars)
        self.ch2idx = {c: i for i, c in enumerate(chars)}
        self.idx2ch = {i: c for i, c in enumerate(chars)}

    # ------------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> List[int]:
        """
        Convert a string into a list of integer token indices.

        Characters not present in the vocabulary are silently skipped.

        :param text: The string to encode.
        :type text: str
        :return: List of integer indices, one per known character.
        :rtype: list[int]

        **Example:**

        .. code-block:: python

            tok = CharTokenizer("abcde")
            tok.encode("ace")   # [0, 2, 4]
            tok.encode("axe")   # [0, 4]  ('x' skipped — not in vocab)
        """
        return [self.ch2idx[c] for c in text if c in self.ch2idx]

    def decode(self, indices: List[int]) -> str:
        """
        Convert a list of integer indices back into a string.

        Unknown indices are replaced with ``'?'``.

        :param indices: List of token indices to decode.
        :type indices: list[int]
        :return: Reconstructed string.
        :rtype: str

        **Example:**

        .. code-block:: python

            tok = CharTokenizer("abcde")
            tok.decode([0, 2, 4])   # "ace"
            tok.decode([0, 99])     # "a?"  (99 is unknown)
        """
        return "".join(self.idx2ch.get(i, "?") for i in indices)

    # ------------------------------------------------------------------
    # One-hot encoding
    # ------------------------------------------------------------------

    def one_hot(self, idx: int) -> List[float]:
        """
        Return a one-hot vector for a single character index.

        The returned list has length ``vocab_size``, with ``1.0`` at
        position ``idx`` and ``0.0`` everywhere else. This is the format
        expected by the NeuralNetwork input layer.

        :param idx: The character index to encode. Must be in
            ``0 ≤ idx < vocab_size``.
        :type idx: int
        :return: One-hot float vector of length ``vocab_size``.
        :rtype: list[float]

        **Example:**

        .. code-block:: python

            tok = CharTokenizer("abc")   # vocab size = 3
            tok.one_hot(0)  # [1.0, 0.0, 0.0]
            tok.one_hot(2)  # [0.0, 0.0, 1.0]
        """
        vec      = [0.0] * self.size
        vec[idx] = 1.0
        return vec

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save the vocabulary to a JSON file.

        Only the character list is stored; all lookup dicts are rebuilt
        on load. The file is human-readable.

        :param path: File path to write to (e.g. ``"vocab.json"``).
        :type path: str

        **Example:**

        .. code-block:: python

            tok.save("my_vocab.json")
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        """
        Load a previously saved tokenizer from a JSON file.

        Reconstructs all lookup dicts from the stored vocab list.
        The returned object behaves identically to one created from text.

        :param path: Path to the JSON file created by :meth:`save`.
        :type path: str
        :return: A fully initialised ``CharTokenizer``.
        :rtype: CharTokenizer
        :raises FileNotFoundError: If ``path`` does not exist.

        **Example:**

        .. code-block:: python

            tok = CharTokenizer.load("my_vocab.json")
            tok.encode("hello")
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Bypass __init__ so we don't need a text string
        tok         = cls.__new__(cls)
        tok.vocab   = data["vocab"]
        tok.size    = len(tok.vocab)
        tok.ch2idx  = {c: i for i, c in enumerate(tok.vocab)}
        tok.idx2ch  = {i: c for i, c in enumerate(tok.vocab)}
        return tok

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the vocabulary size (``len(tok)``)."""
        return self.size

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.size})"
