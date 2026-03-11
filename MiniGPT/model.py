"""
model.py
========
The core ``MiniGPT`` class: builds, trains, generates text, and
saves/loads the character-level language model.

Typical usage::

    from model import MiniGPT

    # --- Train ---
    model = MiniGPT(context_size=8, hidden_layers=[256, 128])
    model.train("wiki_dataset.txt", epochs=5)
    model.save("gpt_weights.json")

    # --- Generate ---
    model = MiniGPT.load("gpt_weights.json")
    print(model.generate(prompt="Democracy is", length=300))
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from typing import List, Optional

import numpy as np

try:
    from Neural_Network import NeuralNetwork
except ImportError:
    print("ERROR: Neural_Network.py not found.")
    print("Place Neural_Network.py in the same folder as model.py.")
    sys.exit(1)

from tokenizer import CharTokenizer
from data import load_text, make_samples, simplify_text, make_index_arrays


class MiniGPT:
    """
    A character-level language model that uses :class:`NeuralNetwork` as its
    learning backbone.

    **Architecture overview:**

    .. code-block:: text

        Raw text
            │
            ▼
        CharTokenizer          (character → integer index)
            │
            ▼
        One-hot encoding       (index → flat binary vector)
            │  context_size × vocab_size floats
            ▼
        NeuralNetwork          (your feedforward net)
            │  hidden_layers with chosen activation
            ▼
        Softmax output         (probability over vocab_size characters)
            │
            ▼
        Sampled next character

    The model is trained as a multi-class classifier: given the last
    ``context_size`` characters, predict the next one.  At generation
    time the prediction is sampled repeatedly to produce new text.

    :param context_size: How many preceding characters the model sees
        before predicting the next one.  Larger = more coherent output
        but slower training (input size grows linearly).
        Good values: ``6`` (fast) – ``16`` (better quality).
    :type context_size: int

    :param hidden_layers: Neuron counts for each hidden layer of the
        underlying NeuralNetwork.  Deeper / wider networks learn better
        patterns but take longer to train.
        Default: ``[256, 128]``.
    :type hidden_layers: list[int]

    :param activation: Activation function for hidden layers.
        One of ``"relu"`` (default), ``"tanh"``, ``"sigmoid"``,
        ``"leaky_relu"``.
    :type activation: str

    :param learning_rate: Step size for each gradient update.
        Too high → unstable training.  Too low → very slow learning.
        Default: ``0.005``.
    :type learning_rate: float

    **Quick example — train and generate:**

    .. code-block:: python

        from model import MiniGPT

        model = MiniGPT(context_size=8, hidden_layers=[256, 128])
        model.train("wiki_dataset.txt", epochs=5)
        model.save("gpt_weights.json")

        text = model.generate(prompt="Science is", length=200)
        print(text)

    **Quick example — load and generate:**

    .. code-block:: python

        model = MiniGPT.load("gpt_weights.json")
        print(model.generate(prompt="History shows", length=300, temperature=0.7))
    """

    def __init__(
        self,
        context_size:  int             = 8,
        hidden_layers: Optional[List[int]] = None,
        activation:    str             = "relu",
        learning_rate: float           = 0.005,
        embed_dim:     int             = 64,
    ) -> None:
        self.context_size  = context_size
        self.hidden_layers = hidden_layers if hidden_layers is not None else [256, 128]
        self.activation    = activation
        self.learning_rate = learning_rate
        self.embed_dim     = embed_dim

        self.tokenizer: Optional[CharTokenizer] = None
        self.nn:        Optional[NeuralNetwork] = None

    # ------------------------------------------------------------------
    # Internal: build the NeuralNetwork
    # ------------------------------------------------------------------

    def _build(self, tokenizer: CharTokenizer) -> None:
        """
        Instantiate the :class:`NeuralNetwork` once the vocabulary size is known.

        Called automatically by :meth:`train`.  You do not normally need
        to call this yourself.

        :param tokenizer: A fitted :class:`~tokenizer.CharTokenizer`.
            Its ``size`` attribute determines the input and output
            dimensions of the network.
        :type tokenizer: CharTokenizer
        """
        self.tokenizer = tokenizer
        input_size     = self.context_size * tokenizer.size

        self.nn = NeuralNetwork(
            input_size    = input_size,
            hidden_layers = self.hidden_layers,
            output_size   = tokenizer.size,
            activation    = self.activation,
            learning_rate = self.learning_rate,
            use_embedding = True,
            vocab_size    = tokenizer.size,
            context_size  = self.context_size,
            embed_dim     = self.embed_dim,
        )
        self.nn.summary()

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(
        self,
        text_or_path: str,
        epochs:       int   = 3,
        max_samples:  int   = 20_000,
        max_chars:    int   = 500_000,
        log_every:    int   = 1,
        simple_vocab: bool  = False,
        save_every:   int   = 0,
        save_path:    str   = "checkpoint.json",
    ) -> None:
        """
        Train the model on a text string or a path to a ``.txt`` file.

        Pass either raw text or a file path — the method detects which
        you mean automatically (if the string ends in ``.txt`` and the
        file exists, it is loaded from disk).

        **What happens internally:**

        1. Text is loaded and tokenized with :class:`~tokenizer.CharTokenizer`.
        2. Up to ``max_samples`` sliding-window training pairs are built.
        3. :meth:`NeuralNetwork.train` runs for ``epochs`` passes.

        :param text_or_path: Either a raw text string **or** a file path
            to a ``.txt`` dataset (e.g. ``"wiki_dataset.txt"``).
        :type text_or_path: str

        :param epochs: Full passes over the training samples.
            More epochs → lower loss → better quality output, but slower.
            Start with ``3–5`` and increase if output is incoherent.
        :type epochs: int

        :param max_samples: Maximum number of training examples to
            generate from the corpus.  Caps memory and training time.
            ``20_000`` is a good starting point; try ``50_000`` for
            noticeably better results.
        :type max_samples: int

        :param max_chars: If loading from a file, only read this many
            characters.  Ignored when ``text_or_path`` is raw text.
        :type max_chars: int

        :param log_every: Print the total loss every N epochs.
            Set to ``0`` to silence all output.
        :type log_every: int

        **Example — train from file:**

        .. code-block:: python

            model = MiniGPT(context_size=8, hidden_layers=[256, 128])
            model.train("wiki_dataset.txt", epochs=5, max_samples=30_000)

        **Example — train from a string:**

        .. code-block:: python

            model = MiniGPT(context_size=6, hidden_layers=[64, 32])
            model.train("hello world " * 500, epochs=20)
        """
        # Accept either a file path or a raw string
        if os.path.isfile(text_or_path):
            text = load_text(text_or_path, max_chars=max_chars)
        else:
            text = text_or_path

        if simple_vocab:
            before = len(set(text))
            text   = simplify_text(text)
            after  = len(set(text))
            print(f"Simple vocab: reduced from {before} → {after} unique chars")

        print("\nTokenizing corpus...")
        tokenizer = CharTokenizer(text)
        self._build(tokenizer)

        encoded = tokenizer.encode(text)
        print(f"Vocab size : {tokenizer.size} characters")
        print(f"Corpus     : {len(encoded):,} tokens")

        print(f"\nBuilding up to {max_samples:,} training samples...")
        index_data = make_index_arrays(encoded, self.context_size, max_samples)
        N = index_data[0].shape[1]
        print(f"Training samples : {N:,}")
        print(f"Input dimension  : {self.context_size} × {tokenizer.size} = "
              f"{self.context_size * tokenizer.size}")

        print(f"\nTraining for {epochs} epoch(s)...\n")
        t0 = time.time()
        self.nn.train(index_data, epochs=epochs, log_every=log_every,
                      save_every=save_every, save_path=save_path)
        print(f"\nTraining finished in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt:      str   = "",
        length:      int   = 200,
        temperature: float = 0.8,
    ) -> str:
        """
        Generate new text autoregressively from an optional seed prompt.

        **How autoregressive generation works:**

        At each step the model:

        1. Takes the last ``context_size`` characters as input.
        2. One-hot encodes them into a flat feature vector.
        3. Runs a forward pass through the NeuralNetwork.
        4. Applies temperature scaling to the output probabilities.
        5. Samples the next character from the scaled distribution.
        6. Appends it to the output and repeats.

        :param prompt: Seed text to start generation from.
            Can be any string.  Characters not in the vocabulary are
            silently dropped.  If empty, generation starts from a random
            character.
        :type prompt: str

        :param length: Number of *new* characters to generate (the
            prompt itself is not counted).
        :type length: int

        :param temperature: Controls how random the output is.

            - ``< 1.0`` (e.g. ``0.5``) — more focused and repetitive.
              The model favours its top predictions strongly.
            - ``1.0`` — unmodified probabilities.
            - ``> 1.0`` (e.g. ``1.5``) — more diverse and creative, but
              also more likely to produce nonsense.

            A value of ``0.7–0.9`` is usually a good starting point.
        :type temperature: float

        :return: The prompt (if any) followed by ``length`` generated
            characters.
        :rtype: str
        :raises RuntimeError: If called before :meth:`train` or
            :meth:`load`.

        **Example:**

        .. code-block:: python

            # Focused output
            print(model.generate("Democracy", length=300, temperature=0.6))

            # Creative / varied output
            print(model.generate("Science", length=300, temperature=1.2))

            # No seed — starts from a random character
            print(model.generate(length=200))
        """
        if self.nn is None or self.tokenizer is None:
            raise RuntimeError(
                "Model is not ready. Call train() or MiniGPT.load() first."
            )

        tok = self.tokenizer

        # Build the initial context window from the prompt
        seed_indices = tok.encode(prompt) if prompt else [random.randint(0, tok.size - 1)]

        # Truncate to the last context_size characters, left-pad if shorter
        ctx = seed_indices[-self.context_size:]
        while len(ctx) < self.context_size:
            ctx = [0] + ctx

        generated = list(prompt) if prompt else []

        for _ in range(length):
            # Flatten one-hot encodings of each context character
            features = []
            for idx in ctx:
                features.extend(tok.one_hot(idx))

            # Forward pass → raw probabilities
            _, _, probs = self.nn.predict(features)

            # Temperature scaling
            if temperature != 1.0:
                logits = np.log(np.array(probs) + 1e-9) / temperature
                probs  = np.exp(logits - logits.max())
                probs /= probs.sum()

            # Sample next character
            next_idx = int(np.random.choice(len(probs), p=probs))
            generated.append(tok.idx2ch[next_idx])

            # Slide the context window one step to the right
            ctx = ctx[1:] + [next_idx]

        return "".join(generated)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, weights_path: str = "gpt_weights.json") -> None:
        """
        Persist the trained model to disk as three JSON files.

        The three files saved are:

        - ``<weights_path>``               — NeuralNetwork weights & biases.
        - ``<weights_path>_tokenizer.json``— vocabulary list.
        - ``<weights_path>_config.json``   — MiniGPT hyper-parameters.

        All three files are required to reload the model with
        :meth:`load`.

        :param weights_path: Base path for the weights file.
            Sibling files are derived from this name automatically.
            Defaults to ``"gpt_weights.json"``.
        :type weights_path: str

        **Example:**

        .. code-block:: python

            model.save("my_model.json")
            # Creates: my_model.json
            #          my_model_tokenizer.json
            #          my_model_config.json
        """
        if self.nn is None:
            print("Nothing to save — model has not been trained.")
            return

        self.nn.save_weights(weights_path)

        tok_path = weights_path.replace(".json", "_tokenizer.json")
        self.tokenizer.save(tok_path)
        print(f"Tokenizer saved to '{tok_path}'.")

        cfg_path = weights_path.replace(".json", "_config.json")
        with open(cfg_path, "w") as f:
            json.dump({
                "context_size":  self.context_size,
                "hidden_layers": self.hidden_layers,
                "activation":    self.activation,
                "learning_rate": self.learning_rate,
                "embed_dim":     self.embed_dim,
            }, f, indent=2)
        print(f"Config saved to '{cfg_path}'.")

    @classmethod
    def load(cls, weights_path: str = "gpt_weights.json") -> "MiniGPT":
        """
        Load a previously saved model from disk.

        Reads the three files created by :meth:`save` and fully
        reconstructs the model, including vocabulary and weights.

        :param weights_path: Path to the main weights JSON file.
            The tokenizer and config files must exist in the same
            directory with the expected ``_tokenizer.json`` and
            ``_config.json`` suffixes.
        :type weights_path: str
        :return: A fully loaded and ready-to-use ``MiniGPT`` instance.
        :rtype: MiniGPT
        :raises FileNotFoundError: If any of the three required files
            are missing.

        **Example:**

        .. code-block:: python

            model = MiniGPT.load("gpt_weights.json")
            print(model.generate("Civil rights", length=200))
        """
        cfg_path = weights_path.replace(".json", "_config.json")
        tok_path = weights_path.replace(".json", "_tokenizer.json")

        for p in (weights_path, tok_path, cfg_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Required model file not found: '{p}'\n"
                    "Make sure all three files (weights, tokenizer, config) are present."
                )

        with open(cfg_path) as f:
            cfg = json.load(f)

        model = cls(
            context_size  = cfg["context_size"],
            hidden_layers = cfg["hidden_layers"],
            activation    = cfg["activation"],
            learning_rate = cfg["learning_rate"],
            embed_dim     = cfg.get("embed_dim", 64),
        )
        model.tokenizer = CharTokenizer.load(tok_path)

        input_size = model.context_size * model.tokenizer.size
        model.nn   = NeuralNetwork(
            input_size    = input_size,
            hidden_layers = model.hidden_layers,
            output_size   = model.tokenizer.size,
            activation    = model.activation,
            learning_rate = model.learning_rate,
            use_embedding = True,
            vocab_size    = model.tokenizer.size,
            context_size  = model.context_size,
            embed_dim     = model.embed_dim,
        )
        model.nn.load_weights(weights_path)
        print("miniGPT loaded successfully.")
        return model

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "trained" if self.nn is not None else "untrained"
        return (
            f"MiniGPT(context={self.context_size}, "
            f"hidden={self.hidden_layers}, "
            f"activation='{self.activation}', "
            f"lr={self.learning_rate}, "
            f"status={status})"
        )