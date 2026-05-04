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

    # --- Generate (tools on by default when trained with tool data) ---
    model = MiniGPT.load("gpt_weights.json")
    print(model.generate(prompt="Democracy is", length=300))

    # --- Opt out of tools ---
    print(model.generate(prompt="Democracy is", tool_registry=None))

    # --- Skip specific tools ---
    from tool_definitions import TOOL_REGISTRY
    print(model.generate(
        prompt       = "5 km equals",
        skip_tools   = {"search", "lookup"},
    ))
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
from data import load_text, simplify_text, make_index_arrays


class MiniGPT:
    """
    A character-level language model that uses :class:`NeuralNetwork` as its
    learning backbone.

    **Architecture overview:**

    .. code-block:: text

        Raw text
            |
            v
        CharTokenizer          (character -> integer index)
            |
            v
        One-hot encoding       (index -> flat binary vector)
            |  context_size x vocab_size floats
            v
        NeuralNetwork          (your feedforward net)
            |  hidden_layers with chosen activation
            v
        Softmax output         (probability over vocab_size characters)
            |
            v
        Sampled next character

    The model is trained as a multi-class classifier: given the last
    ``context_size`` characters, predict the next one.  At generation
    time the prediction is sampled repeatedly to produce new text.

    :param context_size: How many preceding characters the model sees
        before predicting the next one.  Larger = more coherent output
        but slower training (input size grows linearly).
        Good values: ``6`` (fast) - ``16`` (better quality).
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
        Too high -> unstable training.  Too low -> very slow learning.
        Default: ``0.005``.
    :type learning_rate: float

    **Quick example -- train and generate:**

    .. code-block:: python

        from model import MiniGPT

        model = MiniGPT(context_size=8, hidden_layers=[256, 128])
        model.train("wiki_dataset.txt", epochs=5)
        model.save("gpt_weights.json")

        text = model.generate(prompt="Science is", length=200)
        print(text)

    **Quick example -- load and generate:**

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
        batch_size:    int             = 1024,
        num_blocks:    int             = 2,
        num_heads:     int             = 4,
        dropout:       float           = 0.0,
        weight_tying:  bool            = True,
        grad_clip:     float           = 1.0,
    ) -> None:
        self.context_size  = context_size
        self.hidden_layers = hidden_layers if hidden_layers is not None else [256, 128]
        self.activation    = activation
        self.learning_rate = learning_rate
        self.embed_dim     = embed_dim
        self.batch_size    = batch_size
        self.num_blocks    = num_blocks
        self.num_heads     = num_heads
        self.dropout       = dropout
        self.weight_tying  = weight_tying
        self.grad_clip     = grad_clip

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
            batch_size    = self.batch_size,
            num_blocks    = self.num_blocks,
            num_heads     = self.num_heads,
            dropout       = self.dropout,
            weight_tying  = self.weight_tying,
            grad_clip     = self.grad_clip,
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
        Pass either raw text or a file path -- the method detects which
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
            More epochs -> lower loss -> better quality output, but slower.
            Start with ``3-5`` and increase if output is incoherent.
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

        **Example -- train from file:**

        .. code-block:: python

            model = MiniGPT(context_size=8, hidden_layers=[256, 128])
            model.train("wiki_dataset.txt", epochs=5, max_samples=30_000)

        **Example -- train from a string:**

        .. code-block:: python

            model = MiniGPT(context_size=6, hidden_layers=[64, 32])
            model.train("hello world " * 500, epochs=20)
        """
        # Accept either a file path or a raw string
        if os.path.isfile(text_or_path):
            text = load_text(text_or_path, max_chars=max_chars)
        elif text_or_path.endswith(".txt"):
            raise FileNotFoundError(f"Dataset not found: {text_or_path}")
        else:
            text = text_or_path

        if simple_vocab:
            before = len(set(text))
            text   = simplify_text(text)
            after  = len(set(text))
            print(f"Simple vocab: reduced from {before} -> {after} unique chars")

        print("\nTokenizing corpus...")
        tokenizer = CharTokenizer(text)
        if self.nn is None:
            # Fresh training: build a new NeuralNetwork with the known vocab size
            self._build(tokenizer)
        else:
            # Resuming: keep the existing nn (weights + Adam state intact),
            # just update the tokenizer reference and reprint the summary.
            self.tokenizer = tokenizer
            self.nn.summary()

        encoded = tokenizer.encode(text)
        print(f"Vocab size : {tokenizer.size} characters")
        print(f"Corpus     : {len(encoded):,} tokens")

        print(f"\nBuilding up to {max_samples:,} training samples...")
        index_data = make_index_arrays(encoded, self.context_size, max_samples)
        N = index_data[0].shape[1]
        print(f"Training samples : {N:,}")
        print(f"Input dimension  : {self.context_size} x {tokenizer.size} = "
              f"{self.context_size * tokenizer.size}")

        # Write tokenizer + config BEFORE training starts so that any
        # mid-training checkpoint saved by save_every is immediately loadable
        # via MiniGPT.load().  Both files are static -- they never change
        # during training -- so writing them once up-front is safe.
        if save_every and save_path:
            tok_path = save_path.replace(".json", "_tokenizer.json")
            cfg_path = save_path.replace(".json", "_config.json")
            if not os.path.exists(tok_path):
                tokenizer.save(tok_path)
                print(f"Checkpoint header saved: '{tok_path}'")
            if not os.path.exists(cfg_path):
                with open(cfg_path, "w") as _f:
                    json.dump({
                        "context_size":  self.context_size,
                        "hidden_layers": self.hidden_layers,
                        "activation":    self.activation,
                        "learning_rate": self.learning_rate,
                        "embed_dim":     self.embed_dim,
                        "num_blocks":    self.num_blocks,
                        "num_heads":     self.num_heads,
                        "dropout":       self.dropout,
                        "weight_tying":  self.weight_tying,
                        "grad_clip":     self.grad_clip,
                    }, _f, indent=2)
                print(f"Checkpoint header saved: '{cfg_path}'")

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
        prompt:         str              = "",
        length:         int              = 200,
        temperature:    float            = 0.8,
        tool_registry:  "Optional[dict]" = "DEFAULT",
        skip_tools:     "Optional[set]"  = None,
        max_tool_calls: int              = 8,
        verbose:        bool             = True,
    ) -> str:
        """
        Generate new text autoregressively from an optional seed prompt.

        Pass ``tool_registry`` to enable Toolformer-style interleaved tool
        execution; leave it as ``None`` (the default) for plain generation.

        **When NOT to use tools**

        - The model was trained on plain text only (no ``[TOOL:...]``
          patterns in the dataset) — tool delimiters will never appear in
          the output so there is no benefit.
        - You want deterministic / reproducible output for tests or evals.
        - You are generating very short snippets where a tool call would
          dominate the output.
        - You are benchmarking speed — the tool path has small overhead
          from the pattern scanner even when no calls fire.

        In all other cases it is safe to pass a registry; if the model was
        trained with tool data it will call tools when useful, and if not
        it simply won't emit the delimiter pattern.

        **How autoregressive generation works**

        At each step the model takes the last ``context_size`` characters,
        runs a forward pass, applies temperature scaling, and samples the
        next character.  When ``tool_registry`` is supplied, the output
        tail is scanned after each token; a complete ``[TOOL:name|arg]``
        pattern triggers the executor and injects ``[RESULT:...]`` back
        into context before generation continues.

        :param prompt: Seed text.  Characters absent from the vocabulary
            are silently dropped.  Empty → start from a random character.
        :type prompt: str
        :param length: New characters to generate (prompt not counted).
            Injected tool result tokens do not count toward this budget.
        :type length: int
        :param temperature: Sampling temperature.
            ``< 1.0`` focused, ``1.0`` raw, ``> 1.0`` creative/noisy.
            ``0.6–0.9`` is a good starting range.
        :type temperature: float
        :param tool_registry: ``{name: ToolDef}`` from
            ``tool_definitions.TOOL_REGISTRY``, or a subset.
            ``None`` (default) disables tool use entirely.
        :type tool_registry: dict[str, ToolDef] | None
        :param max_tool_calls: Hard cap on tool invocations per call.
            Ignored when ``tool_registry`` is ``None``.
        :type max_tool_calls: int
        :param verbose: Print a summary line for each tool call when
            tools are active.  Ignored otherwise.
        :type verbose: bool
        :return: Prompt (if any) followed by generated characters.
        :rtype: str
        :raises RuntimeError: If called before :meth:`train` / :meth:`load`.

        **Examples:**

        .. code-block:: python

            # Plain generation — no tools needed
            print(model.generate("Democracy", length=300, temperature=0.6))

            # All tools enabled (model must have been trained with tool data)
            from tool_definitions import TOOL_REGISTRY
            print(model.generate(
                "The square root of 144 is",
                tool_registry = TOOL_REGISTRY,
            ))

            # Single tool
            print(model.generate(
                "5 km equals",
                tool_registry = {"convert": TOOL_REGISTRY["convert"]},
            ))
        """
        from tool_definitions import TOOL_REGISTRY as _DEFAULT_REGISTRY  # type: ignore

        if self.nn is None or self.tokenizer is None:
            raise RuntimeError(
                "Model is not ready. Call train() or MiniGPT.load() first."
            )

        # Resolve active registry: default→all, None→disabled, skip→subset
        if tool_registry == "DEFAULT":
            tool_registry = dict(_DEFAULT_REGISTRY)
        if tool_registry and skip_tools:
            tool_registry = {k: v for k, v in tool_registry.items()
                             if k not in skip_tools} or None

        tok = self.tokenizer
        seed_indices = tok.encode(prompt) if prompt else [random.randint(0, tok.size - 1)]
        ctx = seed_indices[-self.context_size:]
        while len(ctx) < self.context_size:
            ctx = [0] + ctx

        # ── Tool-enabled path ──────────────────────────────────────────────
        if tool_registry:
            from Neural_Network import ensure_tool_vocab  # type: ignore
            tok.char2idx = ensure_tool_vocab(tok.char2idx)
            tok.idx2ch   = {v: k for k, v in tok.char2idx.items()}
            for name, tdef in tool_registry.items():
                self.nn.register_tool(name, tdef.executor)
            out_ids, tool_log = self.nn.generate_with_tools(
                context        = ctx,
                idx2char       = tok.idx2ch,
                char2idx       = tok.char2idx,
                max_new        = length,
                temperature    = temperature,
                max_tool_calls = max_tool_calls,
            )
            if verbose and tool_log:
                for entry in tool_log:
                    print(f"[tool] {entry['tool']}({entry['arg']!r}) → {entry['result']!r}")
            generated = "".join(tok.idx2ch.get(i, "") for i in out_ids)
            return (prompt + generated) if prompt else generated

        # ── Plain generation path ──────────────────────────────────────────
        generated = list(prompt) if prompt else []
        for _ in range(length):
            _, _, probs = self.nn.predict(ctx)
            if temperature != 1.0:
                logits = np.log(np.array(probs) + 1e-9) / temperature
                probs  = np.exp(logits - logits.max())
                probs /= probs.sum()
            next_idx = int(np.random.choice(len(probs), p=probs))
            generated.append(tok.idx2ch[next_idx])
            ctx = ctx[1:] + [next_idx]
        return "".join(generated)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, weights_path: str = "gpt_weights.json") -> None:
        """
        Persist the trained model to disk as three JSON files.

        The three files saved are:

        - ``<weights_path>``               -- NeuralNetwork weights & biases.
        - ``<weights_path>_tokenizer.json``-- vocabulary list.
        - ``<weights_path>_config.json``   -- MiniGPT hyper-parameters.

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
            print("Nothing to save -- model has not been trained.")
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
                "num_blocks":    self.num_blocks,
                "num_heads":     self.num_heads,
                "dropout":       self.dropout,
                "weight_tying":  self.weight_tying,
                "grad_clip":     self.grad_clip,
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
            embed_dim     = cfg.get("embed_dim",    64),
            num_blocks    = cfg.get("num_blocks",   2),
            num_heads     = cfg.get("num_heads",    1),
            dropout       = cfg.get("dropout",      0.0),
            weight_tying  = cfg.get("weight_tying", False),
            grad_clip     = cfg.get("grad_clip",    1.0),
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