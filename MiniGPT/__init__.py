"""
miniGPT
=======
A character-level language model built on top of ``Neural_Network.py``.

**Package structure:**

.. code-block:: text

    miniGPT/
    ├── __init__.py        ← you are here  (public API)
    ├── tokenizer.py       ← CharTokenizer class
    ├── data.py            ← load_text(), make_samples()
    ├── model.py           ← MiniGPT class
    ├── cli.py             ← command-line interface
    └── Neural_Network.py  ← your feedforward NN (must be present)

**Quick start — import the package:**

.. code-block:: python

    from miniGPT import MiniGPT

    model = MiniGPT(context_size=8, hidden_layers=[256, 128])
    model.train("wiki_dataset.txt", epochs=5)
    model.save("gpt_weights.json")

    text = model.generate(prompt="Civil rights", length=300)
    print(text)

**Or load a pre-trained model:**

.. code-block:: python

    from miniGPT import MiniGPT

    model = MiniGPT.load("gpt_weights.json")
    print(model.generate("Democracy is", length=200, temperature=0.7))
"""

from model     import MiniGPT
from tokenizer import CharTokenizer
from data      import load_text, make_samples

__all__ = [
    "MiniGPT",
    "CharTokenizer",
    "load_text",
    "make_samples",
]

__version__ = "1.0.0"
__author__  = "miniGPT project"
