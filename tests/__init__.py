"""
tests/
======
Test suite for the miniGPT project.

Folder layout assumed:
    GPT/
    ├── Neural_Network.py
    ├── build_dataset.py
    ├── miniGPT/
    │   ├── tokenizer.py
    │   ├── data.py
    │   ├── model.py
    │   └── cli.py
    └── tests/          <- you are here
        ├── __init__.py
        ├── run_all_tests.py
        ├── test_tokenizer.py
        ├── test_data.py
        ├── test_neural_network.py
        ├── test_model.py
        ├── test_build_dataset.py
        ├── test_integration.py
        └── test_performance.py

Run everything:
    python -m unittest discover -s tests -p "test_*.py" -v

Run one module:
    python -m unittest tests.test_tokenizer -v

Run one class:
    python -m unittest tests.test_tokenizer.TestEncoding -v
"""