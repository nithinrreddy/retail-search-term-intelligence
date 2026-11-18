"""
Check that the project environment is ready.

This script imports every library the project uses and prints its version.
It also loads spaCy's English model and tokenises one sample query.
If anything is missing, the script stops with a clear error message.

Run from the project root, with the virtual environment active:
    python src/check_setup.py
"""

import importlib
import sys

# Libraries used across the project. Each one is needed in a later milestone.
REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "sklearn",      # scikit-learn: models and evaluation metrics
    "nltk",         # text preprocessing
    "spacy",        # tokenisation and linguistic features
    "matplotlib",   # static charts
    "plotly",       # interactive charts
    "streamlit",    # dashboard
    "lime",         # explanations for individual predictions
]


def check_packages():
    """Import each package and print its version. Return True if all succeed."""
    all_ok = True
    for name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "version not listed")
            print(f"  OK    {name:12} {version}")
        except ImportError as error:
            all_ok = False
            print(f"  FAIL  {name:12} {error}")
    return all_ok


def check_spacy_model():
    """Load the spaCy English model and tokenise a sample query."""
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("  FAIL  en_core_web_sm model not found.")
        print("        Run: python -m spacy download en_core_web_sm")
        return False

    sample = "running shoes for men"
    tokens = [token.text for token in nlp(sample)]
    print(f"  OK    spaCy model loaded. Tokens for '{sample}': {tokens}")
    return True


def main():
    print(f"Python version: {sys.version.split()[0]}")

    print("\nChecking packages:")
    packages_ok = check_packages()

    print("\nChecking spaCy language model:")
    model_ok = check_spacy_model() if packages_ok else False

    if packages_ok and model_ok:
        print("\nAll checks passed. The environment is ready.")
    else:
        print("\nSome checks failed. Copy the lines marked FAIL and send them to me.")
        sys.exit(1)


if __name__ == "__main__":
    main()
