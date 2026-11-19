"""Text cleaning for search queries."""

import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

for resource in ["punkt_tab", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    # lowercase, strip punctuation/digits, collapse whitespace
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str, remove_stopwords: bool = True) -> list:
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens


if __name__ == "__main__":
    samples = [
        "Running Shoes!!",
        "Vitamin C Serum for Dark Spots (30ml)",
        "  extra   spaces   here  ",
        "123 running shoes 2024",
    ]
    for s in samples:
        print(f"{s!r:45} -> clean: {clean_text(s)!r:35} tokens: {tokenize(s)}")
