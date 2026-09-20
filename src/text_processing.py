"""Transformaciones de texto compartidas por entrenamiento y despliegue."""

import re


_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_corpus(corpus):
    """Sustituye URLs y normaliza espacios sin eliminar información lingüística."""
    cleaned = []
    for document in corpus:
        document = str(document)
        document = _URL_PATTERN.sub(" url ", document)
        document = _WHITESPACE_PATTERN.sub(" ", document).strip()
        cleaned.append(document)
    return cleaned
