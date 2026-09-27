import ollama

from .config import EMBEDDING_MODEL


def embed_text(text: str) -> list[float]:
    """Generate an embedding for a single piece of text."""

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=text,
    )

    return response.embeddings[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for multiple pieces of text."""

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    return response.embeddings