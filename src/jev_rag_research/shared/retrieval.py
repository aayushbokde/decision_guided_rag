import numpy as np


def cosine_similarity(
    query_embedding: list[float],
    document_embedding: list[float],
) -> float:
    """Calculate cosine similarity between two embeddings."""

    query = np.array(query_embedding)
    document = np.array(document_embedding)

    query_norm = np.linalg.norm(query)
    document_norm = np.linalg.norm(document)

    if query_norm == 0 or document_norm == 0:
        return 0.0

    return float(
        np.dot(query, document)
        / (query_norm * document_norm)
    )


def retrieve(
    query_embedding: list[float],
    document_embeddings: list[list[float]],
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """
    Retrieve the top-k most similar documents.

    Returns:
        List of (document_index, similarity_score)
    """

    scores = []

    for index, document_embedding in enumerate(document_embeddings):
        score = cosine_similarity(
            query_embedding,
            document_embedding,
        )

        scores.append((index, score))

    scores.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    return scores[:top_k]