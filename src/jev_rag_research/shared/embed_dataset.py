import json
import time
from pathlib import Path

from .embeddings import embed_texts


CHUNKS_FILE = Path("data/processed/chunks.json")
EMBEDDINGS_FILE = Path(
    "data/embeddings/chunk_embeddings.json"
)


def main():

    print("Loading chunks...")

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        chunks = json.load(f)

    print(f"Chunks to embed: {len(chunks)}")

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("\nGenerating embeddings with Ollama...")
    print("Model: nomic-embed-text")

    start_time = time.perf_counter()

    embeddings = embed_texts(texts)

    elapsed = time.perf_counter() - start_time

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            "Number of embeddings does not match "
            "number of chunks."
        )

    # ---------------------------------------------------------
    # Store chunk ID together with embedding.
    # ---------------------------------------------------------

    output = []

    for chunk, embedding in zip(
        chunks,
        embeddings,
    ):

        output.append(
            {
                "chunk_id": chunk["id"],
                "embedding": embedding,
            }
        )

    EMBEDDINGS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        EMBEDDINGS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
        )

    print("\nEmbedding generation complete.")

    print(f"Embeddings generated: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"Time taken: {elapsed:.2f} seconds")

    print("\nSaved:")
    print(f"  {EMBEDDINGS_FILE}")


if __name__ == "__main__":
    main()