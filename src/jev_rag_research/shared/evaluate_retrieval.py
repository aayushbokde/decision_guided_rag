import json
from pathlib import Path

from .embeddings import embed_text
from .retrieval import retrieve


QUESTIONS_FILE = Path(
    "data/processed/chunked_questions.json"
)

EMBEDDINGS_FILE = Path(
    "data/embeddings/chunk_embeddings.json"
)

TOP_K_VALUES = [1, 3, 5]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():

    print("Loading questions...")
    questions = load_json(QUESTIONS_FILE)

    print("Loading chunk embeddings...")
    embedding_data = load_json(EMBEDDINGS_FILE)

    # Map chunk ID -> embedding
    embedding_map = {
        item["chunk_id"]: item["embedding"]
        for item in embedding_data
    }

    chunk_ids = list(embedding_map.keys())
    embeddings = [
        embedding_map[chunk_id]
        for chunk_id in chunk_ids
    ]

    print(f"Questions: {len(questions)}")
    print(f"Chunks: {len(chunks := chunk_ids)}")

    hits = {
        k: 0
        for k in TOP_K_VALUES
    }

    print("\nEvaluating retrieval...")

    for i, question in enumerate(questions):

        query = question["question"]
        relevant_chunk_id = question["relevant_chunk_id"]

        query_embedding = embed_text(query)

        retrieved = retrieve(
            query_embedding,
            embeddings,
            top_k=max(TOP_K_VALUES),
        )

        retrieved_chunk_ids = [
            chunk_ids[index]
            for index, _ in retrieved
        ]

        for k in TOP_K_VALUES:

            if relevant_chunk_id in retrieved_chunk_ids[:k]:
                hits[k] += 1

        if (i + 1) % 25 == 0:
            print(
                f"Processed {i + 1}/{len(questions)}"
            )

    print("\nRetrieval Results")
    print("=" * 40)

    for k in TOP_K_VALUES:

        recall = hits[k] / len(questions)

        print(
            f"Recall@{k}: "
            f"{recall:.4f} "
            f"({hits[k]}/{len(questions)})"
        )


if __name__ == "__main__":
    main()