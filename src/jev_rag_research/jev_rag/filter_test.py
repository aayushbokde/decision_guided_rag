import json
import time
from pathlib import Path

from ..shared.config import TOP_K
from ..shared.embeddings import embed_text
from ..shared.retrieval import retrieve
from .oracle import KeywordOracle


QUESTIONS_FILE = Path(
    "data/processed/chunked_questions.json"
)

CHUNKS_FILE = Path(
    "data/processed/chunks.json"
)

EMBEDDINGS_FILE = Path(
    "data/embeddings/chunk_embeddings.json"
)


def load_json(path: Path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def main():

    questions = load_json(
        QUESTIONS_FILE
    )

    chunks = load_json(
        CHUNKS_FILE
    )

    embedding_data = load_json(
        EMBEDDINGS_FILE
    )

    embedding_map = {
        item["chunk_id"]: item["embedding"]
        for item in embedding_data
    }

    chunk_ids = list(
        embedding_map.keys()
    )

    chunk_embeddings = [
        embedding_map[chunk_id]
        for chunk_id in chunk_ids
    ]

    oracle = KeywordOracle(
        threshold=0.10
    )

    # Only inspect the first 20 questions.
    test_questions = questions[:20]

    total_retrieved = 0
    total_kept = 0
    gold_survived = 0

    total_context_before = 0
    total_context_after = 0

    total_decision_time = 0.0

    print("JEV-RAG filtering pipeline test")
    print("=" * 60)

    for i, question_data in enumerate(
        test_questions
    ):

        question = question_data[
            "question"
        ]

        gold_chunk = question_data[
            "relevant_chunk_id"
        ]

        # ---------------------------------
        # Query embedding
        # ---------------------------------

        query_embedding = embed_text(
            question
        )

        # ---------------------------------
        # Retrieval
        # ---------------------------------

        retrieved = retrieve(
            query_embedding,
            chunk_embeddings,
            top_k=TOP_K,
        )

        # ---------------------------------
        # JEV/oracle filtering
        # ---------------------------------

        kept_chunks = []

        decision_start = (
            time.perf_counter()
        )

        for index, score in retrieved:

            chunk = chunks[index]

            decision = oracle.decide(
                question,
                chunk["text"],
            )

            if decision.keep:

                kept_chunks.append(
                    (
                        chunk["id"],
                        decision.confidence,
                    )
                )

        decision_time = (
            time.perf_counter()
            - decision_start
        )

        total_decision_time += (
            decision_time
        )

        # ---------------------------------
        # Determine whether gold survived
        # ---------------------------------

        retrieved_ids = [
            chunks[index]["id"]
            for index, _ in retrieved
        ]

        kept_ids = [
            chunk_id
            for chunk_id, _ in kept_chunks
        ]

        gold_kept = (
            gold_chunk in kept_ids
        )

        # ---------------------------------
        # Print detailed failure cases
        # ---------------------------------

        if not gold_kept:

            print("\n")
            print("!" * 60)
            print("GOLD CHUNK FILTERED OUT")
            print("!" * 60)

            print(
                f"\nQuestion:\n{question}"
            )

            print(
                f"\nGold chunk:\n{gold_chunk}"
            )

            print(
                "\nRetrieved chunks:"
            )

            for index, score in retrieved:

                chunk = chunks[index]

                decision = oracle.decide(
                    question,
                    chunk["text"],
                )

                print("\n" + "-" * 60)

                print(
                    f"Chunk ID: "
                    f"{chunk['id']}"
                )

                print(
                    f"Similarity: "
                    f"{score:.4f}"
                )

                print(
                    f"Decision: "
                    f"{decision}"
                )

                print(
                    f"\nChunk text:\n"
                    f"{chunk['text']}"
                )

        # ---------------------------------
        # Statistics
        # ---------------------------------

        total_retrieved += len(
            retrieved
        )

        total_kept += len(
            kept_chunks
        )

        if gold_kept:
            gold_survived += 1

        # ---------------------------------
        # Context size
        # ---------------------------------

        context_before = sum(
            len(chunks[index]["text"])
            for index, _ in retrieved
        )

        context_after = sum(
            len(
                chunks[
                    next(
                        j
                        for j, c
                        in enumerate(chunks)
                        if c["id"] == chunk_id
                    )
                ]["text"]
            )
            for chunk_id, _ in kept_chunks
        )

        total_context_before += (
            context_before
        )

        total_context_after += (
            context_after
        )

        # ---------------------------------
        # Per-question summary
        # ---------------------------------

        print(
            f"\n[{i + 1}/20]"
        )

        print(
            f"Question: {question}"
        )

        print(
            f"Retrieved: "
            f"{len(retrieved)}"
        )

        print(
            f"Kept: "
            f"{len(kept_chunks)}"
        )

        print(
            f"Gold chunk survived: "
            f"{gold_kept}"
        )

        print(
            f"Decision time: "
            f"{decision_time * 1000:.2f} ms"
        )

    # -------------------------------------
    # Final summary
    # -------------------------------------

    print("\n")
    print("=" * 60)
    print("FILTERING SUMMARY")
    print("=" * 60)

    avg_kept = (
        total_kept
        / len(test_questions)
    )

    avg_context_before = (
        total_context_before
        / len(test_questions)
    )

    avg_context_after = (
        total_context_after
        / len(test_questions)
    )

    context_reduction = (
        1
        - (
            avg_context_after
            / avg_context_before
        )
    )

    gold_recall = (
        gold_survived
        / len(test_questions)
    )

    avg_decision_time = (
        total_decision_time
        / len(test_questions)
    )

    print(
        f"Average retrieved: "
        f"{TOP_K:.2f}"
    )

    print(
        f"Average kept: "
        f"{avg_kept:.2f}"
    )

    print(
        f"Gold chunk survival: "
        f"{gold_survived}/20 "
        f"({gold_recall * 100:.2f}%)"
    )

    print(
        f"Average context before: "
        f"{avg_context_before:.0f} chars"
    )

    print(
        f"Average context after: "
        f"{avg_context_after:.0f} chars"
    )

    print(
        f"Context reduction: "
        f"{context_reduction * 100:.2f}%"
    )

    print(
        f"Average decision time: "
        f"{avg_decision_time * 1000:.2f} ms"
    )


if __name__ == "__main__":
    main()