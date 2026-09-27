import json
from pathlib import Path

from ..shared.config import TOP_K
from ..shared.retrieval import retrieve
from ..shared.embeddings import embed_text
from ..jev_rag.oracle import KeywordOracle


QUESTIONS_FILE = Path(
    "data/processed/chunked_questions.json"
)

CHUNKS_FILE = Path(
    "data/processed/chunks.json"
)

EMBEDDINGS_FILE = Path(
    "data/embeddings/chunk_embeddings.json"
)

OUTPUT_FILE = Path(
    "results/oracle_threshold_sweep.json"
)

THRESHOLDS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
]


def load_json(path: Path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def build_context_length(chunks, indices):
    """
    Calculate context length using the same structure
    as the actual filtered RAG context builder.
    """

    parts = []

    for index in indices:
        chunk = chunks[index]

        parts.append(
            f"[Chunk {chunk['id']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(parts)

    return len(context)


def main():

    print("Loading data...")

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

    # -----------------------------------------------------
    # Query embeddings
    #
    # Each question is embedded only once.
    # The same embedding is reused for every threshold.
    # -----------------------------------------------------

    print(
        f"\nEmbedding {len(questions)} questions..."
    )

    query_embeddings = {}

    for i, question_data in enumerate(
        questions
    ):

        question_id = question_data["id"]
        question = question_data["question"]

        print(
            f"\r[{i + 1}/{len(questions)}]",
            end="",
            flush=True,
        )

        query_embeddings[question_id] = (
            embed_text(question)
        )

    print("\n")

    # -----------------------------------------------------
    # Retrieval
    #
    # Retrieval is independent of the threshold, so
    # Top-K retrieval is performed only once per question.
    # -----------------------------------------------------

    print("Computing Top-K retrieval...")

    retrieval_results = {}

    for question_data in questions:

        question_id = question_data["id"]

        retrieved = retrieve(
            query_embeddings[question_id],
            chunk_embeddings,
            top_k=TOP_K,
        )

        retrieval_results[question_id] = (
            retrieved
        )

    # -----------------------------------------------------
    # Threshold sweep
    # -----------------------------------------------------

    sweep_results = []

    print()
    print("=" * 70)
    print("ORACLE THRESHOLD SWEEP")
    print("=" * 70)

    for threshold in THRESHOLDS:

        print(
            f"\nRunning threshold: "
            f"{threshold:.2f}"
        )

        oracle = KeywordOracle(
            threshold=threshold
        )

        total_selected = 0
        total_context_before = 0
        total_context_after = 0

        gold_survival_count = 0
        fallback_count = 0
        total_decisions = 0

        question_results = []

        # -------------------------------------------------
        # Evaluate every question
        # -------------------------------------------------

        for question_data in questions:

            question_id = question_data["id"]
            question = question_data["question"]

            retrieved = retrieval_results[
                question_id
            ]

            selected_indices = []

            # ---------------------------------------------
            # Apply oracle to each retrieved chunk
            # ---------------------------------------------

            for index, similarity in retrieved:

                chunk = chunks[index]

                decision = oracle.decide(
                    question,
                    chunk["text"],
                )

                total_decisions += 1

                if decision.keep:
                    selected_indices.append(
                        index
                    )

            # ---------------------------------------------
            # Same fallback behavior as filtered RAG
            # ---------------------------------------------

            fallback_used = False

            if not selected_indices:

                selected_indices = [
                    retrieved[0][0]
                ]

                fallback_used = True
                fallback_count += 1

            # ---------------------------------------------
            # Context sizes
            # ---------------------------------------------

            retrieved_indices = [
                index
                for index, _ in retrieved
            ]

            context_before = (
                build_context_length(
                    chunks,
                    retrieved_indices,
                )
            )

            context_after = (
                build_context_length(
                    chunks,
                    selected_indices,
                )
            )

            # ---------------------------------------------
            # Gold chunk survival
            # ---------------------------------------------

            gold_chunk = question_data[
                "relevant_chunk_id"
            ]

            selected_chunk_ids = [
                chunks[index]["id"]
                for index in selected_indices
            ]

            gold_survived = (
                gold_chunk
                in selected_chunk_ids
            )

            if gold_survived:
                gold_survival_count += 1

            # ---------------------------------------------
            # Aggregate statistics
            # ---------------------------------------------

            total_selected += len(
                selected_indices
            )

            total_context_before += (
                context_before
            )

            total_context_after += (
                context_after
            )

            # ---------------------------------------------
            # Per-question result
            # ---------------------------------------------

            question_results.append(
                {
                    "question_id": question_id,
                    "num_retrieved": len(
                        retrieved
                    ),
                    "num_selected": len(
                        selected_indices
                    ),
                    "context_before": (
                        context_before
                    ),
                    "context_after": (
                        context_after
                    ),
                    "gold_survived": (
                        gold_survived
                    ),
                    "fallback_used": (
                        fallback_used
                    ),
                }
            )

        # -------------------------------------------------
        # Aggregate threshold metrics
        # -------------------------------------------------

        num_questions = len(
            questions
        )

        avg_selected = (
            total_selected
            / num_questions
        )

        avg_context_before = (
            total_context_before
            / num_questions
        )

        avg_context_after = (
            total_context_after
            / num_questions
        )

        context_reduction = (
            1
            - (
                avg_context_after
                / avg_context_before
            )
        )

        gold_survival_rate = (
            gold_survival_count
            / num_questions
        )

        avg_decision_count = (
            total_decisions
            / num_questions
        )

        fallback_rate = (
            fallback_count
            / num_questions
        )

        # -------------------------------------------------
        # Store threshold result
        # -------------------------------------------------

        result = {
            "threshold": threshold,
            "questions": num_questions,

            "average_chunks_retrieved": (
                TOP_K
            ),

            "average_chunks_selected": (
                avg_selected
            ),

            "average_context_before": (
                avg_context_before
            ),

            "average_context_after": (
                avg_context_after
            ),

            "context_reduction": (
                context_reduction
            ),

            "gold_chunk_survival": (
                gold_survival_rate
            ),

            "average_decisions": (
                avg_decision_count
            ),

            "fallback_rate": (
                fallback_rate
            ),

            "question_results": (
                question_results
            ),
        }

        sweep_results.append(
            result
        )

        # -------------------------------------------------
        # Print threshold summary
        # -------------------------------------------------

        print(
            f"  Chunks: "
            f"{TOP_K:.2f} → "
            f"{avg_selected:.2f}"
        )

        print(
            f"  Context reduction: "
            f"{context_reduction * 100:.2f}%"
        )

        print(
            f"  Gold survival: "
            f"{gold_survival_rate * 100:.2f}%"
        )

        print(
            f"  Fallback rate: "
            f"{fallback_rate * 100:.2f}%"
        )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            sweep_results,
            f,
            indent=2,
        )

    # -----------------------------------------------------
    # Final summary table
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print("THRESHOLD SWEEP SUMMARY")
    print("=" * 80)

    print(
        f"{'Threshold':<12}"
        f"{'Chunks':<12}"
        f"{'Context Red.':<16}"
        f"{'Gold Survival':<16}"
        f"{'Fallback':<12}"
    )

    print("-" * 80)

    for result in sweep_results:

        print(
            f"{result['threshold']:<12.2f}"
            f"{result['average_chunks_selected']:<12.2f}"
            f"{result['context_reduction'] * 100:<16.2f}"
            f"{result['gold_chunk_survival'] * 100:<16.2f}"
            f"{result['fallback_rate'] * 100:<12.2f}"
        )

    print()
    print(
        f"Results saved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()