import json
import time
from pathlib import Path

import ollama

from ..shared.config import GENERATION_MODEL, TOP_K
from ..shared.embeddings import embed_text
from ..shared.retrieval import retrieve
from .oracle import KeywordOracle


# ============================================================
# Configuration
# ============================================================

QUESTIONS_FILE = Path(
    "data/processed/chunked_questions.json"
)

CHUNKS_FILE = Path(
    "data/processed/chunks.json"
)

EMBEDDINGS_FILE = Path(
    "data/embeddings/chunk_embeddings.json"
)

# Development-only backend.
#
# IMPORTANT:
# This is NOT JEV.
THRESHOLD = 0.15

MAX_RETRIEVAL_K = TOP_K

RESULTS_FILE = Path(
    "results/oracle_early_exit_accumulated_results_200.json"
)


# ============================================================
# Utilities
# ============================================================

def load_json(path: Path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def generate_answer(
    question: str,
    context: str,
):

    prompt = f"""Answer the question using only the provided context.

If the answer cannot be determined from the context,
say "Insufficient information."

Context:

{context}

Question:

{question}

Answer:
"""

    response = ollama.chat(
        model=GENERATION_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response[
        "message"
    ][
        "content"
    ].strip()


def build_context(
    chunks,
    selected_indices,
):

    parts = []

    for index in selected_indices:

        chunk = chunks[index]

        parts.append(
            f"[Chunk {chunk['id']}]\n"
            f"{chunk['text']}"
        )

    return "\n\n".join(parts)


# ============================================================
# Early-Exit RAG
# ============================================================

def run_early_exit_rag(
    question,
    chunks,
    chunk_embeddings,
    decision_backend,
):

    pipeline_start = time.perf_counter()

    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

    embedding_start = time.perf_counter()

    query_embedding = embed_text(
        question
    )

    embedding_time = (
        time.perf_counter()
        - embedding_start
    )

    # --------------------------------------------------------
    # Retrieve candidate pool
    # --------------------------------------------------------

    retrieval_start = time.perf_counter()

    candidate_results = retrieve(
        query_embedding,
        chunk_embeddings,
        top_k=MAX_RETRIEVAL_K,
    )

    retrieval_time = (
        time.perf_counter()
        - retrieval_start
    )

    # --------------------------------------------------------
    # Accumulated-evidence early exit
    # --------------------------------------------------------

    decision_start = time.perf_counter()

    selected_indices = []

    decisions = []

    exit_k = MAX_RETRIEVAL_K

    exited_early = False

    for k in range(
        1,
        MAX_RETRIEVAL_K + 1,
    ):

        index, similarity = candidate_results[
            k - 1
        ]

        # Add current chunk to accumulated evidence.
        selected_indices.append(
            index
        )

        # ----------------------------------------------------
        # Build accumulated context
        # ----------------------------------------------------

        accumulated_context = build_context(
            chunks,
            selected_indices,
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # The oracle now evaluates the QUESTION against the
        # ACCUMULATED EVIDENCE rather than only the current
        # chunk.
        # ----------------------------------------------------

        decision = decision_backend.decide(
            question,
            accumulated_context,
        )

        decisions.append(
            {
                "rank": k,
                "chunk_id": chunks[index]["id"],
                "similarity": similarity,
                "accumulated_chunks": [
                    chunks[i]["id"]
                    for i in selected_indices
                ],
                "keep": decision.keep,
                "confidence": decision.confidence,
                "backend": decision.backend,
            }
        )

        # ----------------------------------------------------
        # Exit when accumulated evidence is sufficient.
        # ----------------------------------------------------

        if decision.keep:

            exit_k = k

            exited_early = (
                k < MAX_RETRIEVAL_K
            )

            break

    decision_time = (
        time.perf_counter()
        - decision_start
    )

    # --------------------------------------------------------
    # Final context
    # --------------------------------------------------------

    context = build_context(
        chunks,
        selected_indices,
    )

    full_context = build_context(
        chunks,
        [
            index
            for index, _ in candidate_results
        ],
    )

    # --------------------------------------------------------
    # LLM generation
    # --------------------------------------------------------

    llm_start = time.perf_counter()

    answer = generate_answer(
        question,
        context,
    )

    llm_time = (
        time.perf_counter()
        - llm_start
    )

    total_time = (
        time.perf_counter()
        - pipeline_start
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {

        "answer": answer,

        "retrieved_candidates": [
            {
                "chunk_id": chunks[index]["id"],
                "similarity": similarity,
                "rank": rank + 1,
            }
            for rank, (
                index,
                similarity,
            )
            in enumerate(
                candidate_results
            )
        ],

        "decisions": decisions,

        "selected_chunks": [
            chunks[index]["id"]
            for index in selected_indices
        ],

        "num_candidates": len(
            candidate_results
        ),

        "num_selected": len(
            selected_indices
        ),

        "retrieval_depth": exit_k,

        "exited_early": exited_early,

        "context_characters": len(
            context
        ),

        "full_context_characters": len(
            full_context
        ),

        "context_reduction": (
            1
            - len(context)
            / len(full_context)
            if full_context
            else 0.0
        ),

        "embedding_time": embedding_time,

        "retrieval_time": retrieval_time,

        "decision_time": decision_time,

        "llm_time": llm_time,

        "total_time": total_time,
    }


# ============================================================
# Main Evaluation
# ============================================================

def main():

    print(
        "Loading data..."
    )

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

    # --------------------------------------------------------
    # Development-only backend
    # --------------------------------------------------------

    decision_backend = KeywordOracle(
        threshold=THRESHOLD
    )

    results = []

    print()
    print("=" * 70)
    print(
        "ACCUMULATED-EVIDENCE "
        "ORACLE EARLY-EXIT RAG"
    )
    print("=" * 70)

    print(
        f"Questions: "
        f"{len(questions)}"
    )

    print(
        f"Decision backend: "
        f"{decision_backend.__class__.__name__}"
    )

    print(
        f"Threshold: "
        f"{THRESHOLD:.2f}"
    )

    print(
        f"Maximum retrieval depth: "
        f"{MAX_RETRIEVAL_K}"
    )

    print()

    # --------------------------------------------------------
    # Evaluation loop
    # --------------------------------------------------------

    for i, question_data in enumerate(
        questions
    ):

        question = question_data[
            "question"
        ]

        print(
            f"[{i + 1}/{len(questions)}] "
            f"{question}"
        )

        result = run_early_exit_rag(
            question,
            chunks,
            chunk_embeddings,
            decision_backend,
        )

        # ----------------------------------------------------
        # Gold-chunk evaluation
        # ----------------------------------------------------

        gold_chunk = question_data[
            "relevant_chunk_id"
        ]

        selected_chunks = result[
            "selected_chunks"
        ]

        gold_survived = (
            gold_chunk
            in selected_chunks
        )

        result["question_id"] = (
            question_data["id"]
        )

        result["reference_answer"] = (
            question_data["answer"]
        )

        result["relevant_chunk_id"] = (
            gold_chunk
        )

        result["gold_chunk_survived"] = (
            gold_survived
        )

        result["threshold"] = (
            THRESHOLD
        )

        results.append(
            result
        )

        print(
            f"  Retrieval depth: "
            f"{result['retrieval_depth']}"
        )

        print(
            f"  Early exit: "
            f"{result['exited_early']}"
        )

        print(
            f"  Selected chunks: "
            f"{result['num_selected']}"
        )

        print(
            f"  Context: "
            f"{result['context_characters']} chars"
        )

        print(
            f"  Gold survived: "
            f"{gold_survived}"
        )

        print(
            f"  Decision time: "
            f"{result['decision_time'] * 1000:.3f} ms"
        )

        print(
            f"  LLM: "
            f"{result['llm_time']:.2f}s"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    # ========================================================
    # Aggregate metrics
    # ========================================================

    count = len(results)

    avg_depth = (
        sum(
            r["retrieval_depth"]
            for r in results
        )
        / count
    )

    early_exit_rate = (
        sum(
            r["exited_early"]
            for r in results
        )
        / count
    )

    avg_context = (
        sum(
            r["context_characters"]
            for r in results
        )
        / count
    )

    avg_full_context = (
        sum(
            r["full_context_characters"]
            for r in results
        )
        / count
    )

    avg_context_reduction = (
        sum(
            r["context_reduction"]
            for r in results
        )
        / count
    )

    gold_survival = (
        sum(
            r["gold_chunk_survived"]
            for r in results
        )
        / count
    )

    avg_embedding = (
        sum(
            r["embedding_time"]
            for r in results
        )
        / count
    )

    avg_retrieval = (
        sum(
            r["retrieval_time"]
            for r in results
        )
        / count
    )

    avg_decision = (
        sum(
            r["decision_time"]
            for r in results
        )
        / count
    )

    avg_llm = (
        sum(
            r["llm_time"]
            for r in results
        )
        / count
    )

    avg_total = (
        sum(
            r["total_time"]
            for r in results
        )
        / count
    )

    # --------------------------------------------------------
    # Retrieval-depth distribution
    # --------------------------------------------------------

    depth_distribution = {}

    for result in results:

        depth = result[
            "retrieval_depth"
        ]

        depth_distribution[depth] = (
            depth_distribution.get(
                depth,
                0,
            )
            + 1
        )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 70)
    print(
        "ACCUMULATED-EVIDENCE "
        "EARLY-EXIT SUMMARY"
    )
    print("=" * 70)

    print(
        f"Questions: "
        f"{count}"
    )

    print(
        f"Threshold: "
        f"{THRESHOLD:.2f}"
    )

    print(
        f"Maximum retrieval depth: "
        f"{MAX_RETRIEVAL_K}"
    )

    print(
        f"Average retrieval depth: "
        f"{avg_depth:.2f}"
    )

    print(
        f"Early-exit rate: "
        f"{early_exit_rate * 100:.2f}%"
    )

    print(
        f"Average context: "
        f"{avg_context:.0f} chars"
    )

    print(
        f"Full Top-{MAX_RETRIEVAL_K} context: "
        f"{avg_full_context:.0f} chars"
    )

    print(
        f"Context reduction: "
        f"{avg_context_reduction * 100:.2f}%"
    )

    print(
        f"Gold chunk survival: "
        f"{gold_survival * 100:.2f}%"
    )

    print()
    print(
        "Retrieval depth distribution:"
    )

    for depth in sorted(
        depth_distribution
    ):

        count_at_depth = (
            depth_distribution[depth]
        )

        percentage = (
            count_at_depth
            / count
            * 100
        )

        print(
            f"  Top-{depth}: "
            f"{count_at_depth} "
            f"({percentage:.2f}%)"
        )

    print()

    print(
        f"Average embedding time: "
        f"{avg_embedding:.3f}s"
    )

    print(
        f"Average retrieval time: "
        f"{avg_retrieval:.3f}s"
    )

    print(
        f"Average decision time: "
        f"{avg_decision * 1000:.3f} ms"
    )

    print(
        f"Average LLM time: "
        f"{avg_llm:.3f}s"
    )

    print(
        f"Average total time: "
        f"{avg_total:.3f}s"
    )

    print()

    print(
        f"Results saved to: "
        f"{RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()