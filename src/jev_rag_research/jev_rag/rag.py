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

# Development-only decision backend.
#
# IMPORTANT:
# This is NOT JEV.
# It is a deterministic lexical oracle used as a
# substitute while developing and evaluating the pipeline.
THRESHOLD = 0.15

# Save this experiment separately so the previous
# threshold=0.10 experiment is preserved.
RESULTS_FILE = Path(
    "results/oracle_filtered_rag_results_015.json"
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

    return response["message"]["content"].strip()


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
# Filtered RAG pipeline
# ============================================================

def run_filtered_rag(
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
    # Retrieval
    # --------------------------------------------------------

    retrieval_start = time.perf_counter()

    retrieved = retrieve(
        query_embedding,
        chunk_embeddings,
        top_k=TOP_K,
    )

    retrieval_time = (
        time.perf_counter()
        - retrieval_start
    )

    # --------------------------------------------------------
    # Decision filtering
    # --------------------------------------------------------

    decision_start = time.perf_counter()

    selected_indices = []
    decisions = []

    for index, similarity in retrieved:

        chunk = chunks[index]

        decision = decision_backend.decide(
            question,
            chunk["text"],
        )

        decisions.append(
            {
                "chunk_id": chunk["id"],
                "similarity": similarity,
                "keep": decision.keep,
                "confidence": decision.confidence,
                "backend": decision.backend,
            }
        )

        if decision.keep:
            selected_indices.append(
                index
            )

    decision_time = (
        time.perf_counter()
        - decision_start
    )

    # --------------------------------------------------------
    # Empty-context fallback
    # --------------------------------------------------------

    if not selected_indices:

        # Keep the highest-scoring chunk rather than
        # sending an empty context to the generator.
        selected_indices = [
            retrieved[0][0]
        ]

        fallback_used = True

    else:

        fallback_used = False

    # --------------------------------------------------------
    # Context construction
    # --------------------------------------------------------

    context = build_context(
        chunks,
        selected_indices,
    )

    context_before = build_context(
        chunks,
        [
            index
            for index, _ in retrieved
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

        "retrieved": [
            {
                "chunk_id": chunks[index]["id"],
                "similarity": similarity,
            }
            for index, similarity in retrieved
        ],

        "decisions": decisions,

        "selected_chunks": [
            chunks[index]["id"]
            for index in selected_indices
        ],

        "num_retrieved": len(
            retrieved
        ),

        "num_selected": len(
            selected_indices
        ),

        "context_characters_before": len(
            context_before
        ),

        "context_characters_after": len(
            context
        ),

        "context_reduction": (
            1
            - len(context) / len(context_before)
            if context_before
            else 0.0
        ),

        "fallback_used": fallback_used,

        "embedding_time": embedding_time,
        "retrieval_time": retrieval_time,
        "decision_time": decision_time,
        "llm_time": llm_time,
        "total_time": total_time,
    }


# ============================================================
# Main evaluation
# ============================================================

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

    # --------------------------------------------------------
    # Development-only decision backend
    # --------------------------------------------------------

    # This is a deterministic lexical oracle.
    #
    # It is NOT JEV and must not be reported as JEV
    # experimental results.

    decision_backend = KeywordOracle(
        threshold=THRESHOLD
    )

    test_questions = questions

    results = []

    print()
    print("=" * 70)
    print("FILTERED RAG EVALUATION")
    print("=" * 70)

    print(
        f"Questions: "
        f"{len(test_questions)}"
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
        f"Output: "
        f"{RESULTS_FILE}"
    )

    print()

    # --------------------------------------------------------
    # Run evaluation
    # --------------------------------------------------------

    for i, question_data in enumerate(
        test_questions
    ):

        question = question_data[
            "question"
        ]

        print(
            f"[{i + 1}/{len(test_questions)}] "
            f"{question}"
        )

        result = run_filtered_rag(
            question,
            chunks,
            chunk_embeddings,
            decision_backend,
        )

        gold_chunk = question_data[
            "relevant_chunk_id"
        ]

        selected_chunks = result[
            "selected_chunks"
        ]

        gold_survived = (
            gold_chunk in selected_chunks
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

        result["threshold"] = THRESHOLD

        results.append(result)

        print(
            f"  Chunks: "
            f"{result['num_retrieved']} → "
            f"{result['num_selected']}"
        )

        print(
            f"  Context reduction: "
            f"{result['context_reduction'] * 100:.2f}%"
        )

        print(
            f"  Gold survived: "
            f"{gold_survived}"
        )

        print(
            f"  Decision: "
            f"{result['decision_time'] * 1000:.3f} ms"
        )

        print(
            f"  LLM: "
            f"{result['llm_time']:.2f}s"
        )

    # --------------------------------------------------------
    # Save results
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

    # --------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------

    count = len(results)

    avg_selected = (
        sum(
            r["num_selected"]
            for r in results
        )
        / count
    )

    avg_context_before = (
        sum(
            r["context_characters_before"]
            for r in results
        )
        / count
    )

    avg_context_after = (
        sum(
            r["context_characters_after"]
            for r in results
        )
        / count
    )

    avg_reduction = (
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

    fallback_rate = (
        sum(
            r["fallback_used"]
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
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FILTERED RAG SUMMARY")
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
        f"Average chunks: "
        f"{TOP_K:.2f} → "
        f"{avg_selected:.2f}"
    )

    print(
        f"Average context: "
        f"{avg_context_before:.0f} → "
        f"{avg_context_after:.0f} chars"
    )

    print(
        f"Context reduction: "
        f"{avg_reduction * 100:.2f}%"
    )

    print(
        f"Gold chunk survival: "
        f"{gold_survival * 100:.2f}%"
    )

    print(
        f"Fallback rate: "
        f"{fallback_rate * 100:.2f}%"
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