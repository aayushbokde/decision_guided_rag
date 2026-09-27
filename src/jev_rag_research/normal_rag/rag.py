# import json
# import time
# from pathlib import Path

# import ollama

# from ..shared.config import (
#     GENERATION_MODEL,
#     TOP_K,
# )
# from ..shared.embeddings import embed_text
# from ..shared.retrieval import retrieve


# QUESTIONS_FILE = Path(
#     "data/processed/chunked_questions.json"
# )

# CHUNKS_FILE = Path(
#     "data/processed/chunks.json"
# )

# EMBEDDINGS_FILE = Path(
#     "data/embeddings/chunk_embeddings.json"
# )


# def load_json(path: Path):
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)


# def build_context(
#     retrieved,
#     chunks,
# ):
#     """Build LLM context from retrieved chunks."""

#     context_parts = []

#     for index, score in retrieved:

#         chunk = chunks[index]

#         context_parts.append(
#             f"[Chunk {chunk['id']}]\n"
#             f"{chunk['text']}"
#         )

#     return "\n\n".join(context_parts)


# def generate_answer(
#     question: str,
#     context: str,
# ):
#     """Generate an answer using the local LLM."""

#     prompt = f"""Answer the question using only the provided context.

# If the answer cannot be determined from the context,
# say "Insufficient information."

# Context:
# {context}

# Question:
# {question}

# Answer:
# """

#     response = ollama.chat(
#         model=GENERATION_MODEL,
#         messages=[
#             {
#                 "role": "user",
#                 "content": prompt,
#             }
#         ],
#     )

#     return response["message"]["content"]


# def run_rag(
#     question: str,
#     chunks,
#     chunk_embeddings,
# ):
#     """Run the complete Normal RAG pipeline."""

#     start = time.perf_counter()

#     query_embedding = embed_text(question)

#     retrieval_start = time.perf_counter()

#     retrieved = retrieve(
#         query_embedding,
#         chunk_embeddings,
#         top_k=TOP_K,
#     )

#     retrieval_time = (
#         time.perf_counter()
#         - retrieval_start
#     )

#     context = build_context(
#         retrieved,
#         chunks,
#     )

#     llm_start = time.perf_counter()

#     answer = generate_answer(
#         question,
#         context,
#     )

#     llm_time = (
#         time.perf_counter()
#         - llm_start
#     )

#     total_time = (
#         time.perf_counter()
#         - start
#     )

#     return {
#         "question": question,
#         "answer": answer,
#         "retrieved": retrieved,
#         "context": context,
#         "retrieval_time": retrieval_time,
#         "llm_time": llm_time,
#         "total_time": total_time,
#     }


# def main():

#     print("Loading data...")

#     questions = load_json(
#         QUESTIONS_FILE
#     )

#     chunks = load_json(
#         CHUNKS_FILE
#     )

#     embedding_data = load_json(
#         EMBEDDINGS_FILE
#     )

#     embedding_map = {
#         item["chunk_id"]: item["embedding"]
#         for item in embedding_data
#     }

#     chunk_ids = list(
#         embedding_map.keys()
#     )

#     chunk_embeddings = [
#         embedding_map[chunk_id]
#         for chunk_id in chunk_ids
#     ]

#     print(
#         f"Questions available: {len(questions)}"
#     )

#     print(
#         f"Chunks available: {len(chunks)}"
#     )

#     # Test with one question first.
#     question = questions[0]

#     print("\nRunning Normal RAG...")
#     print("-" * 50)

#     result = run_rag(
#         question["question"],
#         chunks,
#         chunk_embeddings,
#     )

#     print("\nQuestion:")
#     print(question["question"])

#     print("\nRetrieved chunks:")

#     for index, score in result["retrieved"]:
#         print(
#             f"  {chunks[index]['id']} "
#             f"(score={score:.4f})"
#         )

#     print("\nAnswer:")
#     print(result["answer"])

#     print("\nTiming:")
#     print(
#         f"Retrieval: "
#         f"{result['retrieval_time']:.3f}s"
#     )

#     print(
#         f"LLM: "
#         f"{result['llm_time']:.3f}s"
#     )

#     print(
#         f"Total: "
#         f"{result['total_time']:.3f}s"
#     )


# if __name__ == "__main__":
#     main()

import json
import time
from pathlib import Path

import ollama

from ..shared.config import GENERATION_MODEL, TOP_K
from ..shared.embeddings import embed_text
from ..shared.retrieval import retrieve


QUESTIONS_FILE = Path(
    "data/processed/chunked_questions.json"
)

CHUNKS_FILE = Path(
    "data/processed/chunks.json"
)

EMBEDDINGS_FILE = Path(
    "data/embeddings/chunk_embeddings.json"
)

RESULTS_FILE = Path(
    "results/normal_rag_results.json"
)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_context(retrieved, chunks):
    context_parts = []

    for index, score in retrieved:
        chunk = chunks[index]

        context_parts.append(
            f"[Chunk {chunk['id']}]\n"
            f"{chunk['text']}"
        )

    return "\n\n".join(context_parts)


def generate_answer(question: str, context: str):

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


def run_rag(
    question: str,
    chunks,
    chunk_embeddings,
):

    pipeline_start = time.perf_counter()

    # -------------------------
    # Query embedding
    # -------------------------

    embedding_start = time.perf_counter()

    query_embedding = embed_text(question)

    embedding_time = (
        time.perf_counter()
        - embedding_start
    )

    # -------------------------
    # Retrieval
    # -------------------------

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

    # -------------------------
    # Context construction
    # -------------------------

    context = build_context(
        retrieved,
        chunks,
    )

    # -------------------------
    # Generation
    # -------------------------

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

    return {
        "answer": answer,
        "retrieved": [
            {
                "chunk_id": chunks[index]["id"],
                "score": score,
            }
            for index, score in retrieved
        ],
        "context_characters": len(context),
        "num_chunks": len(retrieved),
        "embedding_time": embedding_time,
        "retrieval_time": retrieval_time,
        "llm_time": llm_time,
        "total_time": total_time,
    }


def main():

    print("Loading dataset...")

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

    print(
        f"Questions: {len(questions)}"
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    print(
        f"Top-K: {TOP_K}"
    )

    print(
        f"Generation model: {GENERATION_MODEL}"
    )

    print("\nStarting Normal RAG evaluation...")
    print("=" * 60)

    results = []

    evaluation_start = time.perf_counter()

    for i, question_data in enumerate(questions):

        question = question_data["question"]

        print(
            f"\n[{i + 1}/{len(questions)}] "
            f"{question}"
        )

        result = run_rag(
            question,
            chunks,
            chunk_embeddings,
        )

        record = {
            "question_id": question_data["id"],
            "question": question,
            "reference_answer": question_data["answer"],
            "relevant_chunk_id": question_data[
                "relevant_chunk_id"
            ],
            **result,
        }

        results.append(record)

        print(
            f"  LLM: {result['llm_time']:.2f}s | "
            f"Total: {result['total_time']:.2f}s"
        )

        # Save after every question.
        # If something crashes, we don't lose
        # the entire experiment.
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

    evaluation_time = (
        time.perf_counter()
        - evaluation_start
    )

    # -------------------------
    # Summary
    # -------------------------

    avg_total = sum(
        r["total_time"]
        for r in results
    ) / len(results)

    avg_llm = sum(
        r["llm_time"]
        for r in results
    ) / len(results)

    avg_embedding = sum(
        r["embedding_time"]
        for r in results
    ) / len(results)

    avg_retrieval = sum(
        r["retrieval_time"]
        for r in results
    ) / len(results)

    avg_context = sum(
        r["context_characters"]
        for r in results
    ) / len(results)

    print("\n\n")
    print("=" * 60)
    print("NORMAL RAG EVALUATION COMPLETE")
    print("=" * 60)

    print(
        f"Questions evaluated: {len(results)}"
    )

    print(
        f"Average embedding time: "
        f"{avg_embedding:.3f}s"
    )

    print(
        f"Average retrieval time: "
        f"{avg_retrieval:.3f}s"
    )

    print(
        f"Average LLM time: "
        f"{avg_llm:.3f}s"
    )

    print(
        f"Average total time: "
        f"{avg_total:.3f}s"
    )

    print(
        f"Average context size: "
        f"{avg_context:.0f} characters"
    )

    print(
        f"Total evaluation time: "
        f"{evaluation_time:.2f}s"
    )

    print("\nResults saved to:")
    print(RESULTS_FILE)


if __name__ == "__main__":
    main()