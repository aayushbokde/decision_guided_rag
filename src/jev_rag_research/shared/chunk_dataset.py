import json
from pathlib import Path

from .config import CHUNK_SIZE, CHUNK_OVERLAP


DOCUMENTS_FILE = Path("data/processed/documents.json")
QUESTIONS_FILE = Path("data/processed/questions.json")

CHUNKS_FILE = Path("data/processed/chunks.json")
CHUNKED_QUESTIONS_FILE = Path("data/processed/chunked_questions.json")


def create_chunks(text: str, document_id: str):
    """
    Split a document into overlapping character-based chunks.
    """

    chunks = []

    start = 0
    chunk_index = 0

    while start < len(text):

        end = min(
            start + CHUNK_SIZE,
            len(text),
        )

        chunk_text = text[start:end]

        chunks.append(
            {
                "id": f"{document_id}_chunk_{chunk_index:03d}",
                "document_id": document_id,
                "text": chunk_text,
                "start": start,
                "end": end,
            }
        )

        if end == len(text):
            break

        start = end - CHUNK_OVERLAP
        chunk_index += 1

    return chunks


def find_answer_chunk(
    chunks,
    answer_start: int,
    answer_text: str,
):
    """
    Find the chunk containing the answer span.
    """

    answer_end = answer_start + len(answer_text)

    # First try to find a chunk containing
    # the entire answer.
    for chunk in chunks:

        if (
            chunk["start"] <= answer_start
            and chunk["end"] >= answer_end
        ):
            return chunk["id"]

    # Fallback: find a chunk containing the
    # beginning of the answer.
    for chunk in chunks:

        if (
            chunk["start"]
            <= answer_start
            < chunk["end"]
        ):
            return chunk["id"]

    return None


def main():

    print("Loading processed dataset...")

    with open(
        DOCUMENTS_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        documents = json.load(f)

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        questions = json.load(f)

    # ---------------------------------------------------------
    # Create chunks.
    # ---------------------------------------------------------

    all_chunks = []

    chunks_by_document = {}

    for document in documents:

        chunks = create_chunks(
            document["text"],
            document["id"],
        )

        chunks_by_document[
            document["id"]
        ] = chunks

        all_chunks.extend(chunks)

    # ---------------------------------------------------------
    # Map every question to its relevant chunk.
    # ---------------------------------------------------------

    chunked_questions = []

    skipped = 0

    for question in questions:

        document_chunks = chunks_by_document[
            question["document_id"]
        ]

        relevant_chunk_id = find_answer_chunk(
            document_chunks,
            question["answer_start"],
            question["answer"],
        )

        if relevant_chunk_id is None:
            skipped += 1
            continue

        chunked_questions.append(
            {
                "id": question["id"],
                "question": question["question"],
                "answer": question["answer"],
                "document_id": question["document_id"],
                "relevant_chunk_id": relevant_chunk_id,
            }
        )

    # ---------------------------------------------------------
    # Save.
    # ---------------------------------------------------------

    with open(
        CHUNKS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            all_chunks,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        CHUNKED_QUESTIONS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            chunked_questions,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nChunking complete.")

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(all_chunks)}")
    print(f"Questions: {len(questions)}")
    print(f"Mapped questions: {len(chunked_questions)}")
    print(f"Skipped questions: {skipped}")

    print("\nSaved:")
    print(f"  {CHUNKS_FILE}")
    print(f"  {CHUNKED_QUESTIONS_FILE}")


if __name__ == "__main__":
    main()