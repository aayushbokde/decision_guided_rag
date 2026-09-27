import json
import random
from pathlib import Path


RAW_FILE = Path("data/raw/squad/dev-v1.1.json")
OUTPUT_DIR = Path("data/processed")

DOCUMENTS_FILE = OUTPUT_DIR / "documents.json"
QUESTIONS_FILE = OUTPUT_DIR / "questions.json"

TARGET_QUESTIONS = 200
SEED = 42


def main():
    print(f"Loading SQuAD from: {RAW_FILE}")

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_FILE}"
        )

    with open(RAW_FILE, "r", encoding="utf-8") as f:
        squad = json.load(f)

    # ---------------------------------------------------------
    # Collect all unique contexts and their questions.
    # ---------------------------------------------------------

    contexts = {}

    for article in squad["data"]:

        title = article["title"]

        for paragraph in article["paragraphs"]:

            context = paragraph["context"]

            if context not in contexts:
                contexts[context] = {
                    "title": title,
                    "qas": []
                }

            contexts[context]["qas"].extend(
                paragraph["qas"]
            )

    print(f"Unique contexts available: {len(contexts)}")

    # ---------------------------------------------------------
    # Randomize contexts deterministically.
    # ---------------------------------------------------------

    context_items = list(contexts.items())

    random.seed(SEED)
    random.shuffle(context_items)

    # ---------------------------------------------------------
    # Select diverse contexts until we reach our target.
    # ---------------------------------------------------------

    selected_contexts = []
    question_count = 0

    for context, metadata in context_items:

        if question_count >= TARGET_QUESTIONS:
            break

        selected_contexts.append(
            (context, metadata)
        )

        question_count += len(metadata["qas"])

    # ---------------------------------------------------------
    # Build documents and questions.
    # ---------------------------------------------------------

    documents = []
    questions = []

    context_to_document_id = {}

    for context, metadata in selected_contexts:

        document_id = f"doc_{len(documents):04d}"

        context_to_document_id[context] = document_id

        documents.append(
            {
                "id": document_id,
                "title": metadata["title"],
                "text": context,
            }
        )

        for qa in metadata["qas"]:

            if len(questions) >= TARGET_QUESTIONS:
                break

            answer = qa["answers"][0]

            questions.append(
                {
                    "id": qa["id"],
                    "question": qa["question"],
                    "answer": answer["text"],
                    "answer_start": answer["answer_start"],
                    "document_id": document_id,
                }
            )

        if len(questions) >= TARGET_QUESTIONS:
            break

    # ---------------------------------------------------------
    # Save.
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        DOCUMENTS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            documents,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        QUESTIONS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            questions,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nDataset preparation complete.")
    print(f"Questions: {len(questions)}")
    print(f"Documents: {len(documents)}")

    print("\nSaved:")
    print(f"  {DOCUMENTS_FILE}")
    print(f"  {QUESTIONS_FILE}")


if __name__ == "__main__":
    main()