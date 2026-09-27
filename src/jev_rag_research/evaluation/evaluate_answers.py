import json
import re
from collections import Counter
from pathlib import Path


RESULTS_FILE = Path(
    "results/normal_rag_results.json"
)


def normalize_answer(text: str) -> str:
    """
    Normalize an answer using the standard SQuAD-style
    normalization procedure.
    """

    text = text.lower()

    # Remove punctuation
    text = re.sub(
        r"[^\w\s]",
        "",
        text,
    )

    # Remove articles
    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text,
    )

    # Normalize whitespace
    text = " ".join(text.split())

    return text


def tokenize(text: str) -> list[str]:

    normalized = normalize_answer(text)

    return normalized.split()


def exact_match(
    prediction: str,
    reference: str,
) -> float:

    return float(
        normalize_answer(prediction)
        == normalize_answer(reference)
    )


def f1_score(
    prediction: str,
    reference: str,
) -> float:

    prediction_tokens = tokenize(
        prediction
    )

    reference_tokens = tokenize(
        reference
    )

    if not prediction_tokens or not reference_tokens:
        return float(
            prediction_tokens == reference_tokens
        )

    prediction_counter = Counter(
        prediction_tokens
    )

    reference_counter = Counter(
        reference_tokens
    )

    common = (
        prediction_counter
        & reference_counter
    )

    num_common = sum(
        common.values()
    )

    if num_common == 0:
        return 0.0

    precision = (
        num_common
        / len(prediction_tokens)
    )

    recall = (
        num_common
        / len(reference_tokens)
    )

    return (
        2 * precision * recall
        / (precision + recall)
    )


def main():

    print("Loading Normal RAG results...")

    with open(
        RESULTS_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        results = json.load(f)

    print(
        f"Questions evaluated: {len(results)}"
    )

    exact_matches = []
    f1_scores = []

    for result in results:

        prediction = result["answer"]
        reference = result["reference_answer"]

        em = exact_match(
            prediction,
            reference,
        )

        f1 = f1_score(
            prediction,
            reference,
        )

        exact_matches.append(em)
        f1_scores.append(f1)

    avg_em = (
        sum(exact_matches)
        / len(exact_matches)
    )

    avg_f1 = (
        sum(f1_scores)
        / len(f1_scores)
    )

    print("\n")
    print("=" * 50)
    print("NORMAL RAG ANSWER QUALITY")
    print("=" * 50)

    print(
        f"Exact Match: {avg_em:.4f} "
        f"({avg_em * 100:.2f}%)"
    )

    print(
        f"Token F1:    {avg_f1:.4f} "
        f"({avg_f1 * 100:.2f}%)"
    )

    # Show a few examples
    print("\nExample evaluations:")
    print("-" * 50)

    for i in range(min(5, len(results))):

        result = results[i]

        print(
            f"\nQuestion: "
            f"{result['question']}"
        )

        print(
            f"Reference: "
            f"{result['reference_answer']}"
        )

        print(
            f"Generated: "
            f"{result['answer']}"
        )

        print(
            f"EM: {exact_matches[i]:.0f} | "
            f"F1: {f1_scores[i]:.4f}"
        )


if __name__ == "__main__":
    main()