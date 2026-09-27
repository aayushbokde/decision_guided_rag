import json
import re
from collections import Counter
from pathlib import Path


RESULTS_FILE = Path(
    "results/oracle_filtered_rag_results_200.json"
)


def normalize_answer(text: str) -> str:
    text = text.lower()

    text = re.sub(
        r"[^\w\s]",
        "",
        text,
    )

    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text,
    )

    text = " ".join(
        text.split()
    )

    return text


def tokenize(text: str) -> list[str]:
    return normalize_answer(text).split()


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

    prediction_tokens = tokenize(prediction)
    reference_tokens = tokenize(reference)

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

    print(
        "Loading oracle-filtered results..."
    )

    with open(
        RESULTS_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        results = json.load(f)

    if not results:
        print("No results found.")
        return

    print(
        f"Questions evaluated: "
        f"{len(results)}"
    )

    em_scores = []
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

        em_scores.append(em)
        f1_scores.append(f1)

    avg_em = (
        sum(em_scores)
        / len(em_scores)
    )

    avg_f1 = (
        sum(f1_scores)
        / len(f1_scores)
    )

    print()
    print("=" * 50)
    print(
        "ORACLE-FILTERED RAG ANSWER QUALITY"
    )
    print("=" * 50)

    print(
        f"Exact Match: "
        f"{avg_em:.4f} "
        f"({avg_em * 100:.2f}%)"
    )

    print(
        f"Token F1:    "
        f"{avg_f1:.4f} "
        f"({avg_f1 * 100:.2f}%)"
    )

    print()
    print("Example evaluations:")
    print("-" * 50)

    for i in range(
        min(5, len(results))
    ):

        result = results[i]

        print(
            f"\nQuestion ID: "
            f"{result['question_id']}"
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
            f"EM: {em_scores[i]:.0f} | "
            f"F1: {f1_scores[i]:.4f}"
        )


if __name__ == "__main__":
    main()