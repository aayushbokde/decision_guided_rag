import json
import re
import string
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

RESULTS_FILE = Path(
    "results/oracle_early_exit_results_200.json"
)


# ============================================================
# SQuAD-style normalization
# ============================================================

def normalize_answer(text: str) -> str:
    """
    Normalize an answer using the standard SQuAD-style
    normalization:

    1. Lowercase
    2. Remove punctuation
    3. Remove articles
    4. Normalize whitespace
    """

    def remove_articles(text):
        return re.sub(
            r"\b(a|an|the)\b",
            " ",
            text,
        )

    def white_space_fix(text):
        return " ".join(
            text.split()
        )

    def remove_punc(text):
        return "".join(
            ch
            for ch in text
            if ch not in string.punctuation
        )

    def lower(text):
        return text.lower()

    return white_space_fix(
        remove_articles(
            remove_punc(
                lower(text)
            )
        )
    )


# ============================================================
# Exact Match
# ============================================================

def exact_match(
    prediction: str,
    reference: str,
) -> int:

    return int(
        normalize_answer(prediction)
        == normalize_answer(reference)
    )


# ============================================================
# Token F1
# ============================================================

def token_f1(
    prediction: str,
    reference: str,
) -> float:

    prediction_tokens = (
        normalize_answer(
            prediction
        ).split()
    )

    reference_tokens = (
        normalize_answer(
            reference
        ).split()
    )

    if (
        not prediction_tokens
        or not reference_tokens
    ):
        return float(
            prediction_tokens
            == reference_tokens
        )

    common = set(
        prediction_tokens
    ) & set(
        reference_tokens
    )

    num_same = sum(
        min(
            prediction_tokens.count(token),
            reference_tokens.count(token),
        )
        for token in common
    )

    if num_same == 0:
        return 0.0

    precision = (
        num_same
        / len(prediction_tokens)
    )

    recall = (
        num_same
        / len(reference_tokens)
    )

    return (
        2 * precision * recall
        / (precision + recall)
    )


# ============================================================
# Load results
# ============================================================

def load_results():

    print(
        "Loading oracle early-exit results..."
    )

    with open(
        RESULTS_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# Evaluation
# ============================================================

def main():

    results = load_results()

    print(
        f"Questions evaluated: "
        f"{len(results)}"
    )

    print()

    exact_scores = []
    f1_scores = []

    evaluations = []

    # --------------------------------------------------------
    # Evaluate every question
    # --------------------------------------------------------

    for result in results:

        prediction = result[
            "answer"
        ]

        reference = result[
            "reference_answer"
        ]

        em = exact_match(
            prediction,
            reference,
        )

        f1 = token_f1(
            prediction,
            reference,
        )

        exact_scores.append(
            em
        )

        f1_scores.append(
            f1
        )

        evaluations.append(
            {
                "question_id": result[
                    "question_id"
                ],
                "reference": reference,
                "prediction": prediction,
                "exact_match": em,
                "token_f1": f1,
                "retrieval_depth": result[
                    "retrieval_depth"
                ],
                "gold_chunk_survived": result[
                    "gold_chunk_survived"
                ],
            }
        )

    # --------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------

    total = len(results)

    exact_match_score = (
        sum(exact_scores)
        / total
    )

    f1_score = (
        sum(f1_scores)
        / total
    )

    # --------------------------------------------------------
    # Additional diagnostics
    # --------------------------------------------------------

    gold_survival = (
        sum(
            result[
                "gold_chunk_survived"
            ]
            for result in results
        )
        / total
    )

    early_exit_count = sum(
        result[
            "exited_early"
        ]
        for result in results
    )

    early_exit_rate = (
        early_exit_count
        / total
    )

    average_depth = (
        sum(
            result[
                "retrieval_depth"
            ]
            for result in results
        )
        / total
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("=" * 55)
    print(
        "ORACLE EARLY-EXIT RAG ANSWER QUALITY"
    )
    print("=" * 55)

    print(
        f"Exact Match: "
        f"{exact_match_score:.4f} "
        f"({exact_match_score * 100:.2f}%)"
    )

    print(
        f"Token F1:    "
        f"{f1_score:.4f} "
        f"({f1_score * 100:.2f}%)"
    )

    print()

    print(
        f"Average retrieval depth: "
        f"{average_depth:.2f}"
    )

    print(
        f"Early-exit rate: "
        f"{early_exit_rate * 100:.2f}%"
    )

    print(
        f"Gold chunk survival: "
        f"{gold_survival * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Example evaluations
    # --------------------------------------------------------

    print()
    print("## Example evaluations:")
    print()

    for evaluation in evaluations[:5]:

        print(
            f"Question ID: "
            f"{evaluation['question_id']}"
        )

        print(
            f"Reference: "
            f"{evaluation['reference']}"
        )

        print(
            f"Generated: "
            f"{evaluation['prediction']}"
        )

        print(
            f"EM: "
            f"{evaluation['exact_match']} "
            f"| F1: "
            f"{evaluation['token_f1']:.4f}"
        )

        print()

    # --------------------------------------------------------
    # Failure analysis
    # --------------------------------------------------------

    failed_gold = [
        evaluation
        for evaluation in evaluations
        if not evaluation[
            "gold_chunk_survived"
        ]
    ]

    print(
        f"Questions where gold chunk "
        f"was not retained: "
        f"{len(failed_gold)}"
    )

    # --------------------------------------------------------
    # Save evaluation results
    # --------------------------------------------------------

    output_file = Path(
        "results/"
        "oracle_early_exit_answer_evaluation.json"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation_output = {
        "questions": total,
        "exact_match": exact_match_score,
        "token_f1": f1_score,
        "average_retrieval_depth": average_depth,
        "early_exit_rate": early_exit_rate,
        "gold_chunk_survival": gold_survival,
        "evaluations": evaluations,
    }

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            evaluation_output,
            f,
            indent=2,
        )

    print()
    print(
        f"Evaluation saved to: "
        f"{output_file}"
    )


if __name__ == "__main__":
    main()