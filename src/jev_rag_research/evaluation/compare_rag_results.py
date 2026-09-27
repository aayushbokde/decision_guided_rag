import json
import math
import re
from collections import Counter
from pathlib import Path


NORMAL_RESULTS_FILE = Path(
    "results/normal_rag_results.json"
)

FILTERED_RESULTS_FILE = Path(
    "results/oracle_filtered_rag_results_200.json"
)

OUTPUT_FILE = Path(
    "results/paired_comparison.json"
)


# ---------------------------------------------------------
# SQuAD-style answer evaluation
# ---------------------------------------------------------

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

    return " ".join(text.split())


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


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def load_json(path: Path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def mean(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


def pearson_correlation(x, y):
    if len(x) < 2 or len(y) < 2:
        return None

    x_mean = mean(x)
    y_mean = mean(y)

    numerator = sum(
        (a - x_mean) * (b - y_mean)
        for a, b in zip(x, y)
    )

    denominator_x = math.sqrt(
        sum(
            (a - x_mean) ** 2
            for a in x
        )
    )

    denominator_y = math.sqrt(
        sum(
            (b - y_mean) ** 2
            for b in y
        )
    )

    denominator = (
        denominator_x * denominator_y
    )

    if denominator == 0:
        return None

    return numerator / denominator


# ---------------------------------------------------------
# Main analysis
# ---------------------------------------------------------

def main():

    print("Loading result files...")

    normal_results = load_json(
        NORMAL_RESULTS_FILE
    )

    filtered_results = load_json(
        FILTERED_RESULTS_FILE
    )

    print(
        f"Normal RAG results: "
        f"{len(normal_results)}"
    )

    print(
        f"Filtered RAG results: "
        f"{len(filtered_results)}"
    )

    # -----------------------------------------------------
    # Build lookup tables using question_id
    # -----------------------------------------------------

    normal_by_id = {
        result["question_id"]: result
        for result in normal_results
    }

    filtered_by_id = {
        result["question_id"]: result
        for result in filtered_results
    }

    normal_ids = set(normal_by_id)
    filtered_ids = set(filtered_by_id)

    common_ids = normal_ids & filtered_ids

    missing_from_normal = (
        filtered_ids - normal_ids
    )

    missing_from_filtered = (
        normal_ids - filtered_ids
    )

    print()
    print("=" * 70)
    print("DATASET ALIGNMENT")
    print("=" * 70)

    print(
        f"Common question IDs: "
        f"{len(common_ids)}"
    )

    print(
        f"Missing from Normal: "
        f"{len(missing_from_normal)}"
    )

    print(
        f"Missing from Filtered: "
        f"{len(missing_from_filtered)}"
    )

    if missing_from_normal or missing_from_filtered:
        print(
            "\nWARNING: Result files are not perfectly aligned."
        )

    # -----------------------------------------------------
    # Paired metrics
    # -----------------------------------------------------

    normal_em = []
    filtered_em = []

    normal_f1 = []
    filtered_f1 = []

    f1_improved = 0
    f1_worsened = 0
    f1_equal = 0

    paired_results = []

    context_reductions = []
    chunks_removed = []

    latency_normal = []
    latency_filtered = []

    failed_gold_survival = []

    for question_id in sorted(common_ids):

        normal = normal_by_id[question_id]
        filtered = filtered_by_id[question_id]

        reference = filtered["reference_answer"]

        normal_answer = normal["answer"]
        filtered_answer = filtered["answer"]

        # Answer metrics
        n_em = exact_match(
            normal_answer,
            reference,
        )

        f_em = exact_match(
            filtered_answer,
            reference,
        )

        n_f1 = f1_score(
            normal_answer,
            reference,
        )

        f_f1 = f1_score(
            filtered_answer,
            reference,
        )

        normal_em.append(n_em)
        filtered_em.append(f_em)

        normal_f1.append(n_f1)
        filtered_f1.append(f_f1)

        # F1 comparison
        f1_difference = f_f1 - n_f1

        if f1_difference > 1e-12:
            f1_improved += 1
        elif f1_difference < -1e-12:
            f1_worsened += 1
        else:
            f1_equal += 1

        # Context
        context_reduction = filtered.get(
            "context_reduction",
            0.0,
        )

        context_reductions.append(
            context_reduction
        )

        normal_chunks = len(normal["retrieved"])

        filtered_chunks = filtered.get(
            "num_selected",
            0,
        )

        chunks_removed.append(
            normal_chunks - filtered_chunks
        )

        # Latency
        if "total_time" in normal:
            latency_normal.append(
                normal["total_time"]
            )

        if "total_time" in filtered:
            latency_filtered.append(
                filtered["total_time"]
            )

        # Gold chunk survival
        if not filtered.get(
            "gold_chunk_survived",
            True,
        ):
            failed_gold_survival.append(
                {
                    "question_id": question_id,
                    "reference_answer": reference,
                    "normal_answer": normal_answer,
                    "filtered_answer": filtered_answer,
                    "normal_f1": n_f1,
                    "filtered_f1": f_f1,
                    "f1_difference": f1_difference,
                    "relevant_chunk_id": filtered.get(
                        "relevant_chunk_id"
                    ),
                    "selected_chunks": filtered.get(
                        "selected_chunks"
                    ),
                }
            )

        paired_results.append(
            {
                "question_id": question_id,
                "reference_answer": reference,
                "normal_f1": n_f1,
                "filtered_f1": f_f1,
                "f1_difference": f1_difference,
                "context_reduction": context_reduction,
                "chunks_removed": (
                    normal_chunks - filtered_chunks
                ),
            }
        )

    # -----------------------------------------------------
    # Aggregate metrics
    # -----------------------------------------------------

    avg_normal_em = mean(normal_em)
    avg_filtered_em = mean(filtered_em)

    avg_normal_f1 = mean(normal_f1)
    avg_filtered_f1 = mean(filtered_f1)

    avg_context_reduction = mean(
        context_reductions
    )

    avg_chunks_removed = mean(
        chunks_removed
    )

    avg_normal_latency = mean(
        latency_normal
    )

    avg_filtered_latency = mean(
        latency_filtered
    )

    latency_difference = (
        avg_filtered_latency
        - avg_normal_latency
    )

    f1_differences = [
        item["f1_difference"]
        for item in paired_results
    ]

    correlation = pearson_correlation(
        context_reductions,
        f1_differences,
    )

    # -----------------------------------------------------
    # Print summary
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("PAIRED NORMAL RAG vs ORACLE-FILTERED RAG")
    print("=" * 70)

    print(
        f"Paired questions: "
        f"{len(common_ids)}"
    )

    print()
    print("ANSWER QUALITY")
    print("-" * 70)

    print(
        f"Normal EM:   "
        f"{avg_normal_em:.4f} "
        f"({avg_normal_em * 100:.2f}%)"
    )

    print(
        f"Filtered EM: "
        f"{avg_filtered_em:.4f} "
        f"({avg_filtered_em * 100:.2f}%)"
    )

    print(
        f"EM difference: "
        f"{(avg_filtered_em - avg_normal_em) * 100:+.2f} percentage points"
    )

    print()

    print(
        f"Normal F1:   "
        f"{avg_normal_f1:.4f} "
        f"({avg_normal_f1 * 100:.2f}%)"
    )

    print(
        f"Filtered F1: "
        f"{avg_filtered_f1:.4f} "
        f"({avg_filtered_f1 * 100:.2f}%)"
    )

    print(
        f"F1 difference: "
        f"{(avg_filtered_f1 - avg_normal_f1) * 100:+.2f} percentage points"
    )

    print()
    print("PER-QUESTION F1 CHANGES")
    print("-" * 70)

    print(
        f"Improved:  {f1_improved}"
    )

    print(
        f"Worsened:  {f1_worsened}"
    )

    print(
        f"Equal:     {f1_equal}"
    )

    print()
    print("EFFICIENCY")
    print("-" * 70)

    print(
        f"Average chunks removed: "
        f"{avg_chunks_removed:.2f}"
    )

    print(
        f"Average context reduction: "
        f"{avg_context_reduction * 100:.2f}%"
    )

    if latency_normal and latency_filtered:

        print(
            f"Normal avg total time: "
            f"{avg_normal_latency:.3f}s"
        )

        print(
            f"Filtered avg total time: "
            f"{avg_filtered_latency:.3f}s"
        )

        print(
            f"Latency difference: "
            f"{latency_difference:+.3f}s"
        )

    print()
    print("GOLD CHUNK SURVIVAL")
    print("-" * 70)

    gold_survival_count = (
        len(common_ids)
        - len(failed_gold_survival)
    )

    gold_survival_rate = (
        gold_survival_count
        / len(common_ids)
        if common_ids
        else 0.0
    )

    print(
        f"Survived: "
        f"{gold_survival_count}/{len(common_ids)} "
        f"({gold_survival_rate * 100:.2f}%)"
    )

    print(
        f"Lost: "
        f"{len(failed_gold_survival)}"
    )

    print()
    print(
        "CORRELATION: CONTEXT REDUCTION vs F1 CHANGE"
    )
    print("-" * 70)

    if correlation is None:
        print(
            "Correlation could not be calculated."
        )
    else:
        print(
            f"Pearson r: {correlation:.4f}"
        )

    # -----------------------------------------------------
    # Gold chunk failures
    # -----------------------------------------------------

    if failed_gold_survival:

        print()
        print("=" * 70)
        print("QUESTIONS WHERE GOLD CHUNK WAS FILTERED OUT")
        print("=" * 70)

        for failure in failed_gold_survival:

            print()
            print(
                f"Question ID: "
                f"{failure['question_id']}"
            )

            print(
                f"Reference: "
                f"{failure['reference_answer']}"
            )

            print(
                f"Relevant chunk: "
                f"{failure['relevant_chunk_id']}"
            )

            print(
                f"Normal F1: "
                f"{failure['normal_f1']:.4f}"
            )

            print(
                f"Filtered F1: "
                f"{failure['filtered_f1']:.4f}"
            )

            print(
                f"F1 difference: "
                f"{failure['f1_difference']:+.4f}"
            )

            print(
                f"Selected chunks: "
                f"{failure['selected_chunks']}"
            )

    # -----------------------------------------------------
    # Save analysis
    # -----------------------------------------------------

    summary = {
        "paired_questions": len(common_ids),

        "normal": {
            "exact_match": avg_normal_em,
            "f1": avg_normal_f1,
        },

        "filtered": {
            "exact_match": avg_filtered_em,
            "f1": avg_filtered_f1,
        },

        "differences": {
            "exact_match": (
                avg_filtered_em
                - avg_normal_em
            ),
            "f1": (
                avg_filtered_f1
                - avg_normal_f1
            ),
        },

        "f1_per_question": {
            "improved": f1_improved,
            "worsened": f1_worsened,
            "equal": f1_equal,
        },

        "efficiency": {
            "average_chunks_removed": avg_chunks_removed,
            "average_context_reduction": avg_context_reduction,
            "normal_average_total_time": avg_normal_latency,
            "filtered_average_total_time": avg_filtered_latency,
            "latency_difference": latency_difference,
        },

        "gold_chunk": {
            "survived": gold_survival_count,
            "lost": len(failed_gold_survival),
            "survival_rate": gold_survival_rate,
            "failures": failed_gold_survival,
        },

        "correlation": {
            "context_reduction_vs_f1_change": correlation,
        },

        "paired_results": paired_results,
    }

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
            summary,
            f,
            indent=2,
        )

    print()
    print(
        f"Analysis saved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()