from .oracle import KeywordOracle


def main():

    oracle = KeywordOracle(
        threshold=0.20
    )

    question = (
        "What metal was used in "
        "Inalchuq's execution?"
    )

    relevant_chunk = (
        "Inalchuq was executed by having "
        "molten silver poured into his eyes "
        "and ears."
    )

    irrelevant_chunk = (
        "Genghis Khan established a postal "
        "system across the Mongol Empire."
    )

    print("Decision backend test")
    print("=" * 50)

    print("\nRelevant chunk:")

    decision = oracle.decide(
        question,
        relevant_chunk,
    )

    print(decision)

    print("\nIrrelevant chunk:")

    decision = oracle.decide(
        question,
        irrelevant_chunk,
    )

    print(decision)


if __name__ == "__main__":
    main()