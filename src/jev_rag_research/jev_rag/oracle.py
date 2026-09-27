import re
from collections import Counter

from .decision import Decision, DecisionBackend


class KeywordOracle(DecisionBackend):
    """
    Deterministic relevance oracle used only for
    pipeline development and debugging.

    This is NOT JEV.

    It uses lexical overlap between the question
    and retrieved chunk.
    """

    def __init__(
        self,
        threshold: float = 0.20,
    ):
        self.threshold = threshold

    @staticmethod
    def tokenize(text: str) -> list[str]:

        text = text.lower()

        text = re.sub(
            r"[^\w\s]",
            " ",
            text,
        )

        return text.split()

    def decide(
        self,
        question: str,
        chunk: str,
    ) -> Decision:

        question_tokens = Counter(
            self.tokenize(question)
        )

        chunk_tokens = set(
            self.tokenize(chunk)
        )

        if not question_tokens:
            return Decision(
                keep=False,
                confidence=0.0,
                backend="keyword_oracle",
            )

        matched = sum(
            count
            for token, count
            in question_tokens.items()
            if token in chunk_tokens
        )

        total = sum(
            question_tokens.values()
        )

        score = matched / total

        return Decision(
            keep=score >= self.threshold,
            confidence=score,
            backend="keyword_oracle",
        )
