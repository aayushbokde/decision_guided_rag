from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Decision:
    """
    Result returned by a decision backend.
    """

    keep: bool
    confidence: float
    backend: str


class DecisionBackend(ABC):

    @abstractmethod
    def decide(
        self,
        question: str,
        chunk: str,
    ) -> Decision:
        """
        Decide whether a retrieved chunk should be kept.
        """
        pass