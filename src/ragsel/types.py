"""Data contracts. Gold labels are a separate type that retrieval code never receives."""
from __future__ import annotations

from dataclasses import dataclass, field

ANSWER_KINDS = ("span", "yesno", "option", "binary")


@dataclass(frozen=True)
class Corpus:
    dataset: str
    docs: tuple[str, ...]
    titles: tuple[str, ...]          # "" when the source has no title


@dataclass(frozen=True)
class PublicQuery:
    """What a deployed system would see: the question and, if the task has them, answer options."""
    qid: str
    dataset: str
    group: str                       # source document/group: the bootstrap block
    text: str                        # used for retrieval
    prompt_text: str                 # shown to the generator (adds options/candidate when the task has them)


@dataclass(frozen=True)
class Gold:
    """Evaluation-only labels. Never passed to routes, index or features."""
    qid: str
    answers: tuple[str, ...]
    support: tuple[int, ...]         # doc ids in Corpus.docs; () when the source has no support labels
    hops: int | None                 # annotated hop count where the dataset provides it
    kind: str                        # one of ANSWER_KINDS
    options: tuple[str, ...] = ()    # all answer options for kind == "option"

    def __post_init__(self) -> None:
        if self.kind not in ANSWER_KINDS:
            raise ValueError(f"unknown answer kind {self.kind!r}")
        if not self.answers:
            raise ValueError(f"{self.qid}: gold has no answers")


@dataclass(frozen=True)
class RetrievalResult:
    doc_ids: tuple[int, ...]
    search_calls: int
    llm_calls: int
    seconds: float
    fallback: bool = False
    extra: dict = field(default_factory=dict, compare=False)
