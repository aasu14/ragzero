"""Step 8: Continuous evaluation harness.

Two metric families:

1. Retrieval quality (deterministic, fast)
   - recall@k: fraction of queries where any ground-truth chunk is in top-k
   - mrr@k: mean reciprocal rank of the first relevant chunk

2. Generation quality (requires the pipeline)
   - hallucination_rate: fraction of non-refused answers whose claims
     don't appear in the cited chunks (string-overlap proxy; production
     would use an NLI model)
   - refusal_rate: fraction of queries that triggered the fallback
   - over_refusal_rate: refusals on queries that DID have valid evidence
     in the corpus (we know because the test case provides expected_doc_ids)

The harness is designed to be run continuously in CI. New corpus
versions, new prompts, and new thresholds should all gate on these
metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .interfaces import Answer, ScoredChunk
from .retrieval import RetrievalResult


@dataclass(frozen=True)
class EvalCase:
    """One test query with ground truth."""
    query: str
    expected_doc_ids: set[str]                  # any of these in top-k = recall hit
    expected_answer_substrings: list[str] = field(default_factory=list)
    is_adversarial: bool = False                # if True, expects a refusal
    notes: str = ""


@dataclass
class EvalReport:
    n_cases: int
    recall_at_k: float
    mrr_at_k: float
    refusal_rate: float
    over_refusal_rate: float
    hallucination_rate: float
    failures: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"cases={self.n_cases} recall@k={self.recall_at_k:.3f} "
            f"mrr={self.mrr_at_k:.3f} refusal={self.refusal_rate:.3f} "
            f"over_refusal={self.over_refusal_rate:.3f} "
            f"hallucination={self.hallucination_rate:.3f}"
        )


class Evaluator:
    """Runs an eval suite against a pipeline.

    The pipeline is supplied as a callable so the evaluator stays
    decoupled from the orchestrator — useful for evaluating retrieval
    in isolation as well.
    """

    def __init__(
        self,
        retrieve_fn: Callable[[str], RetrievalResult],
        answer_fn: Callable[[str], Answer] | None = None,
        k: int = 5,
    ) -> None:
        self.retrieve_fn = retrieve_fn
        self.answer_fn = answer_fn
        self.k = k

    def run(self, cases: list[EvalCase]) -> EvalReport:
        recall_hits = 0
        mrr_sum = 0.0
        refusals = 0
        over_refusals = 0
        hallucinations = 0
        non_refused = 0
        failures: list[str] = []

        for case in cases:
            # Retrieval metrics
            result = self.retrieve_fn(case.query)
            top_k_doc_ids = [sc.chunk.doc_id for sc in result.chunks[: self.k]]
            relevant_positions = [
                i for i, doc_id in enumerate(top_k_doc_ids) if doc_id in case.expected_doc_ids
            ]
            if relevant_positions:
                recall_hits += 1
                mrr_sum += 1.0 / (relevant_positions[0] + 1)
            elif not case.is_adversarial and case.expected_doc_ids:
                failures.append(f"recall_miss: {case.query!r}")

            # End-to-end answer metrics
            if self.answer_fn is not None:
                answer = self.answer_fn(case.query)
                if answer.refused:
                    refusals += 1
                    if not case.is_adversarial and case.expected_doc_ids:
                        over_refusals += 1
                        failures.append(f"over_refusal: {case.query!r}")
                else:
                    non_refused += 1
                    if self._is_hallucination(answer, result.chunks[: self.k]):
                        hallucinations += 1
                        failures.append(f"hallucination: {case.query!r}")
                    # Substring check (very loose — production uses NLI)
                    for needle in case.expected_answer_substrings:
                        if needle.lower() not in answer.text.lower():
                            failures.append(
                                f"missing_expected_substring {needle!r} in: {case.query!r}"
                            )

        n = len(cases)
        return EvalReport(
            n_cases=n,
            recall_at_k=recall_hits / n if n else 0.0,
            mrr_at_k=mrr_sum / n if n else 0.0,
            refusal_rate=refusals / n if n else 0.0,
            over_refusal_rate=over_refusals / n if n else 0.0,
            hallucination_rate=hallucinations / non_refused if non_refused else 0.0,
            failures=failures,
        )

    @staticmethod
    def _is_hallucination(answer: Answer, retrieved: list[ScoredChunk]) -> bool:
        """Proxy: count claims whose key terms appear in no cited chunk.

        Production replacement: NLI model checking entailment of each
        sentence against the cited chunks (e.g., a DeBERTa-large MNLI head).
        """
        if not answer.citations:
            return True  # cited nothing — by definition ungrounded
        cited_ids = {c.chunk_id for c in answer.citations}
        cited_texts = " ".join(
            sc.chunk.text.lower() for sc in retrieved if sc.chunk.chunk_id in cited_ids
        )
        # Crude: if more than half the answer's content words are absent
        # from the cited text, flag it.
        content_words = [
            w.strip(".,;:!?").lower() for w in answer.text.split() if len(w) > 4
        ]
        if not content_words:
            return False
        missing = sum(1 for w in content_words if w not in cited_texts)
        return (missing / len(content_words)) > 0.5
