"""Multilingual decorator.

Wraps any other Strategy with:
- Detect query language
- Translate query → English (or whatever the corpus language is)
- Run the inner strategy
- Translate the answer → target language

This is what makes composition work: GraphStrategy + multilingual = ask in
Hindi, traverse English graph, answer in Hindi. Same for Agentic + multilingual.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Iterator

from ..interfaces import Answer, Citation
from .base import Strategy, StrategyContext, StrategyEvent, make_event


class MultilingualStrategy(Strategy):
    id = "multilingual"
    label = "Multilingual wrapper"

    def __init__(
        self,
        inner: Strategy,
        translator,                       # rag.multilingual.Translator
        corpus_language: str = "en",
        output_language: str = "auto",    # "auto" = match the query's language
    ) -> None:
        self.inner = inner
        self.translator = translator
        self.corpus_language = corpus_language
        self.output_language = output_language

    @property
    def label(self) -> str:  # type: ignore[override]
        return f"Multilingual({self.inner.label})"

    def run(self, query: str, ctx: StrategyContext) -> Iterator[StrategyEvent]:
        yield make_event("info", "Multilingual pre-processing")

        # 1. Detect and translate query
        detected = self.translator.detect_language(query)
        yield make_event(
            "translation",
            f"Detected query language: {detected}",
            detected_language=detected,
        )

        if detected != self.corpus_language:
            t = self.translator.translate(query, target_lang=self.corpus_language, source_lang=detected)
            yield make_event(
                "translation",
                f"Translated query {detected} → {self.corpus_language}",
                original=query,
                translated=t.text,
            )
            working_query = t.text
        else:
            working_query = query

        # 2. Run the inner strategy and collect events
        final_event = None
        for ev in self.inner.run(working_query, ctx):
            if ev.kind == "final":
                final_event = ev
            else:
                yield ev

        if final_event is None:
            # Inner strategy didn't produce a final — emit our own error final
            yield make_event(
                "final",
                "Inner strategy produced no final answer",
                answer=Answer(
                    text="", citations=[], confidence=0.0,
                    refused=True, refusal_reason="inner strategy did not finalize",
                    trace_id=ctx.trace_id,
                ),
            )
            return

        # 3. Translate the answer back if needed
        inner_answer: Answer = final_event.data["answer"]
        target_lang = self.output_language if self.output_language != "auto" else detected
        if (
            not inner_answer.refused
            and inner_answer.text
            and target_lang != self.corpus_language
        ):
            yield make_event("info", f"Translating answer {self.corpus_language} → {target_lang}")
            t = self.translator.translate(
                inner_answer.text, target_lang=target_lang, source_lang=self.corpus_language
            )
            yield make_event(
                "translation",
                f"Answer translated → {target_lang}",
                original=inner_answer.text,
                translated=t.text,
            )
            translated_answer = Answer(
                text=t.text,
                citations=inner_answer.citations,
                confidence=inner_answer.confidence,
                refused=False,
                trace_id=inner_answer.trace_id,
            )
            yield make_event("final", "Done (translated)", answer=translated_answer)
            return

        yield make_event("final", "Done", answer=inner_answer)
