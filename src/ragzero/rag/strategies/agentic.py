"""Agentic RAG strategy — reasoning loop.

The agent:
1. Plans (decomposes the query into sub-questions)
2. Iteratively retrieves evidence for each sub-question
3. Reflects on whether enough evidence has been gathered
4. Synthesizes a final answer with citations

Each iteration emits a "thought" event so the UI can show the agent's work.

The reasoning loop has a hard `max_iterations` cap to prevent runaway costs.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterator

from ..generation import to_answer
from ..fallback import refusal_answer
from ..interfaces import ScoredChunk
from .base import Strategy, StrategyContext, StrategyEvent, make_event


PLAN_PROMPT = """\
You are an agent answering a question by searching a document corpus. Break the
question into 1-4 focused sub-questions that, together, will let you answer.

Return ONLY a JSON object:
{{"sub_questions": ["...", "..."], "reasoning": "<one sentence on why these>"}}

Question: {query}

JSON:"""


REFLECT_PROMPT = """\
You are an agent collecting evidence to answer a question.

Original question: {query}

Evidence collected so far (from previous searches):
{evidence_summary}

Decide what to do next. Return ONLY a JSON object:
{{
  "decision": "<one of: 'have_enough', 'need_more'>",
  "next_question": "<if need_more: the next focused search query, else empty>",
  "reasoning": "<one sentence>"
}}

JSON:"""


@dataclass
class _AgentStep:
    sub_question: str
    chunk_ids: list[str]
    snippet: str  # short summary for the reflection prompt


class AgenticStrategy(Strategy):
    id = "agentic"
    label = "Agentic RAG (reason → search → reflect → answer)"

    def __init__(self, max_iterations: int = 3) -> None:
        self.max_iterations = max_iterations

    def run(self, query: str, ctx: StrategyContext) -> Iterator[StrategyEvent]:
        filters = ctx.options.get("filters") if ctx.options else None
        yield make_event("info", "Starting agentic loop", query=query, max_iterations=self.max_iterations)

        # Step 1: plan
        plan = self._plan(query, ctx)
        yield make_event(
            "thought",
            f"Plan: decomposed into {len(plan['sub_questions'])} sub-question(s)",
            sub_questions=plan["sub_questions"],
            reasoning=plan["reasoning"],
        )

        # Step 2: iterative retrieval
        steps: list[_AgentStep] = []
        all_scored: dict[str, ScoredChunk] = {}  # chunk_id -> ScoredChunk

        sub_qs = list(plan["sub_questions"]) or [query]
        iteration = 0
        while iteration < self.max_iterations and sub_qs:
            current = sub_qs.pop(0)
            iteration += 1
            yield make_event(
                "tool_call",
                f"retrieve(iteration={iteration})",
                sub_question=current,
            )

            retrieval = ctx.pipeline.retriever.search(current, filters=filters)
            yield make_event(
                "tool_result",
                f"Retrieved {len(retrieval.chunks)} chunk(s) for sub-question",
                chunk_ids=[sc.chunk.chunk_id for sc in retrieval.chunks[:5]],
            )

            for sc in retrieval.chunks[:5]:
                # Track best score per chunk
                existing = all_scored.get(sc.chunk.chunk_id)
                if not existing or sc.retrieval_score > existing.retrieval_score:
                    all_scored[sc.chunk.chunk_id] = sc

            snippet = self._summarize_for_reflection(retrieval.chunks[:3])
            steps.append(_AgentStep(
                sub_question=current,
                chunk_ids=[sc.chunk.chunk_id for sc in retrieval.chunks[:5]],
                snippet=snippet,
            ))

            # Step 3: reflect — should we continue?
            if iteration < self.max_iterations:
                reflection = self._reflect(query, steps, ctx)
                yield make_event(
                    "thought",
                    f"Reflection: {reflection['decision']}",
                    reasoning=reflection["reasoning"],
                    next_question=reflection.get("next_question", ""),
                )
                if reflection["decision"] == "have_enough":
                    break
                next_q = reflection.get("next_question", "").strip()
                if next_q and next_q not in [s.sub_question for s in steps]:
                    sub_qs.insert(0, next_q)

        # Step 4: synthesize with constrained generation over ALL collected chunks
        merged_scored = list(all_scored.values())
        merged_scored.sort(key=lambda s: s.retrieval_score, reverse=True)

        # Re-score with confidence
        from ..retrieval import RetrievalResult
        merged_result = RetrievalResult(
            chunks=merged_scored,
            bm25_ranks={sc.chunk.chunk_id: i for i, sc in enumerate(merged_scored)},
            dense_ranks={sc.chunk.chunk_id: i for i, sc in enumerate(merged_scored)},
            top_bm25_score=merged_scored[0].retrieval_score if merged_scored else 0.0,
            top_dense_score=merged_scored[0].retrieval_score if merged_scored else 0.0,
        )
        scored, aggregate = ctx.pipeline.scorer.score(merged_result)
        yield make_event(
            "info",
            f"Synthesis: {len(scored)} chunks, aggregate confidence {aggregate:.3f}",
            aggregate=aggregate,
            n_iterations=iteration,
        )

        gen_out = ctx.pipeline.generator.generate(query, scored)
        yield make_event(
            "generation",
            "Constrained generation complete",
            n_citations=len(gen_out.citations),
            insufficient_evidence=gen_out.insufficient_evidence,
        )

        decision = ctx.pipeline.gate.evaluate(scored, aggregate, gen_out)
        if decision.refuse:
            answer = refusal_answer(decision.reason or "Refused", aggregate, ctx.trace_id)
            yield make_event("final", "Refused by fallback gate", answer=answer)
            return

        answer = to_answer(gen_out, aggregate, ctx.trace_id)
        yield make_event("final", "Done", answer=answer)

    # ---- internal helpers ----

    def _plan(self, query: str, ctx: StrategyContext) -> dict:
        try:
            raw = ctx.pipeline.generator.llm.generate(
                prompt=PLAN_PROMPT.format(query=query),
                context=[],
                max_tokens=512,
                temperature=0.0,
            )
            parsed = self._parse_json(raw)
            if parsed and "sub_questions" in parsed:
                # Clean up the sub-questions
                subs = [s.strip() for s in parsed["sub_questions"] if isinstance(s, str) and s.strip()]
                return {
                    "sub_questions": subs[:4] if subs else [query],
                    "reasoning": parsed.get("reasoning", "(no reasoning)"),
                }
        except Exception:
            pass
        # Fallback: treat the original query as the only sub-question
        return {"sub_questions": [query], "reasoning": "could not decompose, using original query"}

    def _reflect(self, query: str, steps: list[_AgentStep], ctx: StrategyContext) -> dict:
        evidence_summary = "\n".join(
            f"Q{i+1}: {s.sub_question}\nFound: {s.snippet[:300]}"
            for i, s in enumerate(steps)
        )
        try:
            raw = ctx.pipeline.generator.llm.generate(
                prompt=REFLECT_PROMPT.format(query=query, evidence_summary=evidence_summary),
                context=[],
                max_tokens=256,
                temperature=0.0,
            )
            parsed = self._parse_json(raw)
            if parsed and "decision" in parsed:
                return {
                    "decision": parsed["decision"] if parsed["decision"] in ("have_enough", "need_more") else "have_enough",
                    "next_question": parsed.get("next_question", ""),
                    "reasoning": parsed.get("reasoning", ""),
                }
        except Exception:
            pass
        return {"decision": "have_enough", "next_question": "", "reasoning": "could not reflect"}

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        try:
            return json.loads(text)
        except Exception:
            pass
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None

    @staticmethod
    def _summarize_for_reflection(scored_chunks: list[ScoredChunk]) -> str:
        """Build a compact text summary of retrieved chunks for the reflection prompt."""
        parts = []
        for sc in scored_chunks:
            parts.append(f"- {sc.chunk.text[:200].strip()}")
        return "\n".join(parts) if parts else "(no chunks found)"
