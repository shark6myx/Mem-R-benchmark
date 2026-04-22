from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from evidence_program import AnswerStyle, EvidenceProgram, ProgramType
from evidence_verifier import verify_evidence_support
from retrieval_types import RetrievalResult


@dataclass
class PipelineOutcome:
    response_text: str
    user_prompt: str
    raw_context: str
    retrieval_result: RetrievalResult
    verification: Dict[str, Any]
    compressed_evidence_count: int


class ProgramGuidedQAPipeline:
    def __init__(
        self,
        memory_system: Any,
        llm_controller: Any,
        prompt_builder: Callable[[str, str, EvidenceProgram, Optional[Dict[str, Any]]], str],
    ) -> None:
        self.memory_system = memory_system
        self.llm_controller = llm_controller
        self.prompt_builder = prompt_builder

    def _note_text(self, note: Any) -> str:
        if note is None:
            return ""
        return " ".join(
            [
                getattr(note, "content", "") or "",
                getattr(note, "context", "") or "",
                " ".join(getattr(note, "keywords", []) or []),
                " ".join(getattr(note, "tags", []) or []),
            ]
        ).strip()

    def _query_coverage(self, question: str, notes: List[Any]) -> float:
        query_terms = set(self.memory_system._program_query_terms(question))
        if not query_terms:
            return 0.0

        covered = set()
        for note in notes:
            note_terms = set(self.memory_system._program_query_terms(self._note_text(note)))
            covered.update(query_terms & note_terms)
        return len(covered) / max(len(query_terms), 1)

    def _append_trace_step(self, retrieval_result: RetrievalResult, name: str, **details: Any) -> None:
        if not isinstance(retrieval_result.execution_trace, dict):
            retrieval_result.execution_trace = {}
        steps = retrieval_result.execution_trace.setdefault("steps", [])
        if not isinstance(steps, list):
            steps = []
            retrieval_result.execution_trace["steps"] = steps
        entry = {"step": name}
        entry.update(details)
        steps.append(entry)

    def _should_try_answer_subgraph(self, program: EvidenceProgram) -> bool:
        return (
            program.program_type in {ProgramType.MULTI_HOP, ProgramType.VERIFY_UNSUPPORTED}
            or program.answer_style in {
                AnswerStyle.LIST_SPAN,
                AnswerStyle.SUMMARY_SHORT,
                AnswerStyle.STRICT_ENTITY_SPAN,
                AnswerStyle.REASON_PHRASE,
            }
            or bool((program.metadata or {}).get("prefer_answer_subgraph"))
        )

    def _should_use_answer_subgraph(
        self,
        question: str,
        program: EvidenceProgram,
        fallback_notes: List[Any],
        candidate_subgraph: Any,
    ) -> bool:
        if candidate_subgraph is None:
            return False

        candidate_nodes = list(getattr(candidate_subgraph, "nodes", []) or [])
        if not candidate_nodes:
            return False
        if len(candidate_nodes) > max(1, int(program.max_notes or 1)):
            return False

        fallback_coverage = self._query_coverage(question, fallback_notes)
        candidate_coverage = self._query_coverage(question, candidate_nodes)
        if candidate_coverage + 1e-6 < fallback_coverage:
            return False

        confidence = float(getattr(candidate_subgraph, "confidence", 0.0) or 0.0)
        candidate_valid = bool(getattr(candidate_subgraph, "valid", False))
        if program.program_type == ProgramType.MULTI_HOP:
            if not candidate_valid:
                return False
            if confidence < 0.62:
                return False
            if not (getattr(candidate_subgraph, "bridge_node_ids", []) or getattr(candidate_subgraph, "edges", [])):
                return False
            return True

        if program.program_type == ProgramType.VERIFY_UNSUPPORTED:
            if not candidate_valid:
                return False
            if confidence < 0.86:
                return False
            if candidate_coverage < max(0.34, fallback_coverage):
                return False
            return len(candidate_nodes) <= 2

        if program.answer_style == AnswerStyle.LIST_SPAN:
            return candidate_valid and confidence >= 0.5 and candidate_coverage + 0.02 >= fallback_coverage
        if program.answer_style == AnswerStyle.STRICT_ENTITY_SPAN:
            return candidate_valid and confidence >= 0.46 and candidate_coverage + 0.02 >= fallback_coverage
        if program.answer_style in {AnswerStyle.SUMMARY_SHORT, AnswerStyle.REASON_PHRASE}:
            return candidate_valid and confidence >= 0.44 and candidate_coverage + 0.02 >= fallback_coverage

        return False

    def _materialize_context(
        self,
        retrieval_result: RetrievalResult,
        evidence_notes: List[Any],
        program: EvidenceProgram,
    ) -> str:
        if program.include_community_context:
            community_payload = self.memory_system.build_community_context_for_notes(
                evidence_notes,
                top_c=program.community_top_c,
            )
        else:
            community_payload = {"community_context": "", "communities": []}

        retrieval_result.notes = list(evidence_notes)
        retrieval_result.community_context = community_payload.get("community_context", "")
        retrieval_result.communities = community_payload.get("communities", [])
        return retrieval_result.format_context(program.include_community_context)

    def run(
        self,
        question: str,
        program: EvidenceProgram,
        temperature_c5: float = 0.5,
    ) -> PipelineOutcome:
        retrieval_result = self.memory_system.execute_evidence_program(program, question, k=program.k)
        fallback_notes = self.memory_system.compress_to_minimal_evidence(question, retrieval_result.notes, program)
        evidence_notes = list(fallback_notes)
        selection_mode = "minimal_evidence_fallback"

        if self._should_try_answer_subgraph(program):
            candidate_subgraph = self.memory_system.build_answer_subgraph(question, retrieval_result.notes, program)
            retrieval_result.answer_subgraph = candidate_subgraph
            self._append_trace_step(
                retrieval_result,
                "build_answer_subgraph",
                valid=bool(getattr(candidate_subgraph, "valid", False)),
                confidence=float(getattr(candidate_subgraph, "confidence", 0.0) or 0.0),
                node_count=len(getattr(candidate_subgraph, "nodes", []) or []),
                edge_count=len(getattr(candidate_subgraph, "edges", []) or []),
                reason=str(getattr(candidate_subgraph, "metadata", {}).get("reason", "")),
            )
            if self._should_use_answer_subgraph(question, program, fallback_notes, candidate_subgraph):
                candidate_subgraph.used_as_primary = True
                evidence_notes = list(candidate_subgraph.nodes)
                selection_mode = "answer_subgraph"
                self._append_trace_step(
                    retrieval_result,
                    "select_answer_subgraph",
                    node_count=len(evidence_notes),
                    edge_count=len(getattr(candidate_subgraph, "edges", []) or []),
                )
            else:
                self._append_trace_step(
                    retrieval_result,
                    "keep_minimal_evidence",
                    note_count=len(fallback_notes),
                )

        raw_context = self._materialize_context(retrieval_result, evidence_notes, program)
        retrieval_result.metadata = dict(retrieval_result.metadata)
        retrieval_result.metadata["evidence_selection_mode"] = selection_mode
        retrieval_result.metadata["fallback_note_count"] = len(fallback_notes)
        verification = verify_evidence_support(
            question,
            evidence_notes,
            program,
            llm_controller=self.llm_controller,
        )

        should_reject_subgraph = selection_mode == "answer_subgraph" and verification.get("label") != "supported"
        if (
            should_reject_subgraph
            and program.answer_style in {
                AnswerStyle.LIST_SPAN,
                AnswerStyle.SUMMARY_SHORT,
                AnswerStyle.STRICT_ENTITY_SPAN,
                AnswerStyle.REASON_PHRASE,
            }
            and verification.get("label") == "insufficient"
        ):
            should_reject_subgraph = False

        if should_reject_subgraph:
            if retrieval_result.answer_subgraph is not None:
                retrieval_result.answer_subgraph.used_as_primary = False
            evidence_notes = list(fallback_notes)
            selection_mode = "fallback_after_subgraph_verifier"
            raw_context = self._materialize_context(retrieval_result, evidence_notes, program)
            retrieval_result.metadata["evidence_selection_mode"] = selection_mode
            self._append_trace_step(
                retrieval_result,
                "reject_answer_subgraph",
                verifier_label=str(verification.get("label", "")),
                fallback_note_count=len(fallback_notes),
            )
            verification = verify_evidence_support(
                question,
                evidence_notes,
                program,
                llm_controller=self.llm_controller,
            )

        abstain_capable = program.allow_abstain or program.answer_style == AnswerStyle.ABSTAIN_OR_SPAN
        should_abstain = False
        if abstain_capable and verification.get("label") in {"unsupported", "insufficient"}:
            should_abstain = True
        elif abstain_capable and verification.get("label") == "unsupported" and float(verification.get("confidence", 0.0)) >= 0.8:
            should_abstain = True

        if retrieval_result.rrf_scores:
            max_score = max(retrieval_result.rrf_scores.values()) if retrieval_result.rrf_scores else 0.0
            if max_score < float(program.rrf_confidence_threshold) and verification.get("label") != "supported":
                should_abstain = should_abstain or abstain_capable

        if should_abstain:
            verification = dict(verification)
            verification["abstained"] = True
            return PipelineOutcome(
                response_text=json.dumps({"answer": "Not mentioned in the conversation"}),
                user_prompt="",
                raw_context=raw_context,
                retrieval_result=retrieval_result,
                verification=verification,
                compressed_evidence_count=len(evidence_notes),
            )

        user_prompt = self.prompt_builder(question, raw_context, program, verification)
        temperature = temperature_c5 if program.answer_style == AnswerStyle.ABSTAIN_OR_SPAN else 0.7
        response_text = self.llm_controller.llm.get_completion(
            user_prompt,
            response_format={"type": "json_schema", "json_schema": {
                "name": "response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                        }
                    },
                    "required": ["answer"],
                    "additionalProperties": False
                },
                "strict": True
            }},
            temperature=temperature,
        )
        return PipelineOutcome(
            response_text=response_text,
            user_prompt=user_prompt,
            raw_context=raw_context,
            retrieval_result=retrieval_result,
            verification=verification,
            compressed_evidence_count=len(evidence_notes),
        )
