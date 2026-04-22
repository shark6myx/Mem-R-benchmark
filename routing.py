from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from evidence_program import AnswerStyle, EvidenceProgram, ProgramType


class ProgramRouter:
    _ROLE_TERMS = {
        "grandma", "grandfather", "grandpa", "grandmother", "mother", "father", "mom", "dad",
        "friend", "friends", "family", "kids", "kid", "children", "daughter", "son", "wife",
        "husband", "partner", "photo", "gift", "necklace", "bowl", "shoes", "painting",
        "workshop", "agency",
    }
    _QUESTION_WORDS = {
        "what", "which", "who", "when", "where", "why", "how", "would", "could", "should",
        "did", "does", "do", "is", "are", "was", "were", "can", "will",
    }

    def __init__(self, llm_controller: Any, retrieve_k: int, route_cache: Optional[Dict[str, Dict[str, Any]]] = None):
        self.llm_controller = llm_controller
        self.retrieve_k = retrieve_k
        self.route_cache = route_cache if route_cache is not None else {}

    def _extract_named_entities(self, question: str) -> List[str]:
        names: List[str] = []
        for candidate in re.findall(r"\b[A-Z][a-z]+\b", question or ""):
            if candidate.lower() in self._QUESTION_WORDS:
                continue
            if candidate not in names:
                names.append(candidate)
        return names

    def infer_answer_constraints(self, question: str) -> Dict[str, Any]:
        q_raw = (question or "").strip()
        q = q_raw.lower()
        names = self._extract_named_entities(q_raw)
        starts_yes_no = q.startswith((
            "is ", "are ", "was ", "were ", "did ", "does ", "do ", "can ", "could ",
            "should ", "would ", "will ",
        ))
        expects_summary = (
            q.startswith("how does ")
            or "plans for" in q
            or "plans with respect to" in q
            or "personality traits" in q
            or "prioritize" in q
            or "in what ways" in q
            or ("what are " in q and " plans" in q)
        )
        expects_list = (
            not expects_summary
            and (
                q.startswith("where has ")
                or any(
                    phrase in q for phrase in [
                        "what books",
                        "what events",
                        "what symbols",
                        "what instruments",
                        "what fields",
                        "what activities",
                        "what musical artists",
                        "what artists/bands",
                        "what transgender-specific events",
                        "what lgbtq+ events",
                        "what types of",
                        "what items",
                    ]
                )
            )
        )
        has_reason_phrase_cue = any(
            phrase in q for phrase in [
                "why ",
                "what motivated",
                "what inspired",
                "excited about",
                "what kind of place",
                "what is the reason",
                "reason for",
                "symbolize",
                "symbolise",
            ]
        )

        answer_type = "phrase"
        if starts_yes_no:
            answer_type = "yes_no"
        elif q.startswith("where "):
            answer_type = "location"
        elif q.startswith("when ") or "what date" in q or "how often" in q or "how long" in q:
            answer_type = "temporal"
        elif q.startswith("who ") or " identity" in q or "relationship status" in q:
            answer_type = "person"
        elif q.startswith("how many ") or "how much " in q:
            answer_type = "count"
        elif expects_list:
            answer_type = "list"
        elif has_reason_phrase_cue:
            answer_type = "reason_phrase"

        has_inference_cue = any(
            phrase in q for phrase in [
                "would ", "would be", "would still", "would likely", "likely", "considered",
                "still want", "career option", "political leaning", "religious", "ally",
                "if she hadn't", "if he hadn't", "if they hadn't", "if she had not", "if he had not",
                "be likely to", "be considered",
            ]
        )
        role_terms = sorted(term for term in self._ROLE_TERMS if term in q)
        entity_sensitive = bool(names) and (
            bool(role_terms)
            or "'s " in q
            or " photo" in q
            or " gift" in q
            or " made " in q
            or " from " in q
        )
        strict_entity_match = entity_sensitive and (
            bool(role_terms)
            or " photo" in q
            or " gift" in q
            or " necklace" in q
            or " bowl" in q
            or " painting" in q
            or " shoes" in q
            or " from " in q
            or " with respect to" in q
        )
        explicit_abstain = any(
            phrase in q for phrase in [
                "not mentioned", "mentioned in the conversation", "any evidence", "is there evidence",
                "ever mentioned", "unknown",
            ]
        )
        prefer_fact_style = (
            answer_type in {"location", "person", "count", "list", "reason_phrase", "phrase"}
            and not has_inference_cue
            and not expects_summary
        )
        return {
            "answer_type": answer_type,
            "expects_list": expects_list,
            "expects_summary": expects_summary,
            "starts_yes_no": starts_yes_no,
            "has_inference_cue": has_inference_cue,
            "has_reason_phrase_cue": has_reason_phrase_cue,
            "entity_sensitive": entity_sensitive,
            "strict_entity_match": strict_entity_match,
            "explicit_abstain": explicit_abstain,
            "named_entities": names,
            "role_terms": role_terms,
            "prefer_fact_style": prefer_fact_style,
        }

    def infer_task_attributes(self, question: str) -> Dict[str, bool]:
        q = (question or "").strip().lower()
        constraints = self.infer_answer_constraints(question)
        return {
            "has_temporal_cue": any(t in q for t in [
                "when", "date", "time", "before", "after", "currently", "now", "previously",
                "earlier", "later", "how often", "how long", "year", "month", "week", "yesterday",
                "ago",
            ]),
            "has_multi_hop_cue": constraints["has_inference_cue"] or any(t in q for t in [
                "relationship", "connect", "connected", "what led", "led to", "result of",
                "caused", "cause of", "how is", "how are", "counterfactual", "in order to",
            ]),
            "has_factoid_cue": any(t in q for t in [
                "who", "where", "what", "which", "how many", "did ", "was ", "is ", "are ",
            ]),
            "has_update_cue": any(t in q for t in [
                "latest", "current", "updated", "change", "switched", "no longer",
            ]),
            "has_abstain_cue": constraints["explicit_abstain"],
            "has_inference_cue": constraints["has_inference_cue"],
            "has_reason_phrase_cue": constraints["has_reason_phrase_cue"],
            "entity_sensitive": constraints["entity_sensitive"],
            "expects_list": constraints["expects_list"],
            "expects_summary": constraints["expects_summary"],
            "strict_entity_match": constraints["strict_entity_match"],
        }

    def fallback_task_type_from_question(self, question: str) -> Dict[str, Any]:
        attrs = self.infer_task_attributes(question)
        constraints = self.infer_answer_constraints(question)
        if attrs["has_abstain_cue"]:
            return {"task_type": "adversarial_unanswerable", "reason": "heuristic_abstain_cue", "confidence": 0.45}
        if attrs["has_inference_cue"]:
            return {"task_type": "multi_hop", "reason": "heuristic_inference_cue", "confidence": 0.58}
        if constraints["answer_type"] == "temporal" or (
            attrs["has_temporal_cue"]
            and not attrs["has_multi_hop_cue"]
            and not constraints["prefer_fact_style"]
        ):
            return {"task_type": "temporal", "reason": "heuristic_temporal_cue", "confidence": 0.45}
        if attrs["expects_summary"]:
            return {"task_type": "open_domain", "reason": "heuristic_summary_cue", "confidence": 0.54}
        if attrs["has_multi_hop_cue"]:
            return {"task_type": "multi_hop", "reason": "heuristic_multi_hop_cue", "confidence": 0.45}
        if attrs["has_update_cue"]:
            return {"task_type": "knowledge_update", "reason": "heuristic_update_cue", "confidence": 0.45}
        if attrs["entity_sensitive"] or attrs["has_reason_phrase_cue"] or attrs["expects_list"]:
            return {"task_type": "factoid", "reason": "heuristic_entity_or_reason_phrase", "confidence": 0.52}
        if attrs["has_factoid_cue"]:
            return {"task_type": "factoid", "reason": "heuristic_factoid_cue", "confidence": 0.45}
        return {"task_type": "factoid", "reason": "heuristic_default", "confidence": 0.35}

    def llm_route_question(self, question: str) -> Dict[str, Any]:
        q = (question or "").strip()
        if not q:
            return {"task_type": "factoid", "reason": "empty_question", "confidence": 0.5}

        if q in self.route_cache:
            return self.route_cache[q]

        prompt = f"""You are a routing classifier for memory QA retrieval.
Classify the question into one task type from:
- factoid
- temporal
- multi_hop
- open_domain
- adversarial_unanswerable
- knowledge_update

Use these definitions:
- factoid: direct span extraction, short reason phrase, or entity-sensitive ownership/identity question
- temporal: asks for date, time, duration, recency, order, or frequency
- multi_hop: counterfactual or inferential question such as "would", "likely", "considered", "still want", "if ... hadn't"
- open_domain: broad synthesis across many memories, not just one span
- adversarial_unanswerable: explicitly asks whether something was mentioned or whether evidence exists, or truly lacks direct support
- knowledge_update: asks for the latest/current/updated state after a change

Question: {q}

Return strict JSON:
{{
  "task_type": "<one_of_the_types_above>",
  "reason": "<short reason>",
  "confidence": <float_between_0_and_1>
}}"""
        try:
            response = self.llm_controller.llm.get_completion(
                prompt,
                response_format={"type": "json_schema", "json_schema": {
                    "name": "route_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "task_type": {
                                "type": "string",
                                "enum": [
                                    "factoid", "temporal", "multi_hop", "open_domain",
                                    "adversarial_unanswerable", "knowledge_update",
                                ],
                            },
                            "reason": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["task_type", "reason", "confidence"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }},
                temperature=0.0,
            )
            parsed = json.loads(response)
            parsed["confidence"] = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
            self.route_cache[q] = parsed
            return parsed
        except Exception:
            fallback = self.fallback_task_type_from_question(question)
            self.route_cache[q] = fallback
            return fallback

    def predict_evidence_program(self, question: str) -> EvidenceProgram:
        attrs = self.infer_task_attributes(question)
        constraints = self.infer_answer_constraints(question)
        llm_route = self.llm_route_question(question)
        llm_task = str(llm_route.get("task_type", "") or "")
        llm_confidence = float(llm_route.get("confidence", 0.0) or 0.0)
        llm_reason = str(llm_route.get("reason", "") or "")

        if llm_confidence < 0.55:
            fallback = self.fallback_task_type_from_question(question)
            llm_task = fallback["task_type"]
            llm_reason = f"{llm_reason}; heuristic_override".strip("; ")
            llm_confidence = max(llm_confidence, float(fallback.get("confidence", 0.45) or 0.45))

        temporal_program = (
            constraints["answer_type"] == "temporal"
            or (
                llm_task == "temporal"
                and attrs["has_temporal_cue"]
                and not attrs["has_multi_hop_cue"]
                and not constraints["prefer_fact_style"]
            )
        )

        if llm_task == "adversarial_unanswerable" and not constraints["explicit_abstain"]:
            llm_reason = f"{llm_reason}; answerable_adversarial_guard".strip("; ")

        if constraints["explicit_abstain"]:
            program_type = ProgramType.VERIFY_UNSUPPORTED
        elif constraints["has_inference_cue"]:
            program_type = ProgramType.MULTI_HOP
            if llm_task != "multi_hop":
                llm_reason = f"{llm_reason}; inference_override".strip("; ")
        elif temporal_program:
            program_type = ProgramType.TEMPORAL
        elif llm_task == "multi_hop" and not constraints["prefer_fact_style"]:
            program_type = ProgramType.MULTI_HOP
        elif constraints["expects_summary"] or llm_task == "open_domain":
            program_type = ProgramType.PROFILE
        elif llm_task == "knowledge_update" or attrs["has_update_cue"]:
            program_type = ProgramType.PROFILE
        elif constraints["entity_sensitive"] or constraints["has_reason_phrase_cue"] or constraints["expects_list"]:
            program_type = ProgramType.FACT
            if llm_task in {"multi_hop", "open_domain", "adversarial_unanswerable"}:
                llm_reason = f"{llm_reason}; entity_or_reason_phrase_override".strip("; ")
        else:
            program_type = ProgramType.FACT

        program = EvidenceProgram.from_program_type(
            program_type=program_type,
            retrieve_k=self.retrieve_k,
            routing_reason=llm_reason or "question_heuristic",
            routing_confidence=llm_confidence or 0.45,
            metadata={
                "llm_task": llm_task or program_type.value,
                "attrs": attrs,
                "answer_constraints": constraints,
            },
        )

        answer_style = program.answer_style
        overrides: Dict[str, Any] = {}
        prefer_answer_subgraph = False

        if constraints["explicit_abstain"]:
            answer_style = AnswerStyle.ABSTAIN_OR_SPAN
        elif constraints["has_inference_cue"]:
            answer_style = AnswerStyle.REASONING_LABEL
        elif program.program_type == ProgramType.TEMPORAL:
            answer_style = AnswerStyle.TEMPORAL_SHORT
        elif constraints["expects_summary"] or program.program_type == ProgramType.PROFILE:
            answer_style = AnswerStyle.SUMMARY_SHORT
            overrides.update({
                "max_notes": max(program.max_notes, 5),
                "max_hops": max(program.max_hops, 1),
                "include_community_context": True,
            })
            prefer_answer_subgraph = True
        elif constraints["expects_list"]:
            answer_style = AnswerStyle.LIST_SPAN
            overrides.update({
                "max_notes": max(program.max_notes, 4),
                "need_verifier": True,
                "include_community_context": False,
            })
            prefer_answer_subgraph = True
        elif constraints["strict_entity_match"]:
            answer_style = AnswerStyle.STRICT_ENTITY_SPAN
            overrides.update({
                "max_notes": min(max(program.max_notes, 3), 4),
                "max_hops": 0,
                "enable_ppr": False,
                "include_community_context": False,
                "need_verifier": True,
                "allow_abstain": False,
                "channel_weights": {"dense": 1.0, "bm25": 1.05, "community": 0.12, "ppr": 0.0},
                "community_top_c": 1,
                "rrf_confidence_threshold": min(float(program.rrf_confidence_threshold), 0.015),
            })
            prefer_answer_subgraph = True
        elif constraints["has_reason_phrase_cue"]:
            answer_style = AnswerStyle.REASON_PHRASE
            overrides.update({
                "max_notes": max(program.max_notes, 4),
                "need_verifier": True,
            })
            prefer_answer_subgraph = True

        metadata = dict(program.metadata)
        metadata.update({
            "llm_task": llm_task or program_type.value,
            "attrs": attrs,
            "answer_constraints": constraints,
            "prefer_answer_subgraph": prefer_answer_subgraph,
        })

        return program.with_overrides(
            answer_style=answer_style,
            metadata=metadata,
            **overrides,
        )
