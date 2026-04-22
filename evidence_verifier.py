from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from evidence_program import AnswerStyle, EvidenceProgram, ProgramType


_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "at",
    "by", "is", "are", "was", "were", "be", "been", "being", "did", "do", "does",
    "what", "which", "who", "when", "where", "why", "how", "from", "that", "this",
    "it", "as", "about", "into", "their", "there", "have", "has", "had", "any",
    "ever", "mention", "mentioned", "conversation", "record", "question",
}
_QUESTION_WORDS = {
    "what", "which", "who", "when", "where", "why", "how", "would", "could", "should",
    "did", "does", "do", "is", "are", "was", "were", "can", "will",
}
_ROLE_TERMS = {
    "grandma", "grandfather", "grandpa", "grandmother", "mother", "father", "mom", "dad",
    "friend", "friends", "family", "kids", "kid", "children", "daughter", "son", "wife",
    "husband", "partner", "photo", "gift", "necklace", "bowl", "shoes", "painting",
    "workshop", "agency",
}


def _tokenize(text: str) -> List[str]:
    return [tok for tok in re.findall(r"[A-Za-z0-9_]+", (text or "").lower()) if tok and tok not in _STOPWORDS]


def _note_text(note: Any) -> str:
    if note is None:
        return ""
    if isinstance(note, str):
        return note
    content = getattr(note, "content", "")
    context = getattr(note, "context", "")
    keywords = " ".join(getattr(note, "keywords", []) or [])
    tags = " ".join(getattr(note, "tags", []) or [])
    return " ".join(part for part in [content, context, keywords, tags] if part).strip()


def _supporting_span(note: Any, max_chars: int = 160) -> str:
    text = _note_text(note)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _extract_named_entities(question: str) -> List[str]:
    names: List[str] = []
    for candidate in re.findall(r"\b[A-Z][a-z]+\b", question or ""):
        if candidate.lower() in _QUESTION_WORDS:
            continue
        if candidate not in names:
            names.append(candidate)
    return names


def _infer_question_constraints(question: str) -> Dict[str, Any]:
    q = (question or "").strip()
    q_lower = q.lower()
    starts_yes_no = q_lower.startswith((
        "is ", "are ", "was ", "were ", "did ", "does ", "do ", "can ", "could ",
        "should ", "would ", "will ",
    ))
    expects_summary = (
        q_lower.startswith("how does ")
        or "plans for" in q_lower
        or "plans with respect to" in q_lower
        or "personality traits" in q_lower
        or "prioritize" in q_lower
        or "in what ways" in q_lower
        or ("what are " in q_lower and " plans" in q_lower)
    )
    expects_list = (
        not expects_summary
        and (
            q_lower.startswith("where has ")
            or any(
                phrase in q_lower for phrase in [
                    "what books", "what events", "what symbols", "what instruments", "what fields",
                    "what activities", "what musical artists", "what artists/bands",
                    "what transgender-specific events", "what lgbtq+ events", "what types of", "what items",
                ]
            )
        )
    )
    has_reason_phrase_cue = any(
        phrase in q_lower for phrase in [
            "why ", "what motivated", "what inspired", "excited about", "what kind of place",
            "what is the reason", "reason for", "symbolize", "symbolise",
        ]
    )
    answer_type = "phrase"
    if starts_yes_no:
        answer_type = "yes_no"
    elif q_lower.startswith("where "):
        answer_type = "location"
    elif q_lower.startswith("when ") or "what date" in q_lower or "how often" in q_lower or "how long" in q_lower:
        answer_type = "temporal"
    elif q_lower.startswith("who ") or " identity" in q_lower or "relationship status" in q_lower:
        answer_type = "person"
    elif expects_list:
        answer_type = "list"
    elif has_reason_phrase_cue:
        answer_type = "reason_phrase"

    names = _extract_named_entities(q)
    role_terms = sorted(term for term in _ROLE_TERMS if term in q_lower)
    has_inference_cue = any(
        phrase in q_lower for phrase in [
            "would ", "would be", "would still", "would likely", "likely", "considered",
            "still want", "career option", "political leaning", "religious", "ally",
            "if she hadn't", "if he hadn't", "if they hadn't", "if she had not", "if he had not",
            "be likely to", "be considered",
        ]
    )
    return {
        "starts_yes_no": starts_yes_no,
        "expects_list": expects_list,
        "expects_summary": expects_summary,
        "answer_type": answer_type,
        "named_entities": names,
        "role_terms": role_terms,
        "has_inference_cue": has_inference_cue,
        "strict_entity_match": bool(names) and bool(role_terms),
    }


@dataclass
class EvidenceVerifier:
    llm_controller: Optional[Any] = None

    def _heuristic_verify(self, question: str, evidence_notes: List[Any], program: EvidenceProgram) -> Dict[str, Any]:
        q_terms = set(_tokenize(question))
        q_constraints = _infer_question_constraints(question)
        if not evidence_notes:
            return {
                "label": "insufficient",
                "reason": "no evidence retrieved",
                "supporting_spans": [],
                "confidence": 0.05,
            }

        scored_notes: List[Dict[str, Any]] = []
        covered_terms = set()
        note_texts: List[str] = []
        for note in evidence_notes:
            note_text = _note_text(note)
            note_texts.append(note_text.lower())
            note_terms = set(_tokenize(note_text))
            overlap = q_terms & note_terms
            if overlap:
                covered_terms.update(overlap)
            scored_notes.append(
                {
                    "note": note,
                    "overlap_count": len(overlap),
                    "has_timestamp": bool(getattr(note, "timestamp", None)),
                    "span": _supporting_span(note),
                }
            )

        scored_notes.sort(key=lambda item: (item["overlap_count"], item["has_timestamp"]), reverse=True)
        best_overlap = scored_notes[0]["overlap_count"] if scored_notes else 0
        coverage = (len(covered_terms) / max(len(q_terms), 1)) if q_terms else 0.0
        multi_support_count = sum(1 for item in scored_notes if item["overlap_count"] > 0)

        label = "insufficient"
        reason = "evidence is weakly aligned with the question"
        confidence = min(0.95, 0.15 + coverage)

        if program.program_type == ProgramType.VERIFY_UNSUPPORTED:
            if best_overlap == 0 and coverage < 0.08:
                label = "unsupported"
                reason = "no directly supporting evidence found"
                confidence = 0.78
            elif coverage >= 0.16:
                label = "supported"
                reason = "retrieved evidence directly matches the request"
                confidence = min(0.9, 0.45 + coverage)
            else:
                label = "insufficient"
                reason = "retrieved evidence is ambiguous for a strict abstain program"
        elif program.program_type == ProgramType.MULTI_HOP:
            if coverage >= 0.18 and multi_support_count >= 2:
                label = "supported"
                reason = "multiple evidence notes cover the reasoning chain"
                confidence = min(0.92, 0.5 + coverage)
            elif best_overlap == 0:
                label = "insufficient"
                reason = "missing linked evidence for multi-hop reasoning"
            else:
                label = "insufficient"
                reason = "partial chain retrieved but not enough to close the loop"
        else:
            if coverage >= 0.14 or best_overlap >= 2:
                label = "supported"
                reason = "retrieved evidence is aligned with the question"
                confidence = min(0.9, 0.42 + coverage)
            elif best_overlap == 0 and program.allow_abstain:
                label = "unsupported"
                reason = "no direct lexical support found"
                confidence = 0.72

        supporting_spans = [item["span"] for item in scored_notes[: min(3, len(scored_notes))] if item["span"]]
        name_hits = 0
        if q_constraints["named_entities"]:
            for name in q_constraints["named_entities"]:
                if any(name.lower() in note_text for note_text in note_texts):
                    name_hits += 1
        role_hits = 0
        if q_constraints["role_terms"]:
            for role in q_constraints["role_terms"]:
                if any(role in note_text for note_text in note_texts):
                    role_hits += 1

        if q_constraints["named_entities"] and name_hits == 0:
            label = "insufficient" if program.program_type != ProgramType.VERIFY_UNSUPPORTED else "unsupported"
            reason = "question entity is not grounded in the retrieved evidence"
            confidence = min(float(confidence), 0.42)
        elif q_constraints["role_terms"] and role_hits == 0 and label == "supported":
            label = "insufficient"
            reason = "question role or possession is not preserved in the retrieved evidence"
            confidence = min(float(confidence), 0.5)
        elif q_constraints["strict_entity_match"] and label == "supported" and role_hits == 0:
            label = "insufficient"
            reason = "strict entity question needs a role-preserving evidence span"
            confidence = min(float(confidence), 0.54)
        elif q_constraints["expects_list"] and label == "supported" and multi_support_count < 2 and coverage < 0.35:
            label = "insufficient"
            reason = "list question likely misses one or more supported items"
            confidence = min(float(confidence), 0.58)
        elif q_constraints["expects_summary"] and label == "supported" and multi_support_count < 2 and coverage < 0.22:
            label = "insufficient"
            reason = "summary question likely needs broader evidence coverage"
            confidence = min(float(confidence), 0.56)
        elif q_constraints["starts_yes_no"] and label == "supported" and best_overlap < 2 and coverage < 0.18:
            label = "insufficient"
            reason = "binary claim is not directly grounded in the retrieved evidence"
            confidence = min(float(confidence), 0.54)
        elif q_constraints["has_inference_cue"] and program.program_type != ProgramType.MULTI_HOP and label == "supported":
            label = "insufficient"
            reason = "inference question lacks a dedicated reasoning chain"
            confidence = min(float(confidence), 0.55)

        return {
            "label": label,
            "reason": reason,
            "supporting_spans": supporting_spans,
            "confidence": round(float(confidence), 4),
        }

    def _llm_verify(self, question: str, evidence_notes: List[Any], program: EvidenceProgram) -> Optional[Dict[str, Any]]:
        if self.llm_controller is None or not evidence_notes:
            return None

        evidence_lines = []
        for idx, note in enumerate(evidence_notes[:6], start=1):
            evidence_lines.append(f"[{idx}] {_supporting_span(note, max_chars=220)}")

        prompt = f"""You are an evidence verifier for memory QA.
Question: {question}
Program type: {program.program_type.value}
Evidence:
{chr(10).join(evidence_lines)}

Judge whether the evidence is sufficient to answer the question.
Be strict about preserving the exact named person, family role, and possession in the question.
For list questions, do not mark the evidence supported if it likely covers only part of the answer.
Return JSON with:
- label: supported / unsupported / insufficient
- reason: short reason
- supporting_spans: short quoted snippets copied from the evidence list
- confidence: number between 0 and 1
"""
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "evidence_verification",
                "schema": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "enum": ["supported", "unsupported", "insufficient"],
                        },
                        "reason": {"type": "string"},
                        "supporting_spans": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "confidence": {"type": "number"},
                    },
                    "required": ["label", "reason", "supporting_spans", "confidence"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        try:
            response = self.llm_controller.llm.get_completion(
                prompt,
                response_format=response_format,
                temperature=0.0,
            )
            parsed = json.loads(response)
            parsed["confidence"] = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
            parsed["supporting_spans"] = [str(span) for span in parsed.get("supporting_spans", [])[:3]]
            return parsed
        except Exception:
            return None

    def verify_evidence_support(
        self,
        question: str,
        evidence_notes: List[Any],
        program: EvidenceProgram,
    ) -> Dict[str, Any]:
        heuristic = self._heuristic_verify(question, evidence_notes, program)
        llm_result = None
        if program.need_verifier or program.program_type in {ProgramType.MULTI_HOP, ProgramType.VERIFY_UNSUPPORTED}:
            llm_result = self._llm_verify(question, evidence_notes, program)

        if llm_result is None:
            return heuristic

        if heuristic["label"] == "unsupported" and llm_result.get("label") == "supported":
            llm_result["label"] = "insufficient"
            llm_result["reason"] = f"{llm_result.get('reason', '')}; heuristic saw no direct support".strip("; ")

        if heuristic["label"] == "insufficient" and llm_result.get("label") == "supported":
            heuristic_reason = str(heuristic.get("reason", "") or "")
            if any(marker in heuristic_reason for marker in [
                "entity", "role", "possession", "list question", "binary claim", "reasoning chain",
            ]):
                llm_result["label"] = "insufficient"
                llm_result["reason"] = heuristic_reason
                llm_result["confidence"] = min(float(llm_result.get("confidence", 0.5)), float(heuristic["confidence"]))

        if heuristic["label"] == "supported" and llm_result.get("label") == "insufficient":
            llm_result["confidence"] = min(float(llm_result.get("confidence", 0.5)), float(heuristic["confidence"]))

        if program.answer_style == AnswerStyle.REASONING_LABEL and llm_result.get("label") == "unsupported":
            llm_result["label"] = "insufficient"
            llm_result["reason"] = "reasoning questions should degrade to weak support instead of forced abstain"
            llm_result["confidence"] = min(float(llm_result.get("confidence", 0.5)), 0.56)

        return llm_result


def verify_evidence_support(
    question: str,
    evidence_notes: List[Any],
    program: EvidenceProgram,
    llm_controller: Optional[Any] = None,
) -> Dict[str, Any]:
    verifier = EvidenceVerifier(llm_controller=llm_controller)
    return verifier.verify_evidence_support(question, evidence_notes, program)
