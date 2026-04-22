from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Optional

from retrieval_types import RetrievalEngine


class ProgramType(str, Enum):
    FACT = "fact"
    TEMPORAL = "temporal"
    MULTI_HOP = "multi_hop"
    PROFILE = "profile"
    VERIFY_UNSUPPORTED = "verify_unsupported"


class AnswerStyle(str, Enum):
    EXTRACTIVE_SHORT = "extractive_short"
    TEMPORAL_SHORT = "temporal_short"
    REASONING_LABEL = "reasoning_label"
    ABSTAIN_OR_SPAN = "abstain_or_span"
    LIST_SPAN = "list_span"
    SUMMARY_SHORT = "summary_short"
    STRICT_ENTITY_SPAN = "strict_entity_span"
    REASON_PHRASE = "reason_phrase"


@dataclass
class ExecutionTrace:
    steps: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, name: str, **details: Any) -> None:
        entry = {"step": name}
        entry.update(details)
        self.steps.append(entry)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": list(self.steps),
            "metadata": dict(self.metadata),
        }


@dataclass
class EvidenceProgram:
    program_type: ProgramType
    answer_style: AnswerStyle
    retrieval_engine: RetrievalEngine
    max_notes: int
    max_hops: int
    channel_weights: Dict[str, float]
    enable_ppr: bool
    include_community_context: bool
    need_verifier: bool
    allow_abstain: bool
    k: int = 10
    fusion_alpha: float = 0.68
    dense_weight: float = 0.72
    community_top_c: int = 4
    rrf_confidence_threshold: float = 0.02
    routing_reason: str = ""
    routing_confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_program_type(
        cls,
        program_type: ProgramType,
        retrieve_k: int = 10,
        routing_reason: str = "",
        routing_confidence: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "EvidenceProgram":
        presets: Dict[ProgramType, Dict[str, Any]] = {
            ProgramType.FACT: {
                "answer_style": AnswerStyle.EXTRACTIVE_SHORT,
                "retrieval_engine": RetrievalEngine.RRF_MULTI_CHANNEL,
                "max_notes": 3,
                "max_hops": 0,
                "channel_weights": {"dense": 1.0, "bm25": 0.9, "community": 0.2, "ppr": 0.0},
                "enable_ppr": False,
                "include_community_context": False,
                "need_verifier": False,
                "allow_abstain": False,
                "k": min(retrieve_k, 12),
                "fusion_alpha": 0.82,
                "dense_weight": 0.72,
                "community_top_c": 2,
                "rrf_confidence_threshold": 0.02,
            },
            ProgramType.TEMPORAL: {
                "answer_style": AnswerStyle.TEMPORAL_SHORT,
                "retrieval_engine": RetrievalEngine.RRF_MULTI_CHANNEL,
                "max_notes": 3,
                "max_hops": 1,
                "channel_weights": {"dense": 0.9, "bm25": 1.0, "community": 0.35, "ppr": 0.15},
                "enable_ppr": True,
                "include_community_context": True,
                "need_verifier": False,
                "allow_abstain": False,
                "k": min(retrieve_k, 12),
                "fusion_alpha": 0.76,
                "dense_weight": 0.68,
                "community_top_c": 3,
                "rrf_confidence_threshold": 0.02,
            },
            ProgramType.MULTI_HOP: {
                "answer_style": AnswerStyle.REASONING_LABEL,
                "retrieval_engine": RetrievalEngine.RRF_MULTI_CHANNEL,
                "max_notes": 6,
                "max_hops": 2,
                "channel_weights": {"dense": 1.0, "bm25": 0.7, "community": 0.9, "ppr": 0.55},
                "enable_ppr": True,
                "include_community_context": True,
                "need_verifier": True,
                "allow_abstain": False,
                "k": min(max(retrieve_k, 10), 18),
                "fusion_alpha": 0.66,
                "dense_weight": 0.76,
                "community_top_c": 4,
                "rrf_confidence_threshold": 0.02,
            },
            ProgramType.PROFILE: {
                "answer_style": AnswerStyle.SUMMARY_SHORT,
                "retrieval_engine": RetrievalEngine.RRF_MULTI_CHANNEL,
                "max_notes": 5,
                "max_hops": 1,
                "channel_weights": {"dense": 0.9, "bm25": 0.65, "community": 1.0, "ppr": 0.45},
                "enable_ppr": True,
                "include_community_context": True,
                "need_verifier": False,
                "allow_abstain": False,
                "k": min(max(retrieve_k, 8), 16),
                "fusion_alpha": 0.62,
                "dense_weight": 0.68,
                "community_top_c": 5,
                "rrf_confidence_threshold": 0.02,
            },
            ProgramType.VERIFY_UNSUPPORTED: {
                "answer_style": AnswerStyle.ABSTAIN_OR_SPAN,
                "retrieval_engine": RetrievalEngine.RRF_MULTI_CHANNEL,
                "max_notes": 3,
                "max_hops": 0,
                "channel_weights": {"dense": 1.0, "bm25": 1.0, "community": 0.0, "ppr": 0.0},
                "enable_ppr": False,
                "include_community_context": False,
                "need_verifier": True,
                "allow_abstain": True,
                "k": min(retrieve_k, 8),
                "fusion_alpha": 1.0,
                "dense_weight": 0.62,
                "community_top_c": 0,
                "rrf_confidence_threshold": 0.012,
            },
        }
        preset = dict(presets[program_type])
        preset["routing_reason"] = routing_reason
        preset["routing_confidence"] = max(0.0, min(1.0, float(routing_confidence or 0.0)))
        preset["metadata"] = dict(metadata or {})
        return cls(program_type=program_type, **preset)

    def to_legacy_config(self) -> Dict[str, Any]:
        strategy_suffix = self.program_type.value
        retrieval_mode = "direct_hybrid" if self.retrieval_engine == RetrievalEngine.DIRECT_HYBRID else "advanced"
        return {
            "use_agentic": self.retrieval_engine == RetrievalEngine.AGENTIC,
            "retrieval_mode": retrieval_mode,
            "k": self.k,
            "include_community_context": self.include_community_context,
            "max_hops": self.max_hops,
            "fusion_alpha": self.fusion_alpha,
            "dense_weight": self.dense_weight,
            "strategy_name": f"program_{strategy_suffix}",
            "route_task_type": self.program_type.value,
            "route_reason": self.routing_reason,
            "route_confidence": self.routing_confidence,
            "use_rrf": self.retrieval_engine == RetrievalEngine.RRF_MULTI_CHANNEL,
            "channel_weights": dict(self.channel_weights),
            "enable_ppr": self.enable_ppr,
            "community_top_c": self.community_top_c,
            "rrf_confidence_threshold": self.rrf_confidence_threshold,
            "enable_abstain_gate": self.allow_abstain,
            "answer_style": self.answer_style.value,
            "need_verifier": self.need_verifier,
            "program_type": self.program_type.value,
            "max_notes": self.max_notes,
            "retrieval_engine": self.retrieval_engine.value,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "program_type": self.program_type.value,
            "answer_style": self.answer_style.value,
            "retrieval_engine": self.retrieval_engine.value,
            "max_notes": self.max_notes,
            "max_hops": self.max_hops,
            "channel_weights": dict(self.channel_weights),
            "enable_ppr": self.enable_ppr,
            "include_community_context": self.include_community_context,
            "need_verifier": self.need_verifier,
            "allow_abstain": self.allow_abstain,
            "k": self.k,
            "fusion_alpha": self.fusion_alpha,
            "dense_weight": self.dense_weight,
            "community_top_c": self.community_top_c,
            "rrf_confidence_threshold": self.rrf_confidence_threshold,
            "routing_reason": self.routing_reason,
            "routing_confidence": self.routing_confidence,
            "metadata": dict(self.metadata),
        }

    def __str__(self) -> str:
        return (
            "EvidenceProgram("
            f"type={self.program_type.value}, "
            f"style={self.answer_style.value}, "
            f"engine={self.retrieval_engine.value}, "
            f"notes={self.max_notes}, "
            f"hops={self.max_hops}, "
            f"verifier={self.need_verifier}, "
            f"abstain={self.allow_abstain})"
        )

    def with_overrides(self, **updates: Any) -> "EvidenceProgram":
        return replace(self, **updates)
