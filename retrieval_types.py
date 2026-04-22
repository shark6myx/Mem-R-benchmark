from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RetrievalEngine(str, Enum):
    DIRECT_HYBRID = "direct_hybrid"
    GRAPH_ADVANCED = "graph_advanced"
    RRF_MULTI_CHANNEL = "rrf_multi_channel"
    AGENTIC = "agentic"


@dataclass
class EvidenceSubgraph:
    nodes: List[Any] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    seed_node_ids: List[str] = field(default_factory=list)
    bridge_node_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0
    valid: bool = False
    used_as_primary: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any) -> "EvidenceSubgraph":
        if isinstance(payload, cls):
            return cls(
                nodes=list(payload.nodes),
                edges=list(payload.edges),
                seed_node_ids=list(payload.seed_node_ids),
                bridge_node_ids=list(payload.bridge_node_ids),
                confidence=float(payload.confidence),
                valid=bool(payload.valid),
                used_as_primary=bool(payload.used_as_primary),
                metadata=dict(payload.metadata),
            )

        if not isinstance(payload, dict):
            return cls()

        return cls(
            nodes=list(payload.get("nodes", [])),
            edges=list(payload.get("edges", [])),
            seed_node_ids=[str(node_id) for node_id in payload.get("seed_node_ids", [])],
            bridge_node_ids=[str(node_id) for node_id in payload.get("bridge_node_ids", [])],
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            valid=bool(payload.get("valid", False)),
            used_as_primary=bool(payload.get("used_as_primary", False)),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "nodes": list(self.nodes),
            "edges": list(self.edges),
            "seed_node_ids": list(self.seed_node_ids),
            "bridge_node_ids": list(self.bridge_node_ids),
            "confidence": float(self.confidence),
            "valid": bool(self.valid),
            "used_as_primary": bool(self.used_as_primary),
            "metadata": dict(self.metadata),
        }


@dataclass
class RetrievalResult:
    notes: List[Any] = field(default_factory=list)
    community_context: str = ""
    communities: List[Any] = field(default_factory=list)
    rrf_scores: Dict[str, float] = field(default_factory=dict)
    channel_details: Dict[str, Any] = field(default_factory=dict)
    execution_trace: Dict[str, Any] = field(default_factory=dict)
    program_type: str = ""
    retrieval_engine: str = ""
    answer_subgraph: Optional[EvidenceSubgraph] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        retrieval_engine: Optional[str] = None,
        program_type: str = "",
    ) -> "RetrievalResult":
        if isinstance(payload, cls):
            result = cls(
                notes=list(payload.notes),
                community_context=payload.community_context,
                communities=list(payload.communities),
                rrf_scores=dict(payload.rrf_scores),
                channel_details=dict(payload.channel_details),
                execution_trace=dict(payload.execution_trace),
                program_type=payload.program_type,
                retrieval_engine=payload.retrieval_engine,
                answer_subgraph=EvidenceSubgraph.from_payload(payload.answer_subgraph) if payload.answer_subgraph else None,
                metadata=dict(payload.metadata),
            )
            if retrieval_engine:
                result.retrieval_engine = retrieval_engine
            if program_type:
                result.program_type = program_type
            return result

        if isinstance(payload, str):
            return cls(
                notes=[],
                community_context=payload,
                communities=[],
                program_type=program_type,
                retrieval_engine=retrieval_engine or "",
            )

        if not isinstance(payload, dict):
            return cls(
                notes=[],
                community_context=str(payload),
                communities=[],
                program_type=program_type,
                retrieval_engine=retrieval_engine or "",
            )

        metadata = dict(payload.get("metadata", {}))
        return cls(
            notes=list(payload.get("notes", [])),
            community_context=str(payload.get("community_context", "") or ""),
            communities=list(payload.get("communities", [])),
            rrf_scores=dict(payload.get("rrf_scores", {})),
            channel_details=dict(payload.get("channel_details", {})),
            execution_trace=dict(payload.get("execution_trace", {})),
            program_type=str(payload.get("program_type", program_type) or ""),
            retrieval_engine=str(payload.get("retrieval_engine", retrieval_engine or "") or ""),
            answer_subgraph=EvidenceSubgraph.from_payload(payload.get("answer_subgraph")) if payload.get("answer_subgraph") else None,
            metadata=metadata,
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "notes": list(self.notes),
            "community_context": self.community_context,
            "communities": list(self.communities),
            "rrf_scores": dict(self.rrf_scores),
            "channel_details": dict(self.channel_details),
            "execution_trace": dict(self.execution_trace),
            "program_type": self.program_type,
            "retrieval_engine": self.retrieval_engine,
            "answer_subgraph": self.answer_subgraph.to_payload() if self.answer_subgraph else None,
            "metadata": dict(self.metadata),
        }

    def format_context(self, include_community_context: bool = True) -> str:
        lines: List[str] = []
        for i, note in enumerate(self.notes, 1):
            content = getattr(note, "content", "")
            keywords = getattr(note, "keywords", [])
            kws = ", ".join(keywords[:5]) if keywords else "None"
            ts = getattr(note, "timestamp", "")
            ts_str = f" (recorded: {ts})" if ts else ""
            lines.append(f"[Evidence {i}]{ts_str} Content: {content}\nKeywords: {kws}")

        if include_community_context and self.community_context:
            lines.append(f"[Community Context]\n{self.community_context}")

        return "\n---\n".join(lines)
