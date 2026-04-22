from __future__ import annotations

from typing import Any, Dict

from evidence_program import EvidenceProgram
from retrieval_types import RetrievalEngine


def classify_question_mode_from_program(program: EvidenceProgram) -> Dict[str, Any]:
    retrieval_engine = program.retrieval_engine
    return {
        "use_agentic": retrieval_engine == RetrievalEngine.AGENTIC,
        "retrieval_mode": "direct_hybrid" if retrieval_engine == RetrievalEngine.DIRECT_HYBRID else "advanced",
        "k": program.k,
        "include_community_context": program.include_community_context,
        "max_hops": program.max_hops,
        "fusion_alpha": program.fusion_alpha,
        "dense_weight": program.dense_weight,
        "strategy_name": f"program_{program.program_type.value}",
        "route_task_type": program.metadata.get("llm_task", program.program_type.value),
        "route_reason": program.routing_reason,
        "route_confidence": program.routing_confidence,
        "use_rrf": retrieval_engine == RetrievalEngine.RRF_MULTI_CHANNEL,
        "channel_weights": dict(program.channel_weights),
        "enable_ppr": program.enable_ppr,
        "community_top_c": program.community_top_c,
        "rrf_confidence_threshold": program.rrf_confidence_threshold,
        "enable_abstain_gate": program.allow_abstain,
        "answer_style": program.answer_style.value,
        "need_verifier": program.need_verifier,
        "program_type": program.program_type.value,
        "max_notes": program.max_notes,
        "retrieval_engine": retrieval_engine.value,
    }
