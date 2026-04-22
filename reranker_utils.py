from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import torch

from model_cache_utils import resolve_hf_model_reference


_LAYERWISE_RERANKERS = {
    "BAAI/bge-reranker-v2-minicpm-layerwise",
}
_LLM_RERANKERS = {
    "BAAI/bge-reranker-v2-gemma",
}


def _as_float_array(scores: Sequence[float] | float) -> np.ndarray:
    if isinstance(scores, (float, int)):
        return np.asarray([float(scores)], dtype=float)
    return np.asarray(list(scores), dtype=float)


@dataclass
class CrossEncoderRerankerAdapter:
    model: object

    def predict(self, pairs: Sequence[Sequence[str]]) -> np.ndarray:
        return _as_float_array(self.model.predict(list(pairs)))


@dataclass
class FlagEmbeddingRerankerAdapter:
    model: object
    cutoff_layers: List[int] | None = None

    def predict(self, pairs: Sequence[Sequence[str]]) -> np.ndarray:
        kwargs = {}
        if self.cutoff_layers:
            kwargs["cutoff_layers"] = list(self.cutoff_layers)
        return _as_float_array(self.model.compute_score(list(pairs), **kwargs))


def load_reranker(
    model_name: str,
    cutoff_layer: int = 28,
) -> CrossEncoderRerankerAdapter | FlagEmbeddingRerankerAdapter:
    model_ref = resolve_hf_model_reference(model_name)

    if model_name in _LAYERWISE_RERANKERS:
        from FlagEmbedding import LayerWiseFlagLLMReranker

        model = LayerWiseFlagLLMReranker(
            model_ref,
            use_fp16=torch.cuda.is_available(),
        )
        return FlagEmbeddingRerankerAdapter(model=model, cutoff_layers=[int(cutoff_layer)])

    if model_name in _LLM_RERANKERS:
        from FlagEmbedding import FlagLLMReranker

        model = FlagLLMReranker(
            model_ref,
            use_fp16=torch.cuda.is_available(),
        )
        return FlagEmbeddingRerankerAdapter(model=model)

    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_ref, trust_remote_code=True)
        return CrossEncoderRerankerAdapter(model=model)
    except TypeError:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_ref)
        return CrossEncoderRerankerAdapter(model=model)
