from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer


_DEFAULT_CACHE_ROOT = Path.home() / "models"


def get_model_cache_root() -> Path:
    configured = os.getenv("MEMR_MODEL_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return _DEFAULT_CACHE_ROOT


def get_hf_cache_dir() -> Path:
    return get_model_cache_root() / "huggingface"


def get_sentence_transformer_store_dir() -> Path:
    return get_model_cache_root() / "sentence_transformers"


def get_hf_model_store_dir() -> Path:
    return get_model_cache_root() / "hf_models"


def ensure_model_cache_dirs() -> Path:
    cache_root = get_model_cache_root()
    hf_cache_dir = get_hf_cache_dir()
    st_store_dir = get_sentence_transformer_store_dir()
    hf_model_store_dir = get_hf_model_store_dir()

    cache_root.mkdir(parents=True, exist_ok=True)
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    st_store_dir.mkdir(parents=True, exist_ok=True)
    hf_model_store_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HOME", str(hf_cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache_dir / "transformers"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(hf_cache_dir / "sentence_transformers_home"))
    return cache_root


def _normalize_model_key(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", str(model_name).strip())


def get_local_sentence_transformer_dir(model_name: str) -> Path:
    model_path = Path(str(model_name)).expanduser()
    if model_path.exists():
        return model_path.resolve()
    return get_sentence_transformer_store_dir() / _normalize_model_key(model_name)


def get_local_hf_model_dir(model_name: str) -> Path:
    model_path = Path(str(model_name)).expanduser()
    if model_path.exists():
        return model_path.resolve()
    return get_hf_model_store_dir() / _normalize_model_key(model_name)


def has_local_sentence_transformer_artifacts(model_dir: Path) -> bool:
    return model_dir.is_dir() and (
        (model_dir / "modules.json").exists()
        or (model_dir / "config_sentence_transformers.json").exists()
    )


def has_local_hf_model_artifacts(model_dir: Path) -> bool:
    return model_dir.is_dir() and any(
        (model_dir / filename).exists()
        for filename in [
            "config.json",
            "tokenizer_config.json",
            "pytorch_model.bin",
            "model.safetensors",
            "model.safetensors.index.json",
        ]
    )


def load_sentence_transformer(model_name: str, **kwargs) -> SentenceTransformer:
    ensure_model_cache_dirs()

    local_model_dir = get_local_sentence_transformer_dir(model_name)
    if has_local_sentence_transformer_artifacts(local_model_dir):
        return SentenceTransformer(str(local_model_dir), **kwargs)

    return SentenceTransformer(model_name, cache_folder=str(get_hf_cache_dir()), **kwargs)


def download_sentence_transformer(
    model_name: str,
    target_dir: Optional[str] = None,
    force: bool = False,
    **kwargs,
) -> Path:
    ensure_model_cache_dirs()

    local_model_dir = (
        Path(target_dir).expanduser().resolve()
        if target_dir
        else get_local_sentence_transformer_dir(model_name)
    )
    if has_local_sentence_transformer_artifacts(local_model_dir) and not force:
        return local_model_dir

    model = SentenceTransformer(model_name, cache_folder=str(get_hf_cache_dir()), **kwargs)
    local_model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(local_model_dir))
    return local_model_dir


def resolve_hf_model_reference(model_name: str) -> str:
    ensure_model_cache_dirs()
    local_model_dir = get_local_hf_model_dir(model_name)
    if has_local_hf_model_artifacts(local_model_dir):
        return str(local_model_dir)
    return model_name


def download_hf_snapshot(
    model_name: str,
    target_dir: Optional[str] = None,
    force: bool = False,
) -> Path:
    ensure_model_cache_dirs()

    local_model_dir = (
        Path(target_dir).expanduser().resolve()
        if target_dir
        else get_local_hf_model_dir(model_name)
    )
    if has_local_hf_model_artifacts(local_model_dir) and not force:
        return local_model_dir

    from huggingface_hub import snapshot_download

    local_model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_name,
        local_dir=str(local_model_dir),
        local_dir_use_symlinks=False,
        cache_dir=str(get_hf_cache_dir()),
        resume_download=True,
    )
    return local_model_dir
