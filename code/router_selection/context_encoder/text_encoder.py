"""Text embedding with disk caching for router selection."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import hashlib
import json
import os

import torch


def _normalize_name(value: str) -> str:
    text = str(value or "").strip().lower()
    for token in (" ", "-", "/"):
        text = text.replace(token, "_")
    return text


def _safe_read_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_context_index(context_root: str) -> Dict[Tuple[str, str], Path]:
    """Scan dataset context folders and map (dataset, task_level) to context paths."""
    root = Path(context_root) / "dataset"
    index: Dict[Tuple[str, str], Path] = {}
    if not root.exists():
        return index
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        dataset_path = folder / "dataset_llm_context.json"
        if not dataset_path.exists():
            continue
        task_level = folder.name.split("_")[-1].lower()
        try:
            payload = _safe_read_json(dataset_path)
        except Exception:
            continue
        dataset_name = payload.get("dataset", {}).get("name") or payload.get("dataset", {}).get("dataset") or folder.name
        key = (_normalize_name(dataset_name), task_level)
        index[key] = dataset_path
        index[(_normalize_name(folder.name), task_level)] = dataset_path
    return index


def resolve_dataset_context_path(
    context_index: Dict[Tuple[str, str], Path],
    dataset_name: str,
    task_level: str,
) -> Optional[Path]:
    key = (_normalize_name(dataset_name), str(task_level).lower())
    return context_index.get(key)


def resolve_model_context_path(context_root: str, model_name: str) -> Optional[Path]:
    path = Path(context_root) / "model" / str(model_name) / "model_llm_context.json"
    return path if path.exists() else None


def _assemble_dataset_text(payload: Dict) -> str:
    description = payload.get("description") or ""
    dataset_meta = payload.get("dataset", {})
    name = dataset_meta.get("name") or ""
    domain = dataset_meta.get("domain") or ""
    parts = [part for part in (name, domain, description) if part]
    return "\n".join(parts)


def _assemble_model_text(payload: Dict) -> str:
    parts: List[str] = []
    architecture = payload.get("architecture", {})
    if isinstance(architecture, dict):
        summary = architecture.get("architecture_summary")
        if summary:
            parts.append(str(summary))
        for key in ("model", "task"):
            value = architecture.get(key)
            if value:
                parts.append(str(value))
    for key in ("dataset_usage", "task_summary", "training_summary"):
        value = payload.get(key)
        if value:
            parts.append(str(value))
    pretrain = payload.get("pretrain_config", {})
    if isinstance(pretrain, dict):
        method = pretrain.get("method")
        if method:
            parts.append(f"method: {method}")
        dataset = pretrain.get("dataset", {})
        if isinstance(dataset, dict):
            ds_name = dataset.get("name")
            if ds_name:
                parts.append(f"dataset: {ds_name}")
    if not parts:
        parts.append(json.dumps(payload, sort_keys=True))
    return "\n".join(parts)


def load_context_text(path: Path, kind: str) -> str:
    payload = _safe_read_json(path)
    if kind == "dataset":
        return _assemble_dataset_text(payload)
    if kind == "model":
        return _assemble_model_text(payload)
    return json.dumps(payload, sort_keys=True)


@dataclass
class TextEncoderConfig:
    model_name: str
    cache_dir: str
    max_length: int = 256
    batch_size: int = 16
    device: Optional[str] = None


class TextContextEncoder:
    """Encode text with a transformer model and cache embeddings on disk."""

    def __init__(self, cfg: TextEncoderConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._tokenizer = None
        self._model = None
        self._embed_dim = None

    @property
    def embed_dim(self) -> int:
        if self._embed_dim is None:
            self._ensure_model()
        return int(self._embed_dim or 0)

    def _ensure_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "transformers is required for text embeddings. Install it to run router selection."
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        self._model = AutoModel.from_pretrained(self.cfg.model_name)
        self._model.to(self.device)
        self._model.eval()
        self._embed_dim = getattr(self._model.config, "hidden_size", None)

    def _cache_path(self, key: str) -> Path:
        safe_model = _normalize_name(self.cfg.model_name)
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return Path(self.cfg.cache_dir) / safe_model / f"{digest}.pt"

    def _load_cache(self, key: str) -> Optional[torch.Tensor]:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            payload = torch.load(path, map_location="cpu")
            return payload.get("embedding")
        except Exception:
            return None

    def _save_cache(self, key: str, embedding: torch.Tensor) -> None:
        path = self._cache_path(key)
        os.makedirs(path.parent, exist_ok=True)
        try:
            torch.save({"embedding": embedding.cpu()}, path)
        except Exception:
            pass

    def encode_texts(self, texts: Iterable[str]) -> torch.Tensor:
        self._ensure_model()
        assert self._tokenizer is not None
        assert self._model is not None
        batch_size = max(1, int(self.cfg.batch_size))
        outputs: List[torch.Tensor] = []
        texts_list = list(texts)
        for start in range(0, len(texts_list), batch_size):
            chunk = texts_list[start : start + batch_size]
            tokens = self._tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
            ).to(self.device)
            with torch.no_grad():
                model_out = self._model(**tokens)
                hidden = model_out.last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            outputs.append(pooled.cpu())
        return torch.cat(outputs, dim=0) if outputs else torch.empty((0, self.embed_dim), dtype=torch.float32)

    def embed_text(self, text: str, cache_key: str) -> torch.Tensor:
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached
        embedding = self.encode_texts([text])
        embedding = embedding.squeeze(0)
        self._save_cache(cache_key, embedding)
        return embedding

    def embed_file(self, path: Path, kind: str) -> torch.Tensor:
        cache_key = f"{kind}:{path}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached
        text = load_context_text(path, kind)
        embedding = self.embed_text(text, cache_key)
        return embedding
