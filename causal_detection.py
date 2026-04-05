"""Causal relation detection using a transformer classifier."""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CAUSAL_MODEL_NAME = "FacebookAI/roberta-large-mnli"

_causal_model: Optional[AutoModelForSequenceClassification] = None
_causal_tokenizer: Optional[AutoTokenizer] = None
_loaded_model_name: Optional[str] = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = [
    "detect_causal_edge",
    "find_all_causal_edges",
]


def _batch_chunks(items: List[Tuple[int, int, str, str]], size: int) -> Iterable[List[Tuple[int, int, str, str]]]:
    for idx in range(0, len(items), max(1, size)):
        yield items[idx : idx + max(1, size)]


def _load_causal_model(model_name: str = CAUSAL_MODEL_NAME) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Load and cache the causal classifier."""
    global _causal_model, _causal_tokenizer, _loaded_model_name
    if _causal_model is None or _loaded_model_name != model_name:
        _causal_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _causal_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _causal_model.to(_device)
        _causal_model.eval()
        _loaded_model_name = model_name
    assert _causal_model is not None
    assert _causal_tokenizer is not None
    return _causal_model, _causal_tokenizer


def _causal_label_indices(id2label) -> List[int]:
    """Identify which label ids correspond to causal relations."""
    mapping = {int(k): v for k, v in (id2label or {}).items()}
    if not mapping:
        mapping = {0: "NOT_CAUSE", 1: "CAUSE"}
    indices: List[int] = []
    for idx, label in mapping.items():
        if "cause" in label.lower():
            indices.append(idx)
    if not indices and len(mapping) == 2:
        indices.append(1)
    if not indices:
        indices.append(max(mapping.keys()))
    return indices


def _causal_probabilities(logits: torch.Tensor, id2label) -> List[float]:
    probs = torch.softmax(logits, dim=-1)
    positive_indices = _causal_label_indices(id2label)
    selected = probs[:, positive_indices]
    if selected.ndim == 1:
        selected = selected.unsqueeze(1)
    averaged = selected.mean(dim=1)
    return [float(p) for p in averaged]


def detect_causal_edge(
    sentence_a: str,
    sentence_b: str,
    *,
    threshold: float = 0.5,
    model_name: str = CAUSAL_MODEL_NAME,
) -> Optional[Tuple[str, float]]:
    """Predict whether sentence_a CAUSES sentence_b."""
    model, tokenizer = _load_causal_model(model_name)
    encoded = tokenizer(
        sentence_a,
        sentence_b,
        return_tensors="pt",
        truncation=True,
        padding=True,
    ).to(_device)
    with torch.inference_mode():
        logits = model(**encoded).logits
    prob = _causal_probabilities(logits, model.config.id2label)[0]
    if prob >= threshold:
        return "CAUSES", prob
    return None


def find_all_causal_edges(
    sentences: List[str],
    *,
    threshold: float = 0.5,
    batch_size: int = 8,
    model_name: str = CAUSAL_MODEL_NAME,
) -> List[Tuple[int, int, str, float]]:
    """Return all (i, j, 'CAUSES', score) edges among sentences."""
    if not sentences:
        return []

    model, tokenizer = _load_causal_model(model_name)
    pairs: List[Tuple[int, int, str, str]] = []
    for i, sent_a in enumerate(sentences):
        for j, sent_b in enumerate(sentences):
            if i == j:
                continue
            pairs.append((i, j, sent_a, sent_b))

    edges: List[Tuple[int, int, str, float]] = []
    for chunk in _batch_chunks(pairs, batch_size):
        a_texts = [item[2] for item in chunk]
        b_texts = [item[3] for item in chunk]
        encoded = tokenizer(
            a_texts,
            b_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(_device)
        with torch.inference_mode():
            logits = model(**encoded).logits
        probs = _causal_probabilities(logits, model.config.id2label)
        for (i, j, _, _), score in zip(chunk, probs):
            if score >= threshold:
                edges.append((i, j, "CAUSES", score))
    return edges
