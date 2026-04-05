"""Temporal ordering detection using NLI-based transformers."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_TEMPORAL_MODEL = "FacebookAI/roberta-large-mnli"
ALTERNATIVE_TEMPORAL_MODEL = "microsoft/deberta-v3-large"

_nli_models: Dict[str, AutoModelForSequenceClassification] = {}
_nli_tokenizers: Dict[str, AutoTokenizer] = {}
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = [
    "detect_temporal_edge",
    "find_all_temporal_edges",
]


def _batch_chunks(items: List[Tuple[int, int, str, str, str]], size: int) -> Iterable[List[Tuple[int, int, str, str, str]]]:
    for idx in range(0, len(items), max(1, size)):
        yield items[idx : idx + max(1, size)]


def _load_nli_model(model_name: str = DEFAULT_TEMPORAL_MODEL) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Load and cache an NLI model for temporal reasoning."""
    name = model_name or DEFAULT_TEMPORAL_MODEL
    if name not in _nli_models:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name)
        model.to(_device)
        model.eval()
        _nli_models[name] = model
        _nli_tokenizers[name] = tokenizer
    return _nli_models[name], _nli_tokenizers[name]


def _entailment_index(id2label) -> int:
    mapping = {int(k): v for k, v in (id2label or {}).items()}
    if not mapping:
        mapping = {0: "contradiction", 1: "neutral", 2: "entailment"}
    for idx, label in mapping.items():
        if "entail" in label.lower():
            return idx
    return max(mapping.keys())


def _entailment_probabilities(logits: torch.Tensor, id2label) -> List[float]:
    probs = torch.softmax(logits, dim=-1)
    target_idx = _entailment_index(id2label)
    selected = probs[:, target_idx]
    return [float(p) for p in selected]


def _build_hypotheses(sentence_a: str, sentence_b: str) -> Tuple[str, str]:
    before = f"{sentence_a} happened before {sentence_b}."
    after = f"{sentence_a} happened after {sentence_b}."
    return before, after


def detect_temporal_edge(
    sentence_a: str,
    sentence_b: str,
    *,
    threshold: float = 0.5,
    model_name: str = DEFAULT_TEMPORAL_MODEL,
) -> Optional[Tuple[str, float]]:
    """Predict BEFORE/AFTER relation between two sentences."""
    model, tokenizer = _load_nli_model(model_name)
    hyp_before, hyp_after = _build_hypotheses(sentence_a, sentence_b)

    encoded_before = tokenizer(
        sentence_a,
        hyp_before,
        return_tensors="pt",
        truncation=True,
        padding=True,
    ).to(_device)
    encoded_after = tokenizer(
        sentence_a,
        hyp_after,
        return_tensors="pt",
        truncation=True,
        padding=True,
    ).to(_device)

    with torch.inference_mode():
        logits_before = model(**encoded_before).logits
        logits_after = model(**encoded_after).logits

    score_before = _entailment_probabilities(logits_before, model.config.id2label)[0]
    score_after = _entailment_probabilities(logits_after, model.config.id2label)[0]

    if score_before < threshold and score_after < threshold:
        return None
    if score_before >= score_after:
        return "BEFORE", score_before
    return "AFTER", score_after


def find_all_temporal_edges(
    sentences: List[str],
    *,
    threshold: float = 0.5,
    batch_size: int = 8,
    model_name: str = DEFAULT_TEMPORAL_MODEL,
) -> List[Tuple[int, int, str, float]]:
    """Return all (i, j, 'BEFORE'/'AFTER', score) edges among sentences."""
    if not sentences:
        return []

    model, tokenizer = _load_nli_model(model_name)
    entries: List[Tuple[int, int, str, str, str]] = []
    for i, sent_a in enumerate(sentences):
        for j, sent_b in enumerate(sentences):
            if i == j:
                continue
            hyp_before, hyp_after = _build_hypotheses(sent_a, sent_b)
            entries.append((i, j, "BEFORE", sent_a, hyp_before))
            entries.append((i, j, "AFTER", sent_a, hyp_after))

    scores: Dict[Tuple[int, int], Dict[str, float]] = {}
    for chunk in _batch_chunks(entries, batch_size):
        premises = [item[3] for item in chunk]
        hypotheses = [item[4] for item in chunk]
        encoded = tokenizer(
            premises,
            hypotheses,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(_device)
        with torch.inference_mode():
            logits = model(**encoded).logits
        probs = _entailment_probabilities(logits, model.config.id2label)
        for (i, j, label, _, _), prob in zip(chunk, probs):
            slot = scores.setdefault((i, j), {"BEFORE": 0.0, "AFTER": 0.0})
            slot[label] = prob

    edges: List[Tuple[int, int, str, float]] = []
    for (i, j), label_scores in scores.items():
        before_score = label_scores.get("BEFORE", 0.0)
        after_score = label_scores.get("AFTER", 0.0)
        if before_score < threshold and after_score < threshold:
            continue
        if before_score >= after_score:
            edges.append((i, j, "BEFORE", before_score))
        else:
            edges.append((i, j, "AFTER", after_score))
    return edges
