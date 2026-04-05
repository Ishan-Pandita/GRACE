from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)
try:
    # Optional; model loads only if enabled below
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:  # pragma: no cover
    CrossEncoder = None  # type: ignore
import torch
import torch.nn.functional as F
import networkx as nx
import random
from collections import Counter
import math
import time
import json
import argparse
import os
import re
import shutil
from causal_detection import find_all_causal_edges, CAUSAL_MODEL_NAME
from temporal_reasoning import find_all_temporal_edges, DEFAULT_TEMPORAL_MODEL
from coherence_analysis import run_coherence_analysis

# ---------------- Device ----------------
_CUDA_AVAILABLE = torch.cuda.is_available()
_DEFAULT_DEVICE = torch.device("cuda" if _CUDA_AVAILABLE else "cpu")
# ---------------- Dataset ----------------
dataset = load_dataset("deadcode99/CQR")
split = "train" if "train" in dataset else list(dataset.keys())[0]


# ---------------- Answer helpers ----------------
def _coerce_answer_to_str(val):
    try:
        if val is None:
            return None
        if isinstance(val, (list, tuple)):
            parts = []
            for x in val:
                if isinstance(x, dict):
                    parts.append(str(x.get("text") or x.get("answer") or x))
                else:
                    parts.append(str(x))
            return "; ".join(parts)
        if isinstance(val, dict):
            return str(val.get("text") or val.get("answer") or val)
        return str(val)
    except Exception:
        return None


_answer_keys_pref = [
    "answer",
    "answers",
    "gold",
    "gold_answer",
    "label",
    "target",
    "output",
    "final_answer",
]

_QUESTION_STOPWORDS = {
    "the", "a", "an", "and", "or", "in", "on", "at", "of", "to", "for", "with", "by",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "done",
    "this", "that", "these", "those",
    "it", "its", "their", "there", "then", "than",
    "into", "from", "about", "over", "under", "after", "before",
    "many", "much", "several", "any", "some", "all", "each", "every",
    "give", "list", "name", "tell", "describe", "state", "identify"
}

_MONTH_NAMES = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}

RE_EXTRACT_NAMES = re.compile(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)")

def _extract_answer_text(row: dict):
    ans = None
    for k in _answer_keys_pref:
        if k in row:
            ans = _coerce_answer_to_str(row[k])
            if ans:
                break
    return ans


def _extract_answer_texts(row: dict) -> list[str]:
    """Return a list of plausible answer strings from the row.
    Handles strings, lists/tuples of strings or dicts, and dicts with common keys.
    """
    texts: list[str] = []

    def add_text(x):
        if x is None:
            return
        s = str(x).strip()
        if s:
            texts.append(s)

    def from_dict(d: dict):
        # Try common textual keys in order of preference
        for key in ("text", "answer", "string", "normalized", "normalized_value", "value", "label"):
            if key in d and d[key] is not None:
                v = d[key]
                # Some datasets put arrays under these keys
                if isinstance(v, (list, tuple)):
                    for item in v:
                        add_text(item)
                else:
                    add_text(v)
        # Fallback: if nothing extracted, avoid dumping the raw dict as string

    for k in _answer_keys_pref:
        if k not in row:
            continue
        v = row[k]
        if v is None:
            continue
        if isinstance(v, str):
            add_text(v)
        elif isinstance(v, (list, tuple)):
            for item in v:
                if isinstance(item, str):
                    add_text(item)
                elif isinstance(item, dict):
                    from_dict(item)
                else:
                    add_text(item)
        elif isinstance(v, dict):
            from_dict(v)
        else:
            add_text(v)

        # Stop at first key that yields anything
        if texts:
            break

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for t in texts:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq

def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    pct = max(0.0, min(1.0, pct))
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = pct * (len(sorted_vals) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return sorted_vals[lower]
    weight = rank - lower
    return sorted_vals[lower] * (1.0 - weight) + sorted_vals[upper] * weight


def _adaptive_sentence_k(
    sorted_scores: list[float],
    *,
    total_sentences: int,
    max_k: int | None = None,
    min_k: int = 1,
) -> int:
    """Adaptive K for sentence evidence based on relevance drop and context size.

    Expects scores sorted in descending order.
    """
    if not sorted_scores:
        return 0

    # Context-size constraints
    if total_sentences <= 10:
        context_cap = 5
    elif total_sentences <= 20:
        context_cap = 7
    elif total_sentences <= 40:
        context_cap = 8
    else:
        context_cap = 10

    # Do not exceed available candidates
    context_cap = min(context_cap, len(sorted_scores))

    # Treat CLI sentence_top_k as an upper bound if provided
    if max_k is not None and max_k > 0:
        context_cap = min(context_cap, max_k)

    if context_cap <= 0:
        return 0

    context_cap = max(min_k, context_cap)
    context_cap = min(context_cap, len(sorted_scores))

    if context_cap == 1:
        return 1

    # Threshold-based elbow on relevance drop Rel_i - Rel_{i+1}
    delta = 0.07
    elbow = context_cap
    for i in range(context_cap - 1):
        drop = sorted_scores[i] - sorted_scores[i + 1]
        if drop >= delta and (i + 1) >= min_k:
            elbow = i + 1
            break

    k = max(min_k, min(elbow, context_cap))
    return k

def _sigmoid_scalar(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-float(x)))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _mix_rerank_score(logit: float, cosine: float) -> float:
    cos_norm = (cosine + 1.0) / 2.0
    return 0.75 * _sigmoid_scalar(logit) + 0.25 * cos_norm


def _mmr_select_indices(
    query_emb: torch.Tensor,
    candidate_indices: list[int],
    candidate_embs: list[torch.Tensor],
    *,
    target_k: int,
    lamb: float,
) -> list[int]:
    remaining = list(candidate_indices)
    selected: list[int] = []
    selected_embs: list[torch.Tensor] = []
    while remaining and len(selected) < target_k:
        best_idx: int | None = None
        best_score: float | None = None
        best_pos = 0
        for pos, idx in enumerate(remaining):
            emb = candidate_embs[idx]
            score = mmr_score(query_emb, emb, selected_embs=selected_embs, lamb=lamb)
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
                best_pos = pos
        if best_idx is None:
            break
        selected.append(best_idx)
        selected_embs.append(candidate_embs[best_idx])
        remaining.pop(best_pos)
    return selected

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")

def _tokenize_lower(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_PATTERN.findall(text)]


def _extract_question_signals(question_text: str) -> tuple[set[str], bool]:
    tokens = _tokenize_lower(question_text)
    signals: set[str] = set()
    has_digit = False
    for tok in tokens:
        if not tok:
            continue
        if any(ch.isdigit() for ch in tok):
            has_digit = True
            signals.add(tok)
            continue
        if tok in _QUESTION_STOPWORDS:
            continue
        if len(tok) <= 2:
            continue
        signals.add(tok)
    for match in RE_EXTRACT_NAMES.finditer(question_text):
        name = match.group(0).strip().lower()
        if name:
            signals.add(name)
    question_lower = question_text.lower()
    require_answer_hint = False
    hint_phrases = [
        "how many", "how much", "how old", "how long", "how far", "how tall", "how big", "how high",
        "what year", "which year", "what date", "what day", "what time", "what month", "when",
    ]
    if any(phrase in question_lower for phrase in hint_phrases):
        require_answer_hint = True
    if has_digit:
        require_answer_hint = True
    return signals, require_answer_hint


def _sentence_has_answer_hint(text: str) -> bool:
    if any(ch.isdigit() for ch in text):
        return True
    tokens = _tokenize_lower(text)
    if any(tok in _MONTH_NAMES for tok in tokens):
        return True
    ordinal_keywords = {"first", "second", "third", "fourth", "fifth", "last", "final"}
    if any(tok in ordinal_keywords for tok in tokens):
        return True
    return False


def _signal_in_sentence(signal: str, sentence_lower: str, sentence_tokens: set[str]) -> bool:
    if not signal:
        return False
    if " " in signal:
        return signal in sentence_lower
    return signal in sentence_tokens



QA_MODEL_NAME = "microsoft/deberta-v3-base-squad2"
QA_MAX_SEQ_LEN = 384
QA_MAX_ANSWER_LEN = 48
QA_SCORE_THRESHOLD = 0.5

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
NLI_ENTAILMENT_THRESHOLD = 0.5
NLI_ENTAILMENT_IDX = 2

_QA_TOKENIZER: AutoTokenizer | None = None
_QA_MODEL: AutoModelForQuestionAnswering | None = None
_NLI_TOKENIZER: AutoTokenizer | None = None
_NLI_MODEL: AutoModelForSequenceClassification | None = None

def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out

def _preview_text(text: str | None, limit: int = 160) -> str:
    if not text:
        return '<empty>'
    cleaned = ' '.join(str(text).strip().split())
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 3] + '...'
    return cleaned.encode('utf-8', errors='replace').decode('utf-8')

def _get_qa_components():
    global _QA_TOKENIZER, _QA_MODEL
    if _QA_MODEL is not None and _QA_TOKENIZER is not None:
        return _QA_TOKENIZER, _QA_MODEL
    try:
        tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
        model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME)
        model.eval()
        _QA_TOKENIZER = tokenizer
        _QA_MODEL = model
    except Exception:
        _QA_TOKENIZER = None
        _QA_MODEL = None
    return _QA_TOKENIZER, _QA_MODEL

def _run_extractive_qa(question: str, evidence: list[str]):
    tokenizer, model = _get_qa_components()
    if tokenizer is None or model is None:
        return None
    context = " ".join(evidence).strip()
    if not context:
        return None
    try:
        inputs = tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation=True,
            max_length=QA_MAX_SEQ_LEN,
            return_offsets_mapping=True,
        )
    except Exception:
        return None
    offsets = inputs.pop("offset_mapping")[0]
    with torch.no_grad():
        outputs = model(**inputs)
    start_logits = outputs.start_logits[0].cpu()
    end_logits = outputs.end_logits[0].cpu()
    best_score = float("-inf")
    best_indices: tuple[int, int] | None = None
    for start_idx, (start_offset, start_offset_end) in enumerate(offsets):
        if start_offset == start_offset_end:
            continue
        for end_idx in range(start_idx, min(len(offsets), start_idx + QA_MAX_ANSWER_LEN)):
            end_offset, end_offset_end = offsets[end_idx]
            if end_offset == end_offset_end:
                continue
            logit_sum = start_logits[start_idx] + end_logits[end_idx]
            if float(logit_sum) > best_score:
                best_score = float(logit_sum)
                best_indices = (start_idx, end_idx)
    if best_indices is None:
        return None
    start_idx, end_idx = best_indices
    start_char, _ = offsets[start_idx]
    _, end_char = offsets[end_idx]
    span_text = context[start_char:end_char].strip()
    if not span_text:
        return None
    qa_conf = float(torch.sigmoid(torch.tensor(best_score)).item())
    return {
        "text": span_text,
        "score": qa_conf,
    }

def _get_nli_components():
    global _NLI_TOKENIZER, _NLI_MODEL
    if _NLI_MODEL is not None and _NLI_TOKENIZER is not None:
        return _NLI_TOKENIZER, _NLI_MODEL
    try:
        tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
        model.eval()
        _NLI_TOKENIZER = tokenizer
        _NLI_MODEL = model
    except Exception:
        _NLI_TOKENIZER = None
        _NLI_MODEL = None
    return _NLI_TOKENIZER, _NLI_MODEL

def _compute_entailment_probs(evidence: list[str], answer: str):
    tokenizer, model = _get_nli_components()
    answer = (answer or "").strip()
    clean_evidence = [e for e in evidence if isinstance(e, str) and e.strip()]
    if tokenizer is None or model is None or not clean_evidence or not answer:
        return None
    try:
        inputs = tokenizer(
            clean_evidence,
            [answer] * len(clean_evidence),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
    except Exception:
        return None
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    entail = probs[:, NLI_ENTAILMENT_IDX].cpu().tolist()
    return entail

def _derive_final_answer(question: str, evidence: list[str]):
    clean_evidence = _dedupe_preserve_order([e.strip() for e in evidence if isinstance(e, str)])
    qa_result = _run_extractive_qa(question, clean_evidence)
    final_answer = None
    source = None
    qa_score = None
    if qa_result and qa_result.get("score") is not None and qa_result.get("score") >= QA_SCORE_THRESHOLD:
        final_answer = qa_result.get("text")
        qa_score = qa_result.get("score")
        source = "qa_span"
    elif clean_evidence:
        final_answer = clean_evidence[0]
        source = "evidence_fallback"
    entail_probs = _compute_entailment_probs(clean_evidence, final_answer)
    best_entail = None
    best_idx = None
    if entail_probs:
        best_entail = max(entail_probs)
        best_idx = entail_probs.index(best_entail)
    if entail_probs and (best_entail is None or best_entail < NLI_ENTAILMENT_THRESHOLD):
        if best_idx is not None and clean_evidence:
            final_answer = clean_evidence[best_idx]
            source = "nli_fallback"
    return {
        "answer": final_answer.strip() if isinstance(final_answer, str) else final_answer,
        "qa_score": qa_score,
        "answer_source": source,
        "entailment_scores": entail_probs,
        "evidence": clean_evidence,
    }


def _rewrite_answer_with_flan(
    question: str,
    evidence: list[str],
    base_answer: str | None,
    *,
    max_new_tokens: int = 80,
) -> tuple[str | None, dict | None]:
    """Use FLAN-T5 to rewrite/summarize the answer with provided evidence."""
    tokenizer, model = _get_flan_components()
    if tokenizer is None or model is None:
        return base_answer, None
    evidence_text = " ".join([e.strip() for e in evidence if isinstance(e, str) and e.strip()])
    if not evidence_text:
        return base_answer, None
    prompt = (
        "You are an expert reasoning and summarization model. "
        "Given a QUESTION and EVIDENCE (selected nodes forming a paragraph), "
        "produce a clear, concise, fully rephrased FINAL ANSWER in 1-2 sentences. "
        "Do NOT copy sentences; synthesize and use only supported information. "
        "Include causal/temporal relations if implied. "
        f"QUESTION: {question}\n"
        f"EVIDENCE: {evidence_text}\n"
        f"CURRENT ANSWER: {base_answer or ''}\n"
        "FINAL ANSWER:"
    )
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                temperature=0.3,
                top_p=0.9,
            )
        text = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        return text or base_answer, {
            "model": FLAN_MODEL_NAME,
            "prompt_len": int(inputs["input_ids"].shape[1]),
            "used_beams": 4,
        }
    except Exception:
        return base_answer, None
def _select_sentence_evidence(
    primary_indices: list[int],
    candidate_pool: list[int],
    sent_list: list[str],
    sent_embs: list[torch.Tensor],
    question_signals: set[str],
    require_answer_hint: bool,
    *,
    limit: int,
) -> list[int]:
    coverage_targets = {sig for sig in question_signals if sig}
    if not coverage_targets:
        coverage_targets = set()
    sentence_lower_cache = {}
    sentence_token_cache = {}

    def get_lower(idx: int) -> str:
        if idx not in sentence_lower_cache:
            sentence_lower_cache[idx] = sent_list[idx].lower()
        return sentence_lower_cache[idx]

    def get_tokens(idx: int) -> set[str]:
        if idx not in sentence_token_cache:
            sentence_token_cache[idx] = set(_tokenize_lower(sent_list[idx]))
        return sentence_token_cache[idx]

    ordered_candidates = list(dict.fromkeys(primary_indices + [idx for idx in candidate_pool if idx not in primary_indices]))
    chosen: list[int] = []
    covered: set[str] = set()
    answer_hint_covered = False

    for idx in ordered_candidates:
        if len(chosen) >= limit:
            break
        sent_lower = get_lower(idx)
        sent_tokens = get_tokens(idx)
        newly = {sig for sig in coverage_targets - covered if _signal_in_sentence(sig, sent_lower, sent_tokens)}
        needs_answer_hint = require_answer_hint and not answer_hint_covered
        has_hint = _sentence_has_answer_hint(sent_list[idx])
        should_select = False
        if newly:
            should_select = True
        if needs_answer_hint and has_hint:
            should_select = True
        if not chosen and not coverage_targets:
            should_select = True
        if not should_select and not chosen:
            # ensure at least one sentence is selected even if no signals matched yet
            should_select = True
        if should_select:
            chosen.append(idx)
            covered.update(newly)
            if has_hint:
                answer_hint_covered = True
        if covered >= coverage_targets and (not require_answer_hint or answer_hint_covered):
            break

    if not chosen and ordered_candidates:
        chosen.append(ordered_candidates[0])
    return chosen[:limit]

# ---------------- Embedding models ----------------
# BGE-large powers start-node detection and all graph/evidence embeddings.

_BGE_LARGE_MODEL_NAME = "BAAI/bge-large-en-v1.5"
_FALLBACK_BGE_NAME = "BAAI/bge-base-en-v1.5"
_BGE_LARGE = None


def _get_bge_model():
    """Lazy-load BGE encoder once per process to avoid repeated downloads/installs."""
    global _BGE_LARGE, _BGE_LARGE_MODEL_NAME
    if _BGE_LARGE is not None:
        return _BGE_LARGE
    try:
        _BGE_LARGE = SentenceTransformer(_BGE_LARGE_MODEL_NAME, device=str(_DEFAULT_DEVICE))
        print(f"[Embedding] Loaded BGE-large: {_BGE_LARGE_MODEL_NAME}")
    except Exception as _e_bge:
        print(f"[Embedding] Failed to load {_BGE_LARGE_MODEL_NAME}: {_e_bge}")
        print(f"[Embedding] Falling back to {_FALLBACK_BGE_NAME} for embeddings.")
        _BGE_LARGE_MODEL_NAME = _FALLBACK_BGE_NAME
        _BGE_LARGE = SentenceTransformer(_FALLBACK_BGE_NAME, device=str(_DEFAULT_DEVICE))
    return _BGE_LARGE


def _encode_bge_large(sentences: list[str]) -> torch.Tensor:
    """Encode sentences into L2-normalized embeddings using BGE-large.

    Used only for start-node detection and related scoring.
    """
    if not sentences:
        raise ValueError("_encode_bge_large: sentences list is empty")
    model = _get_bge_model()
    return model.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)

# Reranker toggle and settings
RERANKER_ENABLED = True  # set False to disable reranking
RERANKER_MODEL = "BAAI/bge-reranker-base"
# More permissive default; override via --rerank-top-n
RERANK_TOP_N = 60
RERANK_MMR_MIN = 10
RERANK_MMR_MAX = 20
RERANK_MMR_LAMBDA = 0.35
RERANK_SCORE_TIE_DELTA = 0.05
RERANK_EXPAND_STEP = 20
_RERANKER = None

def _get_reranker():
    global _RERANKER
    if not RERANKER_ENABLED:
        return None
    if CrossEncoder is None:
        return None
    if _RERANKER is not None:
        return _RERANKER
    try:
        _RERANKER = CrossEncoder(RERANKER_MODEL)
    except Exception:
        _RERANKER = None
    return _RERANKER


FLAN_MODEL_NAME = "google/flan-t5-base"
_FLAN_TOKENIZER = None
_FLAN_MODEL = None


def _get_flan_components():
    global _FLAN_TOKENIZER, _FLAN_MODEL
    if _FLAN_MODEL is not None and _FLAN_TOKENIZER is not None:
        return _FLAN_TOKENIZER, _FLAN_MODEL
    try:
        _FLAN_TOKENIZER = AutoTokenizer.from_pretrained(FLAN_MODEL_NAME)
        _FLAN_MODEL = AutoModelForSeq2SeqLM.from_pretrained(
            FLAN_MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        )
        _FLAN_MODEL.eval()
    except Exception:
        _FLAN_TOKENIZER = None
        _FLAN_MODEL = None
    return _FLAN_TOKENIZER, _FLAN_MODEL


CAUSAL_SCORE_THRESHOLD = 0.5
CAUSAL_BATCH_SIZE = 8
TEMPORAL_SCORE_THRESHOLD = 0.5
TEMPORAL_BATCH_SIZE = 8
TEMPORAL_MODEL_NAME = DEFAULT_TEMPORAL_MODEL


def _extract_sentence_nodes(graph: nx.Graph) -> tuple[list[str], list[str]]:
    node_ids: list[str] = []
    sentences: list[str] = []
    for node_id, data in graph.nodes(data=True):
        if node_id == "question":
            continue
        text = data.get("text")
        if not isinstance(text, str):
            continue
        stripped = text.strip()
        if not stripped:
            continue
        node_ids.append(node_id)
        sentences.append(stripped)
    return node_ids, sentences


def _infer_causal_edges(graph: nx.Graph, *, threshold: float = CAUSAL_SCORE_THRESHOLD, batch_size: int = CAUSAL_BATCH_SIZE) -> list[dict]:
    node_ids, sentences = _extract_sentence_nodes(graph)
    if not sentences:
        return []
    raw_edges = find_all_causal_edges(sentences, threshold=threshold, batch_size=batch_size)
    edges: list[dict] = []
    for i, j, rel, score in raw_edges:
        if i >= len(node_ids) or j >= len(node_ids):
            continue
        edges.append(
            {
                "source": node_ids[i],
                "target": node_ids[j],
                "rtype": rel,
                "weight": float(score),
                "verification": {
                    "model": CAUSAL_MODEL_NAME,
                    "score": float(score),
                    "threshold": threshold,
                },
            }
        )
    return edges


def _infer_temporal_edges(
    graph: nx.Graph,
    *,
    threshold: float = TEMPORAL_SCORE_THRESHOLD,
    batch_size: int = TEMPORAL_BATCH_SIZE,
    model_name: str = TEMPORAL_MODEL_NAME,
) -> list[dict]:
    node_ids, sentences = _extract_sentence_nodes(graph)
    if not sentences:
        return []
    raw_edges = find_all_temporal_edges(
        sentences,
        threshold=threshold,
        batch_size=batch_size,
        model_name=model_name,
    )
    edges: list[dict] = []
    for i, j, rel, score in raw_edges:
        if i >= len(node_ids) or j >= len(node_ids):
            continue
        edges.append(
            {
                "source": node_ids[i],
                "target": node_ids[j],
                "rtype": rel,
                "weight": float(score),
                "verification": {
                    "model": model_name,
                    "score": float(score),
                    "threshold": threshold,
                },
            }
        )
    return edges


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = torch.nn.functional.normalize(a, p=2, dim=0)
    b = torch.nn.functional.normalize(b, p=2, dim=0)
    return F.cosine_similarity(a, b, dim=0).item()


def mmr_score(query_emb: torch.Tensor,
              candidate_emb: torch.Tensor,
              selected_embs=None,
              lamb: float = 0.7) -> float:
    sim_q = cosine_sim(candidate_emb, query_emb)
    diversity = 0.0
    if selected_embs:
        diversity = max(cosine_sim(candidate_emb, s) for s in selected_embs)
    return lamb * sim_q - (1.0 - lamb) * diversity


def create_or_update_graph(
    graph: nx.Graph,
    query_id: str,
    query_text: str,
    query_emb: torch.Tensor,
    start_id: str,
    start_text: str,
    start_emb: torch.Tensor,
    relevance_map: dict,
    *,
    mmr_lambda: float = 0.7,
    mmr_threshold: float = 0.35,
    selected_embs=None,
):
    score = mmr_score(query_emb, start_emb, selected_embs=selected_embs, lamb=mmr_lambda)

    if score >= mmr_threshold:
        graph.add_node(query_id, text=query_text, embedding=query_emb)
        graph.add_node(start_id, text=start_text, embedding=start_emb)
        graph.add_edge(query_id, start_id, weight=score, kind="query-start")
        relevance_map[(query_id, start_id)] = score
        return graph, score

    new_graph = nx.Graph()
    new_graph.add_node(query_id, text=query_text, embedding=query_emb)
    relevance_map[(query_id, start_id)] = score
    return new_graph, score


# ---------------- Graph utils ----------------
def _encode_bge(sentences: list[str]) -> torch.Tensor:
    """Encode sentences for evidence / graph using the unified BGE-large encoder."""
    if not sentences:
        raise ValueError("_encode_bge: sentences list is empty")
    return _encode_bge_large(sentences)


def embed_segments(segments: list[str]):
    """Embed segments with BGE-large for downstream graph reasoning."""
    if not segments:
        return []
    embs = _encode_bge(segments)
    return [embs[i] for i in range(embs.size(0))]

_SENT_SPLIT_RE = re.compile(r"(?<=[.?!])\s+(?=\S)")

def split_into_sentences(text: str) -> list[str]:
    try:
        parts = _SENT_SPLIT_RE.split(text.strip())
    except Exception:
        parts = [text]
    out: list[str] = []
    seen = set()
    for p in parts:
        s = p.strip().strip("' ")
        if not s:
            continue
        if len(s) < 3:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def collect_sentences(
    segments: list[str],
    *,
    scope: str = "best",
    best_segment: str | None = None,
) -> tuple[list[str], list[int]]:
    """Return sentences and their source segment indices according to scope.
    - scope == "best": use sentences from the best_segment (if provided), else the first segment.
    - scope == "all": use sentences from every segment.
    """
    indexed_segments: list[tuple[int, str]]
    if scope == "best":
        indexed_segments = []
        if best_segment is not None:
            try:
                idx = segments.index(best_segment)
                indexed_segments.append((idx, best_segment))
            except ValueError:
                pass
        if not indexed_segments and segments:
            indexed_segments.append((0, segments[0]))
    else:
        indexed_segments = [(i, seg) for i, seg in enumerate(segments)]

    out: list[str] = []
    sources: list[int] = []
    seen = set()
    for seg_idx, seg in indexed_segments:
        if not seg:
            continue
        for s in split_into_sentences(str(seg)):
            if len(s) < 3:
                continue
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
            sources.append(seg_idx)
    return out, sources


def _estimate_token_len(text: str) -> int:
    try:
        return max(1, len(str(text).split()))
    except Exception:
        return 1


def _chunk_sentences_sliding_window(
    sentences: list[str],
    *,
    window_tokens: int = 500,
    overlap_ratio: float = 0.3,
) -> list[list[int]]:
    """Fallback sliding-window chunking over sentences.

    Returns a list of chunks, each as a list of sentence indices.
    """
    if not sentences:
        return []

    token_lens = [_estimate_token_len(s) for s in sentences]
    n = len(sentences)
    chunks: list[list[int]] = []
    i = 0
    while i < n:
        cur: list[int] = []
        cur_tokens = 0
        j = i
        while j < n:
            needed = token_lens[j]
            if cur and cur_tokens + needed > window_tokens:
                break
            cur.append(j)
            cur_tokens += needed
            j += 1
        if not cur:
            cur = [i]
            j = i + 1
        chunks.append(cur)
        step = max(1, int(len(cur) * (1.0 - overlap_ratio)))
        i += step
    return chunks


def _chunk_sentences_semantic(
    sentences: list[str],
    *,
    target_cluster_size: int = 20,
    max_tokens_per_chunk: int = 2048,
) -> list[list[int]]:
    """Semantic clustering of sentences into chunks using cosine KMeans.

    Returns chunks as lists of sentence indices. Falls back to a single
    chunk if clustering is not applicable.
    """
    if not sentences:
        return []
    if len(sentences) <= target_cluster_size:
        # Single chunk; enforce max_tokens_per_chunk via simple splitting
        token_lens = [_estimate_token_len(s) for s in sentences]
        chunks: list[list[int]] = []
        cur: list[int] = []
        cur_tokens = 0
        for idx, tl in enumerate(token_lens):
            if cur and cur_tokens + tl > max_tokens_per_chunk:
                chunks.append(cur)
                cur = []
                cur_tokens = 0
            cur.append(idx)
            cur_tokens += tl
        if cur:
            chunks.append(cur)
        return chunks

    try:
        embs = embed_segments(sentences)
        if not embs:
            raise RuntimeError("empty embeddings")
        mat = torch.stack(embs, dim=0)
    except Exception:
        # If embedding or stacking fails, fall back to sliding window
        return _chunk_sentences_sliding_window(sentences, window_tokens=max_tokens_per_chunk)

    n = mat.shape[0]
    target_cluster_size = max(1, target_cluster_size)
    k = max(1, round(n / target_cluster_size))
    if k >= n:
        k = max(1, n // 2) or 1

    # Initialize centers from random sentences
    with torch.no_grad():
        perm = torch.randperm(n)
        centers = mat[perm[:k]].clone()
        centers = torch.nn.functional.normalize(centers, p=2, dim=1)

        labels = torch.zeros(n, dtype=torch.long)
        iters = 8
        for _ in range(iters):
            sims = mat @ centers.t()
            labels = torch.argmax(sims, dim=1)
            new_centers = torch.zeros_like(centers)
            counts = torch.zeros(k, dtype=torch.long)
            for cid in range(k):
                mask = labels == cid
                if mask.any():
                    new_centers[cid] = mat[mask].mean(dim=0)
                    counts[cid] = int(mask.sum().item())
                else:
                    new_centers[cid] = centers[cid]
            centers = torch.nn.functional.normalize(new_centers, p=2, dim=1)

    # Group sentence indices by cluster
    clusters: dict[int, list[int]] = {}
    for idx, cid in enumerate(labels.tolist()):
        clusters.setdefault(int(cid), []).append(idx)

    # Within each cluster, sort by original order and enforce token limits
    token_lens = [_estimate_token_len(s) for s in sentences]
    chunks: list[list[int]] = []
    for cid in sorted(clusters.keys()):
        members = sorted(clusters[cid])
        if not members:
            continue
        cur: list[int] = []
        cur_tokens = 0
        for idx in members:
            tl = token_lens[idx]
            if cur and cur_tokens + tl > max_tokens_per_chunk:
                chunks.append(cur)
                cur = []
                cur_tokens = 0
            cur.append(idx)
            cur_tokens += tl
        if cur:
            chunks.append(cur)

    # Safety: ensure all sentences are covered
    covered = {idx for chunk in chunks for idx in chunk}
    missing = [i for i in range(len(sentences)) if i not in covered]
    for idx in missing:
        chunks.append([idx])

    return chunks


def select_top_mmr_candidates(
    query_emb: torch.Tensor,
    selected_embs: list[torch.Tensor],
    candidate_texts: list[str],
    candidate_embs: list[torch.Tensor],
    top_k: int = 3,
    mmr_lambda: float = 0.7,
    mmr_threshold: float = 0.22,
    cosine_threshold: float = 0.45,
):
    scored: list[tuple[int, float, float, float]] = []
    for i, emb in enumerate(candidate_embs):
        mmr_val = mmr_score(query_emb, emb, selected_embs, lamb=mmr_lambda)
        if mmr_val < mmr_threshold:
            continue
        cos_val = cosine_sim(query_emb, emb)
        if cos_val < cosine_threshold:
            continue
        combined = mmr_val + cos_val
        scored.append((i, combined, mmr_val, cos_val))
    scored.sort(key=lambda item: (item[1], item[3], item[2]), reverse=True)
    return scored[:top_k]


def select_top_cosine_candidates(
    query_emb: torch.Tensor,
    candidate_texts: list[str],
    candidate_embs: list[torch.Tensor],
    top_k: int = 4,
    cosine_threshold: float = 0.55,
):
    scored = []
    for i, emb in enumerate(candidate_embs):
        score = cosine_sim(query_emb, emb)
        if score >= cosine_threshold:
            scored.append((i, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def bfs_expand_graph(
    graph: nx.Graph,
    query_id: str,
    start_id: str,
    all_segments: list[str],
    all_embs: list[torch.Tensor],
    relevance_map: dict,
    *,
    max_nodes: int = 10,
    top_k_per_level: int = 3,
    mmr_lambda: float = 0.7,
    mmr_threshold: float = 0.22,
    question_cosine_threshold: float = 0.45,
):
    query_emb = graph.nodes[query_id]["embedding"]
    selected_ids = {query_id, start_id}
    selected_embs = [graph.nodes[n]["embedding"] for n in selected_ids]

    present_texts = {graph.nodes[n]["text"] for n in graph.nodes}
    pool = [(t, e) for t, e in zip(all_segments, all_embs) if t not in present_texts]
    pool_texts = [t for t, _ in pool]
    pool_embs = [e for _, e in pool]

    order_added: list[str] = []
    queue: list[str] = [start_id]
    while queue and len(graph.nodes) < max_nodes and pool_texts:
        parent = queue.pop(0)
        parent_emb = graph.nodes[parent]["embedding"]

        top = select_top_mmr_candidates(
            query_emb,
            selected_embs,
            pool_texts,
            pool_embs,
            top_k=top_k_per_level,
            mmr_lambda=mmr_lambda,
            mmr_threshold=mmr_threshold,
            cosine_threshold=question_cosine_threshold,
        )
        if not top:
            continue

        for idx, combined, mmr_val, question_cos in top:
            text = pool_texts[idx]
            emb = pool_embs[idx]

            node_id = f"node_{len(graph.nodes)}"
            graph.add_node(node_id, text=text, embedding=emb)

            graph.add_edge(
                query_id,
                node_id,
                weight=mmr_val,
                kind="query-candidate",
                combined_score=combined,
                question_cosine=question_cos,
            )
            relevance_map[(query_id, node_id)] = mmr_val

            local_cos = cosine_sim(parent_emb, emb)
            graph.add_edge(
                parent,
                node_id,
                weight=mmr_val,
                kind="parent-candidate",
                local_cosine=local_cos,
                question_cosine=question_cos,
                combined_score=combined,
            )

            order_added.append(node_id)
            queue.append(node_id)
            selected_embs.append(emb)

            if len(graph.nodes) >= max_nodes:
                break

        kept = {i for i, *_ in top}
        pool = [(t, e) for i, (t, e) in enumerate(pool) if i not in kept]
        pool_texts = [t for t, _ in pool]
        pool_embs = [e for _, e in pool]

    return order_added


def pick_start_node(
    question_text: str,
    question_emb: torch.Tensor,
    segments: list[str],
    *,
    mmr_lambda: float = 0.7,
    cosine_first_top_n: int = RERANK_TOP_N,
):
    """Deprecated: legacy start-node picker (kept for backward compatibility)."""
    # Use the newer pick_start_node_with_rerank_check, which now relies on BGE-large
    # internally. This stub simply forwards to that function without answer_text.
    seg, emb, cos_val, mmr0, _ = pick_start_node_with_rerank_check(
        question_text,
        question_emb,
        segments,
        mmr_lambda=mmr_lambda,
        cosine_first_top_n=cosine_first_top_n,
    )
    return seg, emb, cos_val, mmr0


def pick_start_node_with_rerank_check(
    question_text: str,
    question_emb: torch.Tensor,
    segments: list[str],
    *,
    answer_text: str | None = None,
    mmr_lambda: float = 0.7,
    cosine_first_top_n: int = RERANK_TOP_N,
):
    """Pick start node using BGE-large cosine/MMR prefilter and reranker hybrid scoring.

    The `question_emb` argument is ignored; this function re-embeds question & segments
    with BGE-large so that start-node detection uses the same embedding space as the graph.
    """
    # Embed with BGE-large specifically for start-node selection
    if not segments:
        return None, None, None, None, "empty"
    texts = [question_text] + segments
    embs = _encode_bge_large(texts)
    q_emb_bge = embs[0]
    seg_embs = [embs[i] for i in range(1, embs.size(0))]
    cos_scored: list[float] = [
        cosine_sim(seg_emb, q_emb_bge) for seg_emb in seg_embs
    ]

    if not seg_embs:
        return None, None, None, None, "empty"

    pool_limit = min(cosine_first_top_n, len(segments))
    cos_top_idx = sorted(range(len(segments)), key=lambda i: cos_scored[i], reverse=True)[:pool_limit]
    if not cos_top_idx:
        return None, None, None, None, "empty"

    # Fallback candidate by combined cosine + MMR
    fb_best_sum = -1e9
    fb_idx = cos_top_idx[0]
    fb_mmr0 = None
    for idx in cos_top_idx:
        emb = seg_embs[idx]
        cos_val = cos_scored[idx]
        mmr0 = mmr_score(question_emb, emb, selected_embs=None, lamb=mmr_lambda)
        total = cos_val + mmr0
        if total > fb_best_sum:
            fb_best_sum = total
            fb_idx = idx
            fb_mmr0 = mmr0

    reranker = _get_reranker()
    if reranker is None:
        seg = segments[fb_idx]
        emb = seg_embs[fb_idx]
        return seg, emb, cos_scored[fb_idx], fb_mmr0, "fallback_only"

    pool_idx = cos_top_idx
    if not pool_idx:
        pool_idx = list(range(len(segments)))

    if len(pool_idx) >= RERANK_MMR_MIN:
        initial_k = min(len(pool_idx), RERANK_MMR_MAX)
    else:
        initial_k = len(pool_idx)
    target_k = max(1, initial_k)

    final_rows = None
    while True:
        mmr_indices = _mmr_select_indices(
            question_emb,
            pool_idx,
            seg_embs,
            target_k=target_k,
            lamb=RERANK_MMR_LAMBDA,
        )
        if not mmr_indices:
            mmr_indices = [pool_idx[0]]
        try:
            pairs = [(question_text, segments[i]) for i in mmr_indices]
            logits = reranker.predict(pairs)
            candidate_rows = []
            for idx, logit in zip(mmr_indices, logits):
                logit_f = float(logit)
                cos_val = cos_scored[idx]
                mix_score = _mix_rerank_score(logit_f, cos_val)
                candidate_rows.append({
                    "idx": idx,
                    "mix": mix_score,
                    "cos": cos_val,
                    "logit": logit_f,
                })
            candidate_rows.sort(key=lambda r: (r["mix"], r["cos"]), reverse=True)
            final_rows = candidate_rows
            if (
                len(candidate_rows) > 1
                and (candidate_rows[0]["mix"] - candidate_rows[1]["mix"]) < RERANK_SCORE_TIE_DELTA
                and target_k < len(pool_idx)
            ):
                new_target = min(len(pool_idx), target_k + RERANK_EXPAND_STEP)
                if new_target == target_k:
                    break
                target_k = new_target
                continue
            break
        except Exception:
            final_rows = None
            break

    if not final_rows:
        seg = segments[fb_idx]
        emb = seg_embs[fb_idx]
        if fb_mmr0 is None:
            fb_mmr0 = mmr_score(question_emb, emb, selected_embs=None, lamb=mmr_lambda)
        return seg, emb, cos_scored[fb_idx], fb_mmr0, "fallback"

    final_idx = final_rows[0]["idx"]
    final_method = "ce"

    if answer_text is not None and final_idx != fb_idx:
        try:
            ans_emb = model.encode(str(answer_text), convert_to_tensor=True, normalize_embeddings=True)
            final_ans = cosine_sim(ans_emb, seg_embs[final_idx])
            fallback_ans = cosine_sim(ans_emb, seg_embs[fb_idx])
            if fallback_ans > final_ans:
                final_idx = fb_idx
                final_method = "fallback"
        except Exception:
            pass

    final_emb = seg_embs[final_idx]
    final_cos = cos_scored[final_idx]
    final_mmr0 = mmr_score(question_emb, final_emb, selected_embs=None, lamb=mmr_lambda)
    return segments[final_idx], final_emb, final_cos, final_mmr0, final_method

def add_path_from_start(
    graph: nx.Graph,
    query_id: str,
    start_id: str,
    all_segments: list[str],
    all_embs: list[torch.Tensor],
    relevance_map: dict,
    *,
    local_cosine_threshold: float = 0.40,
    mmr_lambda: float = 0.7,
    mmr_threshold: float = 0.22,
):
    """Iteratively add next nodes chosen by highest (MMR(query,cand) + cosine(start,cand)).
    Stops when no remaining candidate meets both thresholds (MMR and local cosine).
    Returns list of (node_id, text, local_cosine) in the order added.
    """
    query_emb = graph.nodes[query_id]["embedding"]
    parent_id = start_id
    added = []

    present_texts = {graph.nodes[n]["text"] for n in graph.nodes}
    pool = [(t, e) for t, e in zip(all_segments, all_embs) if t not in present_texts]
    if not pool:
        return added

    while pool:
        parent_emb = graph.nodes[parent_id]["embedding"]
        selected_embs = [data.get("embedding") for _, data in graph.nodes(data=True) if data.get("embedding") is not None]

        best_idx = None
        best_local = -1.0
        best_mmr = -1.0
        best_sum = -1.0
        # choose by combined score (mmr + local cosine)
        for i, (t, e) in enumerate(pool):
            local_c = cosine_sim(parent_emb, e)
            mmr = mmr_score(query_emb, e, selected_embs=selected_embs, lamb=mmr_lambda)
            combined = local_c + mmr
            if local_c >= local_cosine_threshold and mmr >= mmr_threshold and combined > best_sum:
                best_sum = combined
                best_idx = i
                best_local = local_c
                best_mmr = mmr

        if best_idx is None:
            break

        text, emb = pool.pop(best_idx)
        node_id = f"node_{len(graph.nodes)}"
        graph.add_node(node_id, text=text, embedding=emb)

        # Global edge with MMR weight
        graph.add_edge(query_id, node_id, weight=best_mmr, kind="query-candidate")
        relevance_map[(query_id, node_id)] = best_mmr
        # Local structural edge to the parent
        graph.add_edge(parent_id, node_id, weight=best_local, kind="parent-candidate")

        added.append((node_id, text, best_local))
        parent_id = node_id

        # Remove any duplicates of this text from pool
        pool = [(t, e) for (t, e) in pool if t != text]

    return added


def add_preselected_nodes(
    graph: nx.Graph,
    query_id: str,
    question_text: str,
    question_emb: torch.Tensor,
    segments: list[str],
    all_embs: list[torch.Tensor],
    *,
    top_n: int,
    use_ce: bool = True,
):
    """Attach top-N preselected candidates to the query so they appear in exports.
    Adds nodes (if missing) and query->node edges with weight from cosine or CE.
    """
    # Build pool excluding already-present texts to reduce duplication
    present_texts = {graph.nodes[n]["text"] for n in graph.nodes if "text" in graph.nodes[n]}
    pool = [(i, t, e) for i, (t, e) in enumerate(zip(segments, all_embs)) if t not in present_texts]
    if not pool:
        return []

    # Cosine scores over all segments
    cos_scored = []
    for i, t, e in pool:
        cos_scored.append((i, t, e, cosine_sim(e, question_emb)))
    cos_scored.sort(key=lambda x: x[3], reverse=True)
    cos_top = cos_scored[: min(top_n, len(cos_scored))]

    # Optional CE scores for the same set
    ce_scores = None
    method = "preselected-cos"
    if use_ce:
        reranker = _get_reranker()
        if reranker is not None and cos_top:
            pairs = [(question_text, t) for (_, t, _, _) in cos_top]
            try:
                ce_vals = reranker.predict(pairs)
                ce_scores = [float(x) for x in ce_vals]
                method = "preselected-ce"
            except Exception:
                ce_scores = None

    added_ids = []
    for pos, (i, t, e, cosw) in enumerate(cos_top):
        node_id = None
        # Try to find an existing node with same text (e.g., start node)
        for n, data in graph.nodes(data=True):
            if data.get("text") == t:
                node_id = n
                if data.get("embedding") is None:
                    graph.nodes[n]["embedding"] = e
                break
        if node_id is None:
            node_id = f"node_{len(graph.nodes)}"
            graph.add_node(node_id, text=t, embedding=e)

        w = float(ce_scores[pos]) if ce_scores is not None else float(cosw)
        graph.add_edge(query_id, node_id, weight=w, kind=method)
        added_ids.append(node_id)

    return added_ids


def _fmt_edge(u: str, v: str, data: dict) -> str:
    kind = data.get("kind", "?")
    w = data.get("weight")
    w_str = f"{w:.4f}" if isinstance(w, (int, float)) else str(w)
    return f"    - {u} -> {v} | kind={kind} | weight={w_str}"


def print_report(
    question_text: str,
    answer_text,
    start_text,
    cosine_sel: float,
    mmr_score_val: float,
    graph: nx.Graph,
    relevance_map: dict,
    added_cos_nodes,
    *,
    start_ans_cos: float | None = None,
    graph_best_ans_cos: float | None = None,
):
    sep = "=" * 60
    dash = "-" * 60
    print(sep)
    print("Question")
    print(f"  {question_text}")
    print(dash)
    print("Dataset Answer")
    if answer_text:
        print(f"  {answer_text}")
    else:
        print("  <not available in this split>")
    if start_ans_cos is not None or graph_best_ans_cos is not None:
        print(f"  Answer vs Start cosine : {start_ans_cos:.4f}" if start_ans_cos is not None else "  Answer vs Start cosine : <n/a>")
        print(f"  Answer vs Graph-best cos: {graph_best_ans_cos:.4f}" if graph_best_ans_cos is not None else "  Answer vs Graph-best cos: <n/a>")
    print(dash)
    print("Start Node (most relevant segment)")
    if start_text is not None:
        print(f"  Text       : {start_text}")
        print(f"  Cosine Sim : {cosine_sel:.4f}")
        print(f"  MMR Score  : {mmr_score_val:.4f}")
    else:
        print("  <no start node selected>")
    print(dash)
    print("Graph Status")
    print(f"  Nodes ({len(graph.nodes)}): {list(graph.nodes)}")
    print(f"  Edges ({len(graph.edges)}):")
    for u, v, data in graph.edges(data=True):
        print(_fmt_edge(u, v, data))
    print(dash)
    print("Relevance Map")
    if relevance_map:
        for (qid, sid), sc in relevance_map.items():
            print(f"  ('{qid}', '{sid}') -> {sc:.4f}")
    else:
        print("  <empty>")
    print(dash)
    print("Added Nodes Above Threshold")
    if added_cos_nodes:
        for node_id, text, score in added_cos_nodes:
            print(f"  - {node_id}: {text} (cos={score:.4f})")
    else:
        print("  <none added>")
    print(f"  Total nodes: {len(graph.nodes)} | Total edges: {len(graph.edges)}")
    print(sep)


# ---------------- Per-row pipeline + metrics ----------------
def _entailment_label(score: float | None) -> str:
    if score is None:
        return "Unknown"
    if score >= 0.65:
        return "Entailed"
    if score <= 0.35:
        return "Contradicted"
    return "Neutral"


def print_sample_summary(dataset_index: int | None, result: dict) -> None:
    sep = "=" * 100
    question = (result.get("question") or "").strip()
    answer = (result.get("final_answer") or "").strip()
    dataset_answer = (result.get("answer") or "").strip()
    entailment_score = result.get("final_answer_entailment")
    evidence = result.get("final_answer_evidence") or []
    num_nodes = result.get("num_nodes")
    num_edges = result.get("num_edges")
    cause_edges = result.get("cause_edges_count")
    time_edges = result.get("time_edges_count")
    centrality = result.get("centrality_score")

    print(sep)
    print(f"Question ID: {dataset_index if dataset_index is not None else 'N/A'}")
    print(f"Question: {question if question else 'N/A'}")
    print(f"Dataset Answer: {dataset_answer if dataset_answer else 'N/A'}")
    print()
    print(f"Predicted Answer: {answer if answer else 'N/A'}")
    print(f"Entailment Status: {_entailment_label(entailment_score)}")
    print()
    print("Top Evidence Sentences:")
    if evidence:
        for idx, sentence in enumerate(evidence, start=1):
            clean_sentence = sentence.strip() if isinstance(sentence, str) else str(sentence)
            print(f"   {idx}. {clean_sentence}")
    else:
        print("   No evidence available.")
    print()
    print("Graph Statistics:")
    print(f"   Nodes: {num_nodes if num_nodes is not None else 'N/A'}")
    print(f"   Edges: {num_edges if num_edges is not None else 'N/A'}")
    print(f"   Causal Edges: {cause_edges if cause_edges is not None else 'N/A'}")
    print(f"   Temporal Edges: {time_edges if time_edges is not None else 'N/A'}")
    if centrality is not None:
        print(f"   Centrality Score: {centrality:.2f}")
    else:
        print("   Centrality Score: N/A")
    print(sep)


def run_pipeline_for_row(
    row: dict,
    *,
    rerank_top_n: int = RERANK_TOP_N,
    expansion: str = "bfs",
    branching: int = 4,
    max_nodes: int = 20,
    local_cosine_threshold: float = 0.45,
    mmr_lambda: float = 0.7,
    mmr_threshold: float = 0.25,
    question_cosine_threshold: float = 0.45,
    include_embeddings: bool = False,
    include_preselected: bool = False,
    preselected_use_ce: bool = True,
    sentence_top_k: int | None = None,
    sentence_scope: str = "all",
    report_top_k: int | None = 3,
    verbose: bool = False,
):
    def log(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    t0 = time.time()
    question_text = row.get("question")
    segments = row.get("segments")
    if not question_text or not segments:
        print("Skipping row: missing question or segments")
        return {"ok": False}

    question_signals, question_needs_answer_hint = _extract_question_signals(question_text)

    # For display: a single joined string; for metrics: a list of candidates used individually
    answer_text = _extract_answer_text(row)
    answer_texts = _extract_answer_texts(row)

    G = nx.Graph()
    relevance_map: dict[tuple[str, str], float] = {}

    # Main question embedding for graph reasoning uses BGE-large.
    q_embs_bge = _encode_bge([question_text])
    question_emb = q_embs_bge[0]

    # Choose start node by evaluating CE rerank vs fallback in BGE-large space;
    # prefer better by answer similarity or combined score.
    best_segment, best_emb, best_cosine, best_mmr0, start_method = pick_start_node_with_rerank_check(
        question_text,
        question_emb,
        segments,
        answer_text=answer_text,
        mmr_lambda=mmr_lambda,
        cosine_first_top_n=rerank_top_n,
    )
    if not isinstance(best_segment, str) or not best_segment:
        print("Skipping row: missing start segment after rerank check")
        return {"ok": False}
    if best_emb is None:
        try:
            best_emb = _encode_bge([best_segment])[0]
        except Exception:
            print("Skipping row: failed to embed start segment")
            return {"ok": False}

    start_id = "start_node"
    query_id = "question"

    # Sentence mode: split best segment into sentences, pick top-K by cosine
    added_cos = None
    added_ids = None
    preselected_ids = []
    cause_edges: list[dict] = []
    time_edges: list[dict] = []
    coherence_result: dict | None = None
    question_top_nodes: list[tuple[str, str | None, float]] = []
    best_bge_node = None
    best_bge_related: list[dict[str, object]] = []
    evidence_texts: list[str] = []
    final_answer_info: dict | None = None
    sentence_k_used: int | None = None
    if sentence_top_k is not None and sentence_top_k > 0 and isinstance(best_segment, str):
        # Collect sentences based on scope; default is "all" to ensure up to K nodes
        sent_list, sent_sources = collect_sentences(
            segments,
            scope=sentence_scope or "all",
            best_segment=best_segment,
        )
        if not sent_list:
            sent_list = [best_segment]
            if isinstance(best_segment, str):
                try:
                    idx = segments.index(best_segment)
                except ValueError:
                    idx = 0
            else:
                idx = 0
            sent_sources = [idx]
        sent_embs = embed_segments(sent_list)
        sent_scores = [cosine_sim(e, question_emb) for e in sent_embs]
        order = sorted(range(len(sent_list)), key=lambda i: sent_scores[i], reverse=True)
        cutoff = 0.55
        pct_score = _percentile(sent_scores, 0.85)
        if pct_score is not None:
            cutoff = max(cutoff, pct_score)
        filtered_order = [idx for idx in order if sent_scores[idx] >= cutoff]
        if not filtered_order and order:
            filtered_order = order[:1]

        # Adaptive K selection based on relevance distribution and context size
        filtered_scores = [sent_scores[idx] for idx in filtered_order]
        adaptive_limit = _adaptive_sentence_k(
            sorted_scores=filtered_scores,
            total_sentences=len(sent_list),
            max_k=sentence_top_k,
            min_k=1,
        )
        if adaptive_limit <= 0:
            adaptive_limit = 1
        limit = min(adaptive_limit, len(filtered_order))
        filtered_order = filtered_order[:limit]
        sentence_k_used = limit

        mmr_lambda_sent = 0.35
        source_cap = 3
        primary_selected: list[int] = []
        selected_embs: list[torch.Tensor] = []
        source_counts: Counter[int] = Counter()
        remaining = filtered_order.copy()
        while remaining and len(primary_selected) < limit:
            best_idx = None
            best_score = None
            best_pos = None
            for pos, idx in enumerate(remaining):
                source = sent_sources[idx] if idx < len(sent_sources) else -1
                if source_counts[source] >= source_cap:
                    other_sources = {sent_sources[r] for r in remaining if sent_sources[r] != source}
                    if any(source_counts[o] == 0 for o in other_sources):
                        continue
                score = mmr_score(question_emb, sent_embs[idx], selected_embs, lamb=mmr_lambda_sent)
                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = idx
                    best_pos = pos
            if best_idx is None:
                break
            primary_selected.append(best_idx)
            selected_embs.append(sent_embs[best_idx])
            source = sent_sources[best_idx] if best_idx < len(sent_sources) else -1
            source_counts[source] += 1
            remaining.pop(best_pos)
        if not primary_selected and filtered_order:
            primary_selected = filtered_order

        evidence_indices = _select_sentence_evidence(
            primary_indices=primary_selected,
            candidate_pool=filtered_order,
            sent_list=sent_list,
            sent_embs=sent_embs,
            question_signals=question_signals,
            require_answer_hint=question_needs_answer_hint,
            limit=limit,
        )
        if evidence_indices:
            order = evidence_indices
        elif primary_selected:
            order = primary_selected
        elif filtered_order:
            order = filtered_order
        else:
            order = [0]
        evidence_texts = [sent_list[idx] for idx in order if 0 <= idx < len(sent_list)]

        # Initialize graph with question
        G.add_node(query_id, text=question_text, embedding=question_emb)

        # First sentence as start node (top overall)
        i0 = order[0]
        best_segment = sent_list[i0]
        best_emb = sent_embs[i0]
        mmr = mmr_score(question_emb, best_emb, selected_embs=None, lamb=mmr_lambda)
        G.add_node(start_id, text=best_segment, embedding=best_emb)
        G.add_edge(query_id, start_id, weight=mmr, kind="query-start")
        relevance_map[(query_id, start_id)] = mmr

        # Append others as a chain
        added_cos = []
        parent_id = start_id
        selected_embs = [best_emb]
        for pos in range(1, len(order)):
            idx = order[pos]
            text = sent_list[idx]
            emb = sent_embs[idx]
            node_id = f"node_{len(G.nodes)}"
            mmr_i = mmr_score(question_emb, emb, selected_embs=selected_embs, lamb=mmr_lambda)
            G.add_node(node_id, text=text, embedding=emb)
            G.add_edge(query_id, node_id, weight=mmr_i, kind="query-candidate")
            relevance_map[(query_id, node_id)] = mmr_i
            local_w = cosine_sim(G.nodes[parent_id]["embedding"], emb)
            G.add_edge(parent_id, node_id, weight=local_w, kind="parent-candidate")
            added_cos.append((node_id, text, local_w))
            selected_embs.append(emb)
            parent_id = node_id
    else:
        G, mmr = create_or_update_graph(
            G,
            query_id=query_id,
            query_text=question_text,
            query_emb=question_emb,
            start_id=start_id,
            start_text=best_segment,
            start_emb=best_emb,
            relevance_map=relevance_map,
            mmr_lambda=mmr_lambda,
            mmr_threshold=mmr_threshold,
            selected_embs=None,
        )
        if isinstance(best_segment, str):
            evidence_texts = [best_segment]


    if not evidence_texts:
        fallback_evidence = []
        if isinstance(best_segment, str):
            fallback_evidence.append(best_segment)
        for _, text, _ in question_top_nodes:
            if isinstance(text, str):
                fallback_evidence.append(text)
        evidence_texts = _dedupe_preserve_order(fallback_evidence)

    try:
        final_answer_info = None
        try:
            final_answer_info = _derive_final_answer(question_text, evidence_texts)
        except Exception:
            final_answer_info = None
        if final_answer_info is None:
            final_answer_info = {
                "answer": None,
                "qa_score": None,
                "answer_source": None,
                "entailment_scores": None,
                "evidence": evidence_texts,
            }
        # Rewrite/summarize final answer using FLAN-T5 if available
        rewritten, rewrite_meta = _rewrite_answer_with_flan(
            question_text,
            final_answer_info.get("evidence") or evidence_texts,
            final_answer_info.get("answer"),
        )
        if isinstance(rewritten, str) and rewritten.strip():
            final_answer_info["answer_rewritten_from"] = final_answer_info.get("answer")
            final_answer_info["answer"] = rewritten.strip()
            final_answer_info["answer_source"] = "flan_rewrite"
            if rewrite_meta:
                final_answer_info["answer_rewrite_meta"] = rewrite_meta
        if not (sentence_top_k is not None and sentence_top_k > 0):
            all_embs = embed_segments(segments)

            # Optionally add preselected (cosine top-N / CE) nodes so they appear in the graph
            preselected_ids = []
            if include_preselected:
                try:
                    preselected_ids = add_preselected_nodes(
                        G,
                        query_id=query_id,
                        question_text=question_text,
                        question_emb=question_emb,
                        segments=segments,
                        all_embs=all_embs,
                        top_n=rerank_top_n,
                        use_ce=preselected_use_ce,
                    )
                except Exception:
                    preselected_ids = []

            added_cos = None
            added_ids = None
            if expansion == "path":
                added_cos = add_path_from_start(
                    G,
                    query_id=query_id,
                    start_id=start_id,
                    all_segments=segments,
                    all_embs=all_embs,
                    relevance_map=relevance_map,
                    local_cosine_threshold=local_cosine_threshold,
                    mmr_lambda=mmr_lambda,
                    mmr_threshold=mmr_threshold,
                )
            else:  # bfs
                added_ids = bfs_expand_graph(
                    G,
                    query_id=query_id,
                    start_id=start_id,
                    all_segments=segments,
                    all_embs=all_embs,
                    relevance_map=relevance_map,
                    max_nodes=max_nodes,
                    top_k_per_level=branching,
                    mmr_lambda=mmr_lambda,
                    mmr_threshold=mmr_threshold,
                    question_cosine_threshold=question_cosine_threshold,
                )

        # Compute answer similarity metrics (max over all answer text candidates)
        start_ans_cos = None
        graph_best_ans_cos = None
        if answer_texts:
            try:
                ans_embs = _encode_bge([str(t) for t in answer_texts])
                if best_emb is not None:
                    start_ans_cos = max(cosine_sim(ae, best_emb) for ae in ans_embs)
                # check across all node embeddings in graph
                node_embs = [graph_data["embedding"] for _, graph_data in G.nodes(data=True) if "embedding" in graph_data]
                if node_embs:
                    graph_best_ans_cos = max(
                        cosine_sim(ae, e) for ae in ans_embs for e in node_embs
                    )
            except Exception:
                pass

        if verbose:
            print_report(
                question_text=question_text,
                answer_text=answer_text,
                start_text=best_segment,
                cosine_sel=best_cosine,
                mmr_score_val=mmr,
                graph=G,
                relevance_map=relevance_map,
                added_cos_nodes=added_cos,
                start_ans_cos=start_ans_cos,
                graph_best_ans_cos=graph_best_ans_cos,
            )
            if final_answer_info.get("answer"):
                ans_src = final_answer_info.get("answer_source") or "unknown"
                qa_score = final_answer_info.get("qa_score")
                details = f" (qa_score={qa_score:.3f})" if isinstance(qa_score, float) else ""
                log(f"Final Answer: {final_answer_info['answer']}{details}")
                log(f"Answer Source: {ans_src}")
            entail_scores = final_answer_info.get("entailment_scores")
            if entail_scores:
                best_ent = max(entail_scores)
                log(f"Max entailment confidence: {best_ent:.3f}")

        # Transformer-based causal and temporal inference
        cause_edges = _infer_causal_edges(G, threshold=CAUSAL_SCORE_THRESHOLD, batch_size=CAUSAL_BATCH_SIZE)
        if cause_edges:
            G.graph["cause_edges"] = cause_edges

        time_edges = _infer_temporal_edges(
            G,
            threshold=TEMPORAL_SCORE_THRESHOLD,
            batch_size=TEMPORAL_BATCH_SIZE,
            model_name=TEMPORAL_MODEL_NAME,
        )
        if time_edges:
            G.graph["time_edges"] = time_edges

        try:
            coherence_result = run_coherence_analysis(
                G,
                query_id=query_id,
                question_text=question_text,
            )
        except Exception as e:
            coherence_result = {}
            log(f"[Coherence] Analysis failed: {e}")
        else:
            if coherence_result:
                G.graph["coherence"] = coherence_result
                coref_ct = len(coherence_result.get("coref_links") or [])
                connective_ct = len(coherence_result.get("connective_links") or [])
                log(f"[Coherence] Resolved pronouns: {coref_ct} | connectives: {connective_ct}")
                summary_txt = coherence_result.get("summary_text")
                if summary_txt:
                    log(f"[Coherence] Summary: {_preview_text(summary_txt, 160)}")
            else:
                coherence_result = {}

        if report_top_k and report_top_k > 0:
            question_top_nodes = []
            try:
                scored_nodes: list[tuple[str, str | None, float]] = []
                for node_id, data in G.nodes(data=True):
                    if node_id == query_id:
                        continue
                    emb = data.get("embedding")
                    if emb is None:
                        continue
                    score = cosine_sim(question_emb, emb)
                    text = data.get("text")
                    scored_nodes.append((node_id, text, float(score)))
                scored_nodes.sort(key=lambda item: item[2], reverse=True)
                limit = min(report_top_k, len(scored_nodes))
                question_top_nodes = scored_nodes[:limit]
            except Exception:
                question_top_nodes = []
            if verbose:
                log("Top Question-Relevant Segments")
                if question_top_nodes:
                    for rank, (node_id, text, score) in enumerate(question_top_nodes, 1):
                        preview_raw = text if isinstance(text, str) else (str(text) if text is not None else "<no text>")
                        preview = preview_raw if len(preview_raw) <= 160 else preview_raw[:157] + "..."
                        preview_safe = preview.encode("utf-8", errors="replace").decode("utf-8")
                        log(f"  {rank}. {node_id} | cos={score:.4f} | {preview_safe}")
                else:
                    log("  <none available>")
        else:
            question_top_nodes = []

        best_entry = question_top_nodes[0] if question_top_nodes else None
        if best_entry is None:
            try:
                best_score_val = float("-inf")
                best_entry = None
                for node_id, data in G.nodes(data=True):
                    if node_id == query_id:
                        continue
                    emb = data.get("embedding")
                    if emb is None:
                        continue
                    score_val = cosine_sim(question_emb, emb)
                    if score_val > best_score_val:
                        best_score_val = score_val
                        best_entry = (node_id, data.get("text"), float(score_val))
            except Exception:
                best_entry = None

        if best_entry is not None:
            best_id, best_text, best_score = best_entry
            best_bge_node = {
                "node_id": best_id,
                "text": best_text,
                "question_cosine": float(best_score),
            }
            related_nodes = []
            seen_related = set()

            def _register_related(category, relation_type, direction, other_id, weight):
                if other_id is None:
                    return
                other_text = G.nodes[other_id].get("text") if G.has_node(other_id) else None
                key = (other_id, relation_type, direction, category)
                if key in seen_related:
                    return
                entry = {
                    "node_id": other_id,
                    "text": other_text,
                    "relation_type": relation_type,
                    "direction": direction,
                    "category": category,
                }
                if isinstance(weight, (int, float)):
                    entry["weight"] = float(weight)
                related_nodes.append(entry)
                seen_related.add(key)

            for edge in cause_edges or []:
                src = edge.get("source")
                tgt = edge.get("target")
                if src == best_id:
                    _register_related("causal", "CAUSES", "outgoing", tgt, edge.get("weight"))
                elif tgt == best_id:
                    _register_related("causal", "CAUSED_BY", "incoming", src, edge.get("weight"))

            for edge in time_edges or []:
                src = edge.get("source")
                tgt = edge.get("target")
                if src == best_id:
                    _register_related("temporal", "BEFORE", "outgoing", tgt, edge.get("weight"))
                elif tgt == best_id:
                    _register_related("temporal", "AFTER", "incoming", src, edge.get("weight"))

            best_bge_related = related_nodes

            def _safe_preview(value):
                if isinstance(value, str):
                    trimmed = value if len(value) <= 160 else value[:157] + "..."
                    return trimmed.encode("utf-8", errors="replace").decode("utf-8")
                if value is None:
                    return "<no text>"
                sval = str(value)
                return sval if len(sval) <= 160 else (sval[:157] + "...")

            preview_safe = _safe_preview(best_text)
            if verbose:
                log("BGE Best Node")
                log(f"  {best_id} | cos={best_score:.4f} | {preview_safe}")
                if related_nodes:
                    log("  Related (causal/temporal):")
                    for rel in related_nodes:
                        weight_val = rel.get("weight")
                        weight_str = f" | w={weight_val:.4f}" if isinstance(weight_val, (int, float)) else ""
                        label = f"{rel['category']}:{rel['relation_type']} ({rel['direction']})"
                        other_preview = _safe_preview(rel.get("text"))
                        log(f"    - {rel['node_id']} {label}{weight_str} | {other_preview}")
                else:
                    log("  Related (causal/temporal): <none>")

        t1 = time.time()
        # Prepare compact output payload for JSON outputs
        graph_nodes_texts = [data.get("text") for _, data in G.nodes(data=True) if data.get("text") is not None]
        if added_cos is not None:
            added_nodes_texts = [text for (_, text, _) in (added_cos or [])]
        elif added_ids is not None:
            added_nodes_texts = [G.nodes[nid].get("text") for nid in (added_ids or [])]
        elif preselected_ids:
            added_nodes_texts = [G.nodes[nid].get("text") for nid in preselected_ids]
        else:
            added_nodes_texts = []

        # Graph analytics: centralities and communities
        try:
            deg_c = nx.degree_centrality(G)
        except Exception:
            deg_c = {}
        try:
            btw_c = nx.betweenness_centrality(G, normalized=True)
        except Exception:
            btw_c = {}
        try:
            pr_c = nx.pagerank(G, weight="weight")
        except Exception:
            try:
                pr_c = nx.pagerank(G)
            except Exception:
                pr_c = {}
        # Communities via greedy modularity
        try:
            comms = list(nx.community.greedy_modularity_communities(G))
            node2comm = {}
            for cid, nodeset in enumerate(comms):
                for n in nodeset:
                    node2comm[n] = cid
        except Exception:
            comms = []
            node2comm = {}

        centrality_score = None
        if isinstance(start_id, str):
            if start_id in pr_c:
                centrality_score = float(pr_c[start_id])
            elif start_id in deg_c:
                centrality_score = float(deg_c[start_id])
            elif start_id in btw_c:
                centrality_score = float(btw_c[start_id])

        # Build a comprehensive, serializable graph representation
        nodes_dump = []
        for n, data in G.nodes(data=True):
            node_entry = {"id": n}
            if "text" in data:
                node_entry["text"] = data["text"]
            emb = data.get("embedding")
            if emb is not None:
                try:
                    node_entry["embedding_dim"] = int(emb.shape[0]) if hasattr(emb, "shape") else None
                    node_entry["embedding_dtype"] = str(emb.dtype) if hasattr(emb, "dtype") else None
                    # Norm summarizes vector magnitude without dumping full tensor
                    node_entry["embedding_norm"] = float(torch.norm(emb).item())
                    if include_embeddings:
                        try:
                            node_entry["embedding"] = emb.detach().cpu().tolist() if hasattr(emb, "detach") else list(emb)
                        except Exception:
                            pass
                except Exception:
                    node_entry["embedding_present"] = True
            # Add centrality/community attributes (if available)
            if n in deg_c:
                node_entry["degree_centrality"] = float(deg_c[n])
            if n in btw_c:
                node_entry["betweenness_centrality"] = float(btw_c[n])
            if n in pr_c:
                node_entry["pagerank"] = float(pr_c[n])
            if n in node2comm:
                node_entry["community"] = int(node2comm[n])
            nodes_dump.append(node_entry)

        edges_dump = []
        for u, v, data in G.edges(data=True):
            ed = {"source": u, "target": v}
            for k, val in data.items():
                if isinstance(val, (int, float)):
                    ed[k] = float(val)
                else:
                    ed[k] = val
            edges_dump.append(ed)

        rel_dump = [
            {"source": qid, "target": sid, "score": float(sc)}
            for (qid, sid), sc in relevance_map.items()
        ]

        if added_cos is not None:
            added_order_dump = [
                {"node_id": nid, "text": text, "local_cosine": float(score)}
                for (nid, text, score) in (added_cos or [])
            ]
        else:
            added_order_dump = []
            for nid in (added_ids or []):
                text = G.nodes[nid].get("text")
                local_cos = None
                mmr_w = None
                question_cos_val = None
                combined_val = None
                try:
                    for _, _, ed in G.edges(nid, data=True):
                        if ed.get("kind") == "parent-candidate":
                            lc = ed.get("local_cosine")
                            if isinstance(lc, (int, float)):
                                local_cos = float(lc)
                            w_val = ed.get("weight")
                            if isinstance(w_val, (int, float)):
                                mmr_w = float(w_val)
                            q_val = ed.get("question_cosine")
                            if isinstance(q_val, (int, float)):
                                question_cos_val = float(q_val)
                            comb = ed.get("combined_score")
                            if isinstance(comb, (int, float)):
                                combined_val = float(comb)
                            break
                except Exception:
                    pass
                entry = {"node_id": nid, "text": text}
                if local_cos is not None:
                    entry["local_cosine"] = local_cos
                if mmr_w is not None:
                    entry["mmr_weight"] = mmr_w
                if question_cos_val is not None:
                    entry["question_cosine"] = question_cos_val
                if combined_val is not None:
                    entry["combined_score"] = combined_val
                added_order_dump.append(entry)

        graph_dump = {
            "summary": {
                "num_nodes": int(len(G.nodes)),
                "num_edges": int(len(G.edges)),
            },
            "ids": {"query_id": query_id, "start_id": start_id},
            "nodes": nodes_dump,
            "edges": edges_dump,
            "cause_edges": cause_edges,
            "time_edges": time_edges,
            "coherence": coherence_result,
            "question_top_k": [
                {"node_id": nid, "text": text, "question_cosine": float(score)}
                for (nid, text, score) in question_top_nodes
            ],
            "question_best": best_bge_node,
            "question_best_related": best_bge_related,
            "relevance_map": rel_dump,
            "added_order": added_order_dump,
        }

        return {
            "ok": True,
            "best_cosine": float(best_cosine) if best_segment is not None else None,
            "final_answer": final_answer_info.get("answer"),
            "final_answer_rewritten_from": final_answer_info.get("answer_rewritten_from"),
            "final_answer_rewrite_meta": final_answer_info.get("answer_rewrite_meta"),
            "final_answer_source": final_answer_info.get("answer_source"),
            "final_answer_score": final_answer_info.get("qa_score"),
            "final_answer_entailment": (max(final_answer_info.get("entailment_scores") or []) if final_answer_info.get("entailment_scores") else None),
            "final_answer_evidence": final_answer_info.get("evidence"),
            "mmr": float(mmr) if best_segment is not None else None,
            "num_nodes": int(len(G.nodes)),
            "num_edges": int(len(G.edges)),
            "cause_edges_count": int(len(cause_edges)),
            "time_edges_count": int(len(time_edges)),
            "centrality_score": centrality_score,
            "added_cos_count": int(len(added_cos)) if added_cos is not None else (int(len(added_ids)) if added_ids is not None else int(len(preselected_ids))),
            "answer_start_cosine": start_ans_cos,
            "answer_graph_best_cosine": graph_best_ans_cos,
            "runtime_ms": (t1 - t0) * 1000.0,
            # Output-focused fields
            "sentence_k": sentence_k_used,
            "evidence_count": len(evidence_texts),
            "question": question_text,
            "answer": answer_text,
            "answer_candidates": answer_texts,
            "start_text": best_segment,
            "graph_nodes_texts": graph_nodes_texts,
            "added_nodes_texts": added_nodes_texts,
            "question_top_k": [
                {"node_id": nid, "text": text, "question_cosine": float(score)}
                for (nid, text, score) in question_top_nodes
            ],
            "best_bge_node": best_bge_node,
            "best_bge_related": best_bge_related,
            # Graph representation for separate export
            "graph": graph_dump,
            "cause_edges": cause_edges,
            "time_edges": time_edges,
            "coherence": coherence_result,
            # Non-serializable handle for downstream file exports
            "graph_nx": G,
            "graph_ids": {"query_id": query_id, "start_id": start_id},
            "relevance_map_items": list(relevance_map.items()),
            "start_select_method": (("sentence-topk-" + (sentence_scope or "all")) if (sentence_top_k is not None and sentence_top_k > 0) else start_method),
            "expansion_mode": ("sentence-chain" if (sentence_top_k is not None and sentence_top_k > 0) else expansion),
        }
    except Exception as e:
        print(f"Cosine addition/report step skipped due to error: {e}")
        t1 = time.time()
        return {"ok": False, "runtime_ms": (t1 - t0) * 1000.0}


# ---------------- Aggregation helpers ----------------
def _avg(values):
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _p50(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    vals.sort()
    mid = len(vals) // 2
    if len(vals) % 2 == 1:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def aggregate_and_print_metrics(results: list[dict]):
    total = len(results)
    ok_count = sum(1 for r in results if r.get("ok"))
    best_cosines = [r.get("best_cosine") for r in results if r.get("ok")]
    mmrs = [r.get("mmr") for r in results if r.get("ok")]
    nodes = [r.get("num_nodes") for r in results if r.get("ok")]
    edges = [r.get("num_edges") for r in results if r.get("ok")]
    added_counts = [r.get("added_cos_count") for r in results if r.get("ok")]
    start_ans = [r.get("answer_start_cosine") for r in results if r.get("ok")]
    graph_best_ans = [r.get("answer_graph_best_cosine") for r in results if r.get("ok")]
    runtimes = [r.get("runtime_ms") for r in results if r.get("runtime_ms") is not None]

    print("\n==================== Overall Metrics ====================")
    print(f"Processed: {ok_count}/{total} successful samples")
    if best_cosines:
        print(f"Start cosine       | avg={_avg(best_cosines):.4f} p50={_p50(best_cosines):.4f} min={min(best_cosines):.4f} max={max(best_cosines):.4f}")
    if mmrs:
        print(f"Start MMR          | avg={_avg(mmrs):.4f} p50={_p50(mmrs):.4f} min={min(mmrs):.4f} max={max(mmrs):.4f}")
    if start_ans and any(v is not None for v in start_ans):
        s = [v for v in start_ans if v is not None]
        print(f"Ans vs Start cos   | avg={_avg(s):.4f} p50={_p50(s):.4f} min={min(s):.4f} max={max(s):.4f} (n={len(s)})")
    if graph_best_ans and any(v is not None for v in graph_best_ans):
        g = [v for v in graph_best_ans if v is not None]
        print(f"Ans vs Graph-best  | avg={_avg(g):.4f} p50={_p50(g):.4f} min={min(g):.4f} max={max(g):.4f} (n={len(g)})")
    if nodes:
        print(f"Nodes              | avg={_avg(nodes):.2f} p50={_p50(nodes):.2f} min={min(nodes)} max={max(nodes)}")
    if edges:
        print(f"Edges              | avg={_avg(edges):.2f} p50={_p50(edges):.2f} min={min(edges)} max={max(edges)}")
    if added_counts:
        print(f"Added (cos)        | avg={_avg(added_counts):.2f} p50={_p50(added_counts):.2f} min={min(added_counts)} max={max(added_counts)}")
    if runtimes:
        print(f"Runtime (ms)       | avg={_avg(runtimes):.1f} p50={_p50(runtimes):.1f} min={min(runtimes):.1f} max={max(runtimes):.1f}")
    print("========================================================\n")

    # Return metrics payload for JSON serialization
    def _agg_stats(arr):
        if not arr:
            return None
        return {
            "avg": _avg(arr),
            "p50": _p50(arr),
            "min": min(arr),
            "max": max(arr),
        }

    metrics_payload = {
        "total": total,
        "successful": ok_count,
        "start_cosine": _agg_stats(best_cosines) if best_cosines else None,
        "start_mmr": _agg_stats(mmrs) if mmrs else None,
        "answer_vs_start_cosine": _agg_stats([v for v in start_ans if v is not None]) if any(v is not None for v in start_ans) else None,
        "answer_vs_graph_best_cosine": _agg_stats([v for v in graph_best_ans if v is not None]) if any(v is not None for v in graph_best_ans) else None,
        "nodes": _agg_stats(nodes) if nodes else None,
        "edges": _agg_stats(edges) if edges else None,
        "added_cos_count": _agg_stats(added_counts) if added_counts else None,
        "runtime_ms": _agg_stats(runtimes) if runtimes else None,
    }

    return metrics_payload


# ---------------- Long-context compression ----------------
def _normalize_scores(raw: dict[str, float]) -> dict[str, float]:
    if not raw:
        return {}
    vals = list(raw.values())
    vmin = min(vals)
    vmax = max(vals)
    if vmax <= vmin:
        return {k: 0.0 for k in raw}
    return {k: (v - vmin) / (vmax - vmin) for k, v in raw.items()}


def _compute_global_node_scores(
    G: nx.Graph,
    *,
    question_emb: torch.Tensor | None,
    cause_edges: list[dict] | None,
    alpha: float = 0.6,
    beta: float = 0.2,
    gamma: float = 0.2,
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[int, float]]:
    """Compute global node scores combining relevance, chunk importance, and causal weight."""
    nodes = [n for n in G.nodes if n != "question"]
    if not nodes:
        return {}, {}, {}

    # Global relevance: cosine to question if available, else PageRank
    relevance: dict[str, float] = {}
    if question_emb is not None:
        for n in nodes:
            emb = G.nodes[n].get("embedding")
            if emb is None:
                continue
            try:
                relevance[n] = cosine_sim(emb, question_emb)
            except Exception:
                continue
    if not relevance:
        try:
            pr = nx.pagerank(G, weight="weight")
            relevance = {n: float(pr.get(n, 0.0)) for n in nodes}
        except Exception:
            relevance = {n: 0.0 for n in nodes}
    rel_norm = _normalize_scores(relevance)

    # Chunk-level importance: average relevance within each chunk
    chunk_acc: dict[int, float] = {}
    chunk_counts: dict[int, int] = {}
    for n in nodes:
        chunks = G.nodes[n].get("chunks") or []
        if isinstance(chunks, int):
            chunks = [chunks]
        for ch in chunks:
            try:
                cid = int(ch)
            except Exception:
                continue
            chunk_acc[cid] = chunk_acc.get(cid, 0.0) + rel_norm.get(n, 0.0)
            chunk_counts[cid] = chunk_counts.get(cid, 0) + 1
    chunk_scores: dict[int, float] = {}
    for cid, total_val in chunk_acc.items():
        count = chunk_counts.get(cid) or 1
        chunk_scores[cid] = total_val / float(count)
    chunk_norm = _normalize_scores({str(k): v for k, v in chunk_scores.items()})
    chunk_norm_int: dict[int, float] = {int(k): v for k, v in chunk_norm.items()}

    chunk_importance: dict[str, float] = {}
    for n in nodes:
        chunks = G.nodes[n].get("chunks") or []
        if isinstance(chunks, int):
            chunks = [chunks]
        vals = [chunk_norm_int.get(int(ch), 0.0) for ch in chunks] if chunks else [0.0]
        chunk_importance[n] = max(vals) if vals else 0.0

    # Causal weight: sum of causal edge weights incident to node
    causal_raw: dict[str, float] = {n: 0.0 for n in nodes}
    for edge in cause_edges or []:
        src = edge.get("source")
        tgt = edge.get("target")
        w = edge.get("weight")
        if not isinstance(w, (int, float)):
            continue
        w = float(w)
        if src in causal_raw:
            causal_raw[src] += w
        if tgt in causal_raw:
            causal_raw[tgt] += w
    causal_norm = _normalize_scores(causal_raw)

    scores: dict[str, float] = {}
    components: dict[str, dict[str, float]] = {}
    for n in nodes:
        r = rel_norm.get(n, 0.0)
        c_imp = chunk_importance.get(n, 0.0)
        c_w = causal_norm.get(n, 0.0)
        s = alpha * r + beta * c_imp + gamma * c_w
        scores[n] = s
        components[n] = {
            "relevance": r,
            "chunk_importance": c_imp,
            "causal_weight": c_w,
            "score": s,
        }

    return scores, components, chunk_scores


def run_long_context_compression(
    text: str,
    *,
    question: str | None = None,
    chunk_method: str = "semantic",
    target_chunk_size: int = 20,
    max_tokens_per_chunk: int = 2048,
    window_tokens: int = 500,
    window_overlap: float = 0.3,
    sentence_top_k: int = 8,
    global_top_k: int = 16,
    global_score_threshold: float = 0.55,
    verbose: bool = False,
) -> dict:
    """End-to-end long-context compression and reasoning pipeline.

    Steps:
      1) Chunk the input text into semantic blocks.
      2) Run the existing graph pipeline independently on each chunk.
      3) Merge per-chunk graphs into a global graph, merging redundant nodes.
      4) Score nodes globally, prune, and reconstruct a compressed summary.
    """
    if not text or not isinstance(text, str):
        return {"summary": "", "global_graph": None, "chunk_summaries": [], "chunk_explanations": []}

    question_text = question or "Summarize the following document."
    sentences = split_into_sentences(text)
    if not sentences:
        return {"summary": "", "global_graph": None, "chunk_summaries": [], "chunk_explanations": []}

    if verbose:
        print(f"[LongContext] Total sentences: {len(sentences)}")

    # Step 1: chunk into semantic blocks (preferred) or sliding window (fallback)
    if chunk_method == "semantic":
        try:
            chunk_indices = _chunk_sentences_semantic(
                sentences,
                target_cluster_size=target_chunk_size,
                max_tokens_per_chunk=max_tokens_per_chunk,
            )
        except Exception:
            chunk_indices = _chunk_sentences_sliding_window(
                sentences,
                window_tokens=window_tokens,
                overlap_ratio=window_overlap,
            )
    else:
        chunk_indices = _chunk_sentences_sliding_window(
            sentences,
            window_tokens=window_tokens,
            overlap_ratio=window_overlap,
        )

    chunks = [[sentences[i] for i in idxs] for idxs in chunk_indices]
    if verbose:
        print(f"[LongContext] Created {len(chunks)} chunks")

    # Step 2: process each chunk with existing pipeline
    chunk_infos: list[dict] = []
    for cid, chunk_sents in enumerate(chunks):
        row = {
            "question": question_text,
            "segments": chunk_sents,
        }
        if verbose:
            print(f"[LongContext] Processing chunk {cid + 1}/{len(chunks)} with {len(chunk_sents)} sentences")
        res = run_pipeline_for_row(
            row,
            rerank_top_n=RERANK_TOP_N,
            expansion="bfs",
            branching=4,
            max_nodes=20,
            local_cosine_threshold=0.45,
            mmr_lambda=0.7,
            mmr_threshold=0.25,
            question_cosine_threshold=0.45,
            include_embeddings=False,
            include_preselected=False,
            preselected_use_ce=True,
            sentence_top_k=(sentence_top_k if sentence_top_k and sentence_top_k > 0 else None),
            sentence_scope="all",
            report_top_k=3,
            verbose=False,
        )
        if not isinstance(res, dict) or not res.get("ok"):
            continue
        coherence_info = res.get("coherence") or {}
        local_summary = coherence_info.get("summary_text")
        if not local_summary:
            ev = res.get("final_answer_evidence") or []
            local_summary = " ".join(str(s).strip() for s in ev if isinstance(s, str))
        chunk_infos.append(
            {
                "chunk_id": cid,
                "sentences": chunk_sents,
                "result": res,
                "local_summary": local_summary,
            }
        )

    if not chunk_infos:
        return {"summary": "", "global_graph": None, "chunk_summaries": [], "chunk_explanations": []}

    # Build combined graph with per-chunk node ids
    q_embs_bge = _encode_bge([question_text])
    question_emb = q_embs_bge[0]
    G_combined = nx.Graph()
    G_combined.add_node("question", text=question_text, embedding=question_emb, role="question")

    for info in chunk_infos:
        cid = info["chunk_id"]
        res = info["result"]
        G_chunk = res.get("graph_nx")
        graph_ids = res.get("graph_ids") or {}
        local_qid = graph_ids.get("query_id", "question")
        if not isinstance(G_chunk, nx.Graph):
            continue

        # Map nodes
        node_map: dict[str, str] = {}
        for n, data in G_chunk.nodes(data=True):
            if n == local_qid:
                node_map[n] = "question"
                continue
            global_id = f"c{cid}_{n}"
            node_map[n] = global_id
            nd = dict(data)
            nd["chunk_id"] = cid
            chunks_attr = nd.get("chunks")
            if chunks_attr is None:
                nd["chunks"] = [cid]
            elif isinstance(chunks_attr, list):
                if cid not in chunks_attr:
                    nd["chunks"] = chunks_attr + [cid]
            else:
                nd["chunks"] = [chunks_attr, cid]
            G_combined.add_node(global_id, **nd)

        # Map edges
        for u, v, ed in G_chunk.edges(data=True):
            gu = node_map.get(u)
            gv = node_map.get(v)
            if gu is None or gv is None or gu == gv:
                continue
            G_combined.add_edge(gu, gv, **ed)

    # Merge semantically redundant nodes across chunks
    # Build new merged graph to simplify edge updates
    G_merged = nx.Graph()
    if G_combined.has_node("question"):
        q_data = G_combined.nodes["question"]
        G_merged.add_node("question", **q_data)

    # Prepare records for non-question nodes
    records: list[tuple[str, dict]] = []
    for n, data in G_combined.nodes(data=True):
        if n == "question":
            continue
        emb = data.get("embedding")
        text = data.get("text")
        if emb is None or text is None:
            continue
        records.append((n, data))

    clusters: list[list[tuple[str, dict]]] = []
    cluster_sums: list[torch.Tensor] = []
    cluster_chunks: list[set[int]] = []

    for node_id, data in records:
        emb = data.get("embedding")
        if not isinstance(emb, torch.Tensor):
            continue
        chunks_attr = data.get("chunks") or []
        if isinstance(chunks_attr, int):
            chunks_set = {chunks_attr}
        else:
            try:
                chunks_set = {int(c) for c in chunks_attr}
            except Exception:
                chunks_set = set()

        if not clusters:
            clusters.append([(node_id, data)])
            cluster_sums.append(emb.detach().clone())
            cluster_chunks.append(set(chunks_set))
            continue

        best_idx = None
        best_sim = None
        for cid, s_emb in enumerate(cluster_sums):
            centroid = s_emb / max(1.0, float(len(clusters[cid])))
            try:
                sim = cosine_sim(emb, centroid)
            except Exception:
                continue
            if best_sim is None or sim > best_sim:
                best_sim = sim
                best_idx = cid

        if best_sim is not None and best_sim >= 0.88 and best_idx is not None:
            clusters[best_idx].append((node_id, data))
            cluster_sums[best_idx] = cluster_sums[best_idx] + emb.detach()
            cluster_chunks[best_idx].update(chunks_set)
        else:
            clusters.append([(node_id, data)])
            cluster_sums.append(emb.detach().clone())
            cluster_chunks.append(set(chunks_set))

    # Create merged nodes
    rep_map: dict[str, str] = {}
    for cid, members in enumerate(clusters):
        rep_id, rep_data = members[0]
        # Compute centroid embedding
        centroid = cluster_sums[cid] / max(1.0, float(len(members)))
        centroid = torch.nn.functional.normalize(centroid, p=2, dim=0)
        text = rep_data.get("text")
        for mid, mdata in members:
            rep_map[mid] = rep_id
            t = mdata.get("text")
            if isinstance(t, str) and isinstance(text, str) and len(t) > len(text):
                text = t
        node_chunks = sorted(cluster_chunks[cid])
        nd = dict(rep_data)
        nd["text"] = text
        nd["embedding"] = centroid
        nd["chunks"] = node_chunks
        G_merged.add_node(rep_id, **nd)

    # Re-map edges from combined graph into merged graph
    edge_acc: dict[tuple[str, str], dict] = {}
    for u, v, ed in G_combined.edges(data=True):
        if u == "question" or v == "question":
            gu = "question"
            gv = rep_map.get(v) if u == "question" else rep_map.get(u)
        else:
            gu = rep_map.get(u)
            gv = rep_map.get(v)
        if gu is None or gv is None or gu == gv:
            continue
        key = tuple(sorted((gu, gv)))
        w = ed.get("weight")
        w_val = float(w) if isinstance(w, (int, float)) else 0.0
        existing = edge_acc.get(key)
        if existing is None or w_val > existing.get("weight", 0.0):
            edge_acc[key] = {
                "weight": w_val,
                "kind": ed.get("kind"),
            }

    for (u, v), ed in edge_acc.items():
        G_merged.add_edge(u, v, **{k: v for k, v in ed.items() if v is not None})

    # Step 3: global causal and temporal edges on merged graph
    cause_edges = _infer_causal_edges(G_merged, threshold=CAUSAL_SCORE_THRESHOLD, batch_size=CAUSAL_BATCH_SIZE)
    time_edges = _infer_temporal_edges(
        G_merged,
        threshold=TEMPORAL_SCORE_THRESHOLD,
        batch_size=TEMPORAL_BATCH_SIZE,
        model_name=TEMPORAL_MODEL_NAME,
    )

    # Step 4: score nodes globally and prune
    scores, components, chunk_scores = _compute_global_node_scores(
        G_merged,
        question_emb=question_emb,
        cause_edges=cause_edges,
    )
    candidate_nodes = [n for n in scores.keys() if n in G_merged.nodes]
    if not candidate_nodes:
        return {"summary": "", "global_graph": None, "chunk_summaries": [], "chunk_explanations": []}

    # Threshold + top-K
    kept = [n for n in candidate_nodes if scores.get(n, 0.0) > global_score_threshold]
    if not kept:
        kept = candidate_nodes
    kept = sorted(kept, key=lambda n: scores.get(n, 0.0), reverse=True)
    if global_top_k and global_top_k > 0:
        kept = kept[: min(global_top_k, len(kept))]

    kept_set = set(kept)
    kept_set.add("question")

    G_final = G_merged.subgraph(kept_set).copy()

    # Filter causal/temporal edges to kept nodes
    cause_final = [
        e for e in (cause_edges or []) if e.get("source") in kept_set and e.get("target") in kept_set
    ]
    time_final = [
        e for e in (time_edges or []) if e.get("source") in kept_set and e.get("target") in kept_set
    ]

    # Build directed graph for topological traversal
    D = nx.DiGraph()
    for n in kept:
        D.add_node(n)
    for e in cause_final:
        src = e.get("source")
        tgt = e.get("target")
        if src in kept_set and tgt in kept_set:
            D.add_edge(src, tgt)
    for e in time_final:
        src = e.get("source")
        tgt = e.get("target")
        if src in kept_set and tgt in kept_set:
            D.add_edge(src, tgt)

    try:
        topo_order = list(nx.topological_sort(D)) if D.number_of_edges() > 0 else []
    except Exception:
        topo_order = []

    if not topo_order:
        topo_order = sorted(kept, key=lambda n: scores.get(n, 0.0), reverse=True)

    # Reconstruct compressed summary text (excluding the question node)
    sentences_out: list[str] = []
    seen_texts: set[str] = set()
    for nid in topo_order:
        if nid == "question":
            continue
        text_val = G_final.nodes[nid].get("text")
        if not isinstance(text_val, str):
            continue
        clean = " ".join(text_val.strip().split())
        if not clean or clean in seen_texts:
            continue
        seen_texts.add(clean)
        sentences_out.append(clean)
    summary_text = " ".join(sentences_out)

    # Chunk contribution stats and explanations
    chunk_contrib: dict[int, dict] = {}
    for nid in kept:
        if nid == "question":
            continue
        data = G_final.nodes[nid]
        chunks_attr = data.get("chunks") or []
        if isinstance(chunks_attr, int):
            chunks_attr = [chunks_attr]
        for ch in chunks_attr:
            try:
                cid = int(ch)
            except Exception:
                continue
            entry = chunk_contrib.setdefault(cid, {"nodes": [], "score_sum": 0.0})
            entry["nodes"].append(nid)
            entry["score_sum"] += float(scores.get(nid, 0.0))

    chunk_explanations: list[dict] = []
    for cid, info in sorted(chunk_contrib.items(), key=lambda item: item[1]["score_sum"], reverse=True):
        local_summary = ""
        for cinf in chunk_infos:
            if cinf["chunk_id"] == cid:
                local_summary = cinf.get("local_summary") or ""
                break
        reason = f"Contributed {len(info['nodes'])} high-scoring nodes (score_sum={info['score_sum']:.3f})."
        if local_summary:
            reason += " Summary focus: " + local_summary[:160]
        chunk_explanations.append(
            {
                "chunk_index": cid,
                "node_count": len(info["nodes"]),
                "score_sum": info["score_sum"],
                "reason": reason,
            }
        )

    # Build lightweight serializable global graph view
    nodes_payload = []
    for nid, data in G_final.nodes(data=True):
        node_entry = {
            "id": nid,
            "text": data.get("text"),
            "chunks": data.get("chunks"),
        }
        if nid != "question":
            node_entry["score"] = float(scores.get(nid, 0.0))
            node_entry["components"] = components.get(nid)
        nodes_payload.append(node_entry)

    edges_payload = []
    for u, v, ed in G_final.edges(data=True):
        w = ed.get("weight")
        edges_payload.append(
            {
                "source": u,
                "target": v,
                "kind": ed.get("kind"),
                "weight": float(w) if isinstance(w, (int, float)) else None,
            }
        )

    global_graph = {
        "nodes": nodes_payload,
        "edges": edges_payload,
        "cause_edges": cause_final,
        "time_edges": time_final,
    }

    chunk_summaries = [
        {
            "chunk_index": info["chunk_id"],
            "sentence_count": len(info["sentences"]),
            "summary": info.get("local_summary"),
        }
        for info in chunk_infos
    ]

    return {
        "summary": summary_text,
        "global_graph": global_graph,
        "chunk_summaries": chunk_summaries,
        "chunk_explanations": chunk_explanations,
    }


# ---------------- Batch run (10 random) ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph-based CQR pipeline runner")
    parser.add_argument("--outputs-json", dest="outputs_json", default="outputs.json", help="Path to write per-sample output JSON")
    parser.add_argument("--metrics-json", dest="metrics_json", default="metrics.json", help="Path to write aggregated metrics JSON")
    parser.add_argument("--graphs-json", dest="graphs_json", default="graphs.json", help="Path to write graph representations JSON")
    parser.add_argument("--include-embeddings", dest="include_embeddings", action="store_true", help="Include full embedding vectors in graph JSON (large)")
    parser.add_argument("--graphs-dir", dest="graphs_dir", default="graphs", help="Directory to write per-sample graph files")
    parser.add_argument("--num-samples", dest="num_samples", type=int, default=10, help="Number of random dataset rows to process")
    parser.add_argument("--report-top-k", dest="report_top_k", type=int, default=3, help="Number of question-relevant nodes to list after analysis (0 to disable)")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print detailed per-sample debug output")
    parser.add_argument("--export-gexf", dest="export_gexf", action="store_true", help="Export .gexf graphs (open in Gephi, Cytoscape)")
    parser.add_argument("--export-graphml", dest="export_graphml", action="store_true", help="Export .graphml graphs")
    parser.add_argument("--export-png", dest="export_png", action="store_true", help="Export PNG images of graphs (requires matplotlib)")
    # Sentence-mode: split best segment into sentences and use an adaptive top-K
    parser.add_argument("--sentence-top-k", dest="sentence_top_k", type=int, default=0, help="If >0, enable sentence-mode with adaptive top-K (this is the max K)")
    parser.add_argument("--sentence-scope", dest="sentence_scope", choices=["best", "all"], default="all", help="Sentence source: only best segment or all segments (default all)")
    # New knobs for complexity / expansion
    parser.add_argument("--rerank-top-n", dest="rerank_top_n", type=int, default=RERANK_TOP_N, help="Top-N from cosine pre-filter to rerank with CE")
    parser.add_argument(
        "--expansion",
        dest="expansion",
        choices=["bfs", "path"],
        default="bfs",
        help="Graph expansion strategy: bfs (branching) or path (greedy chain)",
    )
    parser.add_argument("--branching", dest="branching", type=int, default=4, help="Top-K children per level for BFS expansion")
    parser.add_argument("--max-nodes", dest="max_nodes", type=int, default=20, help="Maximum nodes allowed in the graph")
    parser.add_argument("--local-cosine-threshold", dest="local_cosine_threshold", type=float, default=0.40, help="Minimum cosine(parent,candidate) for path expansion")
    parser.add_argument("--mmr-threshold", dest="mmr_threshold", type=float, default=0.22, help="Minimum MMR(query,candidate) to add a node")
    parser.add_argument("--question-cosine-threshold", dest="question_cosine_threshold", type=float, default=0.45, help="Minimum cosine(question,candidate) required during BFS expansion")
    parser.add_argument("--mmr-lambda", dest="mmr_lambda", type=float, default=0.7, help="MMR tradeoff lambda between relevance and diversity")
    parser.add_argument("--include-preselected", dest="include_preselected", action="store_true", help="Attach top-N preselected candidates to the query so they appear in the graph")
    parser.add_argument("--preselected-use-ce", dest="preselected_use_ce", action="store_true", help="When including preselected nodes, use CrossEncoder scores for query edges (fallback to cosine)")
    # Organization of exported files
    parser.add_argument("--group-into-date-dir", dest="group_into_date_dir", action="store_true", help="Group exported files under a YYYY-MM-DD subfolder")
    parser.add_argument("--per-sample-subdir", dest="per_sample_subdir", action="store_true", help="Export each sample into its own subdirectory")
    args = parser.parse_args()

    if args.num_samples <= 0:
        parser.error("--num-samples must be a positive integer")
    if args.report_top_k is not None and args.report_top_k < 0:
        parser.error("--report-top-k must be >= 0")

    try:
        ds = dataset[split]
        total = len(ds)
        if total == 0:
            print(f"Split '{split}' is empty; nothing to process.")
            raise SystemExit(0)
        k = min(args.num_samples, total)
        indices = random.sample(range(total), k)
        results = []
        outputs_payload = []
        graphs_payload = []
        # ensure graphs dir (and optional date dir) exists if any on-disk export is enabled
        export_root = args.graphs_dir
        if args.group_into_date_dir:
            export_root = os.path.join(export_root, time.strftime("%Y-%m-%d"))
        if args.export_gexf or args.export_graphml or args.export_png:
            try:
                os.makedirs(export_root, exist_ok=True)
            except Exception as e:
                print(f"Failed to create graphs export root '{export_root}': {e}")
        for i, idx in enumerate(indices, start=1):
            if args.verbose:
                print(f"\n===== Sample {i}/{k} | dataset index {idx} =====")
            row = ds[int(idx)]
            res = run_pipeline_for_row(
                row,
                rerank_top_n=args.rerank_top_n,
                expansion=args.expansion,
                branching=args.branching,
                max_nodes=args.max_nodes,
                local_cosine_threshold=args.local_cosine_threshold,
                mmr_lambda=args.mmr_lambda,
                mmr_threshold=args.mmr_threshold,
                question_cosine_threshold=args.question_cosine_threshold,
                include_embeddings=args.include_embeddings,
                include_preselected=args.include_preselected,
                preselected_use_ce=args.preselected_use_ce,
                sentence_top_k=(args.sentence_top_k if args.sentence_top_k and args.sentence_top_k > 0 else None),
                sentence_scope=args.sentence_scope,
                report_top_k=args.report_top_k,
                verbose=args.verbose,
            )
            if isinstance(res, dict):
                results.append(res)
                if res.get("ok"):
                    print_sample_summary(int(idx), res)
                    # Derive export base path (supports optional date dir and per-sample subdir)
                    base_root = export_root
                    if args.per_sample_subdir:
                        sample_dir = os.path.join(base_root, f"graph_{int(idx)}")
                        try:
                            os.makedirs(sample_dir, exist_ok=True)
                        except Exception:
                            pass
                        base = os.path.join(sample_dir, "graph")
                    else:
                        base = os.path.join(base_root, f"graph_{int(idx)}")
                    out_entry = {
                        "dataset_index": int(idx),
                        "question": res.get("question"),
                        "answer": res.get("answer"),
                        "final_answer": res.get("final_answer"),
                        "final_answer_rewritten_from": res.get("final_answer_rewritten_from"),
                        "final_answer_rewrite_meta": res.get("final_answer_rewrite_meta"),
                        "final_answer_source": res.get("final_answer_source"),
                        "final_answer_score": res.get("final_answer_score"),
                        "final_answer_entailment": res.get("final_answer_entailment"),
                        "final_answer_evidence": res.get("final_answer_evidence"),
                        "start_text": res.get("start_text"),
                        "start_method": res.get("start_select_method"),
                        "start_cosine": res.get("best_cosine"),
                        "start_mmr": res.get("mmr"),
                        "sentence_k": res.get("sentence_k"),
                        "evidence_count": res.get("evidence_count"),
                        "graph_nodes_texts": res.get("graph_nodes_texts"),
                        "added_nodes_texts": res.get("added_nodes_texts"),
                        "num_nodes": res.get("num_nodes"),
                        "num_edges": res.get("num_edges"),
                        "cause_edges_count": res.get("cause_edges_count"),
                        "time_edges_count": res.get("time_edges_count"),
                        "centrality_score": res.get("centrality_score"),
                        "question_top_k": res.get("question_top_k"),
                        "best_bge_node": res.get("best_bge_node"),
                        "best_bge_related": res.get("best_bge_related"),
                        "coherence": res.get("coherence"),
                        "coherence_summary": (res.get("coherence") or {}).get("summary_text"),
                    }
                    if args.export_png:
                        out_entry["image_path"] = base + ".png"
                    outputs_payload.append(out_entry)

                    graph_entry = {
                        "dataset_index": int(idx),
                        "graph": res.get("graph"),
                        "coherence": res.get("coherence"),
                    }
                    if args.export_png:
                        graph_entry["image_path"] = base + ".png"
                    graphs_payload.append(graph_entry)
                    # Optional on-disk graph exports (GEXF/GraphML/PNG)
                    try:
                        if args.export_gexf or args.export_graphml or args.export_png:
                            G = res.get("graph_nx")
                            cause_edges = res.get("cause_edges") or []
                            time_edges = res.get("time_edges") or []
                            coherence_info = res.get("coherence") or {}
                            coref_edges = coherence_info.get("coref_links") or []
                            connective_edges = coherence_info.get("connective_links") or []
                            # Build serializable copy with safe attributes
                            H = nx.MultiDiGraph()
                            for n, data in G.nodes(data=True):
                                nd = {"text": data.get("text")}
                                emb = data.get("embedding")
                                if emb is not None:
                                    try:
                                        nd["embedding_dim"] = int(emb.shape[0]) if hasattr(emb, "shape") else None
                                        nd["embedding_norm"] = float(torch.norm(emb).item())
                                    except Exception:
                                        pass
                                H.add_node(n, **{k: v for k, v in nd.items() if v is not None})
                            for u, v, ed in G.edges(data=True):
                                ed_out = {}
                                for k_attr, v_attr in ed.items():
                                    if isinstance(v_attr, (int, float)):
                                        ed_out[k_attr] = float(v_attr)
                                    else:
                                        ed_out[k_attr] = str(v_attr)
                                ed_out["directed"] = False
                                ed_out["edge_type"] = ed.get("kind")
                                H.add_edge(u, v, **ed_out)
                            for edge in cause_edges:
                                src = edge.get("source")
                                tgt = edge.get("target")
                                if not src or not tgt:
                                    continue
                                ed_out = {"edge_type": "cause", "directed": True}
                                w = edge.get("weight")
                                if isinstance(w, (int, float)):
                                    ed_out["weight"] = float(w)
                                cue = edge.get("cue")
                                if cue:
                                    ed_out["cue"] = str(cue)
                                base_cos = edge.get("base_cosine")
                                if isinstance(base_cos, (int, float)):
                                    ed_out["base_cosine"] = float(base_cos)
                                rtype = edge.get("rtype")
                                if rtype:
                                    ed_out["rtype"] = str(rtype)
                                ver = edge.get("verification") or {}
                                verification_model = ver.get("model")
                                if verification_model:
                                    ed_out["verification_model"] = str(verification_model)
                                score = ver.get("score")
                                if isinstance(score, (int, float)):
                                    ed_out["verification_score"] = float(score)
                                threshold = ver.get("threshold")
                                if isinstance(threshold, (int, float)):
                                    ed_out["verification_threshold"] = float(threshold)
                                H.add_edge(src, tgt, **ed_out)
                            for edge in time_edges:
                                src = edge.get("source")
                                tgt = edge.get("target")
                                if not src or not tgt:
                                    continue
                                ed_out = {"edge_type": "time", "directed": True}
                                w = edge.get("weight")
                                if isinstance(w, (int, float)):
                                    ed_out["weight"] = float(w)
                                cue = edge.get("cue")
                                if cue:
                                    ed_out["cue"] = str(cue)
                                base_cos = edge.get("base_cosine")
                                if isinstance(base_cos, (int, float)):
                                    ed_out["base_cosine"] = float(base_cos)
                                rtype = edge.get("rtype")
                                if rtype:
                                    ed_out["rtype"] = str(rtype)
                                ver = edge.get("verification") or {}
                                verification_model = ver.get("model")
                                if verification_model:
                                    ed_out["verification_model"] = str(verification_model)
                                score = ver.get("score")
                                if isinstance(score, (int, float)):
                                    ed_out["verification_score"] = float(score)
                                threshold = ver.get("threshold")
                                if isinstance(threshold, (int, float)):
                                    ed_out["verification_threshold"] = float(threshold)
                                H.add_edge(src, tgt, **ed_out)

                            for edge in coref_edges:
                                src = edge.get("source")
                                tgt = edge.get("target")
                                if not src or not tgt:
                                    continue
                                ed_out = {"edge_type": "coherence-coref", "directed": True}
                                conf = edge.get("confidence")
                                if isinstance(conf, (int, float)):
                                    ed_out["confidence"] = float(conf)
                                pron = edge.get("pronoun")
                                if pron:
                                    ed_out["pronoun"] = str(pron)
                                antecedent = edge.get("antecedent_text")
                                if antecedent:
                                    ed_out["antecedent"] = str(antecedent)
                                dist = edge.get("distance")
                                if isinstance(dist, int):
                                    ed_out["distance"] = int(dist)
                                H.add_edge(src, tgt, **ed_out)

                            for edge in connective_edges:
                                src = edge.get("source")
                                tgt = edge.get("target")
                                if not src or not tgt:
                                    continue
                                ed_out = {"edge_type": "coherence-connective", "directed": True}
                                conf = edge.get("confidence")
                                if isinstance(conf, (int, float)):
                                    ed_out["confidence"] = float(conf)
                                phrase = edge.get("connective")
                                if phrase:
                                    ed_out["connective"] = str(phrase)
                                relation = edge.get("relation")
                                if relation:
                                    ed_out["relation"] = str(relation)
                                span = edge.get("span")
                                if span:
                                    ed_out["span"] = str(span)
                                H.add_edge(src, tgt, **ed_out)

                            # 'base' computed earlier according to user flags
                            if args.export_gexf:
                                try:
                                    nx.write_gexf(H, base + ".gexf")
                                    print(f"  - Wrote {base}.gexf")
                                except Exception as e:
                                    print(f"  - Failed writing GEXF for {idx}: {e}")
                            if args.export_graphml:
                                try:
                                    nx.write_graphml(H, base + ".graphml")
                                    print(f"  - Wrote {base}.graphml")
                                except Exception as e:
                                    print(f"  - Failed writing GraphML for {idx}: {e}")
                            if args.export_png:
                                # Stage images: start node and evidence graph
                                try:
                                    import matplotlib.pyplot as plt  # type: ignore
                                    # Ensure directories
                                    _start_dir = os.path.join(export_root, "start_node")
                                    _evidence_dir = os.path.join(export_root, "evidence_graph")
                                    try:
                                        os.makedirs(_start_dir, exist_ok=True)
                                        os.makedirs(_evidence_dir, exist_ok=True)
                                    except Exception:
                                        pass
                                    # IDs and texts
                                    _ids = res.get("graph_ids") or {}
                                    _qid = _ids.get("query_id") or "question"
                                    _sid = _ids.get("start_id") or "start"
                                    _qtext = res.get("question")
                                    _stext = res.get("start_text")
                                    # 1) Start-node image (two nodes with a single edge weighted by similarity)
                                    try:
                                        Hs = nx.MultiDiGraph()
                                        Hs.add_node(_qid, text=_qtext, role="question")
                                        Hs.add_node(_sid, text=_stext, role="start")
                                        _pos_sn = {_qid: (-0.6, 0.0), _sid: (0.6, 0.0)}
                                        _labels_sn = {n: ((Hs.nodes[n].get("text")[:60] + "...") if isinstance(Hs.nodes[n].get("text"), str) and len(Hs.nodes[n].get("text")) > 60 else (Hs.nodes[n].get("text") or n)) for n in Hs.nodes()}
                                        _colors_sn = ["#ffd166", "#ef476f"]
                                        plt.figure(figsize=(6, 4))
                                        nx.draw_networkx_nodes(Hs, _pos_sn, nodelist=[_qid, _sid], node_size=500, node_color=_colors_sn)
                                        nx.draw_networkx_labels(Hs, _pos_sn, labels=_labels_sn, font_size=8)
                                        # Add a single edge with weight label (use best cosine as weight; fallback to MMR if cosine missing)
                                        _cos, _mmr = res.get("best_cosine"), res.get("mmr")
                                        _edge_w = float(_cos) if isinstance(_cos, (int, float)) else (float(_mmr) if isinstance(_mmr, (int, float)) else None)
                                        Hs.add_edge(_qid, _sid, directed=True, weight=_edge_w)
                                        nx.draw_networkx_edges(Hs, _pos_sn, edgelist=[(_qid, _sid)], edge_color="#999999", width=1.6, alpha=0.8, arrows=True, arrowstyle="-|>", arrowsize=16)
                                        _lbl = (f"{_edge_w:.2f}" if isinstance(_edge_w, float) else None)
                                        if _lbl:
                                            nx.draw_networkx_edge_labels(Hs, _pos_sn, edge_labels={(_qid, _sid): _lbl}, font_size=7, label_pos=0.5)
                                        plt.axis("off"); plt.tight_layout()
                                        _start_png = os.path.join(_start_dir, f"start_node_{int(idx)}.png")
                                        plt.savefig(_start_png, dpi=150)
                                        plt.close()
                                        print(f"  - Wrote {_start_png}")
                                    except Exception as _e_sn:
                                        try: plt.close()
                                        except Exception: pass
                                    # 2) Evidence graph image (base + cause)
                                    try:
                                        He = nx.MultiDiGraph()
                                        for _n, _d in H.nodes(data=True):
                                            _role = "question" if _n == _qid else ("start" if _n == _sid else None)
                                            _payload = {"text": _d.get("text")}
                                            if _role: _payload["role"] = _role
                                            He.add_node(_n, **_payload)
                                        for _u, _v, _d in H.edges(data=True):
                                            if _d.get("edge_type") == "cause" or not _d.get("directed"):
                                                _ed = {k: (float(v0) if isinstance(v0, (int, float)) else v0) for k, v0 in _d.items() if v0 is not None}
                                                He.add_edge(_u, _v, **_ed)
                                        _pos_e = nx.spring_layout(He, seed=42)
                                        _node_order = list(He.nodes())
                                        _colors = [("#ffd166" if n == _qid else ("#ef476f" if n == _sid else "#cfe8ff")) for n in _node_order]
                                        _labels = {n: ((He.nodes[n].get("text")[:40] + "...") if isinstance(He.nodes[n].get("text"), str) and len(He.nodes[n].get("text")) > 40 else (He.nodes[n].get("text") or n)) for n in _node_order}
                                        plt.figure(figsize=(8, 6))
                                        nx.draw_networkx_nodes(He, _pos_e, nodelist=_node_order, node_size=500, node_color=_colors)
                                        nx.draw_networkx_labels(He, _pos_e, labels=_labels, font_size=8)
                                        _base_edges = [(_u, _v, _k, _d) for _u, _v, _k, _d in He.edges(keys=True, data=True) if not _d.get("directed")]
                                        _cause_edges = [(_u, _v, _k, _d) for _u, _v, _k, _d in He.edges(keys=True, data=True) if _d.get("edge_type") == "cause"]
                                        if _base_edges:
                                            nx.draw_networkx_edges(He, _pos_e, edgelist=[(_u, _v) for _u, _v, _, _ in _base_edges], edge_color="#999999", width=1.2, alpha=0.4, arrows=False)
                                        if _cause_edges:
                                            nx.draw_networkx_edges(He, _pos_e, edgelist=[(_u, _v) for _u, _v, _, _ in _cause_edges], edge_color="#ef476f", width=2.0, alpha=0.9, arrows=True, arrowstyle="-|>", arrowsize=18, connectionstyle="arc3,rad=0.08")
                                        _edge_labels = {}
                                        for _u, _v, _k, _d in _base_edges + _cause_edges:
                                            _lbl = f"{_d['weight']:.2f}" if isinstance(_d.get("weight"), (int, float)) else (str(_d.get("cue")) if _d.get("cue") else (str(_d.get("rtype")) if _d.get("rtype") else None))
                                            if _lbl: _edge_labels[(_u, _v, _k)] = _lbl
                                        if _edge_labels:
                                            nx.draw_networkx_edge_labels(He, _pos_e, edge_labels=_edge_labels, font_size=7, label_pos=0.5)
                                        plt.axis("off"); plt.tight_layout()
                                        _evidence_png = os.path.join(_evidence_dir, f"evidence_graph_{int(idx)}.png")
                                        plt.savefig(_evidence_png, dpi=150)
                                        plt.close()
                                        print(f"  - Wrote {_evidence_png}")
                                    except Exception as _e_ev:
                                        try: plt.close()
                                        except Exception: pass
                                except Exception as e:
                                    print(f"  - Stage image export skipped due to error: {e}")
                                try:
                                    import matplotlib.pyplot as plt  # type: ignore
                                    plt.figure(figsize=(8, 6))
                                    pos = nx.spring_layout(H, seed=42)
                                    labels = {}
                                    for n in H.nodes():
                                        t = H.nodes[n].get("text")
                                        if isinstance(t, str) and len(t) > 40:
                                            labels[n] = t[:40] + "..."
                                        else:
                                            labels[n] = t or n

                                    qid = None
                                    sid = None
                                    try:
                                        ids = res.get("graph_ids") or {}
                                        qid = ids.get("query_id")
                                        sid = ids.get("start_id")
                                    except Exception:
                                        pass
                                    base_color = "#cfe8ff"  # default
                                    q_color = "#ffd166"     # question node color (amber)
                                    s_color = "#ef476f"     # start node color (red)
                                    node_order = list(H.nodes())
                                    node_colors = [
                                        (q_color if n == qid else (s_color if n == sid else base_color))
                                        for n in node_order
                                    ]

                                    base_edges = [(u, v, key, data) for u, v, key, data in H.edges(keys=True, data=True) if not data.get("directed")]
                                    cause_edges_draw = [(u, v, key, data) for u, v, key, data in H.edges(keys=True, data=True) if data.get("edge_type") == "cause"]
                                    time_edges_draw = [(u, v, key, data) for u, v, key, data in H.edges(keys=True, data=True) if data.get("edge_type") == "time"]
                                    coherence_coref_draw = [(u, v, key, data) for u, v, key, data in H.edges(keys=True, data=True) if data.get("edge_type") == "coherence-coref"]
                                    coherence_conn_draw = [(u, v, key, data) for u, v, key, data in H.edges(keys=True, data=True) if data.get("edge_type") == "coherence-connective"]

                                    base_edgelist = [(u, v) for u, v, _, _ in base_edges]
                                    if base_edgelist:
                                        nx.draw_networkx_edges(
                                            H,
                                            pos,
                                            edgelist=base_edgelist,
                                            edge_color="#999999",
                                            width=1.2,
                                            alpha=0.4,
                                            arrows=False,
                                        )

                                    if cause_edges_draw:
                                        cause_edgelist = [(u, v) for u, v, _, _ in cause_edges_draw]
                                        nx.draw_networkx_edges(
                                            H,
                                            pos,
                                            edgelist=cause_edgelist,
                                            edge_color="#ef476f",
                                            width=2.0,
                                            alpha=0.9,
                                            arrows=True,
                                            arrowstyle="-|>",
                                            arrowsize=18,
                                            connectionstyle="arc3,rad=0.08",
                                        )

                                    if time_edges_draw:
                                        time_edgelist = [(u, v) for u, v, _, _ in time_edges_draw]
                                        nx.draw_networkx_edges(
                                            H,
                                            pos,
                                            edgelist=time_edgelist,
                                            edge_color="#118ab2",
                                            width=2.0,
                                            alpha=0.9,
                                            arrows=True,
                                            arrowstyle="-|>",
                                            arrowsize=18,
                                            connectionstyle="arc3,rad=-0.08",
                                        )

                                    if coherence_coref_draw:
                                        coref_edgelist = [(u, v) for u, v, _, _ in coherence_coref_draw]
                                        nx.draw_networkx_edges(
                                            H,
                                            pos,
                                            edgelist=coref_edgelist,
                                            edge_color="#8338ec",
                                            width=1.8,
                                            alpha=0.85,
                                            arrows=True,
                                            arrowstyle="-|>",
                                            arrowsize=16,
                                            connectionstyle="arc3,rad=0.12",
                                        )

                                    if coherence_conn_draw:
                                        conn_edgelist = [(u, v) for u, v, _, _ in coherence_conn_draw]
                                        nx.draw_networkx_edges(
                                            H,
                                            pos,
                                            edgelist=conn_edgelist,
                                            edge_color="#06d6a0",
                                            width=1.8,
                                            alpha=0.85,
                                            arrows=True,
                                            arrowstyle="-|>",
                                            arrowsize=16,
                                            connectionstyle="arc3,rad=-0.12",
                                        )

                                    nx.draw_networkx_nodes(H, pos, nodelist=node_order, node_size=500, node_color=node_colors)
                                    nx.draw_networkx_labels(H, pos, labels=labels, font_size=8)

                                    edge_labels = {}
                                    for u, v, key, data in H.edges(keys=True, data=True):
                                        label = None
                                        w_val = data.get("weight")
                                        if isinstance(w_val, (int, float)):
                                            label = f"{w_val:.2f}"
                                        elif data.get("cue"):
                                            label = str(data.get("cue"))
                                        elif data.get("rtype"):
                                            label = str(data.get("rtype"))
                                        if label:
                                            edge_labels[(u, v, key)] = label
                                    if edge_labels:
                                        nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels, font_size=7, label_pos=0.5)

                                    plt.axis("off")
                                    plt.tight_layout()
                                    # Save final graph PNG and additional stage outputs
                                    final_png_base = base + ".png"
                                    # Ensure dirs for final copies
                                    _ct_dir = os.path.join(export_root, "causal_temporal")
                                    _out_dir = os.path.join(export_root, "output")
                                    try:
                                        os.makedirs(_ct_dir, exist_ok=True)
                                        os.makedirs(_out_dir, exist_ok=True)
                                    except Exception:
                                        pass
                                    _ct_png = os.path.join(_ct_dir, f"causal_temporal_{int(idx)}.png")
                                    _out_png = os.path.join(_out_dir, f"causal_temporal_{int(idx)}.png")
                                    plt.savefig(final_png_base, dpi=150)
                                    plt.savefig(_ct_png, dpi=150)
                                    plt.savefig(_out_png, dpi=150)
                                    plt.close()
                                    print(f"  - Wrote {final_png_base}")
                                    print(f"  - Wrote {_ct_png}")
                                    print(f"  - Wrote {_out_png}")
                                except Exception as e:
                                    print(f"  - Failed writing PNG for {idx}: {e}")
                    except Exception as e:
                        print(f"  - Graph export skipped due to error: {e}")

        metrics_payload = aggregate_and_print_metrics(results)

        # Write outputs JSON
        try:
            with open(args.outputs_json, "w", encoding="utf-8") as f:
                json.dump(outputs_payload, f, ensure_ascii=False, indent=2)
            print(f"Wrote outputs to {args.outputs_json}")
        except Exception as e:
            print(f"Failed to write outputs JSON ({args.outputs_json}): {e}")

        # Write metrics JSON
        try:
            with open(args.metrics_json, "w", encoding="utf-8") as f:
                json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
            print(f"Wrote metrics to {args.metrics_json}")
        except Exception as e:
            print(f"Failed to write metrics JSON ({args.metrics_json}): {e}")
        
        # Write graphs JSON
        try:
            with open(args.graphs_json, "w", encoding="utf-8") as f:
                json.dump(graphs_payload, f, ensure_ascii=False, indent=2)
            print(f"Wrote graph representations to {args.graphs_json}")
        except Exception as e:
            print(f"Failed to write graphs JSON ({args.graphs_json}): {e}")
    except Exception as e:
        print(f"Batch processing failed: {e}")
