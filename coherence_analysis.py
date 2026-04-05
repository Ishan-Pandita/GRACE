"""Lightweight coherence analysis utilities for the graph pipeline."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")

_PRONOUN_INFO: Dict[str, Dict[str, str]] = {
    "he": {"category": "person", "number": "singular"},
    "him": {"category": "person", "number": "singular"},
    "his": {"category": "person", "number": "singular"},
    "himself": {"category": "person", "number": "singular"},
    "she": {"category": "person", "number": "singular"},
    "her": {"category": "person", "number": "singular"},
    "hers": {"category": "person", "number": "singular"},
    "herself": {"category": "person", "number": "singular"},
    "they": {"category": "plural", "number": "plural"},
    "them": {"category": "plural", "number": "plural"},
    "their": {"category": "plural", "number": "plural"},
    "theirs": {"category": "plural", "number": "plural"},
    "themselves": {"category": "plural", "number": "plural"},
    "we": {"category": "plural", "number": "plural"},
    "us": {"category": "plural", "number": "plural"},
    "our": {"category": "plural", "number": "plural"},
    "ours": {"category": "plural", "number": "plural"},
    "ourselves": {"category": "plural", "number": "plural"},
    "it": {"category": "thing", "number": "singular"},
    "its": {"category": "thing", "number": "singular"},
    "itself": {"category": "thing", "number": "singular"},
    "this": {"category": "neutral", "number": "singular"},
    "that": {"category": "neutral", "number": "singular"},
    "these": {"category": "neutral", "number": "plural"},
    "those": {"category": "neutral", "number": "plural"},
}

_PRONOUN_REGEX = re.compile(r"\b(" + "|".join(sorted(_PRONOUN_INFO.keys(), key=len, reverse=True)) + r")\b", re.IGNORECASE)
_STOPWORDS = {
    "the", "a", "an", "and", "or", "in", "on", "at", "of", "to", "for", "with", "by",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "done",
    "this", "that", "these", "those",
    "it", "its", "their", "there", "then", "than",
    "into", "from", "about", "over", "under", "after", "before",
    "many", "much", "several", "any", "some", "all", "each", "every",
    "give", "list", "name", "tell", "describe", "state", "identify",
}

_CONNECTIVE_DEFS: Sequence[Tuple[str, str]] = [
    ("however", "contrast"),
    ("although", "contrast"),
    ("though", "contrast"),
    ("but", "contrast"),
    ("meanwhile", "temporal"),
    ("then", "temporal"),
    ("afterward", "temporal"),
    ("thereafter", "temporal"),
    ("subsequently", "temporal"),
    ("furthermore", "addition"),
    ("moreover", "addition"),
    ("in addition", "addition"),
    ("also", "addition"),
    ("because", "cause"),
    ("since", "cause"),
    ("therefore", "cause"),
    ("thus", "cause"),
    ("consequently", "cause"),
]
_CONNECTIVE_PATTERNS = [
    (re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE), phrase, relation)
    for phrase, relation in _CONNECTIVE_DEFS
]
_SUMMARY_SENTENCE_LIMIT = 6


def _tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


def _content_tokens(tokens: Sequence[str]) -> List[str]:
    return [tok for tok in tokens if tok not in _STOPWORDS and tok not in _PRONOUN_INFO]


def _prepare_node(node_id: str, text: str) -> Dict[str, Any]:
    tokens = _tokenize(text)
    content = _content_tokens(tokens)
    mentions = _extract_mentions(text, tokens, content)
    return {
        "node_id": node_id,
        "text": text,
        "tokens": tokens,
        "content_tokens": content,
        "mentions": mentions,
    }


def _extract_mentions(text: str, tokens: Sequence[str], content_tokens: Sequence[str]) -> List[Dict[str, Any]]:
    mentions: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for match in _NAME_RE.finditer(text):
        name = match.group(0).strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        mentions.append({
            "text": name,
            "category": "person",
            "tokens": _tokenize(name),
            "priority": 1.0,
        })
    group_keywords = {
        "team", "teams", "people", "members", "students", "forces", "troops", "countries",
        "leaders", "officials", "workers", "scientists", "soldiers", "companies", "children",
    }
    added_category: set[str] = set(m["category"] for m in mentions)
    for tok in content_tokens:
        if tok in seen:
            continue
        category = "thing"
        priority = 0.6
        if tok in group_keywords or tok.endswith("s"):
            category = "group"
            priority = 0.8
        if category in added_category and category != "group":
            continue
        if category == "group" and "group" in added_category:
            continue
        seen.add(tok)
        mentions.append({
            "text": tok.capitalize(),
            "category": category,
            "tokens": [tok],
            "priority": priority,
        })
        added_category.add(category)
    if not mentions and content_tokens:
        tok = content_tokens[0]
        mentions.append({
            "text": tok.capitalize(),
            "category": "unknown",
            "tokens": [tok],
            "priority": 0.5,
        })
    return mentions


def _pronoun_category(pronoun: str) -> str:
    info = _PRONOUN_INFO.get(pronoun.lower())
    if info:
        return info.get("category", "unknown")
    return "unknown"


def _select_best_mention(pronoun_category: str, candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_score = 0.0
    for mention in candidate.get("mentions", []):
        category = mention.get("category", "unknown")
        priority = float(mention.get("priority", 0.5))
        base = 0.4
        if pronoun_category == "person":
            if category == "person":
                base = 1.0
            elif category == "group":
                base = 0.7
            else:
                base = 0.5
        elif pronoun_category == "plural":
            if category in {"group", "plural"}:
                base = 0.95
            elif category == "person":
                base = 0.75
            else:
                base = 0.55
        elif pronoun_category in {"thing", "neutral"}:
            if category in {"thing", "unknown"}:
                base = 0.9
            elif category == "group":
                base = 0.65
            else:
                base = 0.6
        else:
            base = 0.6
        score = base * priority
        if score > best_score:
            best = mention
            best_score = score
    return best


def _score_coref(pronoun_category: str, mention_category: str, overlap: int, distance: int, mention_priority: float) -> float:
    base = 0.35
    if pronoun_category == "person" and mention_category == "person":
        base += 0.35
    elif pronoun_category == "person" and mention_category == "group":
        base += 0.20
    elif pronoun_category == "plural" and mention_category in {"group", "plural"}:
        base += 0.35
    elif pronoun_category in {"thing", "neutral"} and mention_category in {"thing", "unknown"}:
        base += 0.30
    else:
        base += 0.15
    base += min(0.25, 0.08 * overlap)
    base *= mention_priority
    score = base * (0.90 ** max(0, distance - 1))
    return float(max(0.0, min(score, 0.99)))


def _apply_replacements(text: str, replacements: Sequence[Tuple[int, int, str]]) -> str:
    if not replacements:
        return text
    ordered = sorted(replacements, key=lambda item: item[0])
    out_parts: List[str] = []
    last_pos = 0
    for start, end, repl in ordered:
        if start < last_pos:
            continue
        out_parts.append(text[last_pos:start])
        out_parts.append(repl)
        last_pos = end
    out_parts.append(text[last_pos:])
    return "".join(out_parts)


def _detect_connectives(text: str) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for pattern, phrase, relation in _CONNECTIVE_PATTERNS:
        for match in pattern.finditer(text):
            start = match.start()
            confidence = 0.65 if start <= 8 else 0.55
            hits.append({
                "phrase": phrase,
                "relation": relation,
                "span": match.group(0),
                "position": int(start),
                "confidence": float(confidence),
            })
    return hits


def _candidate_sources(nodes: Sequence[Dict[str, Any]], current_index: int, question_record: Optional[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], int]]:
    sources: List[Tuple[Dict[str, Any], int]] = []
    for prev_index in range(current_index - 1, -1, -1):
        distance = current_index - prev_index
        sources.append((nodes[prev_index], distance))
    if question_record is not None:
        sources.append((question_record, current_index + 1))
    return sources


def run_coherence_analysis(
    graph: Any,
    *,
    query_id: str,
    question_text: Optional[str] = None,
    max_links_per_sentence: int = 2,
) -> Dict[str, Any]:
    prepared_nodes: List[Dict[str, Any]] = []
    for node_id, data in getattr(graph, "nodes", lambda **kwargs: [])(data=True):
        if node_id == query_id:
            continue
        text = data.get("text")
        if not isinstance(text, str):
            continue
        prepared_nodes.append(_prepare_node(str(node_id), text))
    question_record = None
    if isinstance(question_text, str) and question_text.strip():
        question_record = _prepare_node(str(query_id), question_text)

    coref_links: List[Dict[str, Any]] = []
    connective_links: List[Dict[str, Any]] = []
    resolved_sentences: List[Dict[str, Any]] = []
    link_keys: set[Tuple[str, str, str]] = set()

    for idx, node in enumerate(prepared_nodes):
        text = node["text"]
        pron_matches = list(_PRONOUN_REGEX.finditer(text))
        replacements: List[Tuple[int, int, str]] = []
        pronoun_resolutions: List[Dict[str, Any]] = []
        if pron_matches:
            content_set = set(node["content_tokens"])
            candidate_nodes = _candidate_sources(prepared_nodes, idx, question_record)
            links_used = 0
            for match in pron_matches:
                pron_surface = match.group(0)
                pron_lower = pron_surface.lower()
                pron_category = _pronoun_category(pron_lower)
                best_candidate = None
                best_score = 0.0
                for candidate, distance in candidate_nodes:
                    mention = _select_best_mention(pron_category, candidate)
                    if mention is None:
                        continue
                    mention_tokens = set(mention.get("tokens", []))
                    candidate_tokens = set(candidate.get("content_tokens", []))
                    overlap = len(content_set & candidate_tokens)
                    if not overlap:
                        overlap = len(content_set & mention_tokens)
                    score = _score_coref(pron_category, mention.get("category", "unknown"), overlap, distance, float(mention.get("priority", 0.5)))
                    if score > best_score:
                        best_score = score
                        best_candidate = (candidate, mention, distance)
                if best_candidate is None or best_score < 0.35:
                    continue
                candidate_node, mention, distance = best_candidate
                key = (candidate_node["node_id"], node["node_id"], pron_lower)
                if key in link_keys:
                    continue
                replacements.append((match.start(), match.end(), mention["text"]))
                resolution_entry = {
                    "pronoun": pron_surface,
                    "antecedent_id": candidate_node["node_id"],
                    "antecedent_text": mention["text"],
                    "confidence": round(best_score, 3),
                    "distance": int(distance),
                }
                pronoun_resolutions.append(resolution_entry)
                coref_links.append({
                    "source": candidate_node["node_id"],
                    "target": node["node_id"],
                    "pronoun": pron_surface,
                    "antecedent_text": mention["text"],
                    "confidence": round(best_score, 3),
                    "distance": int(distance),
                })
                link_keys.add(key)
                links_used += 1
                if links_used >= max_links_per_sentence:
                    break
        resolved_text = _apply_replacements(text, replacements) if replacements else text
        sentence_connectives = _detect_connectives(text)
        if sentence_connectives and idx > 0:
            prev_id = prepared_nodes[idx - 1]["node_id"]
            for conn in sentence_connectives:
                connective_links.append({
                    "source": prev_id,
                    "target": node["node_id"],
                    "connective": conn["phrase"],
                    "relation": conn["relation"],
                    "confidence": round(conn["confidence"], 3),
                    "span": conn["span"],
                })
        resolved_sentences.append({
            "node_id": node["node_id"],
            "original_text": text,
            "resolved_text": resolved_text,
            "pronoun_resolutions": pronoun_resolutions,
            "connectives": sentence_connectives if idx > 0 else [],
        })

    summary_sentences: List[str] = []
    for entry in resolved_sentences:
        candidate_text = entry.get("resolved_text") or entry.get("original_text")
        if not isinstance(candidate_text, str):
            continue
        clean = " ".join(candidate_text.strip().split())
        if clean:
            summary_sentences.append(clean)
        if len(summary_sentences) >= _SUMMARY_SENTENCE_LIMIT:
            break
    summary_text = " ".join(summary_sentences) if summary_sentences else ""

    result: Dict[str, Any] = {
        "stats": {
            "num_sentences": len(prepared_nodes),
            "coref_links": len(coref_links),
            "connective_links": len(connective_links),
        },
        "coref_links": coref_links,
        "connective_links": connective_links,
        "resolved_sentences": resolved_sentences,
    }
    if summary_text:
        result["summary_text"] = summary_text
    return result
