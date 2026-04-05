"""Compression Engine for prompt summarization in mobile prompt compression."""
from __future__ import annotations

from typing import Dict, List, Any, Optional
import re
import numpy as np
from collections import Counter
from datetime import datetime, timezone

# Import existing components
from temporal_manager import TemporalManager


class CompressionEngine:
    """Compresses relevant nodes into optimized prompts for on-device LLMs."""
    
    def __init__(self, summarizer_model=None, max_tokens: int = 512, compression_ratio: float = 0.5):
        self.summarizer_model = summarizer_model
        self.max_tokens = max_tokens
        self.compression_ratio = compression_ratio
        self.temporal_manager = TemporalManager()
        
        # Compression strategies
        self.strategies = {
            "extractive": self._extractive_compression,
            "abstractive": self._abstractive_compression,
            "hybrid": self._hybrid_compression
        }
        
        # Important sentence indicators
        self.importance_indicators = [
            r'\b(?:important|key|crucial|essential|significant|critical)\b',
            r'\b(?:main|primary|principal|major|chief)\b',
            r'\b(?:therefore|thus|hence|consequently|as a result)\b',
            r'\b(?:because|since|due to|as a result of)\b',
            r'\b(?:however|but|although|despite|in contrast)\b'
        ]
    
    def compress_nodes_to_prompt(self, relevant_nodes: List[Dict[str, Any]], 
                               question_node: Dict[str, Any], 
                               strategy: str = "hybrid") -> str:
        """Create compressed prompt from relevant nodes."""
        if not relevant_nodes:
            return self._create_basic_prompt(question_node)
        
        # Order nodes by timestamp
        ordered_nodes = self.temporal_manager.order_nodes_by_timestamp(
            [node["node_data"] for node in relevant_nodes]
        )
        
        # Apply compression strategy
        if strategy in self.strategies:
            compressed_context = self.strategies[strategy](ordered_nodes, question_node)
        else:
            compressed_context = self.strategies["hybrid"](ordered_nodes, question_node)
        
        # Construct final prompt
        final_prompt = self._construct_final_prompt(compressed_context, question_node)
        
        return final_prompt
    
    def _extractive_compression(self, ordered_nodes: List[Dict[str, Any]], 
                              question_node: Dict[str, Any]) -> str:
        """Extractive compression using sentence scoring."""
        # Extract all sentences with scores
        sentences_with_scores = []
        
        for node in ordered_nodes:
            text = node.get("text", "")
            sentences = self._split_sentences(text)
            
            for sentence in sentences:
                if len(sentence.strip()) > 10:  # Filter very short sentences
                    score = self._score_sentence(sentence, question_node["text"])
                    sentences_with_scores.append({
                        "sentence": sentence.strip(),
                        "score": score,
                        "node_id": node.get("id", "unknown"),
                        "node_type": node.get("type", "unknown")
                    })
        
        # Sort by score and select top sentences
        sentences_with_scores.sort(key=lambda x: x["score"], reverse=True)
        
        # Calculate how many sentences to keep based on token limit
        target_sentences = max(3, int(len(sentences_with_scores) * self.compression_ratio))
        selected_sentences = sentences_with_scores[:target_sentences]
        
        # Reorder by original temporal order
        selected_sentences.sort(key=lambda x: ordered_nodes[0].get("timestamp", ""))
        
        # Combine sentences
        compressed_text = " ".join([s["sentence"] for s in selected_sentences])
        
        return self._ensure_token_limit(compressed_text)
    
    def _abstractive_compression(self, ordered_nodes: List[Dict[str, Any]], 
                               question_node: Dict[str, Any]) -> str:
        """Abstractive compression using summarization model."""
        if self.summarizer_model is None:
            # Fallback to extractive
            return self._extractive_compression(ordered_nodes, question_node)
        
        # Combine all node texts
        combined_text = " ".join([node.get("text", "") for node in ordered_nodes])
        
        try:
            # Generate summary
            max_length = int(self.max_tokens * 0.7)  # Leave room for question
            summary = self.summarizer_model(
                combined_text,
                max_length=max_length,
                min_length=50,
                do_sample=False
            )
            
            if isinstance(summary, dict) and "summary_text" in summary:
                return summary["summary_text"]
            elif isinstance(summary, str):
                return summary
            else:
                # Fallback to extractive
                return self._extractive_compression(ordered_nodes, question_node)
                
        except Exception:
            # Fallback to extractive on error
            return self._extractive_compression(ordered_nodes, question_node)
    
    def _hybrid_compression(self, ordered_nodes: List[Dict[str, Any]], 
                          question_node: Dict[str, Any]) -> str:
        """Hybrid compression combining extractive and abstractive methods."""
        # Start with extractive compression
        extractive_result = self._extractive_compression(ordered_nodes, question_node)
        
        # If we have a summarizer model and the text is still long, apply abstractive
        if (self.summarizer_model is not None and 
            len(extractive_result.split()) > self.max_tokens * 0.6):
            
            try:
                # Further compress using abstractive method
                max_length = int(self.max_tokens * 0.5)
                summary = self.summarizer_model(
                    extractive_result,
                    max_length=max_length,
                    min_length=30,
                    do_sample=False
                )
                
                if isinstance(summary, dict) and "summary_text" in summary:
                    return summary["summary_text"]
                elif isinstance(summary, str):
                    return summary
                    
            except Exception:
                pass  # Fall back to extractive result
        
        return extractive_result
    
    def _score_sentence(self, sentence: str, question_text: str) -> float:
        """Score sentence importance based on multiple factors."""
        score = 0.0
        
        # Length factor (prefer medium-length sentences)
        word_count = len(sentence.split())
        if 5 <= word_count <= 20:
            score += 0.2
        elif word_count > 20:
            score += 0.1
        
        # Question relevance (keyword overlap)
        question_words = set(question_text.lower().split())
        sentence_words = set(sentence.lower().split())
        overlap = len(question_words.intersection(sentence_words))
        score += min(0.3, overlap / max(len(question_words), 1))
        
        # Importance indicators
        sentence_lower = sentence.lower()
        indicator_matches = sum(1 for pattern in self.importance_indicators 
                              if re.search(pattern, sentence_lower))
        score += min(0.2, indicator_matches * 0.05)
        
        # Named entities (capitalized words)
        entities = [word for word in sentence.split() if word[0].isupper() and word.isalpha()]
        score += min(0.1, len(entities) * 0.02)
        
        # Numerical data
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', sentence)
        score += min(0.1, len(numbers) * 0.03)
        
        # Causal/temporal indicators
        causal_words = ['because', 'since', 'due to', 'therefore', 'thus', 'hence']
        temporal_words = ['before', 'after', 'when', 'while', 'then', 'next']
        
        causal_count = sum(1 for word in causal_words if word in sentence_lower)
        temporal_count = sum(1 for word in temporal_words if word in sentence_lower)
        score += min(0.1, (causal_count + temporal_count) * 0.02)
        
        return score
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _ensure_token_limit(self, text: str) -> str:
        """Ensure text is within token limit."""
        words = text.split()
        if len(words) <= self.max_tokens:
            return text
        
        # Truncate to token limit
        truncated = " ".join(words[:self.max_tokens])
        
        # Try to end at sentence boundary
        last_period = truncated.rfind('.')
        if last_period > len(truncated) * 0.8:  # If we're close to the end
            return truncated[:last_period + 1]
        
        return truncated + "..."
    
    def _construct_final_prompt(self, compressed_context: str, question_node: Dict[str, Any]) -> str:
        """Construct the final prompt for the LLM."""
        question_text = question_node.get("text", "").strip()
        
        # Avoid double 'Question:' prefix if user already provided one
        q_header = "USER PROMPT: "
        if question_text.lower().startswith("question:"):
            q_header = ""
            
        # Create structured prompt
        prompt_parts = [
            "### RELEVANT BACKGROUND CONTEXT (Compressed):",
            compressed_context if compressed_context else "(No relevant background found)",
            "",
            "### " + q_header + question_text,
            "",
            "### INSTRUCTION:",
            "Answer the user prompt above using the provided background context ONLY if it is relevant. "
            "Do NOT repeat the prompt in your response. Do NOT generate a new question. "
            "Follow all constraints (length, number of examples, etc.) specified by the user above."
        ]
        
        return "\n".join(prompt_parts)
    
    def _create_basic_prompt(self, question_node: Dict[str, Any]) -> str:
        """Create a basic prompt when no context is available."""
        question_text = question_node.get("text", "").strip()
        
        q_header = "USER PROMPT: "
        if question_text.lower().startswith("question:"):
            q_header = ""
            
        return f"""### {q_header}{question_text}

### INSTRUCTION:
Answer the user prompt above accurately and follow all specified constraints (number of examples, length, etc.). 
Do NOT repeat the question in your response."""
    
    def _extract_key_info(self, node: Dict[str, Any]) -> str:
        """Extract most relevant information from a node."""
        text = node.get("text", "")
        sentences = self._split_sentences(text)
        
        # Score and select top sentences
        scored_sentences = []
        for sentence in sentences:
            score = self._score_sentence(sentence, "")
            scored_sentences.append((sentence, score))
        
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 2-3 sentences
        top_sentences = [s[0] for s in scored_sentences[:3]]
        return ". ".join(top_sentences)
    
    def get_compression_statistics(self, original_text: str, compressed_text: str) -> Dict[str, Any]:
        """Get statistics about compression performance."""
        original_words = len(original_text.split())
        compressed_words = len(compressed_text.split())
        
        compression_ratio = compressed_words / max(original_words, 1)
        
        return {
            "original_length": original_words,
            "compressed_length": compressed_words,
            "compression_ratio": compression_ratio,
            "space_saved": 1.0 - compression_ratio,
            "tokens_saved": original_words - compressed_words
        }
    
    def adaptive_compression(self, relevant_nodes: List[Dict[str, Any]], 
                           question_node: Dict[str, Any]) -> str:
        """Adaptive compression based on content and constraints."""
        # Analyze content complexity
        total_text = " ".join([node["node_data"].get("text", "") for node in relevant_nodes])
        complexity_score = self._analyze_complexity(total_text)
        
        # Choose strategy based on complexity
        if complexity_score > 0.7:
            strategy = "abstractive"  # High complexity needs summarization
        elif complexity_score > 0.4:
            strategy = "hybrid"  # Medium complexity
        else:
            strategy = "extractive"  # Low complexity
        
        # Adjust compression ratio based on available context
        context_size = len(relevant_nodes)
        if context_size > 10:
            self.compression_ratio = 0.3  # Aggressive compression
        elif context_size > 5:
            self.compression_ratio = 0.5  # Moderate compression
        else:
            self.compression_ratio = 0.7  # Light compression
        
        return self.compress_nodes_to_prompt(relevant_nodes, question_node, strategy)
    
    def _analyze_complexity(self, text: str) -> float:
        """Analyze text complexity."""
        if not text:
            return 0.0
        
        words = text.split()
        sentences = self._split_sentences(text)
        
        # Average sentence length
        avg_sentence_length = len(words) / max(len(sentences), 1)
        
        # Vocabulary diversity
        unique_words = len(set(words))
        vocab_diversity = unique_words / max(len(words), 1)
        
        # Complex words (words with > 6 characters)
        complex_words = len([w for w in words if len(w) > 6])
        complex_word_ratio = complex_words / max(len(words), 1)
        
        # Combine factors
        complexity = (
            min(1.0, avg_sentence_length / 20) * 0.4 +
            vocab_diversity * 0.3 +
            complex_word_ratio * 0.3
        )
        
        return complexity
