"""Prompt Processor for semantic unit extraction in mobile prompt compression."""
from __future__ import annotations

from typing import Dict, List, Any, Optional
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import numpy as np
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer

# Download required NLTK data (only once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')


class PromptProcessor:
    """Process user input to extract semantic units and create question nodes."""
    
    def __init__(self, sentence_model: SentenceTransformer):
        self.sentence_model = sentence_model
        self.stop_words = set(stopwords.words('english'))
        
        # Common phrase patterns
        self.phrase_patterns = [
            r'\b(?:the|a|an)\s+[\w\s]+\b',  # Articles + nouns
            r'\b(?:because|since|due to|as a result|therefore|thus|hence)\s+[\w\s]+\b',  # Causal phrases
            r'\b(?:in order to|so that|in order that)\s+[\w\s]+\b',  # Purpose phrases
            r'\b(?:according to|based on|according as)\s+[\w\s]+\b',  # Reference phrases
        ]
        
        # Important word categories
        self.important_pos_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
        
    def extract_semantic_units(self, text: str) -> Dict[str, List[str]]:
        """Extract sentences, key phrases, and important words from text."""
        # Clean text
        cleaned_text = self._clean_text(text)
        
        # Extract sentences
        sentences = self._extract_sentences(cleaned_text)
        
        # Extract key phrases
        phrases = self._extract_key_phrases(cleaned_text)
        
        # Extract important words
        words = self._extract_important_words(cleaned_text)
        
        return {
            "sentences": sentences,
            "phrases": phrases,
            "words": words,
            "cleaned_text": cleaned_text
        }
    
    def create_question_node(self, user_input: str, session_id: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Process user input into question node."""
        # Extract semantic units
        semantic_units = self.extract_semantic_units(user_input)
        
        # Generate embedding
        embedding = self.sentence_model.encode(user_input, convert_to_numpy=True)
        
        # Create node data
        node_data = {
            "text": user_input,
            "semantic_units": semantic_units,
            "embedding": embedding.tolist(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "created_at": datetime.now(timezone.utc).timestamp(),
            "session_id": session_id,
            "metadata": metadata or {},
            "processing_stats": {
                "sentence_count": len(semantic_units["sentences"]),
                "phrase_count": len(semantic_units["phrases"]),
                "word_count": len(semantic_units["words"]),
                "char_count": len(user_input)
            }
        }
        
        return node_data
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\-\(\)\[\]\"\'\/\\]', '', text)
        
        # Normalize quotes
        text = re.sub(r'[""''`]', '"', text)
        text = re.sub(r'[''`]', "'", text)
        
        return text.strip()
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        try:
            sentences = sent_tokenize(text)
            # Filter out very short sentences
            sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
            return sentences
        except Exception:
            # Fallback to simple splitting
            return [s.strip() for s in text.split('.') if len(s.strip()) > 3]
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases using patterns and POS tagging."""
        phrases = []
        
        # Pattern-based extraction
        for pattern in self.phrase_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            phrases.extend(matches)
        
        # POS-based noun phrase extraction
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            # Extract noun phrases
            current_phrase = []
            for word, pos in pos_tags:
                if pos.startswith('NN') or pos.startswith('JJ'):
                    current_phrase.append(word)
                elif current_phrase:
                    # End of phrase
                    if len(current_phrase) > 1:
                        phrases.append(' '.join(current_phrase))
                    current_phrase = []
            
            # Add last phrase if exists
            if len(current_phrase) > 1:
                phrases.append(' '.join(current_phrase))
                
        except Exception:
            pass  # Fallback to pattern-based only
        
        # Named entity extraction
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            tree = ne_chunk(pos_tags)
            
            for subtree in tree:
                if hasattr(subtree, 'label'):
                    entity = ' '.join([token for token, pos in subtree.leaves()])
                    phrases.append(entity)
        except Exception:
            pass  # Named entity extraction is optional
        
        # Clean and deduplicate phrases
        phrases = list(set([p.strip() for p in phrases if len(p.strip()) > 2]))
        return phrases
    
    def _extract_important_words(self, text: str) -> List[str]:
        """Extract important words based on POS tags and frequency."""
        important_words = []
        
        try:
            tokens = word_tokenize(text.lower())
            pos_tags = pos_tag(tokens)
            
            # Filter by POS tags and stop words
            for word, pos in pos_tags:
                if (pos in self.important_pos_tags and 
                    word not in self.stop_words and 
                    len(word) > 2 and 
                    word.isalpha()):
                    important_words.append(word)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_words = []
            for word in important_words:
                if word not in seen:
                    seen.add(word)
                    unique_words.append(word)
            
            return unique_words
            
        except Exception:
            # Fallback to simple word extraction
            words = text.lower().split()
            return [w.strip() for w in words if w.isalpha() and len(w) > 2 and w not in self.stop_words]
    
    def calculate_text_complexity(self, text: str) -> Dict[str, float]:
        """Calculate text complexity metrics."""
        semantic_units = self.extract_semantic_units(text)
        
        # Basic metrics
        word_count = len(semantic_units["words"])
        sentence_count = len(semantic_units["sentences"])
        phrase_count = len(semantic_units["phrases"])
        
        # Complexity scores
        avg_sentence_length = word_count / max(sentence_count, 1)
        phrase_density = phrase_count / max(sentence_count, 1)
        lexical_diversity = len(set(semantic_units["words"])) / max(word_count, 1)
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "phrase_count": phrase_count,
            "avg_sentence_length": avg_sentence_length,
            "phrase_density": phrase_density,
            "lexical_diversity": lexical_diversity,
            "complexity_score": (avg_sentence_length * 0.3 + phrase_density * 0.4 + (1 - lexical_diversity) * 0.3)
        }
    
    def extract_query_intent(self, text: str) -> Dict[str, Any]:
        """Extract query intent and type."""
        text_lower = text.lower().strip()
        
        # Question patterns
        question_patterns = {
            "what": r'\bwhat\b',
            "how": r'\bhow\b',
            "why": r'\bwhy\b',
            "when": r'\bwhen\b',
            "where": r'\bwhere\b',
            "who": r'\bwho\b',
            "which": r'\bwhich\b'
        }
        
        # Command patterns
        command_patterns = {
            "explain": r'\bexplain\b',
            "describe": r'\bdescribe\b',
            "define": r'\bdefine\b',
            "compare": r'\bcompare\b',
            "analyze": r'\banalyze\b',
            "summarize": r'\bsummarize\b'
        }
        
        detected_intent = "unknown"
        intent_confidence = 0.0
        
        # Check question patterns
        for intent, pattern in question_patterns.items():
            if re.search(pattern, text_lower):
                detected_intent = f"question_{intent}"
                intent_confidence = 0.8
                break
        
        # Check command patterns
        if detected_intent == "unknown":
            for intent, pattern in command_patterns.items():
                if re.search(pattern, text_lower):
                    detected_intent = f"command_{intent}"
                    intent_confidence = 0.7
                    break
        
        # Default to statement if no patterns match
        if detected_intent == "unknown":
            detected_intent = "statement"
            intent_confidence = 0.5
        
        return {
            "intent": detected_intent,
            "confidence": intent_confidence,
            "is_question": text_lower.endswith('?') or detected_intent.startswith("question_"),
            "is_command": detected_intent.startswith("command_")
        }
