"""Enhanced Causal Reasoner with multiple detection approaches."""
from __future__ import annotations

from typing import Dict, List, Any, Optional, Tuple
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from causal_detection import find_all_causal_edges, CAUSAL_MODEL_NAME

# Import custom training approach
try:
    from custom_causal_training import CausalModelTrainer
    CUSTOM_TRAINING_AVAILABLE = True
except ImportError:
    CUSTOM_TRAINING_AVAILABLE = False
    print("Custom training not available. Install required packages.")


class EnhancedCausalReasoner:
    """Enhanced causal detection with multiple approaches: ML models and custom training."""
    
    def __init__(self, causal_model: Optional[str] = None, lightweight_model: Optional[str] = None, 
                 use_full_model: bool = True, model_confidence_threshold: float = 0.6,
                 use_custom_model: bool = False,
                 custom_model_path: Optional[str] = None):
        self.causal_model_name = causal_model or CAUSAL_MODEL_NAME
        self.lightweight_model_name = lightweight_model or "distilroberta-base"
        self.use_full_model = use_full_model
        self.model_confidence_threshold = model_confidence_threshold
        self.use_custom_model = use_custom_model and CUSTOM_TRAINING_AVAILABLE
        self.custom_model_path = custom_model_path
        
        # Initialize custom model approach
        self._initialize_custom_model()
        
        self.causal_patterns = [
            r'\bbecause\b', r'\bsince\b', r'\bdue to\b', r'\bas a result\b',
            r'\btherefore\b', r'\bconsequently\b', r'\bthus\b', r'\bhence\b',
            r'\bleads to\b', r'\bcauses\b', r'\bresults in\b', r'\bbrings about\b'
        ]
        
        self.causal_cue_words = {
            'strong': ['because', 'since', 'due to', 'causes', 'leads to'],
            'medium': ['therefore', 'thus', 'hence', 'consequently'],
            'weak': ['results in', 'brings about', 'gives rise to']
        }
        
        self._lightweight_model = None
        self._lightweight_tokenizer = None
    
    def _initialize_custom_model(self):
        """Initialize custom trained model approach."""
        # Custom trained model approach
        if self.use_custom_model and self.custom_model_path:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                self.custom_model = AutoModelForSequenceClassification.from_pretrained(self.custom_model_path)
                self.custom_tokenizer = AutoTokenizer.from_pretrained(self.custom_model_path)
                print("Custom trained model initialized")
            except Exception as e:
                print(f"Custom model failed to initialize: {e}")
                self.use_custom_model = False
        
    def detect_causal_relationships(self, node1: Dict[str, Any], node2: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Multi-approach causal detection: ML models + custom training."""
        text1 = node1.get("text", "")
        text2 = node2.get("text", "")
        
        if not text1 or not text2:
            return []
        
        all_results = []
        
        # 1. Traditional ML approach (existing)
        ml_results = self._traditional_ml_detection(text1, text2)
        all_results.extend(ml_results)
        
        # 2. Custom trained model approach
        if self.use_custom_model:
            custom_results = self._custom_model_detection(text1, text2)
            all_results.extend(custom_results)
        
        # 3. Pattern-based approach (fallback)
        pattern_results = self._pattern_based_detection(text1, text2)
        all_results.extend(pattern_results)
        
        # Combine and rank results
        return self._combine_results(all_results)
    
    def _traditional_ml_detection(self, text1: str, text2: str) -> List[Dict[str, Any]]:
        """Traditional ML-based detection using existing models."""
        results = []
        
        # Only use models if explicitly enabled
        if self.use_full_model:
            try:
                # find_all_causal_edges expects List[str] and returns List[Tuple[int, int, str, float]]
                edge_tuples = find_all_causal_edges([text1, text2], threshold=self.model_confidence_threshold)
                for (i, j, label, score) in edge_tuples:
                    results.append({
                        "detection_method": "traditional_ml_full",
                        "confidence": score,
                        "has_causal_relation": True,
                        "method": "traditional_ml_full",
                        "from_index": i,
                        "to_index": j,
                    })
            except Exception as e:
                print(f"Full model failed, falling back to lightweight: {e}")
        
        # Fallback to lightweight screening only if enabled
        if not results and self.use_full_model:
            quick_result = self._lightweight_causal_check(text1, text2)
            quick_result["detection_method"] = "traditional_ml_lightweight"
            results.append(quick_result)
        
        return results
    
    def _custom_model_detection(self, text1: str, text2: str) -> List[Dict[str, Any]]:
        """Custom trained model detection."""
        if not hasattr(self, 'custom_model'):
            return []
        
        try:
            # Use custom trained model
            input_text = f"{text1} [SEP] {text2}"
            inputs = self.custom_tokenizer(
                input_text, return_tensors="pt", truncation=True, max_length=128
            )
            
            with torch.no_grad():
                outputs = self.custom_model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                confidence = probabilities[0][1].item()  # Causal class
            
            if confidence > self.model_confidence_threshold:
                return [{
                    "detection_method": "custom_trained_model",
                    "confidence": confidence,
                    "has_causal_relation": True
                }]
        except Exception as e:
            print(f"Custom model failed: {e}")
        
        return []
    
    def _pattern_based_detection(self, text1: str, text2: str) -> List[Dict[str, Any]]:
        """Pattern-based detection as fallback."""
        combined_text = f"{text1} {text2}".lower()
        
        # Count causal patterns
        pattern_matches = 0
        for pattern in self.causal_patterns:
            if re.search(pattern, combined_text):
                pattern_matches += 1
        
        # Calculate confidence based on pattern matches
        confidence = min(1.0, pattern_matches / len(self.causal_patterns) * 2)
        
        if confidence > 0.3:  # Lower threshold for pattern-based
            return [{
                "detection_method": "pattern_based",
                "confidence": confidence,
                "has_causal_relation": confidence > 0.5,
                "pattern_matches": pattern_matches
            }]
        
        return []
    
    def _combine_results(self, all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine and rank results from different approaches."""
        if not all_results:
            return []
        
        # Weight different approaches
        method_weights = {
            "traditional_ml_full": 0.4,
            "custom_trained_model": 0.4,
            "traditional_ml_lightweight": 0.15,
            "pattern_based": 0.05
        }
        
        # Calculate weighted confidence
        weighted_confidence = 0.0
        total_weight = 0.0
        method_details = []
        
        for result in all_results:
            method = result.get("detection_method", "unknown")
            weight = method_weights.get(method, 0.1)
            confidence = result.get("confidence", 0.0)
            
            weighted_confidence += confidence * weight
            total_weight += weight
            
            method_details.append({
                "method": method,
                "confidence": confidence,
                "weight": weight
            })
        
        # Normalize confidence
        final_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0
        
        return [{
            "detection_method": "multi_approach_combined",
            "confidence": final_confidence,
            "has_causal_relation": final_confidence > self.model_confidence_threshold,
            "individual_results": all_results,
            "method_details": method_details,
            "approach_count": len(all_results)
        }]
    
    def _lightweight_causal_check(self, text1: str, text2: str) -> Dict[str, Any]:
        """Fast causal detection using patterns."""
        pattern_score = self._pattern_match_score(text1, text2)
        cue_score = self._cue_word_score(text1, text2)
        structural_score = self._structural_causal_score(text1, text2)
        model_score = self._lightweight_model_predict(text1, text2)
        
        combined_score = (
            pattern_score * 0.25 + cue_score * 0.20 + 
            structural_score * 0.15 + model_score * 0.40
        )
        
        return {
            "has_causal_relation": combined_score > 0.5,
            "confidence": float(combined_score),
            "method": "lightweight",
            "pattern_score": float(pattern_score),
            "cue_score": float(cue_score),
            "structural_score": float(structural_score),
            "model_score": float(model_score)
        }
    
    def _pattern_match_score(self, text1: str, text2: str) -> float:
        """Calculate pattern matching score."""
        combined_text = f"{text1} {text2}".lower()
        pattern_matches = sum(1 for pattern in self.causal_patterns if re.search(pattern, combined_text))
        return min(1.0, pattern_matches / len(self.causal_patterns) * 3)
    
    def _cue_word_score(self, text1: str, text2: str) -> float:
        """Calculate cue word score."""
        combined_text = f"{text1} {text2}".lower()
        cue_score = 0.0
        
        for strength, words in self.causal_cue_words.items():
            word_matches = sum(1 for word in words if word in combined_text)
            if strength == 'strong':
                cue_score += word_matches * 0.3
            elif strength == 'medium':
                cue_score += word_matches * 0.2
            else:
                cue_score += word_matches * 0.1
        
        return min(1.0, cue_score)
    
    def _structural_causal_score(self, text1: str, text2: str) -> float:
        """Analyze sentence structure for causal indicators."""
        score = 0.0
        
        if self._has_cause_effect_structure(text1, text2):
            score += 0.4
        if self._has_temporal_ordering(text1, text2):
            score += 0.3
        if self._has_logical_connectors(text1, text2):
            score += 0.3
        
        return min(1.0, score)
    
    def _has_cause_effect_structure(self, text1: str, text2: str) -> bool:
        """Check for cause-effect structure."""
        combined_text = f"{text1} {text2}"
        patterns = [
            r'.+\bbecause\b.+', r'\bbecause\b.+,.+',
            r'.+\bsince\b.+', r'\bsince\b.+,.+',
            r'.+\bdue to\b.+', r'\bdue to\b.+,.+'
        ]
        return any(re.search(pattern, combined_text, re.IGNORECASE) for pattern in patterns)
    
    def _has_temporal_ordering(self, text1: str, text2: str) -> bool:
        """Check for temporal ordering."""
        temporal_words = ['before', 'after', 'when', 'while', 'then', 'next']
        combined_text = f"{text1} {text2}".lower()
        return any(word in combined_text for word in temporal_words)
    
    def _has_logical_connectors(self, text1: str, text2: str) -> bool:
        """Check for logical connectors."""
        connectors = ['if', 'then', 'so', 'thus', 'therefore', 'hence']
        combined_text = f"{text1} {text2}".lower()
        return any(word in combined_text for word in connectors)
    
    def _lightweight_model_predict(self, text1: str, text2: str) -> float:
        """Use lightweight model for prediction."""
        try:
            if self._lightweight_model is None:
                self._load_lightweight_model()
            
            if self._lightweight_model is None:
                return self._pattern_match_score(text1, text2) * 0.7
            
            input_text = f"{text1} [SEP] {text2}"
            inputs = self._lightweight_tokenizer(
                input_text, return_tensors="pt", truncation=True, max_length=128
            )
            
            with torch.no_grad():
                outputs = self._lightweight_model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                return float(probabilities[0][1].item())
                
        except Exception:
            return self._pattern_match_score(text1, text2) * 0.7
    
    def _load_lightweight_model(self):
        """Load lightweight model."""
        # Only load if causal detection is enabled
        if not self.use_full_model:
            print("Causal detection disabled - skipping lightweight model loading")
            self._lightweight_model = None
            self._lightweight_tokenizer = None
            return
            
        try:
            self._lightweight_tokenizer = AutoTokenizer.from_pretrained(self.lightweight_model_name)
            self._lightweight_model = AutoModelForSequenceClassification.from_pretrained(
                self.lightweight_model_name, num_labels=2
            )
            self._lightweight_model.eval()
        except Exception:
            self._lightweight_model = None
            self._lightweight_tokenizer = None

    def get_causal_statistics(self, graph) -> Dict[str, Any]:
        """Get statistics about causal relationships in the graph.
        
        Args:
            graph: NetworkX DiGraph to analyze.
            
        Returns:
            Dictionary with causal edge counts, average confidence, and coverage.
        """
        import networkx as nx
        if not isinstance(graph, nx.DiGraph) or len(graph.edges) == 0:
            return {
                "total_causal_edges": 0,
                "total_edges": len(graph.edges) if hasattr(graph, 'edges') else 0,
                "causal_coverage": 0.0,
                "avg_causal_confidence": 0.0,
                "methods_used": [],
            }

        causal_edges = []
        methods_used = set()
        for u, v, data in graph.edges(data=True):
            edge_type = data.get("type", "")
            if edge_type == "causal" or data.get("has_causal_relation", False):
                causal_edges.append(data)
                method = data.get("detection_method", data.get("method", "unknown"))
                methods_used.add(method)

        total_edges = len(graph.edges)
        total_causal = len(causal_edges)
        avg_confidence = (
            sum(e.get("confidence", 0.0) for e in causal_edges) / total_causal
            if total_causal > 0 else 0.0
        )

        return {
            "total_causal_edges": total_causal,
            "total_edges": total_edges,
            "causal_coverage": total_causal / total_edges if total_edges > 0 else 0.0,
            "avg_causal_confidence": round(avg_confidence, 4),
            "methods_used": sorted(methods_used),
        }

