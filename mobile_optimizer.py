"""Mobile Optimizer for resource management and performance tuning."""
from __future__ import annotations

from typing import Dict, List, Any, Optional, Tuple
import psutil
import gc
import numpy as np
import networkx as nx
from datetime import datetime, timezone
import threading
import time


class MobileOptimizer:
    """Optimizes performance and resource usage for edge devices."""
    
    def __init__(self, memory_limit_mb: int = 512, max_graph_nodes: int = 1000):
        self.memory_limit = memory_limit_mb * 1024 * 1024  # Convert to bytes
        self.max_graph_nodes = max_graph_nodes
        
        # Performance tracking
        self.performance_history = []
        self.optimization_stats = {
            "graph_prunes": 0,
            "memory_cleanups": 0,
            "batch_adjustments": 0,
            "threshold_adjustments": 0
        }
        
        # Adaptive parameters
        self.current_batch_size = 16
        self.current_similarity_threshold = 0.7
        self.current_max_depth = 5
        
        # Resource monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
    def optimize_graph_size(self, graph: nx.DiGraph) -> Tuple[nx.DiGraph, int]:
        """Prune graph to fit memory and node constraints."""
        removed_count = 0
        
        # Check node count limit
        if len(graph.nodes) > self.max_graph_nodes:
            removed_count += self._prune_by_node_count(graph)
        
        # Check memory usage
        current_memory = self._get_memory_usage()
        if current_memory > self.memory_limit:
            removed_count += self._prune_by_memory(graph)
        
        # Update statistics
        if removed_count > 0:
            self.optimization_stats["graph_prunes"] += 1
        
        return graph, removed_count
    
    def _prune_by_node_count(self, graph: nx.DiGraph) -> int:
        """Prune nodes based on relevance and age."""
        nodes_to_remove = []
        
        # Calculate node scores
        node_scores = []
        for node_id, node_data in graph.nodes(data=True):
            score = self._calculate_node_importance(node_id, node_data, graph)
            node_scores.append((node_id, score))
        
        # Sort by score (ascending - remove least important first)
        node_scores.sort(key=lambda x: x[1])
        
        # Remove nodes until under limit
        excess_nodes = len(graph.nodes) - self.max_graph_nodes
        for i in range(min(excess_nodes, len(node_scores))):
            nodes_to_remove.append(node_scores[i][0])
        
        # Remove nodes
        graph.remove_nodes_from(nodes_to_remove)
        
        return len(nodes_to_remove)
    
    def _prune_by_memory(self, graph: nx.DiGraph) -> int:
        """Prune nodes to reduce memory usage."""
        # Estimate memory usage per node
        estimated_node_memory = 2048  # Realistic estimate in bytes
        
        # Don't prune ifgraph is small (memory is likely dominated by models)
        if len(graph.nodes) <= 10:
            return 0
            
        target_reduction = (self._get_memory_usage() - self.memory_limit) // estimated_node_memory
        
        # Cap target_reduction so we don't wipe out the graph completely
        # Always keep at least half the nodes
        max_removal = len(graph.nodes) // 2
        target_reduction = min(target_reduction, max_removal)
        
        if target_reduction <= 0:
            return 0
        
        # Remove least important nodes
        node_scores = []
        for node_id, node_data in graph.nodes(data=True):
            score = self._calculate_node_importance(node_id, node_data, graph)
            node_scores.append((node_id, score))
        
        node_scores.sort(key=lambda x: x[1])
        
        nodes_to_remove = [node_id for node_id, _ in node_scores[:target_reduction]]
        graph.remove_nodes_from(nodes_to_remove)
        
        return len(nodes_to_remove)
    
    def _calculate_node_importance(self, node_id: str, node_data: Dict[str, Any], graph: nx.DiGraph) -> float:
        """Calculate importance score for a node."""
        score = 0.0
        
        # Node type importance
        node_type = node_data.get("type", "unknown")
        if node_type == "question":
            score += 0.3
        elif node_type == "response":
            score += 0.2
        
        # Recency (newer nodes are more important)
        try:
            created_at = node_data.get("created_at", 0)
            age_hours = (time.time() - created_at) / 3600
            recency_score = max(0, 1.0 - age_hours / 168)  # Decay over 1 week
            score += recency_score * 0.2
        except Exception:
            pass
        
        # Connectivity (more connected nodes are important)
        degree = graph.degree(node_id)
        connectivity_score = min(1.0, degree / 10)
        score += connectivity_score * 0.2
        
        # Edge weights
        edge_weight_sum = 0
        for _, _, edge_data in graph.edges(node_id, data=True):
            confidence = edge_data.get("confidence", 0.5)
            edge_weight_sum += confidence
        
        edge_score = min(1.0, edge_weight_sum / 5)
        score += edge_score * 0.2
        
        # Session activity (nodes from active sessions)
        session_id = node_data.get("session_id", "")
        if session_id and self._is_session_recent(session_id):
            score += 0.1
        
        return score
    
    def _is_session_recent(self, session_id: str) -> bool:
        """Check if a session is recent."""
        # This would need session tracking - simplified for now
        return True
    
    def optimize_model_inference(self, model, input_text: str) -> Any:
        """Optimize model inference for mobile."""
        # Batch processing if multiple inputs
        if isinstance(input_text, list):
            return self._batch_inference(model, input_text)
        
        # Single input optimization
        return self._optimized_single_inference(model, input_text)
    
    def _batch_inference(self, model, inputs: List[str]) -> List[Any]:
        """Process multiple inputs in batches."""
        results = []
        
        for i in range(0, len(inputs), self.current_batch_size):
            batch = inputs[i:i + self.current_batch_size]
            try:
                batch_results = model(batch)
                results.extend(batch_results if isinstance(batch_results, list) else [batch_results])
            except Exception:
                # Fallback to individual processing
                for input_text in batch:
                    result = self._optimized_single_inference(model, input_text)
                    results.append(result)
        
        return results
    
    def _optimized_single_inference(self, model, input_text: str) -> Any:
        """Optimized single inference."""
        try:
            # Clear cache before inference
            gc.collect()
            
            # Run inference
            result = model(input_text)
            
            return result
            
        except Exception as e:
            print(f"Inference error: {e}")
            return None
    
    def monitor_resources(self) -> Dict[str, Any]:
        """Monitor system resources."""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.used
            memory_percent = memory.percent
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Process-specific memory
            process = psutil.Process()
            process_memory = process.memory_info().rss
            
            return {
                "memory_usage": memory_usage,
                "memory_percent": memory_percent,
                "process_memory": process_memory,
                "cpu_percent": cpu_percent,
                "within_limits": memory_usage <= self.memory_limit,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}
    
    def adaptive_optimization(self, current_metrics: Dict[str, Any]) -> Optional[str]:
        """Continuously optimize based on performance metrics."""
        optimization_applied = None
        
        # Check memory pressure
        memory_percent = current_metrics.get("memory_percent", 0)
        if memory_percent > 85:
            self._emergency_memory_cleanup()
            optimization_applied = "emergency_memory_cleanup"
        
        elif memory_percent > 75:
            self._reduce_batch_size()
            optimization_applied = "reduce_batch_size"
        
        # Check CPU usage
        cpu_percent = current_metrics.get("cpu_percent", 0)
        if cpu_percent > 80:
            self._increase_similarity_threshold()
            optimization_applied = "increase_similarity_threshold"
        
        # Update performance history
        self.performance_history.append({
            "metrics": current_metrics,
            "optimization": optimization_applied,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Keep history manageable
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]
        
        return optimization_applied
    
    def _emergency_memory_cleanup(self):
        """Emergency memory cleanup."""
        gc.collect()
        self.optimization_stats["memory_cleanups"] += 1
        
        # Aggressively reduce batch size
        self.current_batch_size = max(4, self.current_batch_size // 2)
        self.optimization_stats["batch_adjustments"] += 1
    
    def _reduce_batch_size(self):
        """Reduce batch size to lower memory usage."""
        self.current_batch_size = max(8, self.current_batch_size - 4)
        self.optimization_stats["batch_adjustments"] += 1
    
    def _increase_similarity_threshold(self):
        """Increase similarity threshold to reduce processing."""
        self.current_similarity_threshold = min(0.9, self.current_similarity_threshold + 0.05)
        self.optimization_stats["threshold_adjustments"] += 1
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes for the current process."""
        try:
            return psutil.Process().memory_info().rss
        except Exception:
            return 0
    
    def start_monitoring(self, interval_seconds: int = 30):
        """Start background resource monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop background resource monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
    
    def _monitoring_loop(self, interval_seconds: int):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self.monitor_resources()
                self.adaptive_optimization(metrics)
                time.sleep(interval_seconds)
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(interval_seconds)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        return {
            "current_parameters": {
                "batch_size": self.current_batch_size,
                "similarity_threshold": self.current_similarity_threshold,
                "max_depth": self.current_max_depth,
                "memory_limit_mb": self.memory_limit // (1024 * 1024),
                "max_graph_nodes": self.max_graph_nodes
            },
            "optimization_statistics": self.optimization_stats,
            "performance_history_count": len(self.performance_history),
            "monitoring_active": self.monitoring_active,
            "current_resources": self.monitor_resources()
        }
    
    def reset_optimization_stats(self):
        """Reset optimization statistics."""
        self.optimization_stats = {
            "graph_prunes": 0,
            "memory_cleanups": 0,
            "batch_adjustments": 0,
            "threshold_adjustments": 0
        }
        self.performance_history = []
    
    def optimize_for_mobile(self) -> Dict[str, Any]:
        """Apply mobile-specific optimizations."""
        optimizations = []
        
        # Reduce batch size for mobile
        if self.current_batch_size > 8:
            self.current_batch_size = 8
            optimizations.append("batch_size_reduced")
        
        # Increase similarity threshold for faster processing
        if self.current_similarity_threshold < 0.8:
            self.current_similarity_threshold = 0.8
            optimizations.append("similarity_threshold_increased")
        
        # Reduce max depth for BFS
        if self.current_max_depth > 3:
            self.current_max_depth = 3
            optimizations.append("max_depth_reduced")
        
        # Lower memory limit for mobile
        mobile_memory_limit = 256 * 1024 * 1024  # 256MB
        if self.memory_limit > mobile_memory_limit:
            self.memory_limit = mobile_memory_limit
            optimizations.append("memory_limit_reduced")
        
        # Reduce max graph nodes
        mobile_max_nodes = 500
        if self.max_graph_nodes > mobile_max_nodes:
            self.max_graph_nodes = mobile_max_nodes
            optimizations.append("max_graph_nodes_reduced")
        
        return {
            "applied_optimizations": optimizations,
            "new_parameters": {
                "batch_size": self.current_batch_size,
                "similarity_threshold": self.current_similarity_threshold,
                "max_depth": self.current_max_depth,
                "memory_limit_mb": self.memory_limit // (1024 * 1024),
                "max_graph_nodes": self.max_graph_nodes
            }
        }
