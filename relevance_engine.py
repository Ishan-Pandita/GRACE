"""Relevance Engine for BFS traversal and similarity filtering in mobile prompt compression."""
from __future__ import annotations

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import networkx as nx
from collections import deque
import math


class RelevanceEngine:
    """Find most relevant nodes using BFS traversal and similarity filtering."""
    
    def __init__(self, similarity_threshold: float = 0.7, max_depth: int = 5, max_nodes: int = 50):
        self.similarity_threshold = similarity_threshold
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        
    def find_relevant_nodes(self, graph: nx.DiGraph, question_node: Dict[str, Any]) -> List[Dict[str, Any]]:
        """BFS traversal from question node with explicit similarity filtering at each step."""
        if not question_node or "id" not in question_node:
            return []
        
        relevant_nodes = []
        visited = set()
        queue = deque([(question_node["id"], 0)])  # (node_id, depth)
        
        question_embedding = np.array(question_node["embedding"])
        similarity_comparisons = 0  # Track for performance
        
        while queue and len(relevant_nodes) < self.max_nodes:
            node_id, depth = queue.popleft()
            
            # Check depth limit
            if depth > self.max_depth:
                continue
            
            # Skip if already visited
            if node_id in visited:
                continue
                
            visited.add(node_id)
            
            # Get node data
            if node_id not in graph.nodes:
                continue
                
            current_node = graph.nodes[node_id]
            
            # Skip question node itself but still explore its neighbors
            if node_id == question_node["id"]:
                # Add neighbors to queue for exploration
                for neighbor in graph.neighbors(node_id):
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))
                # CRITICAL: Add question node to relevant_nodes if graph is empty
                if len(relevant_nodes) == 0:
                    relevant_nodes.append({
                        "node_id": node_id,
                        "similarity": 1.0,  # Perfect match (itself)
                        "depth": depth,
                        "node_data": current_node,
                        "relevance_score": 1.0  # Perfect relevance score
                    })
                continue
            
            # CRITICAL: Always compare each traversed node with question node
            current_embedding = np.array(current_node["embedding"])
            similarity = self._cosine_similarity(question_embedding, current_embedding)
            similarity_comparisons += 1
            
            # Only include nodes that meet similarity threshold
            if similarity >= self.similarity_threshold:
                relevant_nodes.append({
                    "node_id": node_id,
                    "similarity": float(similarity),
                    "depth": depth,
                    "node_data": current_node,
                    "relevance_score": self._calculate_relevance_score(similarity, depth),
                    "similarity_comparisons": similarity_comparisons
                })
                
                # Continue exploring neighbors of relevant nodes
                for neighbor in graph.neighbors(node_id):
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))
            else:
                # Optional: Still explore neighbors if similarity is close to threshold
                if similarity >= self.similarity_threshold * 0.8:  # 80% of threshold
                    for neighbor in graph.neighbors(node_id):
                        if neighbor not in visited:
                            queue.append((neighbor, depth + 1))
        
        # Only log if we found meaningful results
        if len(relevant_nodes) > 0 or len(graph.nodes) > 0:
            print(f"BFS: Graph has {len(graph.nodes)} nodes, found {len(relevant_nodes)} relevant nodes")
        
        # Sort by relevance score (descending)
        relevant_nodes.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Add performance metadata
        for node in relevant_nodes:
            node["total_similarity_comparisons"] = similarity_comparisons
        
        return relevant_nodes
    
    def find_relevant_nodes_with_mmr(self, graph: nx.DiGraph, question_node: Dict[str, Any], 
                                   lambda_param: float = 0.7, top_k: int = 20) -> List[Dict[str, Any]]:
        """Find relevant nodes using Maximal Marginal Relevance (MMR)."""
        # First get all relevant nodes using basic similarity
        all_relevant = self.find_relevant_nodes(graph, question_node)
        
        if not all_relevant:
            return []
        
        # Apply MMR to diversify selection
        selected_nodes = []
        remaining_nodes = all_relevant.copy()
        question_embedding = np.array(question_node["embedding"])
        
        # Select the most relevant node first
        if remaining_nodes:
            selected_nodes.append(remaining_nodes.pop(0))
        
        # Select remaining nodes using MMR
        while len(selected_nodes) < top_k and remaining_nodes:
            best_node = None
            best_mmr_score = -float('inf')
            
            for candidate in remaining_nodes:
                # Calculate similarity to query
                query_sim = candidate["similarity"]
                
                # Calculate maximum similarity to already selected nodes
                max_selected_sim = 0
                candidate_embedding = np.array(candidate["node_data"]["embedding"])
                
                for selected in selected_nodes:
                    selected_embedding = np.array(selected["node_data"]["embedding"])
                    sim = self._cosine_similarity(candidate_embedding, selected_embedding)
                    max_selected_sim = max(max_selected_sim, sim)
                
                # Calculate MMR score
                mmr_score = lambda_param * query_sim - (1 - lambda_param) * max_selected_sim
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_node = candidate
            
            if best_node:
                selected_nodes.append(best_node)
                remaining_nodes.remove(best_node)
            else:
                break
        
        return selected_nodes
    
    def find_similar_nodes_by_content(self, graph: nx.DiGraph, query_text: str, 
                                    embedding_model, top_k: int = 10) -> List[Dict[str, Any]]:
        """Find nodes similar to query text by content."""
        # Generate embedding for query text
        query_embedding = embedding_model.encode(query_text, convert_to_numpy=True)
        
        similar_nodes = []
        
        for node_id, node_data in graph.nodes(data=True):
            if "embedding" not in node_data:
                continue
                
            node_embedding = np.array(node_data["embedding"])
            similarity = self._cosine_similarity(query_embedding, node_embedding)
            
            if similarity >= self.similarity_threshold:
                similar_nodes.append({
                    "node_id": node_id,
                    "similarity": float(similarity),
                    "node_data": node_data,
                    "relevance_score": similarity  # For content-based search, similarity = relevance
                })
        
        # Sort by similarity and return top-k
        similar_nodes.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_nodes[:top_k]
    
    def find_nodes_by_temporal_relevance(self, graph: nx.DiGraph, question_node: Dict[str, Any],
                                       time_weight: float = 0.3, max_hours: int = 24) -> List[Dict[str, Any]]:
        """Find relevant nodes with temporal weighting."""
        relevant_nodes = self.find_relevant_nodes(graph, question_node)
        
        if not relevant_nodes:
            return []
        
        question_time = question_node.get("created_at", 0)
        
        # Apply temporal weighting
        for node in relevant_nodes:
            node_time = node["node_data"].get("created_at", 0)
            time_diff_hours = abs(question_time - node_time) / 3600  # Convert to hours
            
            # Calculate temporal score (more recent = higher score)
            if time_diff_hours <= max_hours:
                temporal_score = 1.0 - (time_diff_hours / max_hours)
            else:
                temporal_score = 0.1  # Minimal score for old nodes
            
            # Combine relevance and temporal scores
            node["temporal_score"] = temporal_score
            node["combined_score"] = (
                (1 - time_weight) * node["relevance_score"] + 
                time_weight * temporal_score
            )
        
        # Sort by combined score
        relevant_nodes.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return relevant_nodes
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings."""
        # Handle zero vectors
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
    
    def _calculate_relevance_score(self, similarity: float, depth: int) -> float:
        """Calculate relevance score combining similarity and depth."""
        # Depth penalty: closer nodes are more relevant
        depth_penalty = 1.0 / (1.0 + depth * 0.2)
        
        # Combined relevance score
        relevance_score = similarity * depth_penalty
        
        return relevance_score
    
    def update_similarity_threshold(self, new_threshold: float):
        """Update similarity threshold."""
        self.similarity_threshold = max(0.0, min(1.0, new_threshold))
    
    def get_relevance_statistics(self, relevant_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about relevance results."""
        if not relevant_nodes:
            return {
                "total_nodes": 0,
                "avg_similarity": 0.0,
                "avg_depth": 0.0,
                "similarity_range": (0.0, 0.0),
                "depth_range": (0, 0)
            }
        
        similarities = [node["similarity"] for node in relevant_nodes]
        depths = [node["depth"] for node in relevant_nodes]
        
        return {
            "total_nodes": len(relevant_nodes),
            "avg_similarity": float(np.mean(similarities)),
            "avg_depth": float(np.mean(depths)),
            "similarity_range": (float(min(similarities)), float(max(similarities))),
            "depth_range": (min(depths), max(depths)),
            "node_types": list(set(node["node_data"].get("type", "unknown") for node in relevant_nodes))
        }
    
    def filter_by_node_type(self, relevant_nodes: List[Dict[str, Any]], 
                          node_types: List[str]) -> List[Dict[str, Any]]:
        """Filter relevant nodes by type."""
        return [node for node in relevant_nodes 
                if node["node_data"].get("type") in node_types]
    
    def filter_by_session(self, relevant_nodes: List[Dict[str, Any]], 
                         session_id: str) -> List[Dict[str, Any]]:
        """Filter relevant nodes by session ID."""
        return [node for node in relevant_nodes 
                if node["node_data"].get("session_id") == session_id]
