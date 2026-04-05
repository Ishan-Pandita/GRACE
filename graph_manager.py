"""Graph Manager for dynamic knowledge graphs in mobile prompt compression."""
from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple
import networkx as nx
import numpy as np
from datetime import datetime, timezone
import uuid
import json

from sentence_transformers import SentenceTransformer


class GraphManager:
    """Manages dynamic knowledge graphs for prompt compression."""
    
    def __init__(self, embedding_model: SentenceTransformer, similarity_threshold: float = 0.7):
        self.graph = nx.DiGraph()
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.node_counter = 0
        
    def add_question_node(self, text: str, session_id: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Add user question as special node."""
        node_id = f"question_{self.node_counter}"
        self.node_counter += 1
        
        # Generate embedding
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        
        # Create node data
        node_data = {
            "id": node_id,
            "type": "question",
            "text": text,
            "embedding": embedding.tolist(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "created_at": datetime.now(timezone.utc).timestamp(),
            "session_id": session_id,
            "metadata": metadata or {}
        }
        
        # Add to graph
        self.graph.add_node(node_id, **node_data)
        return node_data
    
    def add_response_node(self, text: str, parent_question_id: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Add LLM response as regular node."""
        node_id = f"response_{self.node_counter}"
        self.node_counter += 1
        
        # Generate embedding
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        
        # Create node data
        node_data = {
            "id": node_id,
            "type": "response",
            "text": text,
            "embedding": embedding.tolist(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "created_at": datetime.now(timezone.utc).timestamp(),
            "parent_question_id": parent_question_id,
            "metadata": metadata or {}
        }
        
        # Add to graph
        self.graph.add_node(node_id, **node_data)
        
        # Add edge from question to response
        self.graph.add_edge(parent_question_id, node_id, edge_type="question_response")
        
        return node_data
    
    def bfs_traverse_by_similarity(self, start_node: Dict[str, Any], threshold: float = None) -> List[Dict[str, Any]]:
        """BFS traversal from start node until similarity drops below threshold."""
        if threshold is None:
            threshold = self.similarity_threshold
            
        relevant_nodes = []
        visited = set()
        queue = [(start_node["id"], 0)]
        start_embedding = np.array(start_node["embedding"])
        
        while queue:
            node_id, depth = queue.pop(0)
            
            if node_id in visited:
                continue
                
            visited.add(node_id)
            
            # Skip the start node itself
            if node_id != start_node["id"]:
                current_node = self.graph.nodes[node_id]
                current_embedding = np.array(current_node["embedding"])
                
                # Calculate similarity
                similarity = np.dot(start_embedding, current_embedding) / (
                    np.linalg.norm(start_embedding) * np.linalg.norm(current_embedding)
                )
                
                if similarity >= threshold:
                    relevant_nodes.append({
                        "node_id": node_id,
                        "similarity": float(similarity),
                        "depth": depth,
                        "node_data": current_node
                    })
                else:
                    # Stop exploring this branch if similarity is too low
                    continue
            
            # Add neighbors to queue
            for neighbor in self.graph.neighbors(node_id):
                if neighbor not in visited:
                    queue.append((neighbor, depth + 1))
        
        return relevant_nodes
    
    def get_nodes_by_timestamp(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """Return nodes sorted by timestamp."""
        nodes = []
        for node_id in node_ids:
            if node_id in self.graph.nodes:
                nodes.append(self.graph.nodes[node_id])
        
        # Sort by timestamp
        sorted_nodes = sorted(nodes, key=lambda x: x["timestamp"])
        return sorted_nodes
    
    def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node data by ID."""
        if node_id in self.graph.nodes:
            return self.graph.nodes[node_id]
        return None
    
    def update_node_metadata(self, node_id: str, metadata: Dict[str, Any]) -> bool:
        """Update node metadata."""
        if node_id in self.graph.nodes:
            self.graph.nodes[node_id]["metadata"].update(metadata)
            return True
        return False
    
    def add_edge(self, source_id: str, target_id: str, edge_type: str, confidence: float = 1.0, **kwargs):
        """Add edge between nodes."""
        if source_id in self.graph.nodes and target_id in self.graph.nodes:
            edge_data = {
                "edge_type": edge_type,
                "confidence": confidence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **kwargs
            }
            self.graph.add_edge(source_id, target_id, **edge_data)
            return True
        return False
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "total_nodes": len(self.graph.nodes),
            "question_nodes": len([n for n, d in self.graph.nodes(data=True) if d["type"] == "question"]),
            "response_nodes": len([n for n, d in self.graph.nodes(data=True) if d["type"] == "response"]),
            "total_edges": len(self.graph.edges),
            "edge_types": list(set(d["edge_type"] for _, _, d in self.graph.edges(data=True)))
        }
    
    def prune_old_nodes(self, max_nodes: int = 1000) -> int:
        """Remove oldest nodes to maintain size limit."""
        if len(self.graph.nodes) <= max_nodes:
            return 0
        
        # Sort nodes by creation time
        nodes_by_age = sorted(
            [(n, d["created_at"]) for n, d in self.graph.nodes(data=True)],
            key=lambda x: x[1]
        )
        
        # Remove oldest nodes
        nodes_to_remove = nodes_by_age[:len(self.graph.nodes) - max_nodes]
        removed_count = 0
        
        for node_id, _ in nodes_to_remove:
            self.graph.remove_node(node_id)
            removed_count += 1
        
        return removed_count
    
    def save_graph(self, filepath: str):
        """Save graph to file."""
        # Convert to serializable format
        graph_data = {
            "nodes": dict(self.graph.nodes(data=True)),
            "edges": [(u, v, dict(d)) for u, v, d in self.graph.edges(data=True)]
        }
        
        with open(filepath, 'w') as f:
            json.dump(graph_data, f, indent=2)
    
    def load_graph(self, filepath: str):
        """Load graph from file."""
        with open(filepath, 'r') as f:
            graph_data = json.load(f)
        
        # Rebuild graph
        self.graph = nx.DiGraph()
        
        # Add nodes
        for node_id, node_data in graph_data["nodes"].items():
            self.graph.add_node(node_id, **node_data)
        
        # Add edges
        for u, v, edge_data in graph_data["edges"]:
            self.graph.add_edge(u, v, **edge_data)
        
        # Update node counter
        max_id = max([int(n.split('_')[1]) for n in self.graph.nodes if '_' in n], default=0)
        self.node_counter = max_id + 1
