"""Enhanced Temporal Manager for timestamp ordering and temporal reasoning."""
from __future__ import annotations

from typing import Dict, List, Any, Optional, Tuple
import networkx as nx
from datetime import datetime, timezone, timedelta
import numpy as np

# Import existing temporal reasoning
from temporal_reasoning import find_all_temporal_edges, DEFAULT_TEMPORAL_MODEL


class TemporalManager:
    """Enhanced temporal management with timestamp ordering and reasoning."""
    
    def __init__(self, temporal_model: Optional[str] = None):
        self.temporal_model = temporal_model or DEFAULT_TEMPORAL_MODEL
        
    def order_nodes_by_timestamp(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort nodes chronologically."""
        return sorted(nodes, key=lambda x: x.get("timestamp", ""))
    
    def order_nodes_by_timestamp_with_ids(self, graph: nx.DiGraph, node_ids: List[str]) -> List[Dict[str, Any]]:
        """Return nodes sorted by timestamp given node IDs."""
        nodes = []
        for node_id in node_ids:
            if node_id in graph.nodes:
                nodes.append(graph.nodes[node_id])
        
        return self.order_nodes_by_timestamp(nodes)
    
    def detect_temporal_relationships(self, node1: Dict[str, Any], node2: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhanced temporal reasoning using existing temporal_reasoning.py."""
        try:
            # Use existing temporal reasoning
            temporal_edges = find_all_temporal_edges([(node1["text"], node2["text"])])
            return temporal_edges
        except Exception as e:
            # Fallback to basic timestamp-based reasoning
            return self._basic_temporal_reasoning(node1, node2)
    
    def add_temporal_edges(self, graph: nx.DiGraph) -> int:
        """Add temporal edges between all node pairs."""
        nodes = list(graph.nodes(data=True))
        edges_added = 0
        
        for i, (node1_id, node1_data) in enumerate(nodes):
            for j, (node2_id, node2_data) in enumerate(nodes):
                if i != j and not graph.has_edge(node1_id, node2_id):
                    temporal_rel = self.detect_temporal_relationships(node1_data, node2_data)
                    
                    if temporal_rel and len(temporal_rel) > 0:
                        # Use the first (most confident) temporal relationship
                        rel = temporal_rel[0]
                        graph.add_edge(
                            node1_id, 
                            node2_id, 
                            edge_type="temporal",
                            confidence=rel.get("confidence", 0.5),
                            relationship=rel.get("relationship", "unknown"),
                            direction=rel.get("direction", "forward")
                        )
                        edges_added += 1
        
        return edges_added
    
    def get_temporal_context(self, graph: nx.DiGraph, center_node_id: str, 
                           time_window_hours: float = 24.0) -> Dict[str, Any]:
        """Get temporal context around a center node."""
        if center_node_id not in graph.nodes:
            return {"context_nodes": [], "time_range": None}
        
        center_node = graph.nodes[center_node_id]
        center_time = datetime.fromisoformat(center_node["timestamp"].replace('Z', '+00:00'))
        
        # Define time window
        time_delta = timedelta(hours=time_window_hours)
        start_time = center_time - time_delta
        end_time = center_time + time_delta
        
        context_nodes = []
        
        for node_id, node_data in graph.nodes(data=True):
            if node_id == center_node_id:
                continue
                
            node_time = datetime.fromisoformat(node_data["timestamp"].replace('Z', '+00:00'))
            
            if start_time <= node_time <= end_time:
                time_diff = (node_time - center_time).total_seconds() / 3600  # hours
                
                context_nodes.append({
                    "node_id": node_id,
                    "node_data": node_data,
                    "time_diff_hours": time_diff,
                    "is_before": time_diff < 0,
                    "is_after": time_diff > 0
                })
        
        # Sort by time difference
        context_nodes.sort(key=lambda x: abs(x["time_diff_hours"]))
        
        return {
            "context_nodes": context_nodes,
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "center": center_time.isoformat(),
                "window_hours": time_window_hours
            }
        }
    
    def create_temporal_sequence(self, graph: nx.DiGraph, node_ids: List[str]) -> List[Dict[str, Any]]:
        """Create a temporally ordered sequence from nodes."""
        ordered_nodes = self.order_nodes_by_timestamp_with_ids(graph, node_ids)
        
        sequence = []
        for i, node in enumerate(ordered_nodes):
            node_time = datetime.fromisoformat(node["timestamp"].replace('Z', '+00:00'))
            
            # Calculate time delta from previous node
            if i > 0:
                prev_time = datetime.fromisoformat(ordered_nodes[i-1]["timestamp"].replace('Z', '+00:00'))
                time_delta = (node_time - prev_time).total_seconds()
            else:
                time_delta = 0
            
            sequence.append({
                "node_id": node["id"],
                "node_data": node,
                "sequence_position": i,
                "time_delta_seconds": time_delta,
                "timestamp": node["timestamp"]
            })
        
        return sequence
    
    def analyze_temporal_patterns(self, graph: nx.DiGraph, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Analyze temporal patterns in the graph."""
        nodes = []
        for node_id, node_data in graph.nodes(data=True):
            if session_id is None or node_data.get("session_id") == session_id:
                nodes.append(node_data)
        
        if not nodes:
            return {"patterns": [], "statistics": {}}
        
        # Sort by timestamp
        ordered_nodes = self.order_nodes_by_timestamp(nodes)
        
        # Calculate time intervals
        intervals = []
        for i in range(1, len(ordered_nodes)):
            prev_time = datetime.fromisoformat(ordered_nodes[i-1]["timestamp"].replace('Z', '+00:00'))
            curr_time = datetime.fromisoformat(ordered_nodes[i]["timestamp"].replace('Z', '+00:00'))
            interval = (curr_time - prev_time).total_seconds()
            intervals.append(interval)
        
        # Analyze patterns
        if intervals:
            avg_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            # Detect patterns
            patterns = []
            
            # Burst activity (short intervals)
            burst_threshold = avg_interval - std_interval
            bursts = [i for i, interval in enumerate(intervals) if interval < burst_threshold]
            if bursts:
                patterns.append({
                    "type": "burst_activity",
                    "positions": bursts,
                    "description": f"Detected {len(bursts)} burst activity periods"
                })
            
            # Long gaps
            gap_threshold = avg_interval + std_interval
            gaps = [i for i, interval in enumerate(intervals) if interval > gap_threshold]
            if gaps:
                patterns.append({
                    "type": "long_gaps",
                    "positions": gaps,
                    "description": f"Detected {len(gaps)} long gaps in activity"
                })
            
            # Regular intervals
            if std_interval < avg_interval * 0.3:  # Low variance
                patterns.append({
                    "type": "regular_intervals",
                    "avg_interval": avg_interval,
                    "description": "Regular interaction pattern detected"
                })
        else:
            avg_interval = 0
            std_interval = 0
            patterns = []
        
        return {
            "patterns": patterns,
            "statistics": {
                "total_nodes": len(ordered_nodes),
                "time_span_hours": (datetime.fromisoformat(ordered_nodes[-1]["timestamp"].replace('Z', '+00:00')) - 
                                  datetime.fromisoformat(ordered_nodes[0]["timestamp"].replace('Z', '+00:00'))).total_seconds() / 3600,
                "avg_interval_seconds": avg_interval,
                "std_interval_seconds": std_interval,
                "interaction_rate": len(ordered_nodes) / max((datetime.fromisoformat(ordered_nodes[-1]["timestamp"].replace('Z', '+00:00')) - 
                                                           datetime.fromisoformat(ordered_nodes[0]["timestamp"].replace('Z', '+00:00'))).total_seconds() / 3600, 1)
            }
        }
    
    def _basic_temporal_reasoning(self, node1: Dict[str, Any], node2: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fallback basic temporal reasoning based on timestamps."""
        try:
            time1 = datetime.fromisoformat(node1["timestamp"].replace('Z', '+00:00'))
            time2 = datetime.fromisoformat(node2["timestamp"].replace('Z', '+00:00'))
            
            if time1 < time2:
                relationship = "before"
                direction = "forward"
            elif time1 > time2:
                relationship = "after"
                direction = "backward"
            else:
                relationship = "simultaneous"
                direction = "neutral"
            
            # Basic confidence based on time difference
            time_diff = abs((time2 - time1).total_seconds())
            confidence = max(0.3, 1.0 - (time_diff / (24 * 3600)))  # Decay over 24 hours
            
            return [{
                "relationship": relationship,
                "direction": direction,
                "confidence": confidence,
                "method": "timestamp_based"
            }]
            
        except Exception:
            return []
    
    def get_temporal_summary(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Get a summary of temporal information in the graph."""
        if not graph.nodes:
            return {"summary": "No nodes in graph"}
        
        timestamps = []
        node_types = {"question": 0, "response": 0, "other": 0}
        
        for node_id, node_data in graph.nodes(data=True):
            timestamps.append(node_data.get("timestamp", ""))
            node_type = node_data.get("type", "other")
            if node_type in node_types:
                node_types[node_type] += 1
            else:
                node_types["other"] += 1
        
        # Convert to datetime objects
        valid_timestamps = []
        for ts in timestamps:
            try:
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                valid_timestamps.append(dt)
            except:
                continue
        
        if valid_timestamps:
            min_time = min(valid_timestamps)
            max_time = max(valid_timestamps)
            time_span = max_time - min_time
            
            return {
                "total_nodes": len(graph.nodes),
                "node_types": node_types,
                "time_range": {
                    "earliest": min_time.isoformat(),
                    "latest": max_time.isoformat(),
                    "span_hours": time_span.total_seconds() / 3600,
                    "span_days": time_span.days
                },
                "temporal_edges": len([e for e in graph.edges(data=True) 
                                     if e[2].get("edge_type") == "temporal"])
            }
        else:
            return {
                "total_nodes": len(graph.nodes),
                "node_types": node_types,
                "time_range": None,
                "temporal_edges": 0
            }
