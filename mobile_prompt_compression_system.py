"""Mobile Prompt Compression System - Main interface for on-device deployment."""
from __future__ import annotations

from typing import Dict, List, Any, Optional
import asyncio
import uuid
from datetime import datetime, timezone
import json

# Import all components
from graph_manager import GraphManager
from prompt_processor import PromptProcessor
from relevance_engine import RelevanceEngine
from temporal_manager import TemporalManager
from enhanced_causal_reasoner import EnhancedCausalReasoner
from compression_engine import CompressionEngine
from mobile_optimizer import MobileOptimizer

# Import existing models
from sentence_transformers import SentenceTransformer


class OnDeviceLLMInterface:
    """Interface for on-device LLM with Ollama integration."""
    
    def __init__(self, model_name: str = "mistral-7b-int4", use_ollama: bool = False):
        self.model_name = model_name
        self.use_ollama = use_ollama
        self.model = None
        self.tokenizer = None
        self.ollama_interface = None
        
        if use_ollama:
            try:
                from ollama_interface import create_ollama_interface
                self.ollama_interface = create_ollama_interface(model_name)
                print(f"Using Ollama with model: {model_name}")
            except Exception as e:
                print(f"Failed to initialize Ollama: {e}")
                self.use_ollama = False
    
    def load_model(self):
        """Load the on-device model."""
        if self.use_ollama and self.ollama_interface:
            self.ollama_interface.load_model()
        else:
            # Placeholder for local model loading
            pass
    
    async def generate_response(self, prompt: str, max_tokens: int = 150) -> str:
        """Generate response from the LLM."""
        if self.use_ollama and self.ollama_interface:
            try:
                return await self.ollama_interface.generate_response(prompt, max_tokens)
            except Exception as e:
                print(f"Ollama generation failed: {e}")
                # Fallback to placeholder
                return await self._generate_placeholder_response(prompt)
        else:
            # Placeholder implementation
            return await self._generate_placeholder_response(prompt)
    
    async def _generate_placeholder_response(self, prompt: str) -> str:
        """Generate placeholder response."""
        await asyncio.sleep(0.1)  # Simulate inference time
        
        if "question" in prompt.lower():
            return "Based on the context provided, here's my answer to your question."
        else:
            return "I understand your input and will provide a relevant response."
    
    def unload_model(self):
        """Unload the model to free memory."""
        if self.use_ollama and self.ollama_interface:
            self.ollama_interface.unload_model()
        self.model = None
        self.tokenizer = None


class MobilePromptCompressionSystem:
    """Main system for mobile prompt compression with graph-based reasoning."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Configuration
        self.config = config or self._get_default_config()
        
        # Initialize components
        self._initialize_components()
        
        # Session management
        self.active_sessions = {}
        
        # Start optimization monitoring
        self.mobile_optimizer.start_monitoring()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for mobile deployment."""
        return {
            "models": {
                "embedding_model": "BAAI/bge-small-en-v1.5",
                "mistral_model": "mistral-7b-int4",
                "causal_model": "FacebookAI/roberta-large-mnli",
                "lightweight_model": "distilroberta-base",
                "summarizer_model": None  # Optional
            },
            "thresholds": {
                "similarity_threshold": 0.7,
                "causal_confidence": 0.6,
                "temporal_confidence": 0.5
            },
            "limits": {
                "max_graph_nodes": 500,
                "max_context_tokens": 512,
                "memory_limit_mb": 256,
                "max_bfs_depth": 3
            },
            "optimization": {
                "batch_size": 8,
                "quantization": "int4",
                "prune_interval": 50,
                "cache_embeddings": True
            }
        }
    
    def _initialize_components(self):
        """Initialize all system components."""
        # Load embedding model
        embedding_model_name = self.config["models"]["embedding_model"]
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize core components
        self.graph_manager = GraphManager(
            self.embedding_model, 
            self.config["thresholds"]["similarity_threshold"]
        )
        
        self.prompt_processor = PromptProcessor(self.embedding_model)
        
        self.relevance_engine = RelevanceEngine(
            self.config["thresholds"]["similarity_threshold"],
            self.config["limits"]["max_bfs_depth"]
        )
        
        self.temporal_manager = TemporalManager()
        
        causal_config = self.config.get("causal_detection", {})
        self.causal_reasoner = EnhancedCausalReasoner(
            causal_model=self.config["models"]["causal_model"],
            lightweight_model=self.config["models"]["lightweight_model"],
            use_full_model=causal_config.get("use_full_model", True),
            use_custom_model=causal_config.get("use_custom_model", False),
            custom_model_path=causal_config.get("custom_model_path", None)
        )
        
        self.compression_engine = CompressionEngine(
            self.config["models"].get("summarizer_model"),
            self.config["limits"]["max_context_tokens"]
        )
        
        self.mobile_optimizer = MobileOptimizer(
            self.config["limits"]["memory_limit_mb"],
            self.config["limits"]["max_graph_nodes"]
        )
        
        # Initialize LLM interface
        self.llm_interface = OnDeviceLLMInterface(
            self.config["models"]["mistral_model"],
            use_ollama=self.config.get("use_ollama", False)
        )
        self.llm_interface.load_model()
        
        # Apply mobile optimizations
        self.mobile_optimizer.optimize_for_mobile()
    
    async def process_user_query(self, user_input: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Main processing pipeline for user queries.
        
        Pipeline:
        1. Create question node with embedding
        2. Find top-K most similar existing nodes (global cosine search)
        3. From most relevant node, BFS traversal collecting nodes while similarity >= threshold
        4. Order collected nodes by timestamp
        5. Compress/summarize into condensed prompt
        6. Send compressed prompt to LLM
        7. Add LLM response as a regular node (not question node)
        8. Detect causal edges using custom model + traditional ML
        9. Detect temporal edges
        10. Add similarity edges for future BFS traversals
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "query_count": 0,
                "cumulative_input_tokens": 0,
                "cumulative_compressed_tokens": 0,
                "cumulative_original_tokens": 0,
                "cumulative_tokens_saved": 0,
                "questions_processed": 0,
            }
        
        session = self.active_sessions[session_id]
        session["query_count"] += 1
        
        # Pipeline trace for frontend
        pipeline_trace = {
            "steps": []
        }
        
        try:
            import time
            start_time = time.time()
            
            # ── EARLY CHECK: If context is already full, refuse immediately ──
            max_ctx = self.config["limits"]["max_context_tokens"]
            if session.get("cumulative_compressed_tokens", 0) >= max_ctx:
                model_name = self.config.get('models', {}).get('mistral_model', 'phi3:mini')
                return {
                    "success": True,
                    "response": "Context window is full. Please clear context to continue.",
                    "context_full": True,
                    "session_id": session_id,
                    "question_node_id": None,
                    "response_node_id": None,
                    "relevant_nodes_found": 0,
                    "token_metrics": {
                        "input_tokens": len(user_input.split()),
                        "context_tokens": 0,
                        "total_original_tokens": session["cumulative_original_tokens"],
                        "compressed_tokens": session["cumulative_compressed_tokens"],
                        "compression_ratio": round((1 - session["cumulative_compressed_tokens"] / max(session["cumulative_original_tokens"], 1)) * 100, 1),
                        "tokens_saved": session["cumulative_tokens_saved"],
                        "max_allowed_tokens": max_ctx,
                        "context_usage_percentage": round((session["cumulative_compressed_tokens"] / max(1, max_ctx)) * 100, 1),
                        "model_name": model_name,
                        "questions_processed": session["questions_processed"],
                        "cumulative_input_tokens": session["cumulative_input_tokens"],
                    },
                    "pipeline_trace": pipeline_trace,
                    "graph_stats": self.graph_manager.get_graph_stats()
                }
            
            # ── Step 1: Create question node ──
            prompt_node_data = self.prompt_processor.create_question_node(user_input, session_id)
            metadata = prompt_node_data.get("metadata", {})
            metadata["semantic_units"] = prompt_node_data.get("semantic_units", {})
            metadata["processing_stats"] = prompt_node_data.get("processing_stats", {})
            
            question_node = self.graph_manager.add_question_node(
                prompt_node_data["text"], session_id, metadata
            )
            
            pipeline_trace["steps"].append({
                "name": "Create Question Node",
                "detail": f"Node ID: {question_node['id']}",
                "duration": round(time.time() - start_time, 3)
            })
            
            # ── Step 2: Find top-K most similar existing nodes (global search) ──
            step2_start = time.time()
            top_k_nodes = self._find_top_k_similar(question_node, top_k=10)
            
            pipeline_trace["steps"].append({
                "name": "Find Top-K Similar Nodes",
                "detail": f"Found {len(top_k_nodes)} similar node(s) from {len(self.graph_manager.graph.nodes) - 1} existing",
                "nodes": [{"id": n["node_id"], "similarity": round(n["similarity"], 3), 
                            "text": n["node_data"].get("text", "")[:80]} for n in top_k_nodes[:5]],
                "duration": round(time.time() - step2_start, 3)
            })
            
            # ── Step 3: BFS from most relevant node with cosine threshold ──
            step3_start = time.time()
            bfs_nodes = []
            if top_k_nodes:
                # Start BFS from the most similar node
                best_match = top_k_nodes[0]
                best_node_data = dict(self.graph_manager.graph.nodes[best_match["node_id"]])
                best_node_data["id"] = best_match["node_id"]
                
                bfs_nodes = self.graph_manager.bfs_traverse_by_similarity(
                    question_node, 
                    threshold=self.config["thresholds"]["similarity_threshold"]
                )
                
                # Merge top-k with BFS results (deduplicate)
                seen_ids = {n["node_id"] for n in bfs_nodes}
                for tk_node in top_k_nodes:
                    if tk_node["node_id"] not in seen_ids:
                        bfs_nodes.append(tk_node)
                        seen_ids.add(tk_node["node_id"])
            
            # Filter out the question node itself from the results
            bfs_nodes = [n for n in bfs_nodes if n["node_id"] != question_node["id"]]
            
            detail_msg = f"Collected {len(bfs_nodes)} node(s) above threshold {self.config['thresholds']['similarity_threshold']}"
            if len(bfs_nodes) == 0:
                detail_msg += " — forming a new disjoint graph segment"
                
            pipeline_trace["steps"].append({
                "name": "BFS Traversal (Cosine Threshold)",
                "detail": detail_msg,
                "bfs_path": [n["node_id"] for n in bfs_nodes[:10]],
                "duration": round(time.time() - step3_start, 3)
            })
            
            # ── Step 4: Order by timestamp ──
            step4_start = time.time()
            if bfs_nodes:
                ordered_nodes = self.temporal_manager.order_nodes_by_timestamp(
                    [n["node_data"] for n in bfs_nodes]
                )
            else:
                ordered_nodes = []
            
            pipeline_trace["steps"].append({
                "name": "Temporal Ordering",
                "detail": f"Ordered {len(ordered_nodes)} node(s) chronologically",
                "duration": round(time.time() - step4_start, 3)
            })
            
            # ── Step 5: Compress/summarize into prompt ──
            step5_start = time.time()
            
            # Calculate original token count from all relevant context
            original_context = " ".join([n["node_data"].get("text", "") for n in bfs_nodes])
            original_tokens = len(original_context.split()) if original_context.strip() else 0
            input_tokens = len(user_input.split())
            
            compressed_prompt = self.compression_engine.adaptive_compression(
                bfs_nodes, question_node
            )
            
            compressed_tokens = len(compressed_prompt.split())
            compression_ratio = round((1 - compressed_tokens / max(original_tokens + input_tokens, 1)) * 100, 1)
            
            # Update cumulative session tracking
            session["cumulative_input_tokens"] += input_tokens
            session["cumulative_original_tokens"] += (original_tokens + input_tokens)
            session["cumulative_compressed_tokens"] += compressed_tokens
            session["cumulative_tokens_saved"] += max(0, (original_tokens + input_tokens) - compressed_tokens)
            session["questions_processed"] += 1
            
            cumulative_ratio = round((1 - session["cumulative_compressed_tokens"] / max(session["cumulative_original_tokens"], 1)) * 100, 1)
            
            token_metrics = {
                "input_tokens": input_tokens,
                "context_tokens": original_tokens,
                "total_original_tokens": original_tokens + input_tokens,
                "compressed_tokens": compressed_tokens,
                "compression_ratio": max(0.0, compression_ratio),
                "tokens_saved": max(0, (original_tokens + input_tokens) - compressed_tokens),
                "max_allowed_tokens": self.config["limits"]["max_context_tokens"],
                "context_usage_percentage": round((session["cumulative_compressed_tokens"] / max(1, self.config["limits"]["max_context_tokens"])) * 100, 1),
                "model_name": self.config.get("models", {}).get("mistral_model", "phi3:mini"),
                "questions_processed": session["questions_processed"],
                "cumulative_input_tokens": session["cumulative_input_tokens"],
                "cumulative_compressed_tokens": session["cumulative_compressed_tokens"],
                "cumulative_original_tokens": session["cumulative_original_tokens"],
                "cumulative_tokens_saved": session["cumulative_tokens_saved"],
                "cumulative_compression_ratio": max(0.0, cumulative_ratio),
            }
            
            pipeline_trace["steps"].append({
                "name": "Prompt Compression",
                "detail": f"{original_tokens + input_tokens} tokens -> {compressed_tokens} tokens ({compression_ratio}% reduction)",
                "compressed_prompt": compressed_prompt,
                "retrieved_nodes": [{"id": n["node_id"], "text": n["node_data"].get("text", "")} for n in bfs_nodes],
                "token_metrics": token_metrics,
                "duration": round(time.time() - step5_start, 3)
            })
            
            # ── Check Context Limit (cumulative) ──
            if session["cumulative_compressed_tokens"] >= self.config["limits"]["max_context_tokens"]:
                print(f"Context Window Full (cumulative {session['cumulative_compressed_tokens']}/{self.config['limits']['max_context_tokens']}). Cleaning up graph.")
                
                # IMPORTANT: Remove the question node we just added to keep graph clean
                if question_node["id"] in self.graph_manager.graph:
                    self.graph_manager.graph.remove_node(question_node["id"])
                
                return {
                    "success": True,
                    "response": "Context window is full. Pipeline halted.",
                    "context_full": True,
                    "session_id": session_id,
                    "question_node_id": None,
                    "response_node_id": None,
                    "relevant_nodes_found": len(bfs_nodes),
                    "token_metrics": token_metrics,
                    "pipeline_trace": pipeline_trace,
                    "graph_stats": self.graph_manager.get_graph_stats()
                }

            # ── Step 6: Generate response from LLM ──
            step6_start = time.time()
            response = await self.llm_interface.generate_response(compressed_prompt)
            
            pipeline_trace["steps"].append({
                "name": "LLM Generation (Ollama)",
                "detail": f"Generated {len(response.split())} tokens in {round(time.time() - step6_start, 2)}s",
                "duration": round(time.time() - step6_start, 3)
            })
            
            # ── Step 7: Add response as a regular node ──
            step7_start = time.time()
            response_node = self.graph_manager.add_response_node(
                response, question_node["id"],
                {
                    "session_id": session_id,
                    "compressed_prompt": compressed_prompt,
                }
            )
            
            pipeline_trace["steps"].append({
                "name": "Add Response Node",
                "detail": f"Node ID: {response_node['id']}",
                "duration": round(time.time() - step7_start, 3)
            })
            
            # ── Step 8: Detect causal relationships ──
            step8_start = time.time()
            relationship_stats = self._update_relationships(question_node, response_node)
            
            # Also detect causal edges between the new response and previously similar nodes
            for sim_node in top_k_nodes[:3]:
                try:
                    sim_node_data = dict(self.graph_manager.graph.nodes[sim_node["node_id"]])
                    sim_node_data["id"] = sim_node["node_id"]
                    cross_rel = self._update_relationships(sim_node_data, response_node)
                    relationship_stats["causal_edges"].extend(cross_rel.get("causal_edges", []))
                    relationship_stats["temporal_edges"].extend(cross_rel.get("temporal_edges", []))
                except Exception:
                    pass
            
            pipeline_trace["steps"].append({
                "name": "Causal & Temporal Detection",
                "detail": f"Found {len(relationship_stats['causal_edges'])} causal, {len(relationship_stats['temporal_edges'])} temporal edge(s)",
                "causal_count": len(relationship_stats["causal_edges"]),
                "temporal_count": len(relationship_stats["temporal_edges"]),
                "duration": round(time.time() - step8_start, 3)
            })
            
            # ── Step 9: Add similarity edges for future BFS ──
            step9_start = time.time()
            sim_edges_added = self._add_similarity_edges(question_node, top_k_nodes)
            
            pipeline_trace["steps"].append({
                "name": "Add Similarity Edges",
                "detail": f"Added {sim_edges_added} similarity edge(s) for future BFS traversal",
                "duration": round(time.time() - step9_start, 3)
            })
            
            # ── Step 10: Optimize ──
            self.graph_manager.graph, removed_count = self.mobile_optimizer.optimize_graph_size(
                self.graph_manager.graph
            )
            current_resources = self.mobile_optimizer.monitor_resources()
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            return {
                "success": True,
                "response": response,
                "session_id": session_id,
                "question_node_id": question_node["id"],
                "response_node_id": response_node["id"],
                "relevant_nodes_found": len(bfs_nodes),
                "relevant_nodes_texts": [n["node_data"].get("text", "")[:120] for n in bfs_nodes],
                "compressed_prompt": compressed_prompt,
                "relationship_stats": relationship_stats,
                "pipeline_trace": pipeline_trace,
                "token_metrics": token_metrics,
                "processing_time": processing_time,
                "graph_stats": self.graph_manager.get_graph_stats(),
                "resource_usage": current_resources
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }
    
    def _find_top_k_similar(self, question_node: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """Find top-K most similar nodes in the entire graph by cosine similarity."""
        import numpy as np
        
        question_embedding = np.array(question_node["embedding"])
        similarities = []
        
        for node_id, node_data in self.graph_manager.graph.nodes(data=True):
            if node_id == question_node["id"]:
                continue
            
            node_embedding = np.array(node_data.get("embedding", []))
            if len(node_embedding) == 0:
                continue
            
            # Cosine similarity
            dot = np.dot(question_embedding, node_embedding)
            norm = np.linalg.norm(question_embedding) * np.linalg.norm(node_embedding)
            if norm == 0:
                continue
            sim = float(dot / norm)
            
            similarities.append({
                "node_id": node_id,
                "similarity": sim,
                "depth": 0,
                "node_data": node_data,
                "relevance_score": sim
            })
        
        # Sort descending by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]
    
    def _add_similarity_edges(self, question_node: Dict[str, Any], similar_nodes: List[Dict[str, Any]]) -> int:
        """Add similarity edges between the question node and similar existing nodes, so BFS can traverse them in future."""
        added = 0
        threshold = self.config["thresholds"]["similarity_threshold"]
        
        for sim_node in similar_nodes:
            if sim_node["similarity"] >= threshold:
                # Add bidirectional similarity edges
                success = self.graph_manager.add_edge(
                    question_node["id"], sim_node["node_id"],
                    "similarity",
                    confidence=sim_node["similarity"]
                )
                if success:
                    added += 1
        
        return added
    
    def _update_relationships(self, question_node: Dict[str, Any], response_node: Dict[str, Any]) -> Dict[str, Any]:
        """Update causal and temporal relationships between nodes."""
        result_stats = {
            "causal_edges": [],
            "temporal_edges": []
        }
        try:
            # Causal relationships
            causal_edges = self.causal_reasoner.detect_causal_relationships(
                question_node, response_node
            )
            
            for edge in causal_edges:
                self.graph_manager.add_edge(
                    question_node["id"],
                    response_node["id"],
                    "causal",
                    edge.get("confidence", 0.5),
                    detection_method=edge.get("method", "unknown"),
                    detected_patterns=edge.get("detected_patterns", [])
                )
                result_stats["causal_edges"].append(edge)
            
            # Temporal relationships
            temporal_edges = self.temporal_manager.detect_temporal_relationships(
                question_node, response_node
            )
            
            for edge in temporal_edges:
                self.graph_manager.add_edge(
                    question_node["id"],
                    response_node["id"],
                    "temporal",
                    edge.get("confidence", 0.5),
                    relationship=edge.get("relationship", "unknown"),
                    direction=edge.get("direction", "forward")
                )
                result_stats["temporal_edges"].append(edge)
                
        except Exception as e:
            print(f"Error updating relationships: {e}")
            
        return result_stats
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a session."""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        session_data = self.active_sessions[session_id]
        
        # Get session nodes from graph
        session_nodes = []
        for node_id, node_data in self.graph_manager.graph.nodes(data=True):
            if node_data.get("session_id") == session_id:
                session_nodes.append({
                    "id": node_id,
                    "type": node_data.get("type", "unknown"),
                    "timestamp": node_data.get("timestamp", ""),
                    "text_preview": node_data.get("text", "")[:100] + "..."
                })
        
        return {
            "session_id": session_id,
            "session_data": session_data,
            "nodes": session_nodes,
            "node_count": len(session_nodes)
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "graph_stats": self.graph_manager.get_graph_stats(),
            "active_sessions": len(self.active_sessions),
            "optimization_report": self.mobile_optimizer.get_optimization_report(),
            "temporal_summary": self.temporal_manager.get_temporal_summary(self.graph_manager.graph),
            "causal_statistics": self.causal_reasoner.get_causal_statistics(self.graph_manager.graph),
            "current_config": self.config
        }
    
    def save_session_data(self, session_id: str, filepath: str) -> bool:
        """Save session data to file."""
        try:
            session_info = self.get_session_info(session_id)
            
            # Extract session subgraph
            session_nodes = [node["id"] for node in session_info["nodes"]]
            session_subgraph = self.graph_manager.graph.subgraph(session_nodes)
            
            # Save data
            save_data = {
                "session_info": session_info,
                "graph_data": {
                    "nodes": dict(session_subgraph.nodes(data=True)),
                    "edges": [(u, v, dict(d)) for u, v, d in session_subgraph.edges(data=True)]
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error saving session data: {e}")
            return False
    
    def load_session_data(self, filepath: str) -> bool:
        """Load session data from file."""
        try:
            with open(filepath, 'r') as f:
                save_data = json.load(f)
            
            # Restore graph data
            graph_data = save_data["graph_data"]
            
            # Add nodes
            for node_id, node_data in graph_data["nodes"].items():
                self.graph_manager.graph.add_node(node_id, **node_data)
            
            # Add edges
            for u, v, edge_data in graph_data["edges"]:
                self.graph_manager.graph.add_edge(u, v, **edge_data)
            
            # Restore session info
            session_info = save_data["session_info"]
            self.active_sessions[session_info["session_id"]] = session_info["session_data"]
            
            return True
            
        except Exception as e:
            print(f"Error loading session data: {e}")
            return False
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a session and its nodes from the graph."""
        try:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            # Remove session nodes from graph
            nodes_to_remove = []
            for node_id, node_data in self.graph_manager.graph.nodes(data=True):
                if node_data.get("session_id") == session_id:
                    nodes_to_remove.append(node_id)
            
            self.graph_manager.graph.remove_nodes_from(nodes_to_remove)
            return True
            
        except Exception as e:
            print(f"Error clearing session: {e}")
            return False
    
    def shutdown(self):
        """Shutdown the system and cleanup resources."""
        try:
            # Stop monitoring
            if hasattr(self, 'mobile_optimizer') and self.mobile_optimizer:
                self.mobile_optimizer.stop_monitoring()
            
            # Unload LLM model
            if hasattr(self, 'llm_interface') and self.llm_interface:
                self.llm_interface.unload_model()
            
            # Save final graph state
            if hasattr(self, 'graph_manager') and self.graph_manager:
                self.graph_manager.save_graph("final_graph_state.json")
            
            print("Mobile Prompt Compression System shutdown complete.")
            
        except Exception as e:
            pass
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            # Only shutdown if builtins are still available (prevents shutdown errors)
            if 'open' in __builtins__ or getattr(__builtins__, 'open', None):
                self.shutdown()
        except Exception:
            pass

# Convenience function for quick initialization
def create_mobile_system(config: Optional[Dict[str, Any]] = None) -> MobilePromptCompressionSystem:
    """Create and initialize a mobile prompt compression system."""
    return MobilePromptCompressionSystem(config)


# Example usage
async def example_usage():
    """Example of how to use the mobile prompt compression system."""
    
    # Initialize system
    system = create_mobile_system()
    
    try:
        # Process a query
        result = await system.process_user_query(
            "What are the main causes of climate change?",
            session_id="example_session"
        )
        
        if result["success"]:
            print(f"Response: {result['response']}")
            print(f"Relevant nodes found: {result['relevant_nodes_found']}")
            print(f"Graph stats: {result['graph_stats']}")
        else:
            print(f"Error: {result['error']}")
        
        # Get system status
        status = system.get_system_status()
        print(f"System status: {status}")
        
    finally:
        # Cleanup
        system.shutdown()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
