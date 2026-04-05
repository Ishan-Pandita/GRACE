from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
from improved_config import get_improved_config
from mobile_prompt_compression_system import create_mobile_system
from pathlib import Path
import logging

from fastapi.middleware.cors import CORSMiddleware

# Mute some verbose logging
logging.getLogger("uvicorn").setLevel(logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite React frontend
        "http://localhost:8000",   # Self-origin
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# System Initialization
config = get_improved_config()
config["use_ollama"] = True
config["models"]["mistral_model"] = "phi3:mini"
config["causal_detection"]["use_full_model"] = True
config["causal_detection"]["fallback_to_lightweight"] = True
config["causal_detection"]["use_custom_model"] = True
config["causal_detection"]["custom_model_path"] = "./custom_causal_model"
config["limits"]["max_context_tokens"] = 4096
config["thresholds"]["similarity_threshold"] = 0.6

system = create_mobile_system(config)
session_id = "web_session_1"

class ChatRequest(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def read_root():
    template = Path("index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=template)

@app.get("/graph_state")
async def get_graph():
    nodes = []
    edges = []
    
    for node_id, node_data in system.graph_manager.graph.nodes(data=True):
        node_type = node_data.get("type", "unknown")
        text = node_data.get("text", "")
        # Truncate text for label
        label = text[:40] + ("..." if len(text) > 40 else "")
        # Remove emojis or characters that can break the visualization logic
        label = "".join([c for c in label if ord(c) < 65536])
        full_text = "".join([c for c in text if ord(c) < 65536])
        
        nodes.append({
            "id": node_id,
            "label": label,
            "full_text": full_text,
            "group": node_type,
            "timestamp": node_data.get("timestamp", "")
        })
        
    for u, v, edge_data in system.graph_manager.graph.edges(data=True):
        edges.append({
            "from": u,
            "to": v,
            "label": edge_data.get("edge_type", "edge"),
            "confidence": edge_data.get("confidence", 1.0),
            "detection_method": edge_data.get("detection_method", "")
        })
    
    return {"nodes": nodes, "edges": edges}

@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.message
    
    result = await system.process_user_query(user_input, session_id)
    graph_state = await get_graph()
    
    response_text = result.get("response", "Error processing request.")
    # Safe encoding for web payload
    response_text = "".join([c for c in response_text if ord(c) < 65536])
    
    return JSONResponse(content={
        "success": result.get("success", False),
        "response": response_text,
        "error": result.get("error", ""),
        "graph": graph_state,
        "question_id": result.get("question_node_id"),
        "response_id": result.get("response_node_id"),
        "relevant_nodes_count": result.get("relevant_nodes_found", 0),
        "relevant_nodes_texts": result.get("relevant_nodes_texts", []),
        "compressed_prompt": result.get("compressed_prompt", ""),
        "relationship_stats": result.get("relationship_stats", {}),
        "pipeline_trace": result.get("pipeline_trace", {}),
        "token_metrics": result.get("token_metrics", {}),
        "graph_stats": result.get("graph_stats", {}),
        "processing_time": result.get("processing_time", 0.0),
        "context_full": result.get("context_full", False)
    })


@app.post("/clear")
async def clear_context():
    """Reset all graph data, session state, and start fresh."""
    import networkx as nx
    system.graph_manager.graph = nx.DiGraph()
    system.graph_manager.node_counter = 0
    system.active_sessions.clear()
    raw_history.clear()
    global session_id
    import time as t
    session_id = f"web_session_{int(t.time())}"
    return JSONResponse(content={"success": True, "message": "All context cleared."})


# ═══════════════════════════════════════════
# COMPARE MODE: Compressed vs Raw
# ═══════════════════════════════════════════
import time as time_module
from ollama_interface import OllamaInterface

raw_ollama = OllamaInterface(model_name=config["models"]["mistral_model"])
raw_history: list[dict] = []  # Accumulated raw conversation history
MODEL_NAME = config["models"]["mistral_model"]


@app.post("/clear_compare")
async def clear_compare():
    """Clear both compressed system and raw history."""
    import networkx as nx
    system.graph_manager.graph = nx.DiGraph()
    system.graph_manager.node_counter = 0
    system.active_sessions.clear()
    raw_history.clear()
    global session_id
    import time as t
    session_id = f"web_session_{int(t.time())}"
    return JSONResponse(content={"success": True})


@app.post("/compare")
async def compare(request: ChatRequest):
    """Run the same prompt through both compressed and raw pipelines."""
    user_input = request.message

    # ── Path A: WITH compression (full pipeline) ──
    a_start = time_module.time()
    result_a = await system.process_user_query(user_input, session_id)
    a_time = time_module.time() - a_start

    response_a = result_a.get("response", "Error")
    response_a = "".join([c for c in response_a if ord(c) < 65536])
    token_metrics_a = result_a.get("token_metrics", {})

    # ── Path B: WITHOUT compression (raw accumulated history → LLM) ──
    raw_history.append({"role": "user", "content": user_input})

    # Build raw context: full conversation history
    raw_context = ""
    for entry in raw_history:
        prefix = "User" if entry["role"] == "user" else "Assistant"
        raw_context += f"{prefix}: {entry['content']}\n"
    raw_context += "Assistant:"

    raw_input_tokens = len(user_input.split())
    raw_context_tokens = len(raw_context.split())
    
    max_raw_context = config["limits"]["max_context_tokens"]
    
    if raw_context_tokens >= max_raw_context:
        response_b = (f"Raw Context Window Full! "
                      f"Tokens: {raw_context_tokens}/{max_raw_context} "
                      f"({round(raw_context_tokens/max_raw_context*100, 1)}%). "
                      f"Model: {MODEL_NAME}. "
                      f"Raw pipeline halted.")
        b_time = 0
        raw_result = {"success": True}
        raw_response_tokens = 0
    else:
        b_start = time_module.time()
        raw_result = raw_ollama.generate_response(raw_context, max_tokens=150)
        b_time = time_module.time() - b_start

        if raw_result["success"]:
            response_b = raw_result["response"]
            response_b = "".join([c for c in response_b if ord(c) < 65536])
            raw_response_tokens = len(response_b.split())
            raw_history.append({"role": "assistant", "content": response_b})
        else:
            response_b = f"Error: {raw_result.get('error', 'Unknown')}"
            raw_response_tokens = 0

    # ── Build comparison payload ──
    graph_state = await get_graph()
    graph_stats = result_a.get("graph_stats", {})

    return JSONResponse(content={
        "model": MODEL_NAME,
        "prompt": user_input,

        # Compressed side
        "compressed": {
            "response": response_a,
            "time": round(a_time, 2),
            "input_tokens": token_metrics_a.get("input_tokens", 0),
            "context_tokens": token_metrics_a.get("context_tokens", 0),
            "total_original_tokens": token_metrics_a.get("total_original_tokens", 0),
            "compressed_tokens": token_metrics_a.get("compressed_tokens", 0),
            "compression_ratio": token_metrics_a.get("compression_ratio", 0),
            "tokens_saved": token_metrics_a.get("tokens_saved", 0),
            "bfs_nodes": result_a.get("relevant_nodes_found", 0),
            "pipeline_steps": len(result_a.get("pipeline_trace", {}).get("steps", [])),
            "pipeline_trace": result_a.get("pipeline_trace", {}),
            "context_full": result_a.get("context_full", False),
        },

        # Raw side
        "raw": {
            "response": response_b,
            "time": round(b_time, 2),
            "token_metrics": {
                "input_tokens": raw_input_tokens,
                "context_tokens": raw_context_tokens,
                "total_original_tokens": raw_context_tokens,
                "compressed_tokens": raw_context_tokens,  # It's raw, no compression
                "compression_ratio": 0,
                "tokens_saved": 0,
                "max_allowed_tokens": max_raw_context,
                "context_usage_percentage": round((raw_context_tokens / max(1, max_raw_context)) * 100, 1)
            },
            "history_length": len(raw_history),
            "context_full": raw_context_tokens >= max_raw_context,
        },

        "graph": graph_state,
        "graph_stats": graph_stats,
    })


# 15 test prompts that build overlapping context clusters
TEST_PROMPTS = [
    # Cluster 1: Photosynthesis (builds context)
    "What is photosynthesis and how do plants use sunlight?",
    "How does chlorophyll absorb light during photosynthesis?",
    "Why do plants need water and carbon dioxide for photosynthesis?",
    # Cluster 2: Completely different (history) — no compression expected
    "Who was Julius Caesar and what did he accomplish?",
    # Cluster 1 callback: Should find photosynthesis context
    "What is the role of glucose produced during photosynthesis?",
    # Cluster 3: Water cycle (new, but 'evaporation' and 'sunlight' link to cluster 1)
    "How does the water cycle work? Explain evaporation and condensation.",
    "What causes rainfall and how does precipitation occur?",
    # Causal test: should detect "because" and "leads to"
    "Deforestation leads to increased carbon dioxide because fewer trees perform photosynthesis.",
    # Cluster 2 callback: Roman history — should find Caesar context
    "What was the structure of the Roman government under the Republic?",
    # Cross-domain: links clusters 1 and 3
    "How does photosynthesis connect to the water cycle through transpiration?",
    # Temporal test: "first...then...finally"
    "First a seed germinates, then the seedling grows leaves, and finally the mature plant begins photosynthesis.",
    # Redundant: very similar to prompt 1 — high compression expected
    "Explain what photosynthesis is and how plants convert sunlight to energy.",
    # Causal + cross-domain: climate change links all clusters
    "Because burning fossil fuels releases CO2, global warming increases, which disrupts both photosynthesis rates and the water cycle.",
    # Complex multi-part
    "Compare the Roman aqueduct system with modern water distribution. How did Roman engineering affect agriculture and food production?",
    # Final callback: should find context from ALL clusters
    "Summarize the connections between plant biology, the water cycle, climate change, and ancient Roman agriculture.",
]


from fastapi.responses import StreamingResponse
import asyncio
import json as json_module

@app.get("/test_pipeline")
async def test_pipeline():
    """Run 15-prompt test pipeline, streaming results one by one."""
    async def generate():
        global session_id
        for idx, prompt in enumerate(TEST_PROMPTS):
            result = await system.process_user_query(prompt, session_id)
            graph_state = await get_graph()

            response_text = result.get("response", "Error")
            response_text = "".join([c for c in response_text if ord(c) < 65536])

            payload = {
                "index": idx,
                "total": len(TEST_PROMPTS),
                "prompt": prompt,
                "success": result.get("success", False),
                "response": response_text,
                "graph": graph_state,
                "question_id": result.get("question_node_id"),
                "response_id": result.get("response_node_id"),
                "relevant_nodes_count": result.get("relevant_nodes_found", 0),
                "relevant_nodes_texts": result.get("relevant_nodes_texts", []),
                "compressed_prompt": result.get("compressed_prompt", ""),
                "relationship_stats": result.get("relationship_stats", {}),
                "pipeline_trace": result.get("pipeline_trace", {}),
                "token_metrics": result.get("token_metrics", {}),
                "graph_stats": result.get("graph_stats", {}),
                "processing_time": result.get("processing_time", 0.0)
            }

            yield f"data: {json_module.dumps(payload, default=str)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    print("\n\n" + "="*50)
    print("🚀 App starting at http://localhost:8000")
    print("="*50 + "\n\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
