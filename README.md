# GRACE — Graph-Based Prompt Compression for Edge LLMs

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/Tests-57%20passed-brightgreen)
![Framework](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)
![Frontend](https://img.shields.io/badge/Frontend-React%20%2B%20TypeScript-61DAFB?logo=react)

A graph-based conversational AI system that compresses context using dynamic knowledge graphs before sending prompts to a local LLM. Instead of dumping the full conversation history into every prompt (which quickly overflows the context window), this system builds a **knowledge graph** of past Q&A pairs and uses **BFS traversal**, **causal reasoning**, and **temporal ordering** to retrieve only the most relevant context — then compresses it before sending it to the model.

This is particularly useful for **edge devices** (phones, tablets, embedded systems, Raspberry Pi) where LLMs run with **small context windows** (2K–4K tokens) and **limited memory**, making it impossible to include full conversation histories in every prompt.

## Why This Was Built

Large Language Models have a fixed context window. When running LLMs on edge devices, this window is often much smaller (2K–4K tokens vs 128K on cloud APIs). In multi-turn conversations, naively appending every previous message causes:

1. **Context overflow** — the conversation gets cut off or errors out after just a few turns.
2. **Token waste** — most of the history is irrelevant to the current question, wasting the already limited window.
3. **Slow inference** — more tokens = longer response times, which is critical on resource-constrained devices.
4. **Memory pressure** — edge devices have limited RAM; large prompts can cause OOM crashes.

This system solves all four by maintaining a **graph of knowledge nodes** (questions + responses) connected by **similarity**, **causal**, and **temporal edges**. When a new question arrives, it traverses the graph to find only relevant nodes, compresses them, and sends a minimal prompt — achieving **significant token savings** while preserving answer quality.

---

## Use Cases

This system is designed for scenarios where LLMs run locally on devices with **limited context windows and resources**:

| Use Case | Problem | How This Helps |
|----------|---------|----------------|
| **On-device AI assistants** (phones, tablets) | Quantized models (e.g., Phi-3 Mini, Mistral 7B INT4) have 2K–4K token context windows that fill up after 3–4 conversation turns | Knowledge graph retains full history; only the relevant compressed context is sent to the LLM |
| **IoT / Embedded systems** (Raspberry Pi, Jetson Nano) | Extremely limited RAM (2–8 GB) means long prompts can crash the device | Graph pruning + memory monitoring keeps resource usage within device limits |
| **Offline chatbots** (field workers, remote areas) | No cloud API access; everything must run locally with small models | Works entirely offline with Ollama; no internet needed after initial setup |
| **Customer support kiosks** | Long support sessions with 50+ messages overflow the context of locally-hosted models | BFS traversal finds only the relevant past exchanges, even across long sessions |
| **Medical / Legal consultation devices** | Conversations involve causal chains ("X caused Y, which led to Z") that must be preserved accurately | Causal and temporal edge detection ensures reasoning chains aren't lost during compression |
| **Educational tutoring systems** | Students revisit topics from earlier in the session; the model needs to recall | Similarity edges connect related topics; BFS retrieves past explanations even from many turns ago |
| **Privacy-sensitive deployments** (healthcare, finance) | Data cannot leave the device; cloud LLM APIs are not an option | Fully local — no data sent externally; graph stored on-device |

---

## Project Modules

| Module | Description |
|--------|-------------|
| `app.py` | FastAPI web server — the main entry point that exposes chat, compare, and graph APIs |
| `main.py` | Standalone dataset evaluation pipeline for benchmarking on the CQR dataset |
| `mobile_prompt_compression_system.py` | Core orchestrator that runs the 10-step query processing pipeline |
| `graph_manager.py` | Manages the NetworkX knowledge graph — add nodes, BFS traversal, edge management |
| `prompt_processor.py` | Extracts semantic units (sentences, key phrases, important words) from user input |
| `relevance_engine.py` | BFS-based relevant node discovery with cosine similarity filtering |
| `compression_engine.py` | Compresses retrieved context using extractive, abstractive, or hybrid strategies |
| `enhanced_causal_reasoner.py` | Detects causal relationships using ML models + pattern matching |
| `causal_detection.py` | Low-level causal edge classifier using RoBERTa-large-MNLI |
| `temporal_manager.py` | Handles temporal ordering of nodes and temporal edge detection |
| `temporal_reasoning.py` | NLI-based temporal relation detection (BEFORE/AFTER) |
| `coherence_analysis.py` | Pronoun resolution and discourse connective detection for the evaluation pipeline |
| `ollama_interface.py` | Wrapper for the Ollama REST API to interact with local LLMs |
| `mobile_optimizer.py` | Memory monitoring, graph pruning, and resource optimization for edge devices |
| `improved_config.py` | Configuration builder with device-aware settings |
| `mobile_config.py` | Configuration presets (mobile, high-performance, low-resource) |
| `custom_causal_training.py` | Fine-tunes a custom causal detection model on e-CARE + COPA datasets |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python 3.10+, FastAPI, Uvicorn |
| **Frontend** | React 18, TypeScript, Vite |
| **Graph Engine** | NetworkX (directed knowledge graphs) |
| **Embeddings** | Sentence-Transformers (`BAAI/bge-small-en-v1.5`) |
| **Causal & Temporal Detection** | HuggingFace Transformers (`FacebookAI/roberta-large-mnli`) |
| **QA & Entailment** | DeBERTa-v3 (QA + NLI), FLAN-T5 (answer rewriting) |
| **LLM Inference** | Ollama (local) with `phi3:mini` model |
| **Graph Visualization** | vis-network (interactive browser graph) |
| **Resource Monitoring** | psutil |

---

## Datasets & Models

### Datasets Used

| Dataset | Source | Used In | Purpose |
|---------|--------|---------|----------|
| **e-CARE** | `12ml/e-CARE` (HuggingFace) | `custom_causal_training.py` | Causal reasoning examples — premise/choice pairs with causal labels. Makes up **70%** of the custom training data. |
| **COPA** | `super_glue/copa` (HuggingFace) | `custom_causal_training.py` | Choice of Plausible Alternatives — cause/effect reasoning. Makes up **30%** of the custom training data. |
| **CQR** | `deadcode99/CQR` (HuggingFace) | `main.py` | Conversational Question Reformulation dataset used for evaluation and benchmarking. |

### Pre-trained Models

| Model | HuggingFace ID | Used In | Purpose |
|-------|----------------|---------|----------|
| BGE-Small | `BAAI/bge-small-en-v1.5` | `graph_manager.py`, `prompt_processor.py` | Sentence embeddings for the web app pipeline |
| BGE-Large | `BAAI/bge-large-en-v1.5` | `main.py` | Sentence embeddings for the evaluation pipeline |
| BGE-Base | `BAAI/bge-base-en-v1.5` | `main.py` | ⬆️ Fallback for BGE-Large if it fails to load |
| BGE-Reranker | `BAAI/bge-reranker-base` | `main.py` | Cross-encoder reranking of candidate nodes |
| RoBERTa-MNLI | `FacebookAI/roberta-large-mnli` | `causal_detection.py`, `temporal_reasoning.py` | Causal edge classification + temporal NLI |
| DeBERTa-v3-Large | `microsoft/deberta-v3-large` | `temporal_reasoning.py` | ⬆️ Alternative temporal model (can replace RoBERTa-MNLI) |
| DistilRoBERTa | `distilroberta-base` | `enhanced_causal_reasoner.py` | ⬆️ Lightweight fallback for causal detection if full model fails |
| DeBERTa-v3 QA | `microsoft/deberta-v3-base-squad2` | `main.py` | Extractive question answering |
| DeBERTa-v3 NLI | `cross-encoder/nli-deberta-v3-base` | `main.py` | Entailment verification of extracted answers |
| FLAN-T5 | `google/flan-t5-base` | `main.py` | Answer rewriting and reformulation |
| Phi-3 Mini | `phi3:mini` (via Ollama) | `ollama_interface.py` | Local LLM for response generation |

### Fallback Chains

The system includes graceful degradation — if a primary model fails to load, a lighter alternative is used:

- **Embeddings:** `BGE-Large` → `BGE-Base` (in `main.py` evaluation pipeline)
- **Causal detection:** `RoBERTa-MNLI` (full model) → `DistilRoBERTa` (lightweight) → pattern-based scoring (no model)
- **Temporal detection:** `RoBERTa-MNLI` (default) → `DeBERTa-v3-Large` (alternative, configurable)

### Custom Trained Model

| Model | Base Model | Training Data | Output |
|-------|-----------|---------------|--------|
| Custom Causal Detector | `FacebookAI/roberta-large-mnli` | 70% e-CARE + 30% COPA (5,000 examples, 3 epochs) | `./custom_causal_model/` (~1.4 GB) |

The custom model is a fine-tuned RoBERTa-large-MNLI trained to detect causal relationships between sentence pairs. It is used by `enhanced_causal_reasoner.py` alongside the base RoBERTa-MNLI and pattern-based detection for multi-approach causal reasoning.

---

## Requirements

### Python Dependencies

```
datasets>=2.20.0
sentence-transformers>=2.7.0
torch>=2.2.0
networkx>=3.2.1
transformers>=4.41.0
matplotlib>=3.8.0
fastapi
uvicorn
pydantic
requests
nltk
psutil
numpy
scikit-learn
```

### External Requirements

- **Python 3.10+** (uses modern type-hint syntax like `str | None`)
- **Node.js 18+** (for the React frontend, optional)
- **Ollama** installed and running locally ([ollama.com](https://ollama.com))
- **GPU recommended** (CUDA-compatible) — runs on CPU but significantly slower
- Internet access on first run (to download HuggingFace model weights and datasets)

---

## How to Run

### Prerequisites Checklist

Before starting, make sure you have:
- [ ] Python 3.10 or newer installed (`python --version`)
- [ ] Ollama installed ([download here](https://ollama.com))
- [ ] Node.js 18+ installed (only if using the React frontend)
- [ ] ~5 GB disk space for model weights (downloaded automatically on first run)

---

### Step 1: Set up the Python environment

Open a terminal in the project root:

```bash
# Create a virtual environment
python -m venv .venv

# Activate it
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
pip install fastapi uvicorn pydantic requests nltk psutil numpy scikit-learn
```

> **Note:** If you have a CUDA GPU, install the GPU version of PyTorch first from [pytorch.org](https://pytorch.org/get-started/locally/), then run the pip commands above.

---

### Step 2: Set up Ollama + pull the LLM model

#### Install Ollama (if not already installed)

1. Go to [https://ollama.com/download](https://ollama.com/download) and download the installer for your OS.
2. Run the installer:
   - **Windows:** Run the `.exe` installer and follow the prompts. Ollama will be added to your PATH automatically.
   - **macOS:** Open the `.dmg` and drag Ollama to Applications.
   - **Linux:** Run the one-line install script:
     ```bash
     curl -fsSL https://ollama.com/install.sh | sh
     ```
3. Verify the installation:
   ```bash
   ollama --version
   ```

> **Note:** On Windows, the Ollama desktop app starts automatically after installation and runs in the system tray. If you see `ollama serve` failing with a "port already in use" error, the server is already running in the background — you can skip the `ollama serve` step below.

#### Start the server and pull the model

You need **two terminals** for this step.

**Terminal 1 — Start the Ollama server:**

```bash
ollama serve
```

Leave this running. You should see `Listening on 127.0.0.1:11434`.

**Terminal 2 — Pull the phi3:mini model (one-time download, ~2.3 GB):**

```bash
ollama pull phi3:mini
```

Wait until it says `success`. This only needs to be done once.

---

### Step 3: Start the backend

In **Terminal 2** (with the venv activated):

```bash
python app.py
```

You will see:
```
==================================================
🚀 App starting at http://localhost:8000
==================================================
```

The first run will take a few minutes as it downloads the embedding model (`BAAI/bge-small-en-v1.5`) and the NLI model (`FacebookAI/roberta-large-mnli`) from HuggingFace. These are cached for subsequent runs.

---

### Step 4: Open the UI

You have two options:

**Option A — Simple HTML UI (no setup needed):**

Just open **http://localhost:8000** in your browser. This serves the built-in `index.html` with a chat panel and live knowledge graph.

**Option B — Full React UI (recommended, has pipeline trace + compare mode):**

Open a **Terminal 3**:

```bash
cd frontend
npm install        # first time only
npm run dev
```

Open **http://localhost:5173** in your browser. This gives you:
- **Chat panel** (left) — talk to the system
- **Pipeline trace** (center) — watch each of the 10 pipeline steps execute in real time with token counts
- **Knowledge graph** (right) — interactive vis-network graph that grows as you chat
- **Compare mode** — switch to side-by-side view showing compressed vs raw (no compression) responses and token savings

---

### Step 5: Run the evaluation pipeline (optional)

This is a separate benchmark script that evaluates the graph pipeline against the `deadcode99/CQR` dataset. It does NOT require Ollama.

```bash
# Quick test with 1 sample
python main.py --num-samples 1

# Full evaluation with 25 samples + graph exports
python main.py --num-samples 25 --export-graphml --export-png --graphs-dir graphs
```

**Common CLI options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--num-samples N` | Number of dataset rows to process | 10 |
| `--export-graphml` | Export graphs as GraphML files | off |
| `--export-png` | Export graph PNG snapshots | off |
| `--graphs-dir DIR` | Directory for graph exports | `graphs` |
| `--report-top-k K` | Top-K relevant nodes to report (0 = disable) | 3 |
| `--local-cosine-threshold` | Cosine similarity pruning threshold | 0.35 |
| `--mmr-lambda` | MMR diversity parameter | 0.35 |

Run `python main.py --help` for the full list with defaults.

---

### Step 6: Train the custom causal model (optional)

If you want to retrain the custom causal detection model from scratch:

```bash
python custom_causal_training.py
```

This will:
1. Download the **e-CARE** and **COPA** datasets from HuggingFace
2. Create a balanced dataset (70% e-CARE, 30% COPA, ~5,000 examples)
3. Fine-tune `FacebookAI/roberta-large-mnli` for 3 epochs
4. Save the trained model to `./custom_causal_model/`
5. Print evaluation metrics (accuracy, F1, precision, recall)

> **Note:** A pre-trained custom model is already included in `./custom_causal_model/`. You only need to retrain if you want to experiment with the training process.

---

## Processing Pipeline

When a user sends a message, the system executes this 10-step pipeline:

```
User Query
  │
  ├─ 1. Prompt Processor      → Clean text, extract semantic units, generate embedding
  ├─ 2. Graph Manager          → Add question node to the knowledge graph
  ├─ 3. Similarity Search      → Top-K cosine similarity against all existing nodes
  ├─ 4. Relevance Engine       → BFS traversal from seeds, prune by similarity threshold
  ├─ 5. Temporal Manager       → Order retrieved nodes chronologically
  ├─ 6. Compression Engine     → Compress context (extractive/abstractive/hybrid)
  ├─ 7. Ollama Interface       → Send compressed prompt to LLM → get response
  ├─ 8. Graph Manager          → Add response node + Q→R edge
  ├─ 9. Causal Reasoner        → Detect causal edges (RoBERTa-MNLI + custom model)
  └─ 10. Temporal Manager      → Detect temporal edges (NLI-based BEFORE/AFTER)
```

---

## API Endpoints

The backend runs on `http://localhost:8000` and exposes the following endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve the built-in HTML UI |
| `GET` | `/graph_state` | Get current graph nodes and edges as JSON |
| `POST` | `/chat` | Send a message, get response + graph update + metrics |
| `POST` | `/compare` | Run same prompt through compressed vs raw pipelines |
| `POST` | `/clear` | Reset all graph data and session state |
| `POST` | `/clear_compare` | Clear both compressed and raw pipelines |
| `GET` | `/test_pipeline` | Stream 15 test prompts through the pipeline (SSE) |

---

### `GET /`

Serves the built-in `index.html` — a standalone HTML UI with a chat panel and live knowledge graph. No frontend build step required.

---

### `GET /graph_state`

Returns the current state of the knowledge graph.

**Response:**

```json
{
  "nodes": [
    {
      "id": "q_1",
      "label": "What is photosynthesis and how do...",
      "full_text": "What is photosynthesis and how do plants use sunlight?",
      "group": "question",
      "timestamp": "2026-03-24T09:30:00"
    }
  ],
  "edges": [
    {
      "from": "q_1",
      "to": "r_1",
      "label": "has_response",
      "confidence": 1.0,
      "detection_method": ""
    }
  ]
}
```

Node groups include `question`, `response`, and `unknown`. Edge labels include `has_response`, `similar_to`, `causes`, `caused_by`, `before`, and `after`.

---

### `POST /chat`

The primary chat endpoint. Sends a user message through the full 10-step compression pipeline, gets an LLM response, updates the knowledge graph, and returns detailed metrics.

**Request body:**

```json
{
  "message": "What is photosynthesis?"
}
```

**Response:**

```json
{
  "success": true,
  "response": "Photosynthesis is the process by which...",
  "error": "",
  "graph": { "nodes": [...], "edges": [...] },
  "question_id": "q_1",
  "response_id": "r_1",
  "relevant_nodes_count": 3,
  "relevant_nodes_texts": ["previous relevant context..."],
  "compressed_prompt": "Context: ... Question: What is photosynthesis?",
  "relationship_stats": { "causal_edges": 1, "temporal_edges": 0 },
  "pipeline_trace": {
    "steps": [
      { "name": "Prompt Processor", "duration": 0.12, "details": "..." }
    ]
  },
  "token_metrics": {
    "input_tokens": 8,
    "context_tokens": 45,
    "total_original_tokens": 120,
    "compressed_tokens": 45,
    "compression_ratio": 62.5,
    "tokens_saved": 75
  },
  "graph_stats": { "total_nodes": 4, "total_edges": 5 },
  "processing_time": 2.34,
  "context_full": false
}
```

When `context_full` is `true`, the context window (default 4096 tokens) has been reached and the pipeline halts.

---

### `POST /compare`

Runs the same prompt through **two parallel pipelines** for side-by-side comparison:
- **Path A (Compressed):** Full GRACE pipeline — graph traversal, BFS retrieval, compression, then LLM.
- **Path B (Raw):** Naively accumulates the entire conversation history and sends it all to the LLM without compression.

This demonstrates the token savings and context window efficiency of graph-based compression.

**Request body:**

```json
{
  "message": "How does chlorophyll absorb light?"
}
```

**Response:**

```json
{
  "model": "phi3:mini",
  "prompt": "How does chlorophyll absorb light?",
  "compressed": {
    "response": "Chlorophyll absorbs light primarily in...",
    "time": 1.85,
    "input_tokens": 7,
    "context_tokens": 42,
    "total_original_tokens": 150,
    "compressed_tokens": 42,
    "compression_ratio": 72.0,
    "tokens_saved": 108,
    "bfs_nodes": 3,
    "pipeline_steps": 10,
    "pipeline_trace": { "steps": [...] },
    "context_full": false
  },
  "raw": {
    "response": "Chlorophyll is a green pigment that...",
    "time": 3.12,
    "token_metrics": {
      "input_tokens": 7,
      "context_tokens": 150,
      "total_original_tokens": 150,
      "compressed_tokens": 150,
      "compression_ratio": 0,
      "tokens_saved": 0,
      "max_allowed_tokens": 4096,
      "context_usage_percentage": 3.7
    },
    "history_length": 4,
    "context_full": false
  },
  "graph": { "nodes": [...], "edges": [...] },
  "graph_stats": { "total_nodes": 6, "total_edges": 8 }
}
```

---

### `POST /clear`

Resets the entire system — clears the knowledge graph, removes all nodes and edges, and starts a new session. Use this to begin a fresh conversation.

**Request body:** None

**Response:**

```json
{
  "success": true,
  "message": "All context cleared."
}
```

---

### `POST /clear_compare`

Same as `/clear`, but specifically resets both the compressed pipeline's graph **and** the raw pipeline's accumulated history used in compare mode.

**Request body:** None

**Response:**

```json
{
  "success": true
}
```

---

### `GET /test_pipeline`

Streams a pre-defined set of **15 test prompts** through the pipeline using **Server-Sent Events (SSE)**. The prompts are designed to test specific scenarios:

- **Cluster 1 (Photosynthesis):** Builds overlapping context for compression
- **Cluster 2 (Roman history):** Tests unrelated topic isolation
- **Cluster 3 (Water cycle):** Tests cross-domain linking via shared terms
- **Causal tests:** Prompts with explicit causal language ("leads to", "because")
- **Temporal tests:** Prompts with temporal markers ("first...then...finally")
- **Redundancy tests:** Near-duplicate prompts to test compression ratio

**Response:** `text/event-stream` — each event is a JSON payload:

```
data: {"index": 0, "total": 15, "prompt": "What is photosynthesis?", "success": true, "response": "...", ...}
data: {"index": 1, "total": 15, "prompt": "How does chlorophyll absorb light?", ...}
...
data: [DONE]
```

Each streamed event contains the same fields as the `/chat` response, plus `index` and `total` for progress tracking.

---

## Project Structure

```
GRACE/
├── app.py                              # FastAPI web server (main entry point)
├── mobile_prompt_compression_system.py # Core 10-step pipeline orchestrator
├── graph_manager.py                    # NetworkX knowledge graph engine
├── prompt_processor.py                 # Semantic unit extraction (NLP)
├── relevance_engine.py                 # BFS traversal + cosine similarity search
├── compression_engine.py               # Extractive/hybrid context compression
├── enhanced_causal_reasoner.py         # Multi-approach causal detection
├── causal_detection.py                 # RoBERTa-MNLI causal edge classifier
├── temporal_manager.py                 # Temporal ordering + edge detection
├── temporal_reasoning.py               # NLI-based temporal relation detection
├── coherence_analysis.py               # Pronoun resolution + connective detection
├── ollama_interface.py                 # Ollama REST API wrapper
├── mobile_optimizer.py                 # Memory monitoring + graph pruning
├── mobile_config.py                    # Device-specific config presets
├── improved_config.py                  # Adaptive configuration builder
├── custom_causal_training.py           # Fine-tune causal model (e-CARE + COPA)
├── main.py                             # Standalone evaluation pipeline (CQR dataset)
├── index.html                          # Simple HTML frontend (fallback UI)
├── requirements.txt                    # Python dependencies
├── requirements_mobile.txt             # Mobile-optimized dependencies
├── pyproject.toml                      # Project metadata + pytest config
├── LICENSE                             # MIT License
├── .gitignore                          # Git ignore rules
├── tests/
│   └── test_grace.py                   # 57 unit + integration tests
├── frontend/                           # React + TypeScript + Vite UI
│   ├── src/
│   │   ├── App.tsx                     # Main chat interface
│   │   ├── Compare.tsx                 # Side-by-side compressed vs raw
│   │   └── GraphVis.tsx                # Interactive graph visualization
│   └── package.json
├── custom_causal_model/                # Pre-trained causal detection model
│   ├── model.safetensors               # Model weights (~1.4 GB)
│   ├── config.json
│   └── tokenizer.json
└── graphs/                             # Generated graph visualizations
```

---

## Testing

The project includes a comprehensive test suite with **57 tests** covering all modules:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test class
python -m pytest tests/test_grace.py::TestGraphManager -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=term-missing
```

### Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| Graph Manager | 9 | Node CRUD, BFS traversal, persistence |
| Prompt Processor | 5 | Text cleaning, semantic extraction |
| Compression Engine | 3 | Extractive, hybrid, token limits |
| Temporal Manager | 4 | Ordering, relationship detection |
| Mobile Optimizer | 8 | Pruning, resource monitoring, adaptive tuning |
| Ollama Interface | 3 | Token counting accuracy |
| Configuration | 11 | Validation, device-specific, save/load |
| Causal Detection | 3 | Pattern matching, statistics |
| Coherence Analysis | 4 | Pronoun resolution, connectives |
| Training Pipeline | 4 | Data generation, mixed datasets |
| Integration | 2 | Full pipeline init, session management |

> **Note:** Tests mock all heavy ML dependencies (torch, transformers, sentence-transformers) so they run on any environment without GPU libraries.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
