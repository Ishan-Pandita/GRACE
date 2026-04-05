"""Comprehensive test suite for the GRACE project.

All heavy ML dependencies (torch, sentence_transformers, transformers) are
mocked at the module level so tests run on ANY Python environment without
needing GPU libraries installed.  Tests validate logic, data flow, and
configuration — not model inference.
"""

import pytest
import sys
import os
import json
import time
import types
import numpy as np
import networkx as nx
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timezone

# ============================================================
# Module-level mocks for heavy ML dependencies
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _create_mock_module(name, attrs=None):
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    return mod


# Mock torch
_mock_torch = _create_mock_module("torch", {
    "cuda": MagicMock(is_available=MagicMock(return_value=False)),
    "device": MagicMock,
    "Tensor": np.ndarray,
    "no_grad": MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock())),
    "softmax": MagicMock(),
    "sigmoid": MagicMock(),
    "float16": "float16",
})
_mock_torch.nn = _create_mock_module("torch.nn", {
    "Module": type("Module", (), {}),
    "functional": MagicMock(),
})
_mock_torch.nn.functional = MagicMock()
sys.modules.setdefault("torch", _mock_torch)
sys.modules.setdefault("torch.nn", _mock_torch.nn)
sys.modules.setdefault("torch.nn.functional", MagicMock())
sys.modules.setdefault("torch.utils", MagicMock())
sys.modules.setdefault("torch.utils.data", MagicMock())

# Mock sentence_transformers
_mock_st = _create_mock_module("sentence_transformers", {
    "SentenceTransformer": MagicMock,
    "CrossEncoder": MagicMock,
})
sys.modules.setdefault("sentence_transformers", _mock_st)

# Mock transformers
_mock_tf = _create_mock_module("transformers", {
    "AutoTokenizer": MagicMock,
    "AutoModelForSequenceClassification": MagicMock,
    "AutoModelForQuestionAnswering": MagicMock,
    "AutoModelForSeq2SeqLM": MagicMock,
    "Trainer": MagicMock,
    "TrainingArguments": MagicMock,
})
sys.modules.setdefault("transformers", _mock_tf)

# Mock datasets
sys.modules.setdefault("datasets", _create_mock_module("datasets", {
    "load_dataset": MagicMock(return_value={}),
}))

# Mock sklearn
_mock_sklearn = _create_mock_module("sklearn")
_mock_sklearn_ms = _create_mock_module("sklearn.model_selection", {
    "train_test_split": MagicMock(return_value=([], [])),
})
_mock_sklearn_m = _create_mock_module("sklearn.metrics", {
    "accuracy_score": MagicMock(return_value=0.0),
    "precision_recall_fscore_support": MagicMock(return_value=(0.0, 0.0, 0.0, None)),
})
sys.modules.setdefault("sklearn", _mock_sklearn)
sys.modules.setdefault("sklearn.model_selection", _mock_sklearn_ms)
sys.modules.setdefault("sklearn.metrics", _mock_sklearn_m)

# Mock bitsandbytes, accelerate
sys.modules.setdefault("bitsandbytes", MagicMock())
sys.modules.setdefault("accelerate", MagicMock())


# ============================================================
# 1. Graph Manager Tests
# ============================================================
class TestGraphManager:
    """Tests for graph_manager.py"""

    @pytest.fixture
    def mock_embedding_model(self):
        model = MagicMock()
        model.encode.return_value = np.random.rand(384).astype(np.float32)
        return model

    @pytest.fixture
    def graph_manager(self, mock_embedding_model):
        from graph_manager import GraphManager
        return GraphManager(mock_embedding_model, similarity_threshold=0.7)

    def test_init_creates_empty_graph(self, graph_manager):
        assert isinstance(graph_manager.graph, nx.DiGraph)
        assert len(graph_manager.graph.nodes) == 0

    def test_add_question_node(self, graph_manager):
        node = graph_manager.add_question_node("What is AI?", "s1", {})
        assert node is not None
        assert "id" in node
        assert node["type"] == "question"
        assert node["text"] == "What is AI?"
        assert len(graph_manager.graph.nodes) == 1

    def test_add_response_node(self, graph_manager):
        q = graph_manager.add_question_node("Q?", "s1", {})
        r = graph_manager.add_response_node("A", q["id"], {"session_id": "s1"})
        assert r is not None
        assert r["type"] == "response"
        assert len(graph_manager.graph.nodes) == 2
        assert graph_manager.graph.has_edge(q["id"], r["id"])

    def test_add_edge_valid(self, graph_manager):
        q = graph_manager.add_question_node("Q?", "s1", {})
        r = graph_manager.add_response_node("A", q["id"], {})
        ok = graph_manager.add_edge(q["id"], r["id"], "causal", confidence=0.9)
        assert ok is True

    def test_add_edge_nonexistent(self, graph_manager):
        ok = graph_manager.add_edge("x", "y", "causal", confidence=0.5)
        assert ok is False

    def test_bfs_traverse(self, graph_manager, mock_embedding_model):
        mock_embedding_model.encode.return_value = np.ones(384, dtype=np.float32)
        q1 = graph_manager.add_question_node("Q1", "s1", {})
        graph_manager.add_response_node("A1", q1["id"], {"session_id": "s1"})
        q2 = graph_manager.add_question_node("Q2", "s1", {})
        result = graph_manager.bfs_traverse_by_similarity(q2, threshold=0.5)
        assert isinstance(result, list)

    def test_get_graph_stats(self, graph_manager):
        stats = graph_manager.get_graph_stats()
        assert "total_nodes" in stats
        assert "total_edges" in stats
        assert stats["total_nodes"] == 0

    def test_multiple_nodes_and_edges(self, graph_manager):
        q1 = graph_manager.add_question_node("Q1?", "s1", {})
        r1 = graph_manager.add_response_node("A1", q1["id"], {})
        q2 = graph_manager.add_question_node("Q2?", "s1", {})
        r2 = graph_manager.add_response_node("A2", q2["id"], {})
        graph_manager.add_edge(q1["id"], q2["id"], "similarity", confidence=0.8)
        stats = graph_manager.get_graph_stats()
        assert stats["total_nodes"] == 4

    def test_save_and_load_graph(self, graph_manager, tmp_path):
        q = graph_manager.add_question_node("Q?", "s1", {})
        graph_manager.add_response_node("A", q["id"], {})
        fp = str(tmp_path / "g.json")
        graph_manager.save_graph(fp)
        assert os.path.exists(fp)

        from graph_manager import GraphManager
        gm2 = GraphManager(graph_manager.embedding_model, 0.7)
        gm2.load_graph(fp)
        assert len(gm2.graph.nodes) == len(graph_manager.graph.nodes)


# ============================================================
# 2. Prompt Processor Tests
# ============================================================
class TestPromptProcessor:
    """Tests for prompt_processor.py"""

    @pytest.fixture
    def processor(self):
        from prompt_processor import PromptProcessor
        model = MagicMock()
        model.encode.return_value = np.random.rand(384).astype(np.float32)
        return PromptProcessor(model)

    def test_process_input(self, processor):
        result = processor.extract_semantic_units("  Hello   World  \n\n  Test  ")
        assert isinstance(result, dict)
        assert "cleaned_text" in result
        assert len(result["cleaned_text"]) > 0

    def test_extract_semantic_units(self, processor):
        units = processor.extract_semantic_units("The cat sat on the mat. It was sunny.")
        assert isinstance(units, dict)
        assert "sentences" in units
        assert len(units["sentences"]) >= 1

    def test_create_question_node(self, processor):
        nd = processor.create_question_node("What is ML?", "s1")
        assert isinstance(nd, dict)
        assert "text" in nd

    def test_calculate_complexity(self, processor):
        result = processor.calculate_text_complexity("What is the capital of France? It is Paris.")
        assert isinstance(result, dict)

    def test_empty_input(self, processor):
        units = processor.extract_semantic_units("")
        assert isinstance(units, dict)


# ============================================================
# 3. Compression Engine Tests
# ============================================================
class TestCompressionEngine:
    """Tests for compression_engine.py"""

    @pytest.fixture
    def engine(self):
        from compression_engine import CompressionEngine
        return CompressionEngine(summarizer_model=None, max_tokens=512)

    def test_adaptive_compression(self, engine):
        nodes = [
            {"node_id": "n1", "node_data": {"text": "Climate is changing. Temperatures are rising."}},
            {"node_id": "n2", "node_data": {"text": "CO2 levels have increased dramatically."}},
        ]
        q = {"id": "q1", "text": "What causes climate change?", "embedding": np.random.rand(384).tolist()}
        result = engine.adaptive_compression(nodes, q)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_nodes(self, engine):
        q = {"id": "q1", "text": "What?", "embedding": np.random.rand(384).tolist()}
        result = engine.adaptive_compression([], q)
        assert isinstance(result, str)

    def test_compression_produces_output(self, engine):
        big = "This is a test sentence. " * 200
        nodes = [{"node_id": "n1", "node_data": {"text": big}}]
        q = {"id": "q1", "text": "Test?", "embedding": np.random.rand(384).tolist()}
        result = engine.adaptive_compression(nodes, q)
        # Compression produces a prompt string (may include template overhead)
        assert isinstance(result, str)
        assert len(result) > 0


# ============================================================
# 4. Temporal Manager Tests
# ============================================================
class TestTemporalManager:
    """Tests for temporal_manager.py"""

    @pytest.fixture
    def manager(self):
        from temporal_manager import TemporalManager
        return TemporalManager()

    def test_order_nodes(self, manager):
        nodes = [
            {"text": "B", "timestamp": "2025-01-02T00:00:00Z", "created_at": 1000},
            {"text": "A", "timestamp": "2025-01-01T00:00:00Z", "created_at": 500},
            {"text": "C", "timestamp": "2025-01-03T00:00:00Z", "created_at": 1500},
        ]
        ordered = manager.order_nodes_by_timestamp(nodes)
        assert isinstance(ordered, list)
        assert len(ordered) == 3

    def test_empty(self, manager):
        assert manager.order_nodes_by_timestamp([]) == []

    def test_detect_temporal(self, manager):
        n1 = {"id": "n1", "text": "Yesterday", "timestamp": "2025-01-01T10:00:00Z", "created_at": time.time() - 3600}
        n2 = {"id": "n2", "text": "Today", "timestamp": "2025-01-02T10:00:00Z", "created_at": time.time()}
        result = manager.detect_temporal_relationships(n1, n2)
        assert isinstance(result, list)

    def test_temporal_summary(self, manager):
        g = nx.DiGraph()
        g.add_node("n1", text="Hello", type="question", timestamp="2025-01-01")
        summary = manager.get_temporal_summary(g)
        assert isinstance(summary, dict)


# ============================================================
# 5. Mobile Optimizer Tests
# ============================================================
class TestMobileOptimizer:
    """Tests for mobile_optimizer.py"""

    @pytest.fixture
    def optimizer(self):
        from mobile_optimizer import MobileOptimizer
        return MobileOptimizer(memory_limit_mb=512, max_graph_nodes=100)

    def test_init(self, optimizer):
        assert optimizer.memory_limit == 512 * 1024 * 1024
        assert optimizer.max_graph_nodes == 100

    def test_under_limit(self, optimizer):
        g = nx.DiGraph()
        for i in range(10):
            g.add_node(f"n{i}", text=f"T{i}", type="response", created_at=time.time())
        g2, removed = optimizer.optimize_graph_size(g)
        assert removed == 0
        assert len(g2.nodes) == 10

    def test_over_limit(self, optimizer):
        g = nx.DiGraph()
        for i in range(150):
            g.add_node(f"n{i}", text=f"T{i}", type="response", created_at=time.time() - i * 60)
        g2, removed = optimizer.optimize_graph_size(g)
        assert removed > 0
        assert len(g2.nodes) <= 100

    def test_resources(self, optimizer):
        r = optimizer.monitor_resources()
        assert isinstance(r, dict)
        assert "memory_usage" in r or "error" in r

    def test_mobile_optimize(self, optimizer):
        r = optimizer.optimize_for_mobile()
        assert "applied_optimizations" in r

    def test_report(self, optimizer):
        r = optimizer.get_optimization_report()
        assert "current_parameters" in r
        assert "optimization_statistics" in r

    def test_adaptive_optimization(self, optimizer):
        metrics = {"memory_percent": 90, "cpu_percent": 50}
        result = optimizer.adaptive_optimization(metrics)
        assert result == "emergency_memory_cleanup"

    def test_node_importance_calculation(self, optimizer):
        g = nx.DiGraph()
        g.add_node("n1", type="question", created_at=time.time())
        g.add_node("n2", type="response", created_at=time.time() - 86400)
        g.add_edge("n1", "n2", confidence=0.9)
        score1 = optimizer._calculate_node_importance("n1", dict(g.nodes["n1"]), g)
        score2 = optimizer._calculate_node_importance("n2", dict(g.nodes["n2"]), g)
        assert isinstance(score1, float)
        assert isinstance(score2, float)


# ============================================================
# 6. Ollama Interface Tests
# ============================================================
class TestOllamaInterface:
    """Tests for ollama_interface.py"""

    def _make_interface(self):
        from ollama_interface import OllamaInterface
        obj = OllamaInterface.__new__(OllamaInterface)
        obj.session = MagicMock()
        obj.base_url = "http://localhost:11434"
        obj.model_name = "test"
        return obj

    def test_count_tokens(self):
        iface = self._make_interface()
        assert iface.count_tokens("Hello world this is a test") >= 6

    def test_count_tokens_empty(self):
        iface = self._make_interface()
        assert iface.count_tokens("") >= 1

    def test_count_single_word(self):
        iface = self._make_interface()
        assert iface.count_tokens("Hello") >= 1


# ============================================================
# 7. Configuration Tests
# ============================================================
class TestConfiguration:
    """Tests for mobile_config.py and improved_config.py"""

    def test_mobile_config_sections(self):
        from mobile_config import MobileConfig
        cfg = MobileConfig.get_config("mobile")
        for s in ("models", "thresholds", "limits", "optimization", "deployment"):
            assert s in cfg

    def test_high_perf(self):
        from mobile_config import MobileConfig
        cfg = MobileConfig.get_config("high_performance")
        assert cfg["limits"]["max_graph_nodes"] >= 5000

    def test_low_resource(self):
        from mobile_config import MobileConfig
        cfg = MobileConfig.get_config("low_resource")
        assert cfg["limits"]["max_graph_nodes"] <= 1000

    def test_validate_valid(self):
        from mobile_config import MobileConfig
        r = MobileConfig.validate_config(MobileConfig.get_config("mobile"))
        assert r["valid"] is True

    def test_validate_invalid_threshold(self):
        from mobile_config import MobileConfig
        cfg = MobileConfig.get_config("mobile")
        cfg["thresholds"]["similarity_threshold"] = 5.0
        r = MobileConfig.validate_config(cfg)
        assert r["valid"] is False

    def test_device_specific(self):
        from mobile_config import MobileConfig
        cfg = MobileConfig.create_device_specific_config(2048, has_gpu=False)
        assert cfg["deployment"]["hardware_acceleration"] is False

    def test_improved_config(self):
        from improved_config import get_improved_config
        cfg = get_improved_config(device_memory_mb=4096)
        assert "causal_detection" in cfg
        assert "traversal" in cfg

    def test_validate_improvements(self):
        from improved_config import get_improved_config, validate_improvements
        cfg = get_improved_config(device_memory_mb=8192)
        r = validate_improvements(cfg)
        assert "concerns_addressed" in r
        assert r["overall_score"] > 0

    def test_save_load(self, tmp_path):
        from mobile_config import MobileConfig
        cfg = MobileConfig.get_config("mobile")
        fp = str(tmp_path / "cfg.json")
        MobileConfig.save_config(cfg, fp)
        loaded = MobileConfig.load_config(fp)
        assert loaded is not None
        assert loaded["limits"]["max_context_tokens"] == cfg["limits"]["max_context_tokens"]

    def test_config_template(self):
        from mobile_config import MobileConfig
        tpl = MobileConfig.create_config_template()
        assert "_version" in tpl
        assert "models" in tpl

    def test_optimization_suggestions(self):
        from mobile_config import MobileConfig
        cfg = MobileConfig.get_config("high_performance")
        cfg["limits"]["max_graph_nodes"] = 9999
        suggestions = MobileConfig.get_optimization_suggestions(cfg)
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0


# ============================================================
# 8. Relevance Engine Tests
# ============================================================
class TestRelevanceEngine:
    """Tests for relevance_engine.py"""

    @pytest.fixture
    def engine(self):
        from relevance_engine import RelevanceEngine
        return RelevanceEngine(similarity_threshold=0.7, max_depth=3)

    def test_init(self, engine):
        assert engine.similarity_threshold == 0.7
        assert engine.max_depth == 3


# ============================================================
# 9. Causal Detection Tests (pattern-matching only, no models)
# ============================================================
class TestCausalDetection:
    """Tests for enhanced_causal_reasoner.py pattern matching."""

    @pytest.fixture
    def reasoner(self):
        from enhanced_causal_reasoner import EnhancedCausalReasoner
        r = EnhancedCausalReasoner(
            use_full_model=False,
            use_custom_model=False,
        )
        return r

    def test_pattern_match_causal(self, reasoner):
        score = reasoner._pattern_match_score(
            "Because the rain was heavy", "the streets were flooded"
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_pattern_match_no_causal(self, reasoner):
        score = reasoner._pattern_match_score(
            "The cat sat on the mat", "The weather was nice"
        )
        assert isinstance(score, float)
        assert score == 0.0 or score < 0.3

    def test_causal_statistics(self, reasoner):
        g = nx.DiGraph()
        g.add_node("n1", text="A")
        g.add_node("n2", text="B")
        g.add_edge("n1", "n2", type="causal", confidence=0.8)
        stats = reasoner.get_causal_statistics(g)
        assert isinstance(stats, dict)
        assert stats["total_causal_edges"] == 1
        assert stats["total_edges"] == 1
        assert stats["avg_causal_confidence"] == 0.8


# ============================================================
# 10. Coherence Analysis Tests
# ============================================================
class TestCoherenceAnalysis:
    """Tests for coherence_analysis.py"""

    def test_basic(self):
        from coherence_analysis import run_coherence_analysis
        g = nx.DiGraph()
        g.add_node("q1", text="Who discovered America?")
        g.add_node("s1", text="Christopher Columbus sailed west in 1492.")
        g.add_node("s2", text="He reached the Caribbean islands.")
        g.add_edge("q1", "s1")
        g.add_edge("s1", "s2")
        result = run_coherence_analysis(g, query_id="q1", question_text="Who discovered America?")
        assert "stats" in result
        assert "coref_links" in result
        assert "resolved_sentences" in result

    def test_pronoun_resolution(self):
        from coherence_analysis import run_coherence_analysis
        g = nx.DiGraph()
        g.add_node("q", text="Tell me about Einstein")
        g.add_node("s1", text="Albert Einstein was a physicist.")
        g.add_node("s2", text="He developed the theory of relativity.")
        g.add_edge("q", "s1")
        g.add_edge("s1", "s2")
        result = run_coherence_analysis(g, query_id="q", question_text="Tell me about Einstein")
        assert result["stats"]["num_sentences"] == 2

    def test_empty_graph(self):
        from coherence_analysis import run_coherence_analysis
        g = nx.DiGraph()
        result = run_coherence_analysis(g, query_id="q1")
        assert result["stats"]["num_sentences"] == 0

    def test_connective_detection(self):
        from coherence_analysis import run_coherence_analysis
        g = nx.DiGraph()
        g.add_node("q", text="Explain")
        g.add_node("s1", text="The economy grew in Q1.")
        g.add_node("s2", text="However, inflation also increased.")
        g.add_edge("q", "s1")
        g.add_edge("s1", "s2")
        result = run_coherence_analysis(g, query_id="q")
        assert isinstance(result["connective_links"], list)


# ============================================================
# 11. Custom Causal Training Tests (data generation only)
# ============================================================
class TestCausalTrainingData:
    """Tests for custom_causal_training.py data generation."""

    def test_causal_data_generator(self):
        from custom_causal_training import CausalDataGenerator
        gen = CausalDataGenerator()
        causal = gen.generate_causal_examples(20)
        assert len(causal) == 20
        assert all(ex["label"] == 1 for ex in causal)

    def test_non_causal_data(self):
        from custom_causal_training import CausalDataGenerator
        gen = CausalDataGenerator()
        non_causal = gen.generate_non_causal_examples(20)
        assert len(non_causal) == 20
        assert all(ex["label"] == 0 for ex in non_causal)

    def test_mixed_dataset(self):
        from custom_causal_training import CausalDataGenerator
        gen = CausalDataGenerator()
        mixed = gen.generate_mixed_dataset(10, 10)
        assert len(mixed) == 20
        labels = [ex["label"] for ex in mixed]
        assert 0 in labels and 1 in labels

    def test_domain_specific_data(self):
        from custom_causal_training import create_domain_specific_data
        data = create_domain_specific_data()
        assert len(data) > 0
        assert all("text1" in ex and "text2" in ex for ex in data)


# ============================================================
# 12. Integration / Pipeline Tests
# ============================================================
class TestMobileSystem:
    """Integration tests for mobile_prompt_compression_system.py"""

    def test_init_with_mocked_models(self):
        from mobile_prompt_compression_system import MobilePromptCompressionSystem
        with patch("mobile_prompt_compression_system.SentenceTransformer") as mock_st, \
             patch("mobile_prompt_compression_system.MobileOptimizer") as mock_mo:
            mock_st.return_value = MagicMock()
            mock_st.return_value.encode.return_value = np.random.rand(384).astype(np.float32)
            mock_mo_inst = MagicMock()
            mock_mo_inst.optimize_for_mobile.return_value = {"applied_optimizations": []}
            mock_mo.return_value = mock_mo_inst
            config = {
                "models": {
                    "embedding_model": "test",
                    "mistral_model": "test",
                    "causal_model": "test",
                    "lightweight_model": "test",
                    "summarizer_model": None,
                },
                "thresholds": {"similarity_threshold": 0.7, "causal_confidence": 0.6, "temporal_confidence": 0.5},
                "limits": {"max_graph_nodes": 50, "max_context_tokens": 256, "memory_limit_mb": 128, "max_bfs_depth": 3},
                "optimization": {"batch_size": 4, "quantization": "int4", "prune_interval": 50, "cache_embeddings": True},
                "use_ollama": False,
            }
            system = MobilePromptCompressionSystem(config)
            assert system is not None
            assert len(system.active_sessions) == 0
            status = system.get_system_status()
            assert "graph_stats" in status
            system.shutdown()

    def test_session_management(self):
        from mobile_prompt_compression_system import MobilePromptCompressionSystem
        with patch("mobile_prompt_compression_system.SentenceTransformer") as mock_st, \
             patch("mobile_prompt_compression_system.MobileOptimizer") as mock_mo:
            mock_st.return_value = MagicMock()
            mock_st.return_value.encode.return_value = np.random.rand(384).astype(np.float32)
            mock_mo_inst = MagicMock()
            mock_mo_inst.optimize_for_mobile.return_value = {"applied_optimizations": []}
            mock_mo.return_value = mock_mo_inst
            config = {
                "models": {"embedding_model": "t", "mistral_model": "t", "causal_model": "t", "lightweight_model": "t", "summarizer_model": None},
                "thresholds": {"similarity_threshold": 0.7, "causal_confidence": 0.6, "temporal_confidence": 0.5},
                "limits": {"max_graph_nodes": 50, "max_context_tokens": 256, "memory_limit_mb": 128, "max_bfs_depth": 3},
                "optimization": {"batch_size": 4, "quantization": "int4", "prune_interval": 50, "cache_embeddings": True},
                "use_ollama": False,
            }
            system = MobilePromptCompressionSystem(config)
            info = system.get_session_info("nonexistent")
            assert "error" in info
            ok = system.clear_session("nonexistent")
            assert ok is True
            system.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
