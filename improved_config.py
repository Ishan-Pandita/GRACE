"""Improved configuration addressing the three key concerns."""
from __future__ import annotations

from typing import Dict, Any, Optional
from mobile_config import MobileConfig


def get_improved_config(device_memory_mb: int = 4096, has_gpu: bool = True, 
                       use_full_causal_model: bool = True, 
                       adaptive_node_limit: bool = True) -> Dict[str, Any]:
    """Get improved configuration addressing the three concerns."""
    
    # Base configuration
    if device_memory_mb >= 6000:
        base_config = MobileConfig.HIGH_PERFORMANCE_CONFIG.copy()
        max_nodes = 5000
        max_depth = 7
    elif device_memory_mb >= 3000:
        base_config = MobileConfig.MOBILE_CONFIG.copy()
        max_nodes = 2000
        max_depth = 5
    else:
        base_config = MobileConfig.LOW_RESOURCE_CONFIG.copy()
        max_nodes = 1000
        max_depth = 3
    
    # Adaptive node limits based on memory
    if adaptive_node_limit:
        # Scale nodes based on available memory (rough estimate: 1KB per node)
        estimated_nodes = min(device_memory_mb * 512, max_nodes)  # 512 nodes per MB
        base_config["limits"]["max_graph_nodes"] = max(100, estimated_nodes)
    else:
        base_config["limits"]["max_graph_nodes"] = max_nodes
    
    # Enhanced causal detection settings
    base_config["causal_detection"] = {
        "use_full_model": use_full_causal_model,
        "model_confidence_threshold": 0.6,
        "always_use_full_model": True,  # Address concern #1
        "fallback_to_lightweight": True,
        "use_custom_model": False  # Custom trained models optional
    }
    
    # Enhanced traversal settings
    base_config["traversal"] = {
        "explicit_question_comparison": True,  # Address concern #3
        "similarity_threshold": 0.7,
        "explore_near_threshold": True,  # Explore nodes at 80% of threshold
        "track_similarity_comparisons": True
    }
    
    # Memory-aware settings
    base_config["limits"]["memory_limit_mb"] = min(device_memory_mb // 4, 512)
    base_config["limits"]["max_bfs_depth"] = max_depth
    
    # Performance optimizations
    base_config["optimization"]["batch_size"] = min(32, max(4, device_memory_mb // 256))
    
    return base_config


def create_high_performance_research_config() -> Dict[str, Any]:
    """Configuration for research/high-performance use cases."""
    config = get_improved_config(
        device_memory_mb=8192,  # 8GB
        has_gpu=True,
        use_full_causal_model=True,
        adaptive_node_limit=True
    )
    
    # Research-specific settings
    config.update({
        "research_mode": True,
        "limits": {
            **config["limits"],
            "max_graph_nodes": 10000,  # Much higher for research
            "max_context_tokens": 1024,
            "max_bfs_depth": 10
        },
        "thresholds": {
            **config["thresholds"],
            "similarity_threshold": 0.5,  # Lower for more comprehensive search
            "causal_confidence": 0.4,
            "temporal_confidence": 0.3
        },
        "detailed_logging": True,
        "save_intermediate_results": True
    })
    
    return config


def create_production_mobile_config() -> Dict[str, Any]:
    """Configuration for production mobile deployment."""
    config = get_improved_config(
        device_memory_mb=2048,  # 2GB typical mobile
        has_gpu=False,  # Conservative assumption
        use_full_causal_model=False,  # Use lightweight for production
        adaptive_node_limit=True
    )
    
    # Production-specific settings
    config.update({
        "production_mode": True,
        "limits": {
            **config["limits"],
            "max_graph_nodes": 1500,  # Conservative for production
            "max_context_tokens": 384,
            "max_bfs_depth": 4
        },
        "thresholds": {
            **config["thresholds"],
            "similarity_threshold": 0.75,  # Higher for faster processing
            "causal_confidence": 0.7,
            "temporal_confidence": 0.6
        },
        "optimization": {
            **config["optimization"],
            "aggressive_pruning": True,
            "cache_embeddings": True,
            "batch_size": 8
        },
        "detailed_logging": False,
        "performance_monitoring": True
    })
    
    return config


def validate_improvements(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that the improvements address the three concerns."""
    validation_results = {
        "concerns_addressed": {},
        "recommendations": [],
        "warnings": []
    }
    
    # Concern 1: Causal detection model usage
    causal_config = config.get("causal_detection", {})
    if causal_config.get("use_full_model", False):
        validation_results["concerns_addressed"]["causal_model_usage"] = "✅ Full model enabled"
    else:
        validation_results["concerns_addressed"]["causal_model_usage"] = "⚠️ Using lightweight model only"
        validation_results["recommendations"].append("Consider enabling full causal model for better accuracy")
    
    # Concern 2: Node limits
    max_nodes = config.get("limits", {}).get("max_graph_nodes", 500)
    if max_nodes >= 1000:
        validation_results["concerns_addressed"]["node_limits"] = f"✅ Increased to {max_nodes} nodes"
    elif max_nodes >= 500:
        validation_results["concerns_addressed"]["node_limits"] = f"⚠️ Moderate: {max_nodes} nodes"
        validation_results["recommendations"].append("Consider increasing max_graph_nodes for better coverage")
    else:
        validation_results["concerns_addressed"]["node_limits"] = f"❌ Low: {max_nodes} nodes"
        validation_results["warnings"].append("Very low node limit may impact system effectiveness")
    
    # Concern 3: Question node comparison
    traversal_config = config.get("traversal", {})
    if traversal_config.get("explicit_question_comparison", False):
        validation_results["concerns_addressed"]["question_comparison"] = "✅ Explicit comparison enabled"
    else:
        validation_results["concerns_addressed"]["question_comparison"] = "❌ No explicit comparison setting"
        validation_results["recommendations"].append("Enable explicit_question_comparison for proper traversal")
    
    # Overall assessment
    total_concerns = len(validation_results["concerns_addressed"])
    addressed_concerns = sum(1 for result in validation_results["concerns_addressed"].values() if "✅" in result)
    
    validation_results["overall_score"] = addressed_concerns / total_concerns
    validation_results["summary"] = f"{addressed_concerns}/{total_concerns} concerns properly addressed"
    
    return validation_results


# Example usage function
def demonstrate_improvements():
    """Demonstrate the improvements in action."""
    print("=== IMPROVEMENTS DEMONSTRATION ===\n")
    
    # High-performance research config
    research_config = create_high_performance_research_config()
    research_validation = validate_improvements(research_config)
    
    print("🔬 RESEARCH CONFIGURATION:")
    print(f"Max nodes: {research_config['limits']['max_graph_nodes']}")
    print(f"Full causal model: {research_config['causal_detection']['use_full_model']}")
    print(f"Explicit comparison: {research_config['traversal']['explicit_question_comparison']}")
    print(f"Validation: {research_validation['summary']}")
    print()
    
    # Production mobile config
    mobile_config = create_production_mobile_config()
    mobile_validation = validate_improvements(mobile_config)
    
    print("📱 PRODUCTION MOBILE CONFIGURATION:")
    print(f"Max nodes: {mobile_config['limits']['max_graph_nodes']}")
    print(f"Full causal model: {mobile_config['causal_detection']['use_full_model']}")
    print(f"Explicit comparison: {mobile_config['traversal']['explicit_question_comparison']}")
    print(f"Validation: {mobile_validation['summary']}")
    print()
    
    # Adaptive config based on device
    adaptive_config = get_improved_config(device_memory_mb=3072, has_gpu=True)
    adaptive_validation = validate_improvements(adaptive_config)
    
    print("🔄 ADAPTIVE CONFIGURATION (3GB device):")
    print(f"Max nodes: {adaptive_config['limits']['max_graph_nodes']}")
    print(f"Full causal model: {adaptive_config['causal_detection']['use_full_model']}")
    print(f"Explicit comparison: {adaptive_config['traversal']['explicit_question_comparison']}")
    print(f"Validation: {adaptive_validation['summary']}")
    
    if mobile_validation["recommendations"]:
        print(f"\n📋 RECOMMENDATIONS:")
        for rec in mobile_validation["recommendations"]:
            print(f"  • {rec}")
    
    if mobile_validation["warnings"]:
        print(f"\n⚠️ WARNINGS:")
        for warning in mobile_validation["warnings"]:
            print(f"  • {warning}")


if __name__ == "__main__":
    demonstrate_improvements()
