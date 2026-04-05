"""Configuration management for mobile prompt compression system."""
from __future__ import annotations

from typing import Dict, Any, Optional
import json
import os
from pathlib import Path


class MobileConfig:
    """Configuration management for mobile deployment."""
    
    # Default mobile configuration
    MOBILE_CONFIG = {
        "models": {
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "mistral_model": "mistralai/Mistral-7B-v0.1-int4",
            "causal_model": "FacebookAI/roberta-large-mnli",
            "lightweight_model": "distilroberta-base",
            "summarizer_model": None
        },
        "thresholds": {
            "similarity_threshold": 0.7,
            "causal_confidence": 0.6,
            "temporal_confidence": 0.5
        },
        "limits": {
            "max_graph_nodes": 2000,  # Increased from 500
            "max_context_tokens": 512,
            "memory_limit_mb": 256,
            "max_bfs_depth": 5  # Increased from 3
        },
        "optimization": {
            "batch_size": 8,
            "quantization": "int4",
            "prune_interval": 50,
            "cache_embeddings": True
        },
        "deployment": {
            "device_type": "mobile",
            "hardware_acceleration": True,
            "offline_mode": True,
            "data_compression": True
        }
    }
    
    # High-performance mobile configuration
    HIGH_PERFORMANCE_CONFIG = {
        "models": {
            "embedding_model": "BAAI/bge-base-en-v1.5",
            "mistral_model": "mistralai/Mistral-7B-v0.1-int4",
            "causal_model": "FacebookAI/roberta-large-mnli",
            "lightweight_model": "distilroberta-base",
            "summarizer_model": "t5-small"
        },
        "thresholds": {
            "similarity_threshold": 0.6,
            "causal_confidence": 0.5,
            "temporal_confidence": 0.4
        },
        "limits": {
            "max_graph_nodes": 5000,  # Increased from 1000
            "max_context_tokens": 768,
            "memory_limit_mb": 512,
            "max_bfs_depth": 7  # Increased from 5
        },
        "optimization": {
            "batch_size": 16,
            "quantization": "int4",
            "prune_interval": 100,
            "cache_embeddings": True
        },
        "deployment": {
            "device_type": "mobile_high_performance",
            "hardware_acceleration": True,
            "offline_mode": True,
            "data_compression": True
        }
    }
    
    # Low-resource mobile configuration
    LOW_RESOURCE_CONFIG = {
        "models": {
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "mistral_model": "microsoft/DialoGPT-medium",  # Smaller model
            "causal_model": "distilroberta-base",
            "lightweight_model": "distilroberta-base",
            "summarizer_model": None
        },
        "thresholds": {
            "similarity_threshold": 0.8,
            "causal_confidence": 0.7,
            "temporal_confidence": 0.6
        },
        "limits": {
            "max_graph_nodes": 1000,  # Increased from 200
            "max_context_tokens": 256,
            "memory_limit_mb": 128,
            "max_bfs_depth": 3  # Kept at 3 for low-resource devices
        },
        "optimization": {
            "batch_size": 4,
            "quantization": "int8",
            "prune_interval": 25,
            "cache_embeddings": False
        },
        "deployment": {
            "device_type": "mobile_low_resource",
            "hardware_acceleration": False,
            "offline_mode": True,
            "data_compression": True
        }
    }
    
    @staticmethod
    def get_config(config_type: str = "mobile") -> Dict[str, Any]:
        """Get configuration by type."""
        configs = {
            "mobile": MobileConfig.MOBILE_CONFIG,
            "high_performance": MobileConfig.HIGH_PERFORMANCE_CONFIG,
            "low_resource": MobileConfig.LOW_RESOURCE_CONFIG
        }
        
        return configs.get(config_type, MobileConfig.MOBILE_CONFIG)
    
    @staticmethod
    def save_config(config: Dict[str, Any], filepath: str):
        """Save configuration to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
    
    @staticmethod
    def load_config(filepath: str) -> Optional[Dict[str, Any]]:
        """Load configuration from file."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config from {filepath}: {e}")
            return None
    
    @staticmethod
    def create_device_specific_config(device_memory_mb: int, has_gpu: bool = True) -> Dict[str, Any]:
        """Create configuration based on device specifications."""
        if device_memory_mb >= 6000:  # 6GB+ RAM
            base_config = MobileConfig.HIGH_PERFORMANCE_CONFIG.copy()
        elif device_memory_mb >= 3000:  # 3GB+ RAM
            base_config = MobileConfig.MOBILE_CONFIG.copy()
        else:  # Less than 3GB RAM
            base_config = MobileConfig.LOW_RESOURCE_CONFIG.copy()
        
        # Adjust based on GPU availability
        base_config["deployment"]["hardware_acceleration"] = has_gpu
        
        # Adjust memory limit
        if device_memory_mb < 1024:  # Less than 1GB
            base_config["limits"]["memory_limit_mb"] = min(128, device_memory_mb // 4)
        else:
            base_config["limits"]["memory_limit_mb"] = min(512, device_memory_mb // 6)
        
        return base_config
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration and return validation results."""
        errors = []
        warnings = []
        
        # Check required sections
        required_sections = ["models", "thresholds", "limits", "optimization", "deployment"]
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")
        
        # Validate thresholds
        if "thresholds" in config:
            thresholds = config["thresholds"]
            for key, value in thresholds.items():
                if not isinstance(value, (int, float)) or not (0 <= value <= 1):
                    errors.append(f"Invalid threshold for {key}: {value}")
        
        # Validate limits
        if "limits" in config:
            limits = config["limits"]
            if limits.get("memory_limit_mb", 0) < 64:
                errors.append("Memory limit too low (minimum 64MB)")
            if limits.get("max_graph_nodes", 0) < 10:
                errors.append("Max graph nodes too low (minimum 10)")
            if limits.get("max_context_tokens", 0) < 128:
                errors.append("Max context tokens too low (minimum 128)")
        
        # Validate optimization
        if "optimization" in config:
            optimization = config["optimization"]
            if optimization.get("batch_size", 0) < 1:
                errors.append("Batch size must be at least 1")
            if optimization.get("quantization") not in ["int4", "int8", "fp16", "fp32"]:
                warnings.append("Unknown quantization type")
        
        # Mobile-specific validations
        if "deployment" in config:
            deployment = config["deployment"]
            if deployment.get("device_type") not in ["mobile", "mobile_high_performance", "mobile_low_resource"]:
                warnings.append("Unknown device type")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    @staticmethod
    def get_optimization_suggestions(config: Dict[str, Any]) -> List[str]:
        """Get optimization suggestions based on current configuration."""
        suggestions = []
        
        # Memory optimization
        memory_limit = config.get("limits", {}).get("memory_limit_mb", 256)
        if memory_limit > 512:
            suggestions.append("Consider reducing memory limit for better mobile performance")
        
        # Graph size optimization
        max_nodes = config.get("limits", {}).get("max_graph_nodes", 500)
        if max_nodes > 1000:
            suggestions.append("Large graph size may impact mobile performance")
        
        # Batch size optimization
        batch_size = config.get("optimization", {}).get("batch_size", 8)
        if batch_size > 16:
            suggestions.append("Large batch size may cause memory issues on mobile")
        
        # Similarity threshold optimization
        similarity_threshold = config.get("thresholds", {}).get("similarity_threshold", 0.7)
        if similarity_threshold < 0.5:
            suggestions.append("Low similarity threshold may increase processing time")
        
        # Model optimization
        embedding_model = config.get("models", {}).get("embedding_model", "")
        if "base" in embedding_model and "small" not in embedding_model:
            suggestions.append("Consider using small embedding models for mobile")
        
        return suggestions
    
    @staticmethod
    def create_config_template() -> Dict[str, Any]:
        """Create a configuration template with comments."""
        template = {
            "_description": "Mobile Prompt Compression System Configuration",
            "_version": "1.0",
            "models": {
                "_description": "Model configurations",
                "embedding_model": "BAAI/bge-small-en-v1.5",
                "mistral_model": "mistralai/Mistral-7B-v0.1-int4",
                "causal_model": "FacebookAI/roberta-large-mnli",
                "lightweight_model": "distilroberta-base",
                "summarizer_model": None
            },
            "thresholds": {
                "_description": "Confidence thresholds for various operations",
                "similarity_threshold": 0.7,
                "causal_confidence": 0.6,
                "temporal_confidence": 0.5
            },
            "limits": {
                "_description": "Resource limits for mobile deployment",
                "max_graph_nodes": 500,
                "max_context_tokens": 512,
                "memory_limit_mb": 256,
                "max_bfs_depth": 3
            },
            "optimization": {
                "_description": "Performance optimization settings",
                "batch_size": 8,
                "quantization": "int4",
                "prune_interval": 50,
                "cache_embeddings": True
            },
            "deployment": {
                "_description": "Deployment-specific settings",
                "device_type": "mobile",
                "hardware_acceleration": True,
                "offline_mode": True,
                "data_compression": True
            }
        }
        
        return template
    
    @staticmethod
    def export_config_for_deployment(config: Dict[str, Any], deployment_dir: str):
        """Export configuration and related files for deployment."""
        deployment_path = Path(deployment_dir)
        deployment_path.mkdir(parents=True, exist_ok=True)
        
        # Save main config
        MobileConfig.save_config(config, str(deployment_path / "config.json"))
        
        # Save config template
        template = MobileConfig.create_config_template()
        MobileConfig.save_config(template, str(deployment_path / "config_template.json"))
        
        # Create deployment manifest
        manifest = {
            "config_version": "1.0",
            "created_at": "2025-02-26T11:47:00Z",
            "device_requirements": {
                "min_memory_mb": config["limits"]["memory_limit_mb"],
                "recommended_memory_mb": config["limits"]["memory_limit_mb"] * 2,
                "storage_required_mb": 2048,  # Estimated
                "hardware_acceleration": config["deployment"]["hardware_acceleration"]
            },
            "model_info": {
                "embedding_model": config["models"]["embedding_model"],
                "llm_model": config["models"]["mistral_model"],
                "quantization": config["optimization"]["quantization"]
            },
            "performance_expectations": {
                "max_inference_time_seconds": 2.0,
                "max_memory_usage_mb": config["limits"]["memory_limit_mb"],
                "expected_accuracy": "high"
            }
        }
        
        with open(deployment_path / "deployment_manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"Configuration exported to {deployment_path}")
        return str(deployment_path)


# Convenience functions
def get_mobile_config() -> Dict[str, Any]:
    """Get default mobile configuration."""
    return MobileConfig.get_config("mobile")


def get_high_performance_config() -> Dict[str, Any]:
    """Get high-performance mobile configuration."""
    return MobileConfig.get_config("high_performance")


def get_low_resource_config() -> Dict[str, Any]:
    """Get low-resource mobile configuration."""
    return MobileConfig.get_config("low_resource")


def create_custom_config(device_memory_mb: int, has_gpu: bool = True) -> Dict[str, Any]:
    """Create custom configuration based on device specs."""
    return MobileConfig.create_device_specific_config(device_memory_mb, has_gpu)
