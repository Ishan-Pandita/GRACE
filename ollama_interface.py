"""Ollama interface for Mistral 7B integration."""
from __future__ import annotations

import requests
import json
import time
from typing import Dict, List, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaInterface:
    """Interface for Ollama API integration."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "mistral"):
        self.base_url = base_url
        self.model_name = model_name
        self.session = requests.Session()
        
    def test_connection(self) -> bool:
        """Test connection to Ollama server."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False
    
    def generate_response(self, prompt: str, temperature: float = 0.7, 
                         max_tokens: int = 150) -> Dict[str, Any]:
        """Generate response from Ollama model."""
        start_time = time.time()
        print(f"🔍 Ollama: Starting generation for {len(prompt)} chars, max_tokens={max_tokens}")
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": "You are a helpful and precise assistant. Answer the question based on the provided context. Follow all formatting and length instructions provided in the prompt.",
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_k": 40,
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120  # Increased from 60 to 120 seconds
            )
            
            if response.status_code == 200:
                result = response.json()
                end_time = time.time()
                generation_time = end_time - start_time
                
                print(f"🔍 Ollama: Generation completed in {generation_time:.2f}s")
                
                return {
                    "success": True,
                    "response": result.get("response", ""),
                    "prompt_eval_count": result.get("prompt_eval_count", 0),
                    "eval_count": result.get("eval_count", 0),
                    "generation_time": generation_time,
                    "model": self.model_name
                }
            else:
                end_time = time.time()
                generation_time = end_time - start_time
                print(f"🔍 Ollama: Failed in {generation_time:.2f}s, status: {response.status_code}")
                
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "generation_time": generation_time
                }
                
        except requests.exceptions.Timeout as e:
            end_time = time.time()
            generation_time = end_time - start_time
            print(f"🔍 Ollama: TIMEOUT after {generation_time:.2f}s")
            
            return {
                "success": False,
                "error": f"Timeout after {generation_time:.2f}s: {str(e)}",
                "generation_time": generation_time
            }
        except Exception as e:
            end_time = time.time()
            generation_time = end_time - start_time
            print(f"🔍 Ollama: ERROR after {generation_time:.2f}s: {e}")
            
            return {
                "success": False,
                "error": f"Generation failed: {str(e)}",
                "generation_time": generation_time
            }
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count using word-based BPE approximation.
        
        Sub-word tokenizers (BPE/SentencePiece) typically produce ~1.3x the
        word count for English text. This is more accurate than char/4.
        """
        words = text.split()
        return max(1, int(len(words) * 1.3))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                for model in models:
                    if model["name"].startswith(self.model_name):
                        return {
                            "name": model["name"],
                            "size": model["size"],
                            "modified_at": model["modified_at"],
                            "digest": model["digest"]
                        }
            return {"error": "Model not found"}
        except Exception as e:
            return {"error": str(e)}


class OllamaLLMInterface:
    """LLM interface compatible with mobile prompt compression system."""
    
    def __init__(self, model_name: str = "mistral", base_url: str = "http://localhost:11434"):
        self.ollama = OllamaInterface(base_url, model_name)
        self.model_name = model_name
        
        # Test connection
        if not self.ollama.test_connection():
            raise ConnectionError(f"Cannot connect to Ollama at {base_url}")
        
        logger.info(f"Connected to Ollama with model: {model_name}")
    
    def load_model(self):
        """Load model (placeholder for compatibility)."""
        logger.info(f"Model {self.model_name} is already loaded in Ollama")
    
    async def generate_response(self, prompt: str, max_tokens: int = 150, 
                               temperature: float = 0.7) -> str:
        """Generate response (async-compatible interface)."""
        result = self.ollama.generate_response(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        if result["success"]:
            return result["response"]
        else:
            raise Exception(f"Generation failed: {result.get('error', 'Unknown error')}")
    
    def unload_model(self):
        """Unload model (placeholder for compatibility)."""
        logger.info("Model unloading not needed with Ollama")
    
    def get_token_usage(self, prompt: str, response: str) -> Dict[str, int]:
        """Get token usage statistics."""
        prompt_tokens = self.ollama.count_tokens(prompt)
        response_tokens = self.ollama.count_tokens(response)
        
        return {
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": prompt_tokens + response_tokens
        }


# Convenience function for quick setup
def create_ollama_interface(model_name: str = "mistral") -> OllamaLLMInterface:
    """Create and initialize Ollama interface."""
    return OllamaLLMInterface(model_name=model_name)


if __name__ == "__main__":
    # Test Ollama interface
    print("🧪 Testing Ollama Interface")
    
    try:
        interface = create_ollama_interface("mistral")
        
        # Test model info
        model_info = interface.ollama.get_model_info()
        print(f"Model info: {model_info}")
        
        # Test generation
        test_prompt = "What is the capital of France? Give a brief answer."
        print(f"Testing with prompt: {test_prompt}")
        
        result = interface.ollama.generate_response(test_prompt)
        
        if result["success"]:
            print(f"✅ Response: {result['response']}")
            print(f"📊 Tokens: {result['prompt_eval_count']} prompt, {result['eval_count']} response")
            print(f"⏱️  Time: {result['response_time']:.2f}s")
        else:
            print(f"❌ Error: {result['error']}")
            
    except Exception as e:
        print(f"❌ Failed to test Ollama interface: {e}")
        print("Make sure Ollama is running: ollama serve")
        print("And Mistral is downloaded: ollama pull mistral")
