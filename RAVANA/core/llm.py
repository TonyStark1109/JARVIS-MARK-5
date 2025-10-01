"""
RAVANA LLM Manager
"""

import sys
import os
import logging
import json
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class LLMManager:
    """Manages LLM interactions."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = self.load_config()
        self.api_keys = self.get_gemini_keys()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration."""
        try:
            with open('config/config.json', 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            self.logger.error(f"Config load error: {e}")
            return {}
    
    def get_gemini_keys(self) -> List[str]:
        """Get Gemini API keys."""
        try:
            gemini_config = self.config.get('gemini', {})
            api_keys = gemini_config.get('api_keys', [])
            
            # Also check environment variables for additional keys
            env_keys = []
            for i in range(1, 21):  # Check for up to 20 environment variables
                env_key = os.getenv(f"GEMINI_API_KEY_{i}")
                if env_key:
                    env_keys.append(env_key)
            
            all_keys = api_keys + env_keys
            self.logger.info(f"Loaded {len(all_keys)} Gemini API keys")
            return all_keys
            
        except Exception as e:
            self.logger.error(f"Gemini keys error: {e}")
            return []

    def generate_response(self, prompt: str, model: str = "gemini") -> str:
        """Generate LLM response."""
        try:
            self.logger.info(f"Generating response with {model}")
            
            if model == "gemini":
                return self._generate_gemini_response(prompt)
            else:
                return f"Response for prompt: {prompt[:50]}..."
                
        except Exception as e:
            self.logger.error(f"Response generation error: {e}")
            return "Error generating response"
    
    def _generate_gemini_response(self, prompt: str) -> str:
        """Generate response using Gemini."""
        try:
            # Simulate Gemini response
            return f"Gemini response for: {prompt[:50]}..."
        except Exception as e:
            self.logger.error(f"Gemini response error: {e}")
            return "Gemini response error"
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return ["gemini", "gpt-3.5", "gpt-4", "claude"]
    
    def validate_api_key(self, key: str) -> bool:
        """Validate API key."""
        try:
            # Simulate key validation
            return len(key) > 10
        except Exception as e:
            self.logger.error(f"Key validation error: {e}")
            return False

def safe_call_llm(prompt: str, model: str = "gemini") -> str:
    """Safe wrapper for LLM calls."""
    try:
        llm = LLMManager()
        return llm.generate_response(prompt, model)
    except Exception as e:
        logger.error(f"Safe LLM call error: {e}")
        return "Error generating response"

def main():
    """Main function."""
    llm = LLMManager()
    print(f"Available models: {llm.get_available_models()}")
    print(f"API keys loaded: {len(llm.api_keys)}")

if __name__ == "__main__":
    main()