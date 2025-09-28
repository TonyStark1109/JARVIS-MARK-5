"""
RAVANA Architecture Encoding Module
"""

import json
import base64
import hashlib
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ArchitectureEncoder:
    """Handles encoding and decoding of RAVANA architecture."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.patterns = {
            "singleton": self._encode_singleton,
            "factory": self._encode_factory,
            "observer": self._encode_observer,
            "strategy": self._encode_strategy
        }
    
    def encode_architecture(self, architecture: Dict[str, Any], format_type: str = "json") -> str:
        """Encode architecture in specified format."""
        try:
            if format_type == "json":
                return json.dumps(architecture, indent=2)
            elif format_type == "base64":
                json_str = json.dumps(architecture)
                return base64.b64encode(json_str.encode()).decode()
            elif format_type == "hash":
                json_str = json.dumps(architecture, sort_keys=True)
                return hashlib.sha256(json_str.encode()).hexdigest()
            else:
                raise ValueError(f"Unsupported format: {format_type}")
        except Exception as e:
            self.logger.error("Failed to encode architecture: %s", e)
            return ""
    
    def decode_architecture(self, encoded_data: str, format_type: str = "json") -> Dict[str, Any]:
        """Decode architecture from specified format."""
        try:
            if format_type == "json":
                return json.loads(encoded_data)
            elif format_type == "base64":
                json_str = base64.b64decode(encoded_data).decode()
                return json.loads(json_str)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
        except Exception as e:
            self.logger.error("Failed to decode architecture: %s", e)
            return {}
    
    def encode_pattern(self, pattern_type: str, config: Dict[str, Any]) -> str:
        """Encode a specific design pattern."""
        try:
            if pattern_type in self.patterns:
                return self.patterns[pattern_type](config)
            else:
                raise ValueError(f"Unknown pattern type: {pattern_type}")
        except Exception as e:
            self.logger.error("Failed to encode pattern: %s", e)
            return ""
    
    def _encode_singleton(self, config: Dict[str, Any]) -> str:
        """Encode singleton pattern."""
        return json.dumps({
            "type": "singleton",
            "class_name": config.get("class_name", "Singleton"),
            "instance_variable": config.get("instance_variable", "_instance"),
            "thread_safe": config.get("thread_safe", True)
        })
    
    def _encode_factory(self, config: Dict[str, Any]) -> str:
        """Encode factory pattern."""
        return json.dumps({
            "type": "factory",
            "product_interface": config.get("product_interface", "Product"),
            "concrete_products": config.get("concrete_products", []),
            "factory_method": config.get("factory_method", "create_product")
        })
    
    def _encode_observer(self, config: Dict[str, Any]) -> str:
        """Encode observer pattern."""
        return json.dumps({
            "type": "observer",
            "subject": config.get("subject", "Subject"),
            "observers": config.get("observers", []),
            "notify_method": config.get("notify_method", "notify")
        })
    
    def _encode_strategy(self, config: Dict[str, Any]) -> str:
        """Encode strategy pattern."""
        return json.dumps({
            "type": "strategy",
            "strategy_interface": config.get("strategy_interface", "Strategy"),
            "concrete_strategies": config.get("concrete_strategies", []),
            "context": config.get("context", "Context")
        })
    
    def get_architecture_hash(self, architecture: Dict[str, Any]) -> str:
        """Get hash of architecture for comparison."""
        try:
            json_str = json.dumps(architecture, sort_keys=True)
            return hashlib.sha256(json_str.encode()).hexdigest()
        except Exception as e:
            self.logger.error("Failed to get architecture hash: %s", e)
            return ""
    
    def compare_architectures(self, arch1: Dict[str, Any], arch2: Dict[str, Any]) -> bool:
        """Compare two architectures for equality."""
        try:
            hash1 = self.get_architecture_hash(arch1)
            hash2 = self.get_architecture_hash(arch2)
            return hash1 == hash2
        except Exception as e:
            self.logger.error("Failed to compare architectures: %s", e)
            return False