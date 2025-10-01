"""
General Assistant Profile for RAVANA

Base profile for various AI assistant configurations
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

class AssistantProfile:
    """Base assistant profile with configurable personality"""
    
    def __init__(self, profile_name: str = "Assistant"):
        self.logger = logging.getLogger(__name__)
        self.profile_name = profile_name
        self.personality = {
            "name": profile_name,
            "personality_type": "assistant",
            "tone": "neutral",
            "response_style": "helpful"
        }
        
        # Base responses
        self.responses = {
            "greeting": f"Hello! I'm {profile_name}. How can I help you?",
            "help": "I'm here to assist you. What would you like to know?",
            "error": "I apologize, but I encountered an issue. Let me try to help.",
            "unknown": "I'm not sure I understand. Could you please clarify?"
        }
        
        # Configuration
        self.config = {
            "max_response_length": 500,
            "response_delay": 0.1,
            "enable_emotions": False,
            "enable_memory": True
        }
    
    def set_personality(self, personality_type: str, tone: str, response_style: str):
        """Set personality configuration"""
        self.personality.update({
            "personality_type": personality_type,
            "tone": tone,
            "response_style": response_style
        })
        self.logger.info(f"Personality updated: {personality_type}, {tone}, {response_style}")
    
    def add_response(self, key: str, response: str):
        """Add custom response"""
        self.responses[key] = response
    
    def get_response(self, key: str) -> str:
        """Get response by key"""
        return self.responses.get(key, self.responses["unknown"])
    
    def process_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input and return analysis"""
        return {
            "input": user_input,
            "timestamp": datetime.now().isoformat(),
            "profile": self.profile_name,
            "personality": self.personality["personality_type"]
        }
    
    def get_profile_info(self) -> Dict[str, Any]:
        """Get profile information"""
        return {
            "name": self.profile_name,
            "personality": self.personality,
            "config": self.config,
            "response_count": len(self.responses)
        }
