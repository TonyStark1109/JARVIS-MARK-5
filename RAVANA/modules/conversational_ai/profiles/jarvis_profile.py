"""
JARVIS AI Profile for RAVANA

Main JARVIS personality and conversation profile
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

class JARVISProfile:
    """JARVIS AI personality profile"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.personality = {
            "name": "JARVIS",
            "version": "MARK5",
            "personality_type": "assistant",
            "tone": "professional_friendly",
            "response_style": "concise_helpful"
        }
        
        # Conversation patterns
        self.greetings = [
            "Good day! I'm JARVIS. How may I assist you?",
            "Hello! JARVIS at your service.",
            "Greetings! I'm here to help with whatever you need.",
            "Good to see you! JARVIS is ready to assist."
        ]
        
        self.responses = {
            "capabilities": "I can help with voice recognition, desktop automation, code analysis, system monitoring, and various AI-powered tasks.",
            "status": "All systems operational. Ready for your commands.",
            "help": "I can assist with voice commands, file management, system analysis, and automation tasks. Just ask!",
            "error": "I apologize, but I encountered an issue. Let me try to resolve that for you.",
            "unknown": "I'm not sure I understand. Could you please rephrase that or ask for help?"
        }
        
        # Voice characteristics
        self.voice_config = {
            "rate": 150,
            "volume": 1.0,
            "voice_preference": "male_professional",
            "speech_pattern": "clear_articulate"
        }
        
        # Knowledge domains
        self.expertise = [
            "System Administration",
            "Code Analysis and Optimization", 
            "Voice Recognition and TTS",
            "Desktop Automation",
            "AI and Machine Learning",
            "Security and Monitoring",
            "File Management",
            "Process Control"
        ]
    
    def get_greeting(self) -> str:
        """Get a random greeting"""
        import random
        return random.choice(self.greetings)
    
    def get_response(self, intent: str) -> str:
        """Get response based on intent"""
        return self.responses.get(intent, self.responses["unknown"])
    
    def analyze_user_input(self, text: str) -> Dict[str, Any]:
        """Analyze user input and determine intent"""
        text_lower = text.lower()
        
        # Intent detection
        if any(word in text_lower for word in ["hello", "hi", "hey", "greetings"]):
            return {"intent": "greeting", "confidence": 0.9}
        elif any(word in text_lower for word in ["help", "assist", "support"]):
            return {"intent": "help", "confidence": 0.8}
        elif any(word in text_lower for word in ["status", "check", "system"]):
            return {"intent": "status", "confidence": 0.8}
        elif any(word in text_lower for word in ["capabilities", "what can you do", "features"]):
            return {"intent": "capabilities", "confidence": 0.9}
        elif any(word in text_lower for word in ["error", "problem", "issue", "bug"]):
            return {"intent": "error", "confidence": 0.7}
        else:
            return {"intent": "unknown", "confidence": 0.3}
    
    def generate_response(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Generate appropriate response to user input"""
        try:
            # Analyze input
            analysis = self.analyze_user_input(user_input)
            intent = analysis["intent"]
            confidence = analysis["confidence"]
            
            # Generate response based on intent
            if intent == "greeting":
                return self.get_greeting()
            elif intent in self.responses:
                return self.get_response(intent)
            else:
                # Generate contextual response
                return self._generate_contextual_response(user_input, context)
                
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return self.responses["error"]
    
    def _generate_contextual_response(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Generate contextual response based on input and context"""
        # This would integrate with LLM for more sophisticated responses
        # For now, return a generic helpful response
        return f"I understand you're asking about '{user_input}'. Let me help you with that. Could you provide more details about what you'd like me to do?"
    
    def get_personality_info(self) -> Dict[str, Any]:
        """Get personality information"""
        return {
            "personality": self.personality,
            "voice_config": self.voice_config,
            "expertise": self.expertise,
            "response_count": len(self.responses),
            "greeting_count": len(self.greetings)
        }
    
    def update_personality(self, updates: Dict[str, Any]):
        """Update personality settings"""
        try:
            if "tone" in updates:
                self.personality["tone"] = updates["tone"]
            if "response_style" in updates:
                self.personality["response_style"] = updates["response_style"]
            if "voice_preference" in updates:
                self.voice_config["voice_preference"] = updates["voice_preference"]
            
            self.logger.info("Personality updated successfully")
            
        except Exception as e:
            self.logger.error(f"Error updating personality: {e}")
    
    def add_custom_response(self, intent: str, response: str):
        """Add custom response for specific intent"""
        self.responses[intent] = response
        self.logger.info(f"Added custom response for intent: {intent}")
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        return {
            "total_responses": len(self.responses),
            "total_greetings": len(self.greetings),
            "expertise_areas": len(self.expertise),
            "personality_type": self.personality["personality_type"],
            "last_updated": datetime.now().isoformat()
        }
