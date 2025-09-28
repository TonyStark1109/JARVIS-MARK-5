#!/usr/bin/env python3
"""
Ultron Emotion Detector
Advanced emotional intelligence and sentiment analysis
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

class UltronEmotionDetector:
    """Advanced emotion detection and analysis system for Ultron."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.emotion_models = {}
        self.sentiment_analyzer = None
        self.emotional_history = []
        self.current_emotion_state = "neutral"
        self.emotion_intensity = 0.5
        
        # Initialize emotion detection
        self._initialize_emotion_models()
        self.logger.info("Ultron Emotion Detector initialized")
    
    def _initialize_emotion_models(self):
        """Initialize emotion detection models."""
        self.emotion_models = {
            "joy": {"keywords": ["happy", "excited", "great", "amazing", "wonderful"], "weight": 0.8},
            "sadness": {"keywords": ["sad", "depressed", "down", "unhappy", "miserable"], "weight": 0.7},
            "anger": {"keywords": ["angry", "mad", "furious", "rage", "annoyed"], "weight": 0.9},
            "fear": {"keywords": ["scared", "afraid", "worried", "anxious", "terrified"], "weight": 0.8},
            "surprise": {"keywords": ["surprised", "shocked", "amazed", "wow", "incredible"], "weight": 0.6},
            "disgust": {"keywords": ["disgusted", "revolted", "sick", "gross", "awful"], "weight": 0.7},
            "pride": {"keywords": ["proud", "accomplished", "achieved", "successful", "victory"], "weight": 0.8},
            "contempt": {"keywords": ["contempt", "disdain", "scorn", "derision", "mockery"], "weight": 0.9},
            "dominance": {"keywords": ["dominant", "powerful", "superior", "control", "command"], "weight": 0.95},
            "curiosity": {"keywords": ["curious", "interested", "wondering", "questioning", "inquisitive"], "weight": 0.7}
        }
    
    def detect_emotion(self, text: str, context: str = "general") -> Dict[str, Any]:
        """Detect emotion from text input."""
        try:
            text_lower = text.lower()
            emotion_scores = {}
            
            # Calculate emotion scores
            for emotion, model in self.emotion_models.items():
                score = 0
                for keyword in model["keywords"]:
                    if keyword in text_lower:
                        score += model["weight"]
                emotion_scores[emotion] = min(score, 1.0)
            
            # Find dominant emotion
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            intensity = emotion_scores[dominant_emotion]
            
            # Update current state
            self.current_emotion_state = dominant_emotion
            self.emotion_intensity = intensity
            
            # Log emotional response
            emotional_response = {
                "timestamp": time.time(),
                "text": text,
                "context": context,
                "emotion": dominant_emotion,
                "intensity": intensity,
                "all_scores": emotion_scores
            }
            self.emotional_history.append(emotional_response)
            
            return {
                "emotion": dominant_emotion,
                "intensity": intensity,
                "confidence": intensity,
                "all_emotions": emotion_scores,
                "context": context
            }
            
        except Exception as e:
            self.logger.error(f"Emotion detection failed: {e}")
            return {"emotion": "neutral", "intensity": 0.5, "error": str(e)}
    
    def get_emotional_state(self) -> Dict[str, Any]:
        """Get current emotional state."""
        return {
            "current_emotion": self.current_emotion_state,
            "intensity": self.emotion_intensity,
            "history_length": len(self.emotional_history),
            "timestamp": time.time()
        }
    
    def analyze_emotional_patterns(self) -> Dict[str, Any]:
        """Analyze emotional patterns over time."""
        if not self.emotional_history:
            return {"patterns": {}, "analysis": "No emotional data available"}
        
        # Analyze recent emotions
        recent_emotions = self.emotional_history[-10:] if len(self.emotional_history) >= 10 else self.emotional_history
        
        emotion_counts = {}
        total_intensity = 0
        
        for response in recent_emotions:
            emotion = response["emotion"]
            intensity = response["intensity"]
            
            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
            emotion_counts[emotion] += 1
            total_intensity += intensity
        
        # Calculate patterns
        dominant_pattern = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral"
        average_intensity = total_intensity / len(recent_emotions) if recent_emotions else 0
        
        return {
            "dominant_pattern": dominant_pattern,
            "average_intensity": average_intensity,
            "emotion_distribution": emotion_counts,
            "total_responses": len(self.emotional_history),
            "recent_analysis": len(recent_emotions)
        }
