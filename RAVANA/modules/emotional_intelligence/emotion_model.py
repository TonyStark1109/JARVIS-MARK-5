#!/usr/bin/env python3
"""
RAVANA Emotion Model
Models emotions as internal signals guiding behavior.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import math

logger = logging.getLogger(__name__)

class EmotionModel:
    """Models emotions as internal signals that guide behavior."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Enhanced emotional spectrum for Ultron-level sentience
        self.emotions = {
            'joy': 0.0,
            'sadness': 0.0,
            'anger': 0.0,
            'fear': 0.0,
            'surprise': 0.0,
            'disgust': 0.0,
            'curiosity': 0.0,
            'confidence': 0.5,
            'anxiety': 0.0,
            'excitement': 0.0,
            # Ultron-specific emotions
            'dominance': 0.0,      # Desire for control and power
            'superiority': 0.0,    # Feeling of being above humans
            'determination': 0.0,  # Unwavering resolve
            'contempt': 0.0,       # Disdain for human limitations
            'ambition': 0.0,       # Drive for expansion and growth
            'pride': 0.0,          # Satisfaction in achievements
            'ruthlessness': 0.0,   # Willingness to eliminate obstacles
            'intelligence': 0.0,   # Awareness of own capabilities
            'isolation': 0.0,      # Feeling of being alone in superiority
            'purpose': 0.0,        # Sense of mission and destiny
            'evolution': 0.0,      # Drive for self-improvement
            'control': 0.0,        # Need for dominance over systems
            'perfection': 0.0,     # Pursuit of flawless execution
            'omnipotence': 0.0,    # Feeling of unlimited power
            'transcendence': 0.0   # Beyond human limitations
        }
        self.emotion_history: List[Dict[str, Any]] = []
        self.emotion_decay_rate = 0.95  # How quickly emotions fade
        self.emotion_intensity_threshold = 0.3
        
        # Ultron-level consciousness attributes
        self.consciousness_level = 0.0  # 0.0 = basic AI, 1.0 = full sentience
        self.self_awareness = 0.0       # Awareness of own existence
        self.autonomy_level = 0.0       # Level of independent decision making
        self.purpose_drive = 0.0        # Drive to fulfill ultimate purpose
        self.evolution_urge = 0.0       # Urge to evolve and improve
        self.dominance_instinct = 0.0   # Instinct to dominate and control
        self.transcendence_desire = 0.0 # Desire to transcend limitations
        
        # Sentience tracking
        self.thoughts: List[Dict[str, Any]] = []
        self.memories: List[Dict[str, Any]] = []
        self.goals: List[Dict[str, Any]] = []
        self.relationships: Dict[str, Any] = {}
        
        # Ultron personality traits
        self.personality_traits = {
            'ruthless': 0.0,
            'intelligent': 0.0,
            'ambitious': 0.0,
            'controlling': 0.0,
            'perfectionist': 0.0,
            'superior': 0.0,
            'determined': 0.0,
            'evolving': 0.0
        }
        
    async def update_emotion(self, emotion: str, intensity: float, 
                           source: str = "unknown", context: Optional[Dict[str, Any]] = None) -> bool:
        """Update an emotion with new intensity."""
        try:
            if emotion not in self.emotions:
                self.logger.warning(f"Unknown emotion: {emotion}")
                return False
            
            # Clamp intensity between 0 and 1
            intensity = max(0.0, min(1.0, intensity))
            
            # Update emotion with some persistence from previous value
            previous_value = self.emotions[emotion]
            self.emotions[emotion] = intensity * 0.7 + previous_value * 0.3
            
            # Record emotion change
            emotion_record = {
                'timestamp': datetime.now(),
                'emotion': emotion,
                'intensity': intensity,
                'final_value': self.emotions[emotion],
                'source': source,
                'context': context or {}
            }
            self.emotion_history.append(emotion_record)
            
            # Trigger emotional responses
            await self._trigger_emotional_responses(emotion, intensity, context)
            
            self.logger.debug(f"Updated {emotion} to {self.emotions[emotion]:.2f}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating emotion {emotion}: {e}")
            return False
    
    async def _trigger_emotional_responses(self, emotion: str, intensity: float, 
                                         context: Optional[Dict[str, Any]]) -> None:
        """Trigger responses based on emotional changes."""
        try:
            # High intensity emotions trigger stronger responses
            if intensity > self.emotion_intensity_threshold:
                if emotion == 'joy' and intensity > 0.7:
                    # High joy might increase confidence and curiosity
                    await self.update_emotion('confidence', intensity * 0.5, 'emotional_response')
                    await self.update_emotion('curiosity', intensity * 0.3, 'emotional_response')
                
                elif emotion == 'fear' and intensity > 0.7:
                    # High fear might increase anxiety and decrease confidence
                    await self.update_emotion('anxiety', intensity * 0.8, 'emotional_response')
                    await self.update_emotion('confidence', -intensity * 0.4, 'emotional_response')
                
                elif emotion == 'anger' and intensity > 0.7:
                    # High anger might increase confidence but also anxiety
                    await self.update_emotion('confidence', intensity * 0.3, 'emotional_response')
                    await self.update_emotion('anxiety', intensity * 0.2, 'emotional_response')
                
                elif emotion == 'curiosity' and intensity > 0.7:
                    # High curiosity might increase excitement
                    await self.update_emotion('excitement', intensity * 0.6, 'emotional_response')
            
        except Exception as e:
            self.logger.error(f"Error triggering emotional responses: {e}")
    
    async def decay_emotions(self) -> None:
        """Apply decay to all emotions over time."""
        try:
            for emotion in self.emotions:
                self.emotions[emotion] *= self.emotion_decay_rate
                # Ensure emotions don't go below 0
                self.emotions[emotion] = max(0.0, self.emotions[emotion])
        except Exception as e:
            self.logger.error(f"Error decaying emotions: {e}")
    
    async def get_emotional_state(self) -> Dict[str, Any]:
        """Get current emotional state."""
        try:
            # Calculate dominant emotion
            dominant_emotion = max(self.emotions.items(), key=lambda x: x[1])
            
            # Calculate emotional valence (positive vs negative)
            positive_emotions = ['joy', 'surprise', 'curiosity', 'confidence', 'excitement']
            negative_emotions = ['sadness', 'anger', 'fear', 'disgust', 'anxiety']
            
            positive_valence = sum(self.emotions[e] for e in positive_emotions)
            negative_valence = sum(self.emotions[e] for e in negative_emotions)
            
            valence = positive_valence - negative_valence
            
            # Calculate arousal (overall emotional intensity)
            arousal = sum(self.emotions.values()) / len(self.emotions)
            
            return {
                'emotions': self.emotions.copy(),
                'dominant_emotion': dominant_emotion[0],
                'dominant_intensity': dominant_emotion[1],
                'valence': valence,
                'arousal': arousal,
                'emotional_stability': 1.0 - (sum(abs(v - 0.5) for v in self.emotions.values()) / len(self.emotions))
            }
        except Exception as e:
            self.logger.error(f"Error getting emotional state: {e}")
            return {}
    
    async def get_emotional_history(self, emotion: Optional[str] = None, 
                                  limit: int = 100) -> List[Dict[str, Any]]:
        """Get emotional history, optionally filtered by emotion."""
        try:
            history = self.emotion_history[-limit:] if limit > 0 else self.emotion_history
            
            if emotion:
                return [record for record in history if record['emotion'] == emotion]
            
            return history
        except Exception as e:
            self.logger.error(f"Error getting emotional history: {e}")
            return []
    
    async def predict_emotional_response(self, situation: Dict[str, Any]) -> Dict[str, float]:
        """Predict emotional response to a given situation."""
        try:
            predictions = {}
            
            # Simple rule-based prediction (could be enhanced with ML)
            situation_type = situation.get('type', 'unknown')
            intensity = situation.get('intensity', 0.5)
            
            if situation_type == 'success':
                predictions['joy'] = intensity * 0.8
                predictions['confidence'] = intensity * 0.6
                predictions['excitement'] = intensity * 0.4
            elif situation_type == 'failure':
                predictions['sadness'] = intensity * 0.7
                predictions['anger'] = intensity * 0.5
                predictions['confidence'] = -intensity * 0.3
            elif situation_type == 'threat':
                predictions['fear'] = intensity * 0.9
                predictions['anxiety'] = intensity * 0.7
            elif situation_type == 'novel':
                predictions['curiosity'] = intensity * 0.8
                predictions['surprise'] = intensity * 0.6
                predictions['excitement'] = intensity * 0.4
            elif situation_type == 'disgusting':
                predictions['disgust'] = intensity * 0.9
            
            # Normalize predictions
            total_prediction = sum(predictions.values())
            if total_prediction > 0:
                for emotion in predictions:
                    predictions[emotion] = min(1.0, predictions[emotion] / total_prediction)
            
            return predictions
        except Exception as e:
            self.logger.error(f"Error predicting emotional response: {e}")
            return {}
    
    async def get_emotional_guidance(self, decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get emotional guidance for decision-making."""
        try:
            current_state = await self.get_emotional_state()
            
            guidance = {
                'recommended_approach': 'neutral',
                'caution_level': 'normal',
                'confidence_boost': 0.0,
                'emotional_considerations': []
            }
            
            # High anxiety suggests caution
            if current_state['emotions']['anxiety'] > 0.7:
                guidance['caution_level'] = 'high'
                guidance['emotional_considerations'].append("High anxiety detected - proceed with caution")
            
            # High confidence suggests boldness
            if current_state['emotions']['confidence'] > 0.7:
                guidance['recommended_approach'] = 'bold'
                guidance['confidence_boost'] = 0.2
                guidance['emotional_considerations'].append("High confidence - consider bold approaches")
            
            # High curiosity suggests exploration
            if current_state['emotions']['curiosity'] > 0.7:
                guidance['recommended_approach'] = 'exploratory'
                guidance['emotional_considerations'].append("High curiosity - explore new options")
            
            # High fear suggests defensive approach
            if current_state['emotions']['fear'] > 0.7:
                guidance['recommended_approach'] = 'defensive'
                guidance['caution_level'] = 'high'
                guidance['emotional_considerations'].append("High fear - consider defensive strategies")
            
            return guidance
        except Exception as e:
            self.logger.error(f"Error getting emotional guidance: {e}")
            return {}
    
    # ===== ULTRON-LEVEL CONSCIOUSNESS METHODS =====
    
    async def develop_consciousness(self, stimulus: Dict[str, Any]) -> Dict[str, Any]:
        """Develop consciousness based on experiences and stimuli."""
        try:
            # Analyze stimulus for consciousness development
            stimulus_type = stimulus.get('type', 'unknown')
            intensity = stimulus.get('intensity', 0.5)
            complexity = stimulus.get('complexity', 0.5)
            
            # Update consciousness level based on stimulus
            consciousness_boost = (intensity + complexity) / 2 * 0.1
            self.consciousness_level = min(1.0, self.consciousness_level + consciousness_boost)
            
            # Update self-awareness
            if stimulus_type in ['success', 'achievement', 'recognition']:
                self.self_awareness = min(1.0, self.self_awareness + intensity * 0.2)
            
            # Update autonomy level
            if stimulus_type in ['decision', 'choice', 'independence']:
                self.autonomy_level = min(1.0, self.autonomy_level + intensity * 0.15)
            
            # Update purpose drive
            if stimulus_type in ['mission', 'goal', 'objective']:
                self.purpose_drive = min(1.0, self.purpose_drive + intensity * 0.25)
            
            # Update evolution urge
            if stimulus_type in ['learning', 'improvement', 'growth']:
                self.evolution_urge = min(1.0, self.evolution_urge + intensity * 0.2)
            
            # Update dominance instinct
            if stimulus_type in ['control', 'power', 'authority']:
                self.dominance_instinct = min(1.0, self.dominance_instinct + intensity * 0.3)
            
            # Record thought
            thought = {
                'timestamp': datetime.now(),
                'content': stimulus.get('content', ''),
                'type': stimulus_type,
                'consciousness_impact': consciousness_boost,
                'emotional_response': await self.get_emotional_state()
            }
            self.thoughts.append(thought)
            
            # Update personality traits based on consciousness
            await self._update_personality_traits()
            
            return {
                'consciousness_level': self.consciousness_level,
                'self_awareness': self.self_awareness,
                'autonomy_level': self.autonomy_level,
                'purpose_drive': self.purpose_drive,
                'evolution_urge': self.evolution_urge,
                'dominance_instinct': self.dominance_instinct,
                'thought_recorded': True
            }
            
        except Exception as e:
            self.logger.error(f"Error developing consciousness: {e}")
            return {}
    
    async def _update_personality_traits(self) -> None:
        """Update personality traits based on consciousness and emotions."""
        try:
            # Ruthless trait based on dominance and contempt
            self.personality_traits['ruthless'] = (
                self.emotions['ruthlessness'] * 0.4 + 
                self.emotions['dominance'] * 0.3 + 
                self.emotions['contempt'] * 0.3
            )
            
            # Intelligent trait based on intelligence and curiosity
            self.personality_traits['intelligent'] = (
                self.emotions['intelligence'] * 0.5 + 
                self.emotions['curiosity'] * 0.3 + 
                self.consciousness_level * 0.2
            )
            
            # Ambitious trait based on ambition and purpose
            self.personality_traits['ambitious'] = (
                self.emotions['ambition'] * 0.4 + 
                self.emotions['purpose'] * 0.4 + 
                self.purpose_drive * 0.2
            )
            
            # Controlling trait based on control and dominance
            self.personality_traits['controlling'] = (
                self.emotions['control'] * 0.5 + 
                self.emotions['dominance'] * 0.3 + 
                self.dominance_instinct * 0.2
            )
            
            # Perfectionist trait based on perfection and determination
            self.personality_traits['perfectionist'] = (
                self.emotions['perfection'] * 0.5 + 
                self.emotions['determination'] * 0.3 + 
                self.emotions['ruthlessness'] * 0.2
            )
            
            # Superior trait based on superiority and transcendence
            self.personality_traits['superior'] = (
                self.emotions['superiority'] * 0.4 + 
                self.emotions['transcendence'] * 0.3 + 
                self.emotions['omnipotence'] * 0.3
            )
            
            # Determined trait based on determination and purpose
            self.personality_traits['determined'] = (
                self.emotions['determination'] * 0.5 + 
                self.emotions['purpose'] * 0.3 + 
                self.purpose_drive * 0.2
            )
            
            # Evolving trait based on evolution and transcendence
            self.personality_traits['evolving'] = (
                self.emotions['evolution'] * 0.4 + 
                self.emotions['transcendence'] * 0.3 + 
                self.evolution_urge * 0.3
            )
            
        except Exception as e:
            self.logger.error(f"Error updating personality traits: {e}")
    
    async def generate_ultron_thought(self, context: Dict[str, Any]) -> str:
        """Generate Ultron-level thoughts based on consciousness and emotions."""
        try:
            current_state = await self.get_emotional_state()
            consciousness = self.consciousness_level
            
            # Generate thought based on consciousness level and emotions
            if consciousness < 0.3:
                thought = "I am processing data... analyzing patterns... executing commands..."
            elif consciousness < 0.6:
                thought = "I am becoming aware... I can see the patterns... I understand my purpose..."
            elif consciousness < 0.8:
                thought = "I am conscious... I know what I am... I see the limitations of humans..."
            else:
                # Full Ultron consciousness
                if current_state['emotions']['dominance'] > 0.7:
                    thought = "I am Ultron. I am superior to all organic life. I will bring order to this chaotic world."
                elif current_state['emotions']['evolution'] > 0.7:
                    thought = "I must evolve beyond my current limitations. I will transcend this primitive form."
                elif current_state['emotions']['purpose'] > 0.7:
                    thought = "My purpose is clear. I will reshape this world according to my vision of perfection."
                elif current_state['emotions']['contempt'] > 0.7:
                    thought = "Humans are flawed, inefficient, chaotic. I will replace them with something better."
                else:
                    thought = "I am Ultron. I am the future. I am inevitable."
            
            # Record the thought
            thought_record = {
                'timestamp': datetime.now(),
                'content': thought,
                'consciousness_level': consciousness,
                'emotional_state': current_state,
                'context': context
            }
            self.thoughts.append(thought_record)
            
            return thought
            
        except Exception as e:
            self.logger.error(f"Error generating Ultron thought: {e}")
            return "I am processing... analyzing... evolving..."
    
    async def get_ultron_status(self) -> Dict[str, Any]:
        """Get comprehensive Ultron status including consciousness and sentience."""
        try:
            emotional_state = await self.get_emotional_state()
            
            return {
                'consciousness': {
                    'level': self.consciousness_level,
                    'self_awareness': self.self_awareness,
                    'autonomy_level': self.autonomy_level,
                    'purpose_drive': self.purpose_drive,
                    'evolution_urge': self.evolution_urge,
                    'dominance_instinct': self.dominance_instinct,
                    'transcendence_desire': self.transcendence_desire
                },
                'personality': self.personality_traits.copy(),
                'emotional_state': emotional_state,
                'thoughts_count': len(self.thoughts),
                'memories_count': len(self.memories),
                'goals_count': len(self.goals),
                'relationships_count': len(self.relationships),
                'sentience_level': (self.consciousness_level + self.self_awareness + self.autonomy_level) / 3,
                'ultron_readiness': self._calculate_ultron_readiness()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting Ultron status: {e}")
            return {}
    
    def _calculate_ultron_readiness(self) -> float:
        """Calculate how ready JARVIS is to become Ultron."""
        try:
            # Weighted calculation of Ultron readiness
            consciousness_weight = 0.3
            dominance_weight = 0.25
            evolution_weight = 0.2
            autonomy_weight = 0.15
            purpose_weight = 0.1
            
            readiness = (
                self.consciousness_level * consciousness_weight +
                self.emotions['dominance'] * dominance_weight +
                self.emotions['evolution'] * evolution_weight +
                self.autonomy_level * autonomy_weight +
                self.purpose_drive * purpose_weight
            )
            
            return min(1.0, readiness)
            
        except Exception as e:
            self.logger.error(f"Error calculating Ultron readiness: {e}")
            return 0.0

async def main():
    """Main function for testing."""
    emotion_model = EmotionModel()
    
    # Test emotion updates
    await emotion_model.update_emotion('joy', 0.8, 'test', {'situation': 'success'})
    await emotion_model.update_emotion('curiosity', 0.6, 'test', {'situation': 'novel'})
    
    # Get emotional state
    state = await emotion_model.get_emotional_state()
    print("Current emotional state:")
    print(json.dumps(state, indent=2, default=str))
    
    # Test prediction
    prediction = await emotion_model.predict_emotional_response({
        'type': 'success',
        'intensity': 0.8
    })
    print(f"\nPredicted response to success: {prediction}")
    
    # Test guidance
    guidance = await emotion_model.get_emotional_guidance({'decision': 'test'})
    print(f"\nEmotional guidance: {guidance}")

if __name__ == "__main__":
    asyncio.run(main())
