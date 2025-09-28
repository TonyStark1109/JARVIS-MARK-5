#!/usr/bin/env python3
"""
RAVANA Decision Maker
Core decision-making engine for autonomous agents.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import random

logger = logging.getLogger(__name__)

class DecisionMaker:
    """Makes decisions based on current state, goals, and constraints."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.decision_history: List[Dict[str, Any]] = []
        self.decision_weights: Dict[str, float] = {
            'safety': 0.3,
            'efficiency': 0.25,
            'curiosity': 0.2,
            'social': 0.15,
            'exploration': 0.1
        }
        
    async def make_decision(self, situation: Dict[str, Any], 
                          options: List[Dict[str, Any]], 
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a decision given a situation and available options."""
        try:
            self.logger.info(f"Making decision for situation: {situation.get('description', 'Unknown')}")
            
            # Evaluate each option
            evaluated_options = []
            for option in options:
                score = await self._evaluate_option(option, situation, context)
                evaluated_options.append({
                    'option': option,
                    'score': score,
                    'confidence': self._calculate_confidence(score)
                })
            
            # Select best option
            best_option = max(evaluated_options, key=lambda x: x['score'])
            
            decision = {
                'id': f"decision_{len(self.decision_history)}",
                'timestamp': datetime.now(),
                'situation': situation,
                'selected_option': best_option['option'],
                'score': best_option['score'],
                'confidence': best_option['confidence'],
                'all_options': evaluated_options,
                'reasoning': await self._generate_reasoning(situation, best_option, context)
            }
            
            self.decision_history.append(decision)
            self.logger.info(f"Decision made with score {best_option['score']:.2f} and confidence {best_option['confidence']:.2f}")
            
            return decision
        except Exception as e:
            self.logger.error(f"Failed to make decision: {e}")
            return {'error': str(e)}
    
    async def _evaluate_option(self, option: Dict[str, Any], 
                             situation: Dict[str, Any], 
                             context: Optional[Dict[str, Any]]) -> float:
        """Evaluate a single option and return a score."""
        try:
            score = 0.0
            
            # Safety evaluation
            safety_score = option.get('safety_rating', 0.5)
            score += safety_score * self.decision_weights['safety']
            
            # Efficiency evaluation
            efficiency_score = option.get('efficiency_rating', 0.5)
            score += efficiency_score * self.decision_weights['efficiency']
            
            # Curiosity evaluation
            curiosity_score = option.get('curiosity_rating', 0.5)
            score += curiosity_score * self.decision_weights['curiosity']
            
            # Social evaluation
            social_score = option.get('social_rating', 0.5)
            score += social_score * self.decision_weights['social']
            
            # Exploration evaluation
            exploration_score = option.get('exploration_rating', 0.5)
            score += exploration_score * self.decision_weights['exploration']
            
            # Context-based adjustments
            if context:
                if context.get('urgency', False):
                    score += 0.1  # Favor faster options
                if context.get('learning_mode', False):
                    score += curiosity_score * 0.2  # Favor learning opportunities
            
            return min(1.0, max(0.0, score))  # Clamp between 0 and 1
        except Exception as e:
            self.logger.error(f"Error evaluating option: {e}")
            return 0.0
    
    def _calculate_confidence(self, score: float) -> float:
        """Calculate confidence based on score and other factors."""
        # Higher scores generally mean higher confidence
        base_confidence = score
        
        # Adjust based on decision history
        if len(self.decision_history) > 0:
            recent_decisions = self.decision_history[-5:]  # Last 5 decisions
            avg_score = sum(d.get('score', 0) for d in recent_decisions) / len(recent_decisions)
            consistency_bonus = 1.0 - abs(score - avg_score)
            base_confidence = (base_confidence + consistency_bonus) / 2
        
        return min(1.0, max(0.0, base_confidence))
    
    async def _generate_reasoning(self, situation: Dict[str, Any], 
                                best_option: Dict[str, Any], 
                                context: Optional[Dict[str, Any]]) -> str:
        """Generate human-readable reasoning for the decision."""
        try:
            reasoning_parts = []
            
            # Primary factors
            if best_option['score'] > 0.8:
                reasoning_parts.append("This option scored highly across multiple criteria")
            elif best_option['score'] > 0.6:
                reasoning_parts.append("This option provides a good balance of factors")
            else:
                reasoning_parts.append("This option was selected as the best available choice")
            
            # Specific strengths
            strengths = []
            if best_option['option'].get('safety_rating', 0) > 0.7:
                strengths.append("safety")
            if best_option['option'].get('efficiency_rating', 0) > 0.7:
                strengths.append("efficiency")
            if best_option['option'].get('curiosity_rating', 0) > 0.7:
                strengths.append("learning potential")
            
            if strengths:
                reasoning_parts.append(f"Particularly strong in: {', '.join(strengths)}")
            
            # Context considerations
            if context:
                if context.get('urgency'):
                    reasoning_parts.append("Urgency was a key factor in the decision")
                if context.get('learning_mode'):
                    reasoning_parts.append("Learning opportunities were prioritized")
            
            return ". ".join(reasoning_parts) + "."
        except Exception as e:
            self.logger.error(f"Error generating reasoning: {e}")
            return "Decision made based on available information."
    
    async def update_weights(self, feedback: Dict[str, Any]) -> bool:
        """Update decision weights based on feedback."""
        try:
            if 'weights' in feedback:
                for key, value in feedback['weights'].items():
                    if key in self.decision_weights:
                        self.decision_weights[key] = value
                        self.logger.info(f"Updated weight for {key} to {value}")
            
            # Normalize weights
            total_weight = sum(self.decision_weights.values())
            if total_weight > 0:
                for key in self.decision_weights:
                    self.decision_weights[key] /= total_weight
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to update weights: {e}")
            return False
