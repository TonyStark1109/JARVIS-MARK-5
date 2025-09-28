#!/usr/bin/env python3
"""
RAVANA Meta Law Inference
Infers meta-laws and patterns from system behavior and data.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import re
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class MetaLawInference:
    """Infers meta-laws and patterns from system behavior."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.observed_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.inferred_laws: List[Dict[str, Any]] = []
        self.confidence_threshold = 0.7
        self.min_occurrences = 3
        
    async def observe_event(self, event: Dict[str, Any]) -> bool:
        """Observe an event for pattern analysis."""
        try:
            event_type = event.get('type', 'unknown')
            self.observed_patterns[event_type].append({
                'timestamp': event.get('timestamp', datetime.now()),
                'data': event.get('data', {}),
                'context': event.get('context', {})
            })
            
            # Trigger pattern analysis if we have enough data
            if len(self.observed_patterns[event_type]) >= self.min_occurrences:
                await self._analyze_patterns(event_type)
            
            return True
        except Exception as e:
            self.logger.error(f"Error observing event: {e}")
            return False
    
    async def _analyze_patterns(self, event_type: str) -> None:
        """Analyze patterns for a specific event type."""
        try:
            events = self.observed_patterns[event_type]
            if len(events) < self.min_occurrences:
                return
            
            # Temporal patterns
            temporal_laws = await self._analyze_temporal_patterns(events)
            
            # Causal patterns
            causal_laws = await self._analyze_causal_patterns(events)
            
            # Behavioral patterns
            behavioral_laws = await self._analyze_behavioral_patterns(events)
            
            # Combine and store laws
            laws = temporal_laws + causal_laws + behavioral_laws
            for law in laws:
                if law['confidence'] >= self.confidence_threshold:
                    law['event_type'] = event_type
                    law['inferred_at'] = datetime.now()
                    self.inferred_laws.append(law)
                    self.logger.info(f"Inferred law: {law['description']}")
            
        except Exception as e:
            self.logger.error(f"Error analyzing patterns for {event_type}: {e}")
    
    async def _analyze_temporal_patterns(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze temporal patterns in events."""
        laws = []
        
        try:
            if len(events) < 2:
                return laws
            
            # Calculate time intervals
            intervals = []
            for i in range(1, len(events)):
                interval = (events[i]['timestamp'] - events[i-1]['timestamp']).total_seconds()
                intervals.append(interval)
            
            if not intervals:
                return laws
            
            # Check for periodicity
            avg_interval = sum(intervals) / len(intervals)
            variance = sum((x - avg_interval)**2 for x in intervals) / len(intervals)
            std_dev = variance**0.5
            
            if std_dev < avg_interval * 0.2:  # Low variance suggests periodicity
                laws.append({
                    'type': 'temporal',
                    'description': f"Events occur approximately every {avg_interval:.2f} seconds",
                    'confidence': min(1.0, 1.0 - (std_dev / avg_interval)),
                    'parameters': {
                        'average_interval': avg_interval,
                        'standard_deviation': std_dev
                    }
                })
            
            # Check for trends
            if len(intervals) >= 3:
                trend = self._calculate_trend(intervals)
                if abs(trend) > 0.1:  # Significant trend
                    trend_direction = "increasing" if trend > 0 else "decreasing"
                    laws.append({
                        'type': 'temporal_trend',
                        'description': f"Event intervals are {trend_direction} over time",
                        'confidence': min(1.0, abs(trend)),
                        'parameters': {
                            'trend_slope': trend,
                            'trend_direction': trend_direction
                        }
                    })
            
        except Exception as e:
            self.logger.error(f"Error analyzing temporal patterns: {e}")
        
        return laws
    
    async def _analyze_causal_patterns(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze causal relationships between events."""
        laws = []
        
        try:
            # Look for common data patterns that precede events
            data_patterns = defaultdict(list)
            
            for event in events:
                data = event.get('data', {})
                for key, value in data.items():
                    data_patterns[key].append(value)
            
            # Check for consistent patterns
            for key, values in data_patterns.items():
                if len(set(values)) == 1:  # All values are the same
                    laws.append({
                        'type': 'causal',
                        'description': f"Event always occurs when {key} = {values[0]}",
                        'confidence': 1.0,
                        'parameters': {
                            'condition': {key: values[0]},
                            'consistency': 1.0
                        }
                    })
                elif len(set(values)) / len(values) < 0.3:  # Low diversity
                    most_common = Counter(values).most_common(1)[0]
                    laws.append({
                        'type': 'causal',
                        'description': f"Event usually occurs when {key} = {most_common[0]} ({most_common[1]}/{len(values)} times)",
                        'confidence': most_common[1] / len(values),
                        'parameters': {
                            'condition': {key: most_common[0]},
                            'frequency': most_common[1] / len(values)
                        }
                    })
            
        except Exception as e:
            self.logger.error(f"Error analyzing causal patterns: {e}")
        
        return laws
    
    async def _analyze_behavioral_patterns(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze behavioral patterns in events."""
        laws = []
        
        try:
            # Analyze context patterns
            context_keys = set()
            for event in events:
                context = event.get('context', {})
                context_keys.update(context.keys())
            
            for key in context_keys:
                values = [event.get('context', {}).get(key) for event in events]
                values = [v for v in values if v is not None]
                
                if len(values) >= self.min_occurrences:
                    # Check for patterns in context values
                    if all(isinstance(v, (int, float)) for v in values):
                        # Numeric patterns
                        avg_val = sum(values) / len(values)
                        if all(abs(v - avg_val) < avg_val * 0.1 for v in values):
                            laws.append({
                                'type': 'behavioral',
                                'description': f"Event occurs when {key} ≈ {avg_val:.2f}",
                                'confidence': 0.9,
                                'parameters': {
                                    'context_key': key,
                                    'average_value': avg_val,
                                    'variance': sum((v - avg_val)**2 for v in values) / len(values)
                                }
                            })
                    else:
                        # Categorical patterns
                        value_counts = Counter(values)
                        most_common = value_counts.most_common(1)[0]
                        if most_common[1] / len(values) > 0.7:
                            laws.append({
                                'type': 'behavioral',
                                'description': f"Event usually occurs when {key} = '{most_common[0]}'",
                                'confidence': most_common[1] / len(values),
                                'parameters': {
                                    'context_key': key,
                                    'most_common_value': most_common[0],
                                    'frequency': most_common[1] / len(values)
                                }
                            })
            
        except Exception as e:
            self.logger.error(f"Error analyzing behavioral patterns: {e}")
        
        return laws
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate the trend (slope) of a series of values."""
        try:
            n = len(values)
            if n < 2:
                return 0.0
            
            x = list(range(n))
            x_mean = sum(x) / n
            y_mean = sum(values) / n
            
            numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean)**2 for i in range(n))
            
            if denominator == 0:
                return 0.0
            
            return numerator / denominator
        except Exception as e:
            self.logger.error(f"Error calculating trend: {e}")
            return 0.0
    
    async def get_inferred_laws(self, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all inferred laws, optionally filtered by event type."""
        if event_type:
            return [law for law in self.inferred_laws if law.get('event_type') == event_type]
        return self.inferred_laws
    
    async def test_law(self, law: Dict[str, Any], new_event: Dict[str, Any]) -> bool:
        """Test if a new event follows an inferred law."""
        try:
            law_type = law.get('type')
            parameters = law.get('parameters', {})
            
            if law_type == 'temporal':
                # Test temporal law (would need historical data)
                return True  # Placeholder
            elif law_type == 'causal':
                condition = parameters.get('condition', {})
                event_data = new_event.get('data', {})
                return all(event_data.get(k) == v for k, v in condition.items())
            elif law_type == 'behavioral':
                context_key = parameters.get('context_key')
                expected_value = parameters.get('most_common_value')
                actual_value = new_event.get('context', {}).get(context_key)
                return actual_value == expected_value
            
            return False
        except Exception as e:
            self.logger.error(f"Error testing law: {e}")
            return False
    
    async def get_system_insights(self) -> Dict[str, Any]:
        """Get high-level insights about the system based on inferred laws."""
        try:
            insights = {
                'total_laws': len(self.inferred_laws),
                'law_types': Counter(law['type'] for law in self.inferred_laws),
                'event_types': Counter(law.get('event_type', 'unknown') for law in self.inferred_laws),
                'average_confidence': sum(law['confidence'] for law in self.inferred_laws) / len(self.inferred_laws) if self.inferred_laws else 0,
                'most_confident_law': max(self.inferred_laws, key=lambda x: x['confidence']) if self.inferred_laws else None
            }
            
            return insights
        except Exception as e:
            self.logger.error(f"Error getting system insights: {e}")
            return {}

async def main():
    """Main function for testing."""
    inference = MetaLawInference()
    
    # Simulate some events
    test_events = [
        {'type': 'user_action', 'data': {'action': 'click', 'button': 'submit'}, 'context': {'page': 'login'}},
        {'type': 'user_action', 'data': {'action': 'click', 'button': 'submit'}, 'context': {'page': 'login'}},
        {'type': 'user_action', 'data': {'action': 'click', 'button': 'submit'}, 'context': {'page': 'login'}},
        {'type': 'system_event', 'data': {'status': 'error', 'code': 500}, 'context': {'service': 'api'}},
        {'type': 'system_event', 'data': {'status': 'error', 'code': 500}, 'context': {'service': 'api'}},
    ]
    
    for event in test_events:
        await inference.observe_event(event)
    
    # Get inferred laws
    laws = await inference.get_inferred_laws()
    print("Inferred Laws:")
    for law in laws:
        print(f"- {law['description']} (confidence: {law['confidence']:.2f})")
    
    # Get insights
    insights = await inference.get_system_insights()
    print(f"\nSystem Insights: {insights}")

if __name__ == "__main__":
    asyncio.run(main())
