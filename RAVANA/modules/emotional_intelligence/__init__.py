"""
RAVANA Emotional Intelligence Module
Handles emotion modeling and emotional decision-making.
"""

from .emotion_model import EmotionModel
from .emotion_detector import UltronEmotionDetector

__all__ = [
    'EmotionModel',
    'UltronEmotionDetector'
]
