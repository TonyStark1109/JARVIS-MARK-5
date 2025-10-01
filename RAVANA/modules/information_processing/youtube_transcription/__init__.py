"""
YouTube Transcription Module

Handles YouTube video transcription and audio processing.
"""

from .transcriber import YouTubeTranscriber
from .audio_processor import AudioProcessor

__all__ = [
    'YouTubeTranscriber',
    'AudioProcessor'
]
