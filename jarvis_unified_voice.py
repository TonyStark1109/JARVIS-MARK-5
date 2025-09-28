#!/usr/bin/env python3
"""
JARVIS Unified Voice System
Consolidated voice recognition and TTS system with Fast Whisper + Threading
Designed for pure voice interaction like a real AI assistant
"""

import os
import sys
import json
import logging
import threading
import queue
import time
import io
import datetime
from typing import Optional, Dict, Any, Callable

import numpy as np
import pyaudio
import pygame
import requests

# Try to import faster_whisper
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    WhisperModel = None

# Try to import pyttsx3 for TTS
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    pyttsx3 = None

# Fast Whisper only - no fallback needed

# Add backend to path
sys.path.append('backend')

# Import JARVIS modules
try:
    from jarvis_desktop_automation import JARVISDesktopInterface
    from backend.modules.weather_api import WeatherAPI
    from backend.modules.hacking import EthicalHacking
except ImportError as e:
    print(f"Warning: Some JARVIS modules not available: {e}")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JARVISUnifiedVoice:
    """
    Unified JARVIS Voice System
    Combines Fast Whisper voice recognition with Paul Bettany TTS
    Designed for pure voice interaction
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Wake words and variations
        self.wake_words = [
            "hey jarvis", "jarvis", "j", "buddy", "assistant", "ai",
            "hey j", "okay jarvis", "ok jarvis", "listen jarvis",
            "wake up jarvis", "jarvis wake up", "hey buddy", "okay buddy",
            "jarvis wake up", "wake up", "hey ai", "okay ai"
        ]

        # Audio configuration for Logitech headset
        self.chunk_size = 1024
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 16000
        self.record_seconds = 3
        self.silence_threshold = 0.1  # Extremely low threshold - process all audio
        self.silence_duration = 1.0

        # Logitech device detection
        self.logitech_device_index = self._find_logitech_device()

        # Threading and queues
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.is_processing = False

        # Audio stream
        self.audio_stream = None
        self.audio = None

        # Whisper model
        self.whisper_model = None
        self._initialize_whisper()

        # Fallback recognizer
        # Fast Whisper only - no fallback needed

        # TTS System (Paul Bettany Voice)
        self.elevenlabs_api_key = self._get_elevenlabs_api_key()
        self.elevenlabs_base_url = "https://api.elevenlabs.io/v1"
        self.voice_config = self._get_voice_config()

        # Initialize pygame for audio playback
        pygame.mixer.init()

        # JARVIS modules
        self.desktop_interface = None
        self.hacking_tools = None
        self.weather_api = None
        self._initialize_modules()

        # Callbacks
        self.on_command_processed = None
        self.on_error = None

        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_audio_thread, daemon=True)
        self.processing_thread.start()

        self.logger.info("JARVIS Unified Voice System initialized")

    def _find_logitech_device(self):
        """Find Logitech headset device index"""
        try:
            audio = pyaudio.PyAudio()
            for i in range(audio.get_device_count()):
                info = audio.get_device_info_by_index(i)
                if 'logitech' in info['name'].lower() and info['maxInputChannels'] > 0:
                    self.logger.info("Found Logitech device: %s (Index: %s)", info['name'], i)
                    audio.terminate()
                    return i
            audio.terminate()
            self.logger.warning("Logitech device not found, using default")
            return None
        except (OSError, IOError, RuntimeError) as e:
            self.logger.error("Error finding Logitech device: %s", e)
            return None

    def _get_elevenlabs_api_key(self):
        """Get ElevenLabs API key from config"""
        try:
            with open('config/config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get('elevenlabs_api_key', '')
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return os.getenv('ELEVENLABS_API_KEY', '')

    def _get_voice_config(self):
        """Get JARVIS voice configuration"""
        return {
            'voice_id': 'pNInz6obpgDQGcFmaJgB',  # Adam voice
            'voice_settings': {
                'stability': 0.88,
                'similarity_boost': 0.92,
                'style': 0.45,
                'use_speaker_boost': True
            },
            'model_id': 'eleven_multilingual_v2',
            'speed': 0.95,
            'pitch': 0.72,
            'emphasis': 0.85,
            'clarity': 0.98,
            'presence': 0.85,
            'digital_processing': 0.5,
            'smoothness': 0.95,
            'intelligence': 0.9,
            'authority': 0.8,
            'warmth': 0.7
        }

    def _initialize_whisper(self):
        """Initialize Fast Whisper model"""
        if not WHISPER_AVAILABLE:
            self.logger.error(
                "Faster Whisper not available - this is required for JARVIS voice system"
            )
            return

        try:
            # Use CPU for now to avoid CUDA library issues
            self.logger.info("Using CPU for Whisper (faster setup)")
            device = "cpu"
            compute_type = "float32"

            self.logger.info("Loading Whisper model: base on %s", device)
            self.whisper_model = WhisperModel(
                "base",
                device=device,
                compute_type=compute_type
            )
            self.logger.info("Whisper model loaded successfully on %s", device)
        except (RuntimeError, OSError, ImportError) as e:
            self.logger.error("Failed to load Whisper model: %s", e)
            self.whisper_model = None


    def _initialize_modules(self):
        """Initialize JARVIS modules"""
        try:
            self.desktop_interface = JARVISDesktopInterface()
            self.hacking_tools = EthicalHacking()
            self.weather_api = WeatherAPI()
            self.logger.info("JARVIS modules initialized")
        except (ImportError, AttributeError, RuntimeError) as e:
            self.logger.warning("Some modules not available: %s", e)

    def set_command_callback(self, callback: Callable[[str], None]):
        """Set callback for command processing"""
        self.on_command_processed = callback

    def set_error_callback(self, callback: Callable[[str], None]):
        """Set callback for errors"""
        self.on_error = callback

    def start_listening(self):
        """Start continuous voice listening"""
        if self.is_listening:
            return

        try:
            self.is_listening = True
            self.is_processing = True

            # Initialize audio
            self.audio = pyaudio.PyAudio()

            # Start audio stream with Logitech device
            self.audio_stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.logitech_device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )

            self.audio_stream.start_stream()

            # Greet user
            self.speak(
                "Good day! I'm JARVIS. I'm now listening continuously. "
                "Just say 'Hey JARVIS' to activate me."
            )

            self.logger.info("Started continuous voice listening")

        except (OSError, RuntimeError, AttributeError) as e:
            self.logger.error("Failed to start listening: %s", e)
            self.is_listening = False
            if self.on_error:
                self.on_error(f"Failed to start listening: {e}")

    def stop_listening(self):
        """Stop voice listening"""
        if not self.is_listening:
            return

        try:
            self.is_listening = False
            self.is_processing = False

            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None

            if self.audio:
                self.audio.terminate()

            self.logger.info("Stopped voice listening")

        except (OSError, RuntimeError, AttributeError) as e:
            self.logger.error("Failed to stop listening: %s", e)

    def _audio_callback(self, in_data, _frame_count=None, _time_info=None, _status=None):
        """Audio stream callback for real-time processing"""
        if self.is_listening:
            # Convert audio data to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.int16)

            # Add to processing queue
            if not self.audio_queue.full():
                self.audio_queue.put(audio_data)

        return (in_data, pyaudio.paContinue)

    def _process_audio_thread(self):
        """Background thread for processing audio"""
        audio_buffer = []
        buffer_size = int(self.sample_rate * self.record_seconds)

        while True:
            try:
                if not self.is_processing:
                    time.sleep(0.1)
                    continue

                # Collect audio data
                if not self.audio_queue.empty():
                    audio_chunk = self.audio_queue.get_nowait()
                    audio_buffer.extend(audio_chunk)

                # Process when buffer is full
                if len(audio_buffer) >= buffer_size:
                    # Convert to numpy array
                    audio_data = np.array(audio_buffer[:buffer_size], dtype=np.int16)
                    audio_buffer = audio_buffer[buffer_size:]

                    # Process audio in separate thread to avoid blocking
                    threading.Thread(
                        target=self._process_audio_chunk,
                        args=(audio_data,),
                        daemon=True
                    ).start()

                time.sleep(0.01)  # Small delay to prevent high CPU usage

            except (RuntimeError, OSError, ValueError) as e:
                self.logger.error("Error in audio processing thread: %s", e)
                if self.on_error:
                    self.on_error(f"Audio processing error: {e}")

    def _process_audio_chunk(self, audio_data: np.ndarray):
        """Process a chunk of audio data"""
        try:
            # Check if audio has enough energy (not silence)
            audio_level = np.max(np.abs(audio_data))
            self.logger.info(
                "Audio level: %.2f (threshold: %.2f)", 
                audio_level,
                self.silence_threshold
            )

            if audio_level < self.silence_threshold:
                self.logger.info("Audio too quiet, skipping")
                return

            # Try Whisper first
            result = self._transcribe_with_whisper(audio_data)

            if result and result.get("text"):
                text = result["text"].strip().lower()
                confidence = result.get("confidence", 0.0)

                # Only process if confidence is reasonable (very low threshold for better detection)
                if confidence > 0.05 and len(text) > 0:
                    self.logger.info("Transcribed: '%s' (confidence: %.2f)", text, confidence)

                    # Check for wake words
                    if self._contains_wake_word(text):
                        self.logger.info("Wake word detected: '%s'", text)
                        self._handle_wake_word_detected(text)
                    else:
                        # Check if it's a command (if we're in command mode)
                        self._handle_command(text)

            else:
                # Whisper failed - log and continue
                self.logger.info("No speech detected in audio chunk or low confidence")

        except (RuntimeError, OSError, ValueError) as e:
            self.logger.error("Error processing audio chunk: %s", e)
            if self.on_error:
                self.on_error(f"Audio processing error: {e}")

    def _transcribe_with_whisper(self, audio_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Transcribe audio using Fast Whisper"""
        if not self.whisper_model:
            return None

        try:
            # Convert to float32 and normalize
            audio_float = audio_data.astype(np.float32) / 32768.0

            # Transcribe with Whisper (optimized for voice detection)
            segments, info = self.whisper_model.transcribe(
                audio_float,
                beam_size=1,  # Faster processing
                language="en",
                condition_on_previous_text=False,
                vad_filter=False,  # Disable VAD - it's removing all audio
                temperature=0.0,  # More deterministic
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.1,  # Very low threshold for speech detection
                word_timestamps=True,  # Better word recognition
                initial_prompt="Hey JARVIS"  # Help with wake word recognition
            )

            # Collect segments
            full_text = ""
            confidence_scores = []

            for segment in segments:
                full_text += segment.text
                if hasattr(segment, 'avg_logprob'):
                    confidence_scores.append(np.exp(segment.avg_logprob))

            if full_text.strip():
                avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
                return {
                    "text": full_text.strip().lower(),
                    "confidence": float(avg_confidence),
                    "language": info.language,
                    "language_probability": float(info.language_probability)
                }

        except (RuntimeError, OSError, ValueError) as e:
            self.logger.error("Whisper transcription error: %s", e)

        return None


    def _contains_wake_word(self, text: str) -> bool:
        """Check if text contains any wake words"""
        text_lower = text.lower().strip()
        for wake_word in self.wake_words:
            if wake_word in text_lower:
                return True
        return False

    def _handle_wake_word_detected(self, _text: str):
        """Handle wake word detection"""
        try:
            # JARVIS responds to wake word
            self.speak("Yes, I'm here. How can I assist you?")

            # Start listening for command
            self._listen_for_command()

        except (RuntimeError, OSError, ValueError) as e:
            self.logger.error("Error handling wake word: %s", e)
            if self.on_error:
                self.on_error(f"Error handling wake word: {e}")

    def _handle_command(self, _text: str):
        """Handle command processing"""
        try:
            # Process command through JARVIS
            response = self._process_command(_text)
            if response:
                self.speak(response)

            if self.on_command_processed:
                self.on_command_processed(_text)

        except (RuntimeError, OSError, ValueError) as e:
            self.logger.error("Error handling command: %s", e)
            if self.on_error:
                self.on_error(f"Error handling command: {e}")

    def _listen_for_command(self):
        """Listen for command after wake word"""
        try:
            # Use a simple approach - wait for next audio chunk
            # This will be handled by the continuous listening
            pass
        except (RuntimeError, OSError, ValueError) as e:
            self.logger.error("Error listening for command: %s", e)

    def _process_command(self, command: str) -> str:
        """Process JARVIS command and return response"""
        try:
            command_lower = command.lower().strip()

            # Weather commands
            if any(word in command_lower for word in ['weather', 'temperature', 'forecast']):
                if self.weather_api:
                    try:
                        weather_data = self.weather_api.get_current_weather("London")
                        condition = weather_data.get('condition', 'unknown')
                        temperature = weather_data.get('temperature', 'unknown')
                        return (
                            f"The current weather is {condition} with a temperature of "
                            f"{temperature}."
                        )
                    except (RuntimeError, OSError, ValueError, AttributeError):
                        return (
                            "I'm sorry, I couldn't retrieve the weather information "
                            "at the moment."
                        )
                else:
                    return "I'm sorry, weather information is not available at the moment."

            # Desktop automation commands
            elif any(word in command_lower for word in ['open', 'launch', 'start']):
                if self.desktop_interface:
                    try:
                        if 'notepad' in command_lower:
                            result = self.desktop_interface.open_application("notepad")
                            if result.get('success'):
                                return "Opening Notepad for you."
                            else:
                                return "Could not open Notepad."
                        elif 'calculator' in command_lower:
                            result = self.desktop_interface.open_application("calculator")
                            if result.get('success'):
                                return "Opening Calculator for you."
                            else:
                                return "Could not open Calculator."
                        else:
                            return (
                                "I can help you open applications. "
                                "Please specify which one you'd like to open."
                            )
                    except (RuntimeError, OSError, ValueError, AttributeError):
                        return "I'm sorry, I couldn't open that application."
                else:
                    return "I'm sorry, desktop automation is not available at the moment."

            # System commands
            elif any(word in command_lower for word in ['time', 'date']):
                now = datetime.datetime.now()
                time_str = now.strftime('%I:%M %p')
                date_str = now.strftime('%B %d, %Y')
                return f"The current time is {time_str} on {date_str}."

            # Greeting responses
            elif any(word in command_lower for word in ['hello', 'hi', 'hey']):
                return "Hello! I'm JARVIS, your AI assistant. How can I help you today?"

            # Help commands
            elif any(word in command_lower for word in ['help', 'what can you do']):
                return ("I can help you with weather information, open applications, "
                        "tell you the time, and assist with various tasks. "
                        "Just ask me what you need!")

            # Default response
            else:
                return (f"I understand you said '{command}'. I'm processing your request "
                        "and will help you with that.")

        except (RuntimeError, OSError, ValueError, AttributeError) as e:
            self.logger.error("Error processing command: %s", e)
            return "I'm sorry, I encountered an error processing your request. Please try again."

    def speak(self, text: str):
        """Speak text using Paul Bettany voice"""
        try:
            if not self.elevenlabs_api_key:
                self.logger.warning("ElevenLabs API key not available, using system TTS")
                self._speak_system_tts(text)
                return

            # Prepare request data
            data = {
                "text": text,
                "model_id": self.voice_config['model_id'],
                "voice_settings": self.voice_config['voice_settings']
            }

            # Make request to ElevenLabs
            url = f"{self.elevenlabs_base_url}/text-to-speech/{self.voice_config['voice_id']}"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.elevenlabs_api_key
            }

            response = requests.post(url, json=data, headers=headers, timeout=30)

            if response.status_code == 200:
                # Play audio
                audio_data = io.BytesIO(response.content)
                pygame.mixer.music.load(audio_data)
                pygame.mixer.music.play()

                # Wait for audio to finish
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)

                self.logger.info("Spoke: %s", text)
            else:
                self.logger.error("TTS request failed: %s", response.status_code)
                self._speak_system_tts(text)

        except (RuntimeError, OSError, ValueError, requests.RequestException) as e:
            self.logger.error("Error speaking: %s", e)
            self._speak_system_tts(text)

    def _speak_system_tts(self, text: str):
        """System TTS using pyttsx3"""
        try:
            if pyttsx3 is not None:
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 0.9)
                engine.say(text)
                engine.runAndWait()
            else:
                print(f"JARVIS: {text}")  # Print as fallback
        except (RuntimeError, OSError, ValueError) as e:
            self.logger.error("System TTS failed: %s", e)
            print(f"JARVIS: {text}")  # Print as last resort

    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "is_listening": self.is_listening,
            "is_processing": self.is_processing,
            "whisper_available": WHISPER_AVAILABLE and self.whisper_model is not None,
            "tts_available": bool(self.elevenlabs_api_key) or TTS_AVAILABLE,
            "wake_words": self.wake_words
        }

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.stop_listening()
            self.logger.info("JARVIS Unified Voice System cleaned up")
        except (RuntimeError, OSError, ValueError) as e:
            self.logger.error("Error during cleanup: %s", e)

class JARVISVoiceManager:
    """Singleton manager for JARVIS voice instance"""
    _instance = None

    @classmethod
    def get_unified_voice(cls) -> JARVISUnifiedVoice:
        """Get global unified voice instance"""
        if cls._instance is None:
            cls._instance = JARVISUnifiedVoice()
        return cls._instance

def get_unified_voice() -> JARVISUnifiedVoice:
    """Get global unified voice instance"""
    return JARVISVoiceManager.get_unified_voice()

def start_voice_listening():
    """Start voice listening"""
    voice_instance = get_unified_voice()
    voice_instance.start_listening()

def stop_voice_listening():
    """Stop voice listening"""
    voice_instance = get_unified_voice()
    voice_instance.stop_listening()

def set_command_callback(callback):
    """Set command callback"""
    voice_instance = get_unified_voice()
    voice_instance.set_command_callback(callback)

def set_error_callback(callback):
    """Set error callback"""
    voice_instance = get_unified_voice()
    voice_instance.set_error_callback(callback)

if __name__ == "__main__":
    # Test the unified voice system
    def on_command(text):
        """Handle command callback"""
        print(f"Command received: '{text}'")

    def on_error(error):
        """Handle error callback"""
        print(f"Error: {error}")

    voice_system = JARVISUnifiedVoice()
    voice_system.set_command_callback(on_command)
    voice_system.set_error_callback(on_error)

    print("Starting JARVIS unified voice system...")
    print("Say 'Hey JARVIS' to activate. Press Ctrl+C to stop.")

    try:
        voice_system.start_listening()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        voice_system.cleanup()
