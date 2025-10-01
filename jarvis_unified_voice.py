#!/usr/bin/env python3
"""
JARVIS Unified Voice System
Consolidated voice recognition and TTS system with Fast Whisper + Threading
Designed for pure voice interaction like a real AI assistant
"""
# pylint: disable=import-error,broad-exception-caught,trailing-whitespace,import-outside-toplevel

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

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pyaudio
except ImportError:
    pyaudio = None

try:
    import pygame
except ImportError:
    pygame = None

try:
    import requests
except ImportError:
    requests = None

# Try to import faster_whisper
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    WhisperModel = None

# Try to import pyttsx3 for TTS
# pyttsx3 removed - using pygame TTS only
pyttsx3 = None
TTS_AVAILABLE = False

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
        
        # TTS engine removed - using pygame TTS only

        # Set CUDA environment variables for better compatibility
        self._setup_cuda_environment()

        # Wake words and variations
        self.wake_words = [
            "hey jarvis", "jarvis", "j", "buddy", "assistant", "ai",
            "hey j", "okay jarvis", "ok jarvis", "listen jarvis",
            "wake up jarvis", "jarvis wake up", "hey buddy", "okay buddy",
            "jarvis wake up", "wake up", "hey ai", "okay ai"
        ]
        
        # Clap detection
        self.clap_detector = None
        self.clap_detection_enabled = True
        
        # Pygame audio interpretation
        self.audio_analysis_enabled = True
        self.audio_events = []

        # Audio configuration for Logitech headset - OPTIMIZED FOR SPEED
        self.chunk_size = 512  # Smaller chunks for faster processing
        self.audio_format = pyaudio.paInt16 if pyaudio else None
        self.channels = 1
        self.sample_rate = 16000
        self.record_seconds = 1.5  # Shorter recording for faster response
        self.silence_threshold = 0.01  # Very low threshold for better voice detection
        self.silence_duration = 0.5  # Shorter silence detection
        self.energy_threshold = 0.005  # Lower energy threshold for better detection

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
        if pygame:
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

        # Test TTS audio output
        self._test_audio_output()
        
        # Initialize clap detection
        self._initialize_clap_detection()
        
        self.logger.info("JARVIS Unified Voice System initialized")

    def _test_audio_output(self):
        """Test and configure audio output for TTS using pygame"""
        try:
            if pygame:
                self.logger.info("Testing pygame TTS audio output...")
                # Test pygame TTS
                self._speak_pygame_tts("Hello! This is JARVIS testing pygame TTS audio output. Can you hear me clearly through your Logitech headset?")
                self.logger.info("Pygame TTS test completed - did you hear JARVIS speaking?")
            else:
                self.logger.warning("pygame not available for audio output test")
        except Exception as e:
            self.logger.error("Audio output test failed: %s", e)

    def _initialize_clap_detection(self):
        """Initialize clap detection system"""
        try:
            # Import clap detector
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'TOOLS', 'AUDIO'))
            from ClapDetection import ClapDetector
            
            # Create clap detector with callback
            self.clap_detector = ClapDetector(callback=self._on_clap_detected)
            
            # Start clap detection if enabled
            if self.clap_detection_enabled:
                self.clap_detector.start_detection()
                self.logger.info("Clap detection initialized and started")
            else:
                self.logger.info("Clap detection initialized but disabled")
                
        except ImportError as e:
            self.logger.warning("Clap detection not available: %s", e)
            self.clap_detector = None
        except Exception as e:
            self.logger.error("Failed to initialize clap detection: %s", e)
            self.clap_detector = None

    def _on_clap_detected(self):
        """Handle clap detection - wake up JARVIS"""
        try:
            self.logger.info("Clap detected! Waking up JARVIS...")
            
            # JARVIS responds to clap
            self.speak("Yes, I heard your clap! How can I assist you?")
            
            # Start listening for command after clap
            self._listen_for_command()
            
        except Exception as e:
            self.logger.error("Error handling clap detection: %s", e)

    def _setup_cuda_environment(self):
        """Setup CUDA environment variables for better compatibility"""
        try:
            # Set CUDA environment variables for better library loading
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For debugging
            # Force CUDA 11.8 compatibility
            os.environ['CUDA_HOME'] = os.environ.get('CUDA_HOME', '')
            self.logger.info("CUDA environment variables set for GPU mode")
        except Exception as e:
            self.logger.warning("Failed to set CUDA environment: %s", e)

    def _find_logitech_device(self):
        """Find Logitech headset device index"""
        try:
            if not pyaudio:
                self.logger.warning("PyAudio not available, using default device")
                return None
            audio = pyaudio.PyAudio()
            for i in range(audio.get_device_count()):
                info = audio.get_device_info_by_index(i)
                if ('logitech' in info['name'].lower() and 
                    isinstance(info['maxInputChannels'], (int, float)) and 
                    info['maxInputChannels'] > 0):
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
        """Initialize Fast Whisper model with GPU support"""
        if not WHISPER_AVAILABLE:
            self.logger.error(
                "Faster Whisper not available - this is required for JARVIS voice system"
            )
            return

        # Force GPU usage as requested by user
        try:
            # Detect GPU availability and configure accordingly
            device, compute_type = self._detect_best_device()
            
            if device == "cuda":
                self.logger.info("Loading Whisper model: base on %s with %s", device, compute_type)
                
                # Try different compute types for CUDA 11.8 compatibility
                compute_types_to_try = [compute_type, "float16", "float32", "int8"]
                
                for ct in compute_types_to_try:
                    try:
                        self.logger.info("Attempting to load with compute_type: %s", ct)
                        self.whisper_model = WhisperModel(
                            "base",
                            device=device,
                            compute_type=ct
                        )
                        self.logger.info("SUCCESS: Whisper model loaded on %s with %s", device, ct)
                        return  # Success, exit the function
                    except Exception as e:
                        self.logger.warning("Failed with compute_type %s: %s", ct, e)
                        continue
                
                # If all compute types failed, raise error
                raise RuntimeError("All GPU compute types failed")
            else:
                raise RuntimeError("GPU mode required but device detection failed")
                
        except Exception as e:
            self.logger.error("Failed to load Whisper model on GPU: %s", e)
            self.logger.error("GPU mode is required - please check CUDA installation")
            self.whisper_model = None
            raise RuntimeError(f"GPU Whisper initialization failed: {e}") from e

    def _detect_best_device(self) -> tuple[str, str]:
        """Detect the best available device for Whisper processing"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                cuda_version = torch.version.cuda
                self.logger.info("CUDA available: %d GPU(s) detected - %s", gpu_count, gpu_name)
                self.logger.info("CUDA version: %s", cuda_version)

                # Check GPU memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.logger.info("GPU Memory: %.1f GB", gpu_memory)
                
                # Test if CUDA libraries are actually working
                try:
                    # Try a simple CUDA operation to test libraries
                    test_tensor = torch.tensor([1.0]).cuda()
                    _ = test_tensor * 2  # Test operation
                    self.logger.info("CUDA libraries working properly")
                    
                    # Force GPU usage regardless of memory (user requested GPU only)
                    if gpu_memory >= 2:  # Lowered threshold to 2GB
                        self.logger.info("Forcing GPU usage as requested")
                        return "cuda", "float16"  # Use half precision for better performance
                    else:
                        self.logger.warning("GPU memory low (%.1f GB), forcing GPU usage", 
                                           gpu_memory)
                        return "cuda", "float32"  # Use float32 for low memory
                except Exception as cuda_error:
                    self.logger.warning("CUDA libraries test failed: %s", cuda_error)
                    # Still try to use GPU even if test fails
                    self.logger.info("Attempting GPU usage despite test failure")
                    return "cuda", "float32"
            else:
                self.logger.error("CUDA not available - this is required for GPU mode")
                raise RuntimeError("CUDA not available - GPU mode required")
        except ImportError as exc:
            self.logger.error("PyTorch not available - this is required for GPU mode")
            raise RuntimeError("PyTorch not available - GPU mode required") from exc
        except Exception as e:
            self.logger.error("Error detecting GPU: %s", e)
            raise RuntimeError(f"GPU detection failed: {e}") from e


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
            if not pyaudio:
                self.logger.error("PyAudio not available, cannot start listening")
                return
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

    def enable_clap_detection(self):
        """Enable clap detection"""
        self.clap_detection_enabled = True
        if self.clap_detector:
            self.clap_detector.start_detection()
            self.logger.info("Clap detection enabled")
        else:
            self.logger.warning("Clap detector not available")

    def disable_clap_detection(self):
        """Disable clap detection"""
        self.clap_detection_enabled = False
        if self.clap_detector:
            self.clap_detector.stop_detection()
            self.logger.info("Clap detection disabled")

    def set_clap_threshold(self, threshold: int):
        """Set clap detection threshold"""
        if self.clap_detector:
            self.clap_detector.set_threshold(threshold)
            self.logger.info("Clap detection threshold set to: %d", threshold)
        else:
            self.logger.warning("Clap detector not available")

    def _audio_callback(self, in_data, _frame_count=None, _time_info=None, _status=None):
        """Audio stream callback for real-time processing"""
        if self.is_listening:
            # Convert audio data to numpy array
            if not np:
                return (in_data, pyaudio.paContinue if pyaudio else None)
            audio_data = np.frombuffer(in_data, dtype=np.int16)

            # Add to processing queue
            if not self.audio_queue.full():
                self.audio_queue.put(audio_data)

        return (in_data, pyaudio.paContinue if pyaudio else None)

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
                    if not np:
                        continue
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

    def _process_audio_chunk(self, audio_data):
        """Process a chunk of audio data"""
        try:
            # Check if audio has enough energy (not silence)
            if not np:
                return
            audio_level = np.max(np.abs(audio_data))
            rms_level = np.sqrt(np.mean(audio_data**2))
            self.logger.info(
                "Audio level: %.4f (max), %.4f (rms) - thresholds: %.4f, %.4f", 
                audio_level, rms_level, self.silence_threshold, self.energy_threshold
            )

            # Use both max and RMS thresholds for better detection
            if audio_level < self.silence_threshold and rms_level < self.energy_threshold:
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

    def _transcribe_with_whisper(self, audio_data) -> Optional[Dict[str, Any]]:
        """Transcribe audio using Fast Whisper"""
        if not self.whisper_model:
            return None

        try:
            # Convert to float32 and normalize
            if not np:
                return None
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Analyze audio with pygame for real-time interpretation
            if self.audio_analysis_enabled:
                self._analyze_audio_with_pygame(audio_data)

            # Transcribe with Whisper (ULTRA-OPTIMIZED for speed)
            segments, info = self.whisper_model.transcribe(
                audio_float,
                beam_size=1,  # Fastest processing
                language="en",
                condition_on_previous_text=False,
                vad_filter=False,  # Disable VAD for speed
                temperature=0.0,  # Deterministic
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.1,  # Higher threshold for faster processing
                word_timestamps=False,  # Disable for speed
                initial_prompt="Hey JARVIS"  # Shorter prompt for speed
            )

            # Collect segments
            full_text = ""
            confidence_scores = []

            for segment in segments:
                full_text += segment.text
                if hasattr(segment, 'avg_logprob') and np:
                    confidence_scores.append(np.exp(segment.avg_logprob))

            if full_text.strip():
                avg_confidence = np.mean(confidence_scores) if confidence_scores and np else 0.0
                return {
                    "text": full_text.strip().lower(),
                    "confidence": float(avg_confidence),
                    "language": info.language,
                    "language_probability": float(info.language_probability)
                }

        except (RuntimeError, OSError, ValueError) as e:
            self.logger.error("Whisper transcription error: %s", e)
            # If CUDA library error, try to reinitialize with CPU
            if "cublas64_12.dll" in str(e) or "CUDA" in str(e):
                self.logger.warning("CUDA library error detected, switching to CPU mode")
                try:
                    self.whisper_model = WhisperModel(
                        "base",
                        device="cpu",
                        compute_type="float32"
                    )
                    self.logger.info("Successfully switched to CPU mode")
                except Exception as cpu_error:
                    self.logger.error("Failed to switch to CPU mode: %s", cpu_error)

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

            if not requests:
                self.logger.warning("Requests not available, using system TTS")
                return self._speak_system_tts(text)
            response = requests.post(url, json=data, headers=headers, timeout=30)

            if response.status_code == 200:
                # Play audio
                if pygame:
                    audio_data = io.BytesIO(response.content)
                    pygame.mixer.music.load(audio_data)
                    pygame.mixer.music.play()

                    # Wait for audio to finish
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                else:
                    self.logger.warning("Pygame not available, using system TTS")
                    self._speak_system_tts(text)

                self.logger.info("Spoke: %s", text)
            else:
                self.logger.error("TTS request failed: %s", response.status_code)
                self._speak_system_tts(text)

        except (RuntimeError, OSError, ValueError) as e:
            self.logger.error("Error speaking: %s", e)
            self._speak_system_tts(text)

    def _speak_system_tts(self, text: str):
        """System TTS using pygame for better audio device routing"""
        try:
            # Use pygame TTS only (pyttsx3 removed)
            if pygame:
                self._speak_pygame_tts(text)
            else:
                print(f"JARVIS: {text}")  # Print as fallback
        except Exception as e:
            self.logger.error("System TTS failed: %s", e)
            print(f"JARVIS: {text}")  # Print as last resort

    def _speak_pygame_tts(self, text: str):
        """Enhanced TTS using pygame with advanced audio features"""
        try:
            import tempfile
            import os
            import time
            
            self.logger.info("Using enhanced pygame TTS for: %s", text)
            
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_wav = temp_file.name
            
            try:
                # Use Windows SAPI to generate high-quality WAV file
                import win32com.client
                speaker = win32com.client.Dispatch("SAPI.SpVoice")
                
                # Configure voice for better quality
                voices = speaker.GetVoices()
                if voices.Count > 0:
                    # Use the first available voice (usually the best one)
                    speaker.Voice = voices.Item(0)
                
                # Set speech rate and volume
                speaker.Rate = 0  # Normal rate
                speaker.Volume = 100  # Maximum volume
                
                file_stream = win32com.client.Dispatch("SAPI.SpFileStream")
                file_stream.Open(temp_wav, 3)  # 3 = write mode
                speaker.AudioOutputStream = file_stream
                speaker.Speak(text)
                file_stream.Close()
                
                # Enhanced pygame audio playback
                if pygame:
                    pygame.mixer.init(
                        frequency=44100,  # Higher quality
                        size=-16,         # 16-bit audio
                        channels=2,       # Stereo
                        buffer=1024       # Larger buffer for smoother playback
                    )
                    
                    # Load and play audio with pygame
                    pygame.mixer.music.load(temp_wav)
                    pygame.mixer.music.set_volume(1.0)  # Maximum volume
                    pygame.mixer.music.play()
                
                # Enhanced playback monitoring with audio interpretation
                self._monitor_playback_with_interpretation()
                
                self.logger.info("Enhanced pygame TTS completed - check your Logitech headset!")
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_wav):
                    try:
                        os.unlink(temp_wav)
                    except OSError:
                        pass  # File might be in use, will be cleaned up later
                    
        except ImportError:
            self.logger.warning("win32com not available, using text output")
            print(f"JARVIS: {text}")
        except Exception as e:
            self.logger.error("Pygame TTS failed: %s", e)
            print(f"JARVIS: {text}")

    def _monitor_playback_with_interpretation(self):
        """Monitor pygame playback with audio interpretation capabilities"""
        try:
            if not pygame:
                return
                
            # Wait for playback to complete while monitoring audio
            while pygame.mixer.music.get_busy():
                # Get current playback position (if available)
                try:
                    # This gives us real-time audio monitoring capability
                    current_time = pygame.time.get_ticks()
                    
                    # We could add audio analysis here in the future
                    # For now, just monitor playback status
                    pygame.time.wait(50)  # Check every 50ms
                    
                except Exception as e:
                    # If position monitoring fails, just wait
                    pygame.time.wait(100)
                    
        except Exception as e:
            self.logger.warning("Playback monitoring error: %s", e)
            # Fallback to simple wait
            if pygame:
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)

    def _analyze_audio_with_pygame(self, audio_data):
        """Analyze audio data using pygame for real-time interpretation"""
        try:
            if not pygame or not self.audio_analysis_enabled:
                return None
            
            # Convert audio data to pygame-compatible format
            import numpy as np
            
            # Convert to numpy array if not already
            if not isinstance(audio_data, np.ndarray):
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
            else:
                audio_array = audio_data
            
            # Analyze audio characteristics
            audio_analysis = {
                'peak_amplitude': np.max(np.abs(audio_array)),
                'rms_amplitude': np.sqrt(np.mean(audio_array**2)),
                'frequency_spectrum': self._get_frequency_spectrum(audio_array),
                'zero_crossing_rate': self._calculate_zero_crossing_rate(audio_array),
                'timestamp': pygame.time.get_ticks()
            }
            
            # Store audio event
            self.audio_events.append(audio_analysis)
            
            # Keep only recent events (last 100)
            if len(self.audio_events) > 100:
                self.audio_events = self.audio_events[-100:]
            
            return audio_analysis
            
        except Exception as e:
            self.logger.warning("Audio analysis failed: %s", e)
            return None

    def _get_frequency_spectrum(self, audio_data):
        """Get frequency spectrum using pygame's audio capabilities"""
        try:
            import numpy as np
            
            # Simple FFT for frequency analysis
            fft = np.fft.fft(audio_data)
            frequencies = np.fft.fftfreq(len(audio_data))
            
            # Get dominant frequency
            magnitude = np.abs(fft)
            dominant_freq_idx = np.argmax(magnitude)
            dominant_frequency = abs(frequencies[dominant_freq_idx])
            
            return {
                'dominant_frequency': dominant_frequency,
                'spectral_centroid': np.sum(frequencies * magnitude) / np.sum(magnitude),
                'spectral_rolloff': self._calculate_spectral_rolloff(magnitude, frequencies)
            }
            
        except Exception as e:
            self.logger.warning("Frequency analysis failed: %s", e)
            return None

    def _calculate_zero_crossing_rate(self, audio_data):
        """Calculate zero crossing rate for audio analysis"""
        try:
            import numpy as np
            
            # Calculate zero crossings
            zero_crossings = np.where(np.diff(np.signbit(audio_data)))[0]
            zcr = len(zero_crossings) / len(audio_data)
            
            return zcr
            
        except Exception as e:
            self.logger.warning("Zero crossing calculation failed: %s", e)
            return 0

    def _calculate_spectral_rolloff(self, magnitude, frequencies, rolloff_threshold=0.85):
        """Calculate spectral rolloff frequency"""
        try:
            import numpy as np
            
            # Calculate cumulative sum
            cumsum = np.cumsum(magnitude)
            total_energy = cumsum[-1]
            
            # Find rolloff frequency
            rolloff_energy = rolloff_threshold * total_energy
            rolloff_idx = np.where(cumsum >= rolloff_energy)[0]
            
            if len(rolloff_idx) > 0:
                return abs(frequencies[rolloff_idx[0]])
            else:
                return abs(frequencies[-1])
                
        except Exception as e:
            self.logger.warning("Spectral rolloff calculation failed: %s", e)
            return 0

    def get_audio_analysis(self):
        """Get recent audio analysis data"""
        return {
            'recent_events': self.audio_events[-10:] if self.audio_events else [],
            'total_events': len(self.audio_events),
            'analysis_enabled': self.audio_analysis_enabled
        }


    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        gpu_info = self._get_gpu_info()
        clap_status = self.clap_detector.get_status() if self.clap_detector else {"is_running": False}
        
        return {
            "is_listening": self.is_listening,
            "is_processing": self.is_processing,
            "whisper_available": WHISPER_AVAILABLE and self.whisper_model is not None,
            "tts_available": bool(self.elevenlabs_api_key) or TTS_AVAILABLE,
            "wake_words": self.wake_words,
            "gpu_info": gpu_info,
            "clap_detection": clap_status,
            "audio_config": {
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "silence_threshold": self.silence_threshold,
                "energy_threshold": self.energy_threshold
            }
        }

    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information for debugging"""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "cuda_available": True,
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                    "current_device": torch.cuda.current_device()
                }
            else:
                return {"cuda_available": False, "reason": "CUDA not available"}
        except ImportError:
            return {"cuda_available": False, "reason": "PyTorch not installed"}
        except Exception as e:
            return {"cuda_available": False, "reason": f"Error: {e}"}

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.stop_listening()
            
            # Stop clap detection
            if self.clap_detector:
                self.clap_detector.stop_detection()
                self.clap_detector = None
            
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
