"""JARVIS Mark 5 - Advanced AI Assistant"""


#!/usr/bin/env python3
"""
JARVIS Voice Agent for RAVANA
Integrates JARVIS voice recognition and TTS capabilities into RAVANA AGI system
"""

import asyncio
import logging
import os
import sys
import time
from typing import Dict, Any
import queue

# Add JARVIS path for imports
jarvis_path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.append(jarvis_path)

logger = logging.getLogger(__name__)

class JARVISVoiceAgent:
    """RAVANA Agent for JARVIS Voice Recognition and TTS capabilities"""

    def __init__(self, ravana_system=None, *args, **kwargs):  # pylint: disable=unused-argument
        self.ravana_system = ravana_system
        self.name = "JARVIS Voice Agent"
        self.capabilities = [
            "voice_recognition",
            "text_to_speech",
            "voice_commands",
            "audio_processing",
            "conversation_flow"
        ]
        self.is_active = False
        self.voice_system = None
        self.whisper_voice = None
        self.audio_queue = queue.Queue()

        # Initialize JARVIS voice systems
        self._initialize_voice_systems()

    def _initialize_voice_systems(self, *args, **kwargs):  # pylint: disable=unused-argument
        """Initialize JARVIS voice recognition and TTS systems"""
        try:
            # Import unified JARVIS voice system
            from jarvis_unified_voice import JARVISUnifiedVoice

            # Initialize unified voice system (handles both recognition and TTS)
            self.voice_system = JARVISUnifiedVoice()
            self.whisper_voice = self.voice_system  # Unified system handles both
            logger.info("✅ JARVIS Unified Voice System initialized")

        except (ValueError, TypeError, AttributeError, ImportError) as e:
            logger.error("Failed to initialize JARVIS unified voice system: %s", e)
            self.voice_system = None
            self.whisper_voice = None

    async def activate(self, *args, **kwargs):  # pylint: disable=unused-argument
        """Activate the voice agent"""
        self.is_active = True
        logger.info("🎤 %s activated", self.name)

        # Start background audio processing
        await self._start_audio_processing()

    async def deactivate(self, *args, **kwargs):  # pylint: disable=unused-argument
        """Deactivate the voice agent"""
        self.is_active = False
        logger.info("🎤 %s deactivated", self.name)

    async def process_voice_input(self, audio_data: bytes) -> Dict[str, Any]:
        """Process voice input and return transcribed text"""
        try:
            if not self.voice_system:
                return {"error": "Voice system not initialized"}

            # Process audio with JARVIS voice system
            result = self.voice_system._recognize_audio_safe(audio_data)

            # Create RAVANA-compatible response
            response = {
                "agent": self.name,
                "capability": "voice_recognition",
                "transcript": result if isinstance(result, str) else result.get("text", "") if result else "",
                "confidence": result.get("confidence", 0.0) if isinstance(result, dict) else 0.0,
                "timestamp": time.time(),
                "language": result.get("language", "en") if isinstance(result, dict) else "en",
                "success": True
            }

            # Send to RAVANA for processing
            if response["transcript"]:
                await self._send_to_ravana(response["transcript"])

            return response

        except (ValueError, TypeError, AttributeError, ImportError) as e:
            logger.error("Voice processing error: %s", e)
            return {"error": str(e), "success": False}

    async def speak_text(self, text: str, voice_type: str = "unified") -> Dict[str, Any]:
        """Convert text to speech using JARVIS Unified Voice System"""
        try:
            if self.voice_system:
                # Use unified voice system (handles both recognition and TTS)
                self.voice_system.speak(text)
                success = True
                voice_system = "unified"
            else:
                return {"error": "No unified voice system available", "success": False}

            return {
                "agent": self.name,
                "capability": "text_to_speech",
                "text": text,
                "voice_system": voice_system,
                "success": success is not None,
                "timestamp": time.time()
            }

        except (ValueError, TypeError, AttributeError, ImportError) as e:
            logger.error("TTS error: %s", e)
            return {"error": str(e), "success": False}

    async def handle_voice_command(self, command: str) -> Dict[str, Any]:
        """Handle voice commands through RAVANA system"""
        try:
            # Process command through RAVANA
            response = ""
            if self.ravana_system and hasattr(self.ravana_system, 'run_single_task'):
                response = await self.ravana_system.run_single_task(f"Voice command: {command}")

            # Speak the response
            tts_result = None
            if response and len(response) > 0:
                tts_result = await self.speak_text(response)

            return {
                "agent": self.name,
                "capability": "voice_commands",
                "command": command,
                "response": response,
                "tts_result": tts_result,
                "timestamp": time.time(),
                "success": True
            }

        except (ValueError, TypeError, AttributeError, ImportError) as e:
            logger.error("Voice command error: %s", e)
            return {"error": str(e), "success": False}

    async def start_listening(self) -> Dict[str, Any]:
        """Start continuous voice listening"""
        try:
            if not self.voice_system:
                return {"error": "Voice system not initialized", "success": False}

            # Start listening in background
            listening_task = asyncio.create_task(self._continuous_listening())

            return {
                "agent": self.name,
                "capability": "voice_commands",
                "action": "started_listening",
                "task": listening_task,
                "timestamp": time.time(),
                "success": True
            }

        except (ValueError, TypeError, AttributeError, ImportError) as e:
            logger.error("Start listening error: %s", e)
            return {"error": str(e), "success": False}

    async def stop_listening(self) -> Dict[str, Any]:
        """Stop continuous voice listening"""
        try:
            # Stop any active listening tasks
            self.is_active = False

            return {
                "agent": self.name,
                "capability": "voice_commands",
                "action": "stopped_listening",
                "timestamp": time.time(),
                "success": True
            }

        except (ValueError, TypeError, AttributeError, ImportError) as e:
            logger.error("Stop listening error: %s", e)
            return {"error": str(e), "success": False}

    async def _start_audio_processing(self, *args, **kwargs):  # pylint: disable=unused-argument
        """Start background audio processing"""
        try:
            # Start audio processing loop
            asyncio.create_task(self._audio_processing_loop())
            logger.info("Audio processing loop started")
        except (ValueError, TypeError, AttributeError, ImportError) as e:
            logger.error("Failed to start audio processing: %s", e)

    async def _audio_processing_loop(self, *args, **kwargs):  # pylint: disable=unused-argument
        """Background audio processing loop"""
        while self.is_active:
            try:
                # Process any queued audio
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get_nowait()
                    await self.process_voice_input(audio_data)

                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

            except (ValueError, TypeError, AttributeError, ImportError) as e:
                logger.error("Audio processing loop error: %s", e)
                await asyncio.sleep(1)

    async def _continuous_listening(self, *args, **kwargs):  # pylint: disable=unused-argument
        """Continuous voice listening loop"""
        try:
            while self.is_active:
                # Capture audio (this would integrate with actual microphone)
                # For now, this is a placeholder for the listening loop
                await asyncio.sleep(0.1)

        except (ValueError, TypeError, AttributeError, ImportError) as e:
            logger.error("Continuous listening error: %s", e)

    async def _send_to_ravana(self, text, *args, **kwargs):  # pylint: disable=unused-argument
        """Send transcribed text to RAVANA for processing"""
        try:
            # Process through RAVANA's decision engine
            if self.ravana_system and hasattr(self.ravana_system, 'process_voice_input'):
                response = await self.ravana_system.process_voice_input(text)

                # Speak the response
                if response:
                    await self.speak_text(response)

        except (ValueError, TypeError, AttributeError, ImportError) as e:
            logger.error("Error sending to RAVANA: %s", e)

    def get_status(self) -> Dict[str, Any]:
        """Get voice agent status"""
        return {
            "name": self.name,
            "active": self.is_active,
            "capabilities": self.capabilities,
            "unified_voice_system_available": self.voice_system is not None,
            "audio_queue_size": self.audio_queue.qsize()
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on voice systems"""
        health_status = {
            "agent": self.name,
            "timestamp": time.time(),
            "overall_health": "healthy",
            "components": {}
        }

        try:
            # Check unified voice system
            if self.voice_system:
                health_status["components"]["unified_voice_system"] = "healthy"
            else:
                health_status["components"]["unified_voice_system"] = "unavailable"
                health_status["overall_health"] = "degraded"

            # Check audio queue
            if self.audio_queue.qsize() < 100:  # Reasonable queue size
                health_status["components"]["audio_queue"] = "healthy"
            else:
                health_status["components"]["audio_queue"] = "overloaded"
                health_status["overall_health"] = "degraded"

        except (ValueError, TypeError, AttributeError, ImportError) as e:
            health_status["overall_health"] = "unhealthy"
            health_status["error"] = str(e)

        return health_status
