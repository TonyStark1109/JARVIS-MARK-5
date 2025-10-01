#!/usr/bin/env python3
"""
JARVIS Mark 5 - Clap Detection Module
Detects clap sounds to wake up JARVIS as an alternative to voice commands
"""

import pyaudio
import numpy as np
import threading
import time
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

class ClapDetector:
    """Clap detection system for JARVIS wake-up"""
    
    def __init__(self, callback: Optional[Callable] = None):
        """
        Initialize clap detector
        
        Args:
            callback: Function to call when clap is detected
        """
        self.logger = logging.getLogger(__name__)
        self.callback = callback
        self.is_running = False
        self.thread = None
        
        # Audio configuration
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.chunk = 1024
        
        # Clap detection parameters
        self.threshold = 3000  # Audio level threshold for clap detection
        self.min_clap_interval = 0.5  # Minimum time between claps (seconds)
        self.last_clap_time = 0
        
        # Audio processing
        self.audio = None
        self.stream = None
        
        # Clap pattern detection (for double clap)
        self.clap_times = []
        self.double_clap_window = 1.0  # Window for double clap detection
        
        self.logger.info("ClapDetector initialized")
    
    def _detect_clap_pattern(self, current_time: float) -> bool:
        """
        Detect clap patterns (single or double clap)
        
        Args:
            current_time: Current timestamp
            
        Returns:
            True if clap pattern detected
        """
        # Add current clap time
        self.clap_times.append(current_time)
        
        # Remove old clap times outside the detection window
        self.clap_times = [t for t in self.clap_times if current_time - t <= self.double_clap_window]
        
        # Check for clap patterns
        if len(self.clap_times) == 1:
            # Single clap
            self.logger.info("Single clap detected")
            return True
        elif len(self.clap_times) == 2:
            # Double clap
            self.logger.info("Double clap detected")
            return True
        elif len(self.clap_times) > 2:
            # Multiple claps - treat as single clap
            self.logger.info("Multiple claps detected - treating as single clap")
            self.clap_times = [current_time]  # Reset to current clap
            return True
        
        return False
    
    def _process_audio_chunk(self, data: bytes) -> bool:
        """
        Process audio chunk to detect claps
        
        Args:
            data: Raw audio data
            
        Returns:
            True if clap detected
        """
        try:
            # Convert to numpy array
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # Calculate audio level
            peak = np.abs(audio_data).max()
            rms = np.sqrt(np.mean(audio_data**2))
            
            # Log audio levels for debugging
            if peak > 1000:  # Only log when there's significant audio
                self.logger.debug("Audio level: peak=%.1f, rms=%.1f, threshold=%.1f", 
                                peak, rms, self.threshold)
            
            # Check if audio level exceeds threshold
            if peak > self.threshold:
                current_time = time.time()
                
                # Check minimum interval between claps
                if current_time - self.last_clap_time < self.min_clap_interval:
                    return False
                
                # Detect clap pattern
                if self._detect_clap_pattern(current_time):
                    self.last_clap_time = current_time
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error("Error processing audio chunk: %s", e)
            return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio stream callback for real-time processing"""
        if self.is_running:
            # Process audio chunk
            if self._process_audio_chunk(in_data):
                # Clap detected - trigger callback
                if self.callback:
                    try:
                        self.callback()
                    except Exception as e:
                        self.logger.error("Error in clap callback: %s", e)
        
        return (in_data, pyaudio.paContinue)
    
    def start_detection(self):
        """Start clap detection"""
        if self.is_running:
            self.logger.warning("Clap detection already running")
            return
        
        try:
            # Initialize PyAudio
            self.audio = pyaudio.PyAudio()
            
            # Open audio stream
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
                stream_callback=self._audio_callback
            )
            
            # Start detection
            self.is_running = True
            self.stream.start_stream()
            
            self.logger.info("Clap detection started - listening for claps...")
            self.logger.info("Threshold: %d, Min interval: %.1fs", self.threshold, self.min_clap_interval)
            
        except Exception as e:
            self.logger.error("Failed to start clap detection: %s", e)
            self.stop_detection()
            raise
    
    def stop_detection(self):
        """Stop clap detection"""
        if not self.is_running:
            return
        
        try:
            self.is_running = False
            
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            
            if self.audio:
                self.audio.terminate()
                self.audio = None
            
            self.logger.info("Clap detection stopped")
            
        except Exception as e:
            self.logger.error("Error stopping clap detection: %s", e)
    
    def set_threshold(self, threshold: int):
        """
        Set audio threshold for clap detection
        
        Args:
            threshold: Audio level threshold (higher = less sensitive)
        """
        self.threshold = threshold
        self.logger.info("Clap detection threshold set to: %d", threshold)
    
    def set_callback(self, callback: Callable):
        """
        Set callback function for clap detection
        
        Args:
            callback: Function to call when clap is detected
        """
        self.callback = callback
        self.logger.info("Clap detection callback set")
    
    def get_status(self) -> dict:
        """Get clap detector status"""
        return {
            "is_running": self.is_running,
            "threshold": self.threshold,
            "min_clap_interval": self.min_clap_interval,
            "has_callback": self.callback is not None
        }

def test_clap_detection():
    """Test clap detection functionality"""
    def on_clap_detected():
        print("CLAP DETECTED! JARVIS would wake up now!")
    
    # Create clap detector
    detector = ClapDetector(callback=on_clap_detected)
    
    try:
        print("Testing clap detection...")
        print("Clap your hands to test detection!")
        print("Press Ctrl+C to stop")
        
        # Start detection
        detector.start_detection()
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping clap detection test...")
        detector.stop_detection()
        print("Test completed!")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    test_clap_detection()
