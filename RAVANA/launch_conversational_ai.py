#!/usr/bin/env python3
"""
RAVANA Conversational AI Launcher

This module provides the main entry point for launching RAVANA's conversational AI system.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the RAVANA directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from main import RAVANAMain
from modules.communication.conversation_manager import ConversationManager
from modules.emotional_intelligence.emotion_detector import EmotionDetector
from modules.episodic_memory.memory import EpisodicMemory

logger = logging.getLogger(__name__)

class ConversationalAILauncher:
    """Launcher for RAVANA Conversational AI system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ravana = RAVANAMain()
        self.conversation_manager = None
        self.emotion_detector = None
        self.memory = None
        
    async def initialize(self):
        """Initialize the conversational AI system"""
        try:
            self.logger.info("Initializing RAVANA Conversational AI...")
            
            # Initialize RAVANA main system
            if not await self.ravana.initialize():
                self.logger.error("Failed to initialize RAVANA main system")
                return False
            
            if not await self.ravana.start():
                self.logger.error("Failed to start RAVANA main system")
                return False
            
            # Initialize conversation manager
            self.conversation_manager = ConversationManager()
            await self.conversation_manager.initialize()
            
            # Initialize emotion detector
            self.emotion_detector = EmotionDetector()
            await self.emotion_detector.initialize()
            
            # Initialize episodic memory
            self.memory = EpisodicMemory()
            await self.memory.initialize()
            
            self.logger.info("RAVANA Conversational AI initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize conversational AI: {e}")
            return False
    
    async def start_conversation(self):
        """Start the conversational AI interface"""
        try:
            self.logger.info("Starting conversational AI interface...")
            
            print("🤖 RAVANA Conversational AI is ready!")
            print("Type 'quit' to exit the conversation.")
            print("=" * 50)
            
            while True:
                try:
                    # Get user input
                    user_input = input("\nYou: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', 'bye']:
                        print("RAVANA: Goodbye! It was nice talking to you.")
                        break
                    
                    if not user_input:
                        continue
                    
                    # Process the input
                    response = await self.process_input(user_input)
                    print(f"RAVANA: {response}")
                    
                except KeyboardInterrupt:
                    print("\n\nRAVANA: Goodbye! It was nice talking to you.")
                    break
                except Exception as e:
                    self.logger.error(f"Error in conversation loop: {e}")
                    print("RAVANA: I'm sorry, I encountered an error. Please try again.")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start conversation: {e}")
            return False
    
    async def process_input(self, user_input: str) -> str:
        """Process user input and generate response"""
        try:
            # Detect emotion
            emotion = await self.emotion_detector.detect_emotion(user_input)
            
            # Store in memory
            await self.memory.store_interaction(user_input, emotion)
            
            # Generate response using conversation manager
            response = await self.conversation_manager.generate_response(
                user_input, 
                emotion=emotion,
                context=await self.memory.get_recent_context()
            )
            
            # Store response in memory
            await self.memory.store_response(response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing input: {e}")
            return "I'm sorry, I'm having trouble processing that right now. Please try again."
    
    async def shutdown(self):
        """Shutdown the conversational AI system"""
        try:
            self.logger.info("Shutting down RAVANA Conversational AI...")
            
            if self.memory:
                await self.memory.shutdown()
            
            if self.emotion_detector:
                await self.emotion_detector.shutdown()
            
            if self.conversation_manager:
                await self.conversation_manager.shutdown()
            
            if self.ravana:
                await self.ravana.shutdown()
            
            self.logger.info("RAVANA Conversational AI shutdown complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            return False

async def main():
    """Main function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    launcher = ConversationalAILauncher()
    
    try:
        if await launcher.initialize():
            await launcher.start_conversation()
        else:
            print("Failed to initialize RAVANA Conversational AI")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        await launcher.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
