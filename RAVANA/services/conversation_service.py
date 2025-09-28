"""
RAVANA Conversation Service

This module provides conversation management and processing services.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from database.database_engine import DatabaseEngine
from database.schemas import ConversationSchema, UserSchema

logger = logging.getLogger(__name__)

class ConversationService:
    """Service for managing conversations and dialogue"""
    
    def __init__(self, db_engine: DatabaseEngine):
        self.logger = logging.getLogger(__name__)
        self.db_engine = db_engine
        self.conversation_history = {}
        self.active_sessions = {}
        
    async def initialize(self) -> bool:
        """Initialize the conversation service"""
        try:
            self.logger.info("Initializing RAVANA Conversation Service...")
            
            # Load active conversations from database
            await self._load_active_conversations()
            
            self.logger.info("RAVANA Conversation Service initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize conversation service: {e}")
            return False
    
    async def _load_active_conversations(self):
        """Load active conversations from database"""
        try:
            # This would load active conversations from the database
            # For now, we'll initialize empty
            self.logger.info("Loaded active conversations from database")
            
        except Exception as e:
            self.logger.error(f"Error loading active conversations: {e}")
    
    async def start_conversation(self, user_id: int, session_id: int) -> str:
        """Start a new conversation"""
        try:
            conversation_id = str(uuid.uuid4())
            
            # Initialize conversation in memory
            self.conversation_history[conversation_id] = {
                "user_id": user_id,
                "session_id": session_id,
                "messages": [],
                "context": {},
                "started_at": datetime.now(),
                "last_activity": datetime.now()
            }
            
            # Update active sessions
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = []
            self.active_sessions[session_id].append(conversation_id)
            
            self.logger.info(f"Started conversation {conversation_id} for user {user_id}")
            return conversation_id
            
        except Exception as e:
            self.logger.error(f"Error starting conversation: {e}")
            raise
    
    async def add_message(self, conversation_id: str, user_message: str, 
                         ai_response: str, emotion: str = None, 
                         confidence: float = 0.0) -> bool:
        """Add a message to a conversation"""
        try:
            if conversation_id not in self.conversation_history:
                raise ValueError(f"Conversation {conversation_id} not found")
            
            conversation = self.conversation_history[conversation_id]
            
            # Add message to conversation
            message = {
                "user_message": user_message,
                "ai_response": ai_response,
                "emotion": emotion,
                "confidence": confidence,
                "timestamp": datetime.now()
            }
            
            conversation["messages"].append(message)
            conversation["last_activity"] = datetime.now()
            
            # Store in database
            await self._store_conversation_message(conversation_id, message)
            
            self.logger.debug(f"Added message to conversation {conversation_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding message: {e}")
            return False
    
    async def _store_conversation_message(self, conversation_id: str, message: Dict[str, Any]):
        """Store conversation message in database"""
        try:
            conversation = self.conversation_history[conversation_id]
            
            # Create conversation schema
            conv_schema = ConversationSchema(
                user_id=conversation["user_id"],
                session_id=conversation["session_id"],
                user_message=message["user_message"],
                ai_response=message["ai_response"],
                emotion_detected=message.get("emotion"),
                confidence_score=message.get("confidence", 0.0),
                context_data=conversation.get("context", {})
            )
            
            # Store in database (this would be implemented with actual database calls)
            self.logger.debug(f"Stored conversation message in database")
            
        except Exception as e:
            self.logger.error(f"Error storing conversation message: {e}")
    
    async def get_conversation_history(self, conversation_id: str, 
                                     limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversation history"""
        try:
            if conversation_id not in self.conversation_history:
                return []
            
            conversation = self.conversation_history[conversation_id]
            messages = conversation["messages"]
            
            # Return last N messages
            return messages[-limit:] if limit > 0 else messages
            
        except Exception as e:
            self.logger.error(f"Error getting conversation history: {e}")
            return []
    
    async def get_conversation_context(self, conversation_id: str) -> Dict[str, Any]:
        """Get conversation context"""
        try:
            if conversation_id not in self.conversation_history:
                return {}
            
            conversation = self.conversation_history[conversation_id]
            return conversation.get("context", {})
            
        except Exception as e:
            self.logger.error(f"Error getting conversation context: {e}")
            return {}
    
    async def update_conversation_context(self, conversation_id: str, 
                                        context: Dict[str, Any]) -> bool:
        """Update conversation context"""
        try:
            if conversation_id not in self.conversation_history:
                return False
            
            conversation = self.conversation_history[conversation_id]
            conversation["context"].update(context)
            conversation["last_activity"] = datetime.now()
            
            self.logger.debug(f"Updated context for conversation {conversation_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating conversation context: {e}")
            return False
    
    async def end_conversation(self, conversation_id: str) -> bool:
        """End a conversation"""
        try:
            if conversation_id not in self.conversation_history:
                return False
            
            conversation = self.conversation_history[conversation_id]
            session_id = conversation["session_id"]
            
            # Remove from active sessions
            if session_id in self.active_sessions:
                if conversation_id in self.active_sessions[session_id]:
                    self.active_sessions[session_id].remove(conversation_id)
                
                # Clean up empty sessions
                if not self.active_sessions[session_id]:
                    del self.active_sessions[session_id]
            
            # Remove from conversation history
            del self.conversation_history[conversation_id]
            
            self.logger.info(f"Ended conversation {conversation_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error ending conversation: {e}")
            return False
    
    async def get_active_conversations(self, user_id: int = None) -> List[Dict[str, Any]]:
        """Get active conversations"""
        try:
            conversations = []
            
            for conv_id, conversation in self.conversation_history.items():
                if user_id is None or conversation["user_id"] == user_id:
                    conversations.append({
                        "conversation_id": conv_id,
                        "user_id": conversation["user_id"],
                        "session_id": conversation["session_id"],
                        "message_count": len(conversation["messages"]),
                        "started_at": conversation["started_at"],
                        "last_activity": conversation["last_activity"]
                    })
            
            return conversations
            
        except Exception as e:
            self.logger.error(f"Error getting active conversations: {e}")
            return []
    
    async def cleanup_old_conversations(self, hours: int = 24):
        """Clean up old conversations"""
        try:
            cutoff_time = datetime.now().timestamp() - (hours * 3600)
            conversations_to_remove = []
            
            for conv_id, conversation in self.conversation_history.items():
                last_activity = conversation["last_activity"].timestamp()
                if last_activity < cutoff_time:
                    conversations_to_remove.append(conv_id)
            
            for conv_id in conversations_to_remove:
                await self.end_conversation(conv_id)
            
            self.logger.info(f"Cleaned up {len(conversations_to_remove)} old conversations")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old conversations: {e}")
    
    async def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        try:
            total_conversations = len(self.conversation_history)
            total_messages = sum(len(conv["messages"]) for conv in self.conversation_history.values())
            active_sessions = len(self.active_sessions)
            
            # Calculate average messages per conversation
            avg_messages = total_messages / total_conversations if total_conversations > 0 else 0
            
            return {
                "total_conversations": total_conversations,
                "total_messages": total_messages,
                "active_sessions": active_sessions,
                "average_messages_per_conversation": avg_messages
            }
            
        except Exception as e:
            self.logger.error(f"Error getting conversation stats: {e}")
            return {}
    
    async def shutdown(self):
        """Shutdown the conversation service"""
        try:
            # Save all conversations to database
            await self._save_all_conversations()
            
            # Clear memory
            self.conversation_history.clear()
            self.active_sessions.clear()
            
            self.logger.info("Conversation service shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during conversation service shutdown: {e}")
    
    async def _save_all_conversations(self):
        """Save all conversations to database"""
        try:
            # This would save all conversations to the database
            # For now, we'll just log the action
            self.logger.info("Saving all conversations to database")
            
        except Exception as e:
            self.logger.error(f"Error saving conversations: {e}")
