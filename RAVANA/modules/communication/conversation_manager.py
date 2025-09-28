#!/usr/bin/env python3
"""
RAVANA Conversation Manager
Manages multi-agent conversations and context tracking.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ConversationManager:
    """Manages conversations between RAVANA agents and external systems."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        self.conversation_history: List[Dict[str, Any]] = []
        self.context_window = 50  # Maximum messages to keep in context
        
    async def start_conversation(self, conversation_id: str, participants: List[str]) -> bool:
        """Start a new conversation with specified participants."""
        try:
            self.active_conversations[conversation_id] = {
                'participants': participants,
                'messages': [],
                'start_time': datetime.now(),
                'context': {},
                'status': 'active'
            }
            self.logger.info(f"Started conversation {conversation_id} with participants: {participants}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start conversation {conversation_id}: {e}")
            return False
    
    async def add_message(self, conversation_id: str, sender: str, content: str, 
                         message_type: str = "text", metadata: Optional[Dict] = None) -> bool:
        """Add a message to an active conversation."""
        try:
            if conversation_id not in self.active_conversations:
                self.logger.warning(f"Conversation {conversation_id} not found")
                return False
            
            message = {
                'id': f"{conversation_id}_{len(self.active_conversations[conversation_id]['messages'])}",
                'sender': sender,
                'content': content,
                'type': message_type,
                'timestamp': datetime.now(),
                'metadata': metadata or {}
            }
            
            self.active_conversations[conversation_id]['messages'].append(message)
            
            # Maintain context window
            if len(self.active_conversations[conversation_id]['messages']) > self.context_window:
                self.active_conversations[conversation_id]['messages'] = \
                    self.active_conversations[conversation_id]['messages'][-self.context_window:]
            
            self.logger.debug(f"Added message to conversation {conversation_id} from {sender}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add message to conversation {conversation_id}: {e}")
            return False
    
    async def get_conversation_context(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get the current context for a conversation."""
        if conversation_id not in self.active_conversations:
            return []
        return self.active_conversations[conversation_id]['messages']
    
    async def end_conversation(self, conversation_id: str) -> bool:
        """End an active conversation and archive it."""
        try:
            if conversation_id not in self.active_conversations:
                return False
            
            conversation = self.active_conversations.pop(conversation_id)
            conversation['end_time'] = datetime.now()
            conversation['status'] = 'ended'
            
            self.conversation_history.append(conversation)
            self.logger.info(f"Ended conversation {conversation_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to end conversation {conversation_id}: {e}")
            return False
    
    async def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Get a summary of a conversation."""
        if conversation_id not in self.active_conversations:
            return {}
        
        conv = self.active_conversations[conversation_id]
        return {
            'id': conversation_id,
            'participants': conv['participants'],
            'message_count': len(conv['messages']),
            'duration': (datetime.now() - conv['start_time']).total_seconds(),
            'status': conv['status']
        }
