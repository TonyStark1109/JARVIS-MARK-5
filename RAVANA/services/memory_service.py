"""
RAVANA Memory Service
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from modules.episodic_memory.memory import EpisodicMemory, MemoryItem

logger = logging.getLogger(__name__)

class MemoryService:
    """Service for managing episodic memory"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.memory = EpisodicMemory()
        
    async def initialize(self) -> bool:
        """Initialize the memory service"""
        try:
            self.logger.info("Initializing RAVANA Memory Service...")
            
            # Load existing memories
            self.memory.load_memories()
            
            self.logger.info("RAVANA Memory Service initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize memory service: {e}")
            return False
    
    async def store_memory(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """Store a new memory"""
        try:
            memory_id = self.memory.add_memory(text, metadata)
            self.logger.info(f"Stored memory: {memory_id}")
            return memory_id
            
        except Exception as e:
            self.logger.error(f"Error storing memory: {e}")
            return None
    
    async def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Get a specific memory"""
        try:
            return self.memory.get_memory(memory_id)
        except Exception as e:
            self.logger.error(f"Error getting memory: {e}")
            return None
    
    async def search_memories(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """Search memories by query"""
        try:
            return self.memory.search_memories(query, limit)
        except Exception as e:
            self.logger.error(f"Error searching memories: {e}")
            return []
    
    async def update_memory(self, memory_id: str, text: str = None, 
                          metadata: Dict[str, Any] = None) -> bool:
        """Update an existing memory"""
        try:
            return self.memory.update_memory(memory_id, text, metadata)
        except Exception as e:
            self.logger.error(f"Error updating memory: {e}")
            return False
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory"""
        try:
            return self.memory.delete_memory(memory_id)
        except Exception as e:
            self.logger.error(f"Error deleting memory: {e}")
            return False
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        try:
            return self.memory.get_memory_stats()
        except Exception as e:
            self.logger.error(f"Error getting memory stats: {e}")
            return {}
    
    async def clear_memories(self):
        """Clear all memories"""
        try:
            self.memory.clear_memories()
            self.logger.info("Cleared all memories")
        except Exception as e:
            self.logger.error(f"Error clearing memories: {e}")
    
    async def get_recent_memories(self, limit: int = 10) -> List[MemoryItem]:
        """Get recent memories"""
        try:
            all_memories = self.memory.get_all_memories()
            # Sort by timestamp and return most recent
            all_memories.sort(key=lambda x: x.timestamp, reverse=True)
            return all_memories[:limit]
        except Exception as e:
            self.logger.error(f"Error getting recent memories: {e}")
            return []
    
    async def consolidate_memories(self) -> bool:
        """Consolidate similar memories"""
        try:
            # Simple consolidation - remove very similar memories
            all_memories = self.memory.get_all_memories()
            memories_to_remove = []
            
            for i, memory1 in enumerate(all_memories):
                for j, memory2 in enumerate(all_memories[i+1:], i+1):
                    # Simple similarity check
                    if memory1.text.lower() == memory2.text.lower():
                        memories_to_remove.append(memory2.id)
            
            # Remove duplicate memories
            for memory_id in memories_to_remove:
                await self.delete_memory(memory_id)
            
            self.logger.info(f"Consolidated {len(memories_to_remove)} duplicate memories")
            return True
            
        except Exception as e:
            self.logger.error(f"Error consolidating memories: {e}")
            return False
    
    async def export_memories(self, format_type: str = "json") -> str:
        """Export all memories"""
        try:
            all_memories = self.memory.get_all_memories()
            
            if format_type == "json":
                memories_data = []
                for memory in all_memories:
                    memories_data.append({
                        "id": memory.id,
                        "text": memory.text,
                        "metadata": memory.metadata,
                        "timestamp": memory.timestamp.isoformat()
                    })
                return json.dumps(memories_data, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            self.logger.error(f"Error exporting memories: {e}")
            return ""
    
    async def import_memories(self, data: str, format_type: str = "json") -> bool:
        """Import memories from data"""
        try:
            if format_type == "json":
                memories_data = json.loads(data)
                
                for memory_data in memories_data:
                    await self.store_memory(
                        memory_data["text"],
                        memory_data.get("metadata", {})
                    )
                
                self.logger.info(f"Imported {len(memories_data)} memories")
                return True
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            self.logger.error(f"Error importing memories: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the memory service"""
        try:
            # Save memories before shutdown
            self.memory.save_memories()
            self.logger.info("Memory service shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during memory service shutdown: {e}")