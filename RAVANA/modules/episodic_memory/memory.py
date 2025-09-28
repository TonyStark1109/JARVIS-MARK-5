"""
RAVANA Episodic Memory Module
"""

import os
import logging
import requests
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class MemoryItem:
    """Represents a memory item."""
    id: str
    text: str
    similarity: float
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "similarity": self.similarity,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {}
        }
    
    def model_dump(self, *args, **kwargs):  # pylint: disable=unused-argument
        """Model dump for compatibility."""
        return self.to_dict()

class EpisodicMemory:
    """Manages episodic memory for RAVANA."""
    
    def __init__(self, memory_file: str = "episodic_memory.json"):
        self.logger = logging.getLogger(__name__)
        self.memory_file = memory_file
        self.memories = []
        self.embedding_model = None
        self.multimodal_service = None
        self.load_memories()
    
    def load_memories(self):
        """Load memories from file."""
        try:
            if os.path.exists(self.memory_file):
                import json
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.memories = [
                        MemoryItem(
                            id=item['id'],
                            text=item['text'],
                            similarity=item.get('similarity', 0.0),
                            timestamp=datetime.fromisoformat(item['timestamp']),
                            metadata=item.get('metadata', {})
                        )
                        for item in data.get('memories', [])
                    ]
                self.logger.info(f"Loaded {len(self.memories)} memories")
            else:
                self.logger.info("No memory file found, starting fresh")
        except Exception as e:
            self.logger.error(f"Failed to load memories: {e}")
            self.memories = []
    
    def save_memories(self):
        """Save memories to file."""
        try:
            import json
            data = {
                "memories": [memory.to_dict() for memory in self.memories]
            }
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"Saved {len(self.memories)} memories")
        except Exception as e:
            self.logger.error(f"Failed to save memories: {e}")
    
    def add_memory(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """Add a new memory."""
        try:
            memory_id = f"memory_{len(self.memories)}_{datetime.now().timestamp()}"
            memory = MemoryItem(
                id=memory_id,
                text=text,
                similarity=0.0,
                timestamp=datetime.now(),
                metadata=metadata or {}
            )
            self.memories.append(memory)
            self.save_memories()
            self.logger.info(f"Added memory: {memory_id}")
            return memory_id
        except Exception as e:
            self.logger.error(f"Failed to add memory: {e}")
            return ""
    
    def search_memories(self, query: str, limit: int = 10) -> List[MemoryItem]:
        """Search memories by similarity."""
        try:
            if not self.memories:
                return []
            
            # Simple text-based similarity search
            query_lower = query.lower()
            scored_memories = []
            
            for memory in self.memories:
                text_lower = memory.text.lower()
                # Simple word overlap similarity
                query_words = set(query_lower.split())
                text_words = set(text_lower.split())
                overlap = len(query_words.intersection(text_words))
                similarity = overlap / max(len(query_words), 1)
                
                memory.similarity = similarity
                scored_memories.append(memory)
            
            # Sort by similarity and return top results
            scored_memories.sort(key=lambda x: x.similarity, reverse=True)
            return scored_memories[:limit]

    except Exception as e:
            self.logger.error(f"Failed to search memories: {e}")
        return []
    
    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Get a specific memory by ID."""
        try:
            for memory in self.memories:
                if memory.id == memory_id:
                    return memory
            return None
    except Exception as e:
            self.logger.error(f"Failed to get memory {memory_id}: {e}")
            return None
    
    def update_memory(self, memory_id: str, text: str = None, metadata: Dict[str, Any] = None) -> bool:
        """Update an existing memory."""
        try:
            memory = self.get_memory(memory_id)
            if not memory:
                return False
            
            if text is not None:
                memory.text = text
            if metadata is not None:
                memory.metadata.update(metadata)
            
            memory.timestamp = datetime.now()
            self.save_memories()
            self.logger.info(f"Updated memory: {memory_id}")
            return True
    except Exception as e:
            self.logger.error(f"Failed to update memory {memory_id}: {e}")
            return False
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        try:
            for i, memory in enumerate(self.memories):
                if memory.id == memory_id:
                    del self.memories[i]
                    self.save_memories()
                    self.logger.info(f"Deleted memory: {memory_id}")
                    return True
            return False
    except Exception as e:
            self.logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
    
    def get_all_memories(self) -> List[MemoryItem]:
        """Get all memories."""
        return self.memories.copy()
    
    def clear_memories(self):
        """Clear all memories."""
        try:
            self.memories.clear()
            self.save_memories()
            self.logger.info("Cleared all memories")
    except Exception as e:
            self.logger.error(f"Failed to clear memories: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        try:
            return {
                "total_memories": len(self.memories),
                "oldest_memory": min(m.timestamp for m in self.memories).isoformat() if self.memories else None,
                "newest_memory": max(m.timestamp for m in self.memories).isoformat() if self.memories else None,
                "memory_file": self.memory_file
            }
    except Exception as e:
            self.logger.error(f"Failed to get memory stats: {e}")
            return {}

def main():
    """Main function."""
    memory = EpisodicMemory()
    
    # Example usage
    memory.add_memory("I learned about Python programming today", {"topic": "programming"})
    memory.add_memory("The weather was sunny and warm", {"topic": "weather"})
    
    results = memory.search_memories("Python programming")
    print(f"Found {len(results)} memories about Python programming")
    
    stats = memory.get_memory_stats()
    print(f"Memory stats: {stats}")

if __name__ == "__main__":
    main()