"""
RAVANA Knowledge Service
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class KnowledgeService:
    """Service for managing knowledge and information"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.knowledge_cache = {}
        
    async def initialize(self) -> bool:
        """Initialize the knowledge service"""
        try:
            self.logger.info("Initializing RAVANA Knowledge Service...")
            self.logger.info("RAVANA Knowledge Service initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize knowledge service: {e}")
            return False
    
    async def store_knowledge(self, knowledge_id: str, content: str, 
                            metadata: Dict[str, Any] = None) -> bool:
        """Store knowledge in the system"""
        try:
            knowledge_data = {
                "id": knowledge_id,
                "content": content,
                "metadata": metadata or {},
                "created_at": datetime.now(),
                "access_count": 0
            }
            
            self.knowledge_cache[knowledge_id] = knowledge_data
            self.logger.info(f"Stored knowledge: {knowledge_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing knowledge: {e}")
            return False
    
    async def get_knowledge(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """Get knowledge by ID"""
        try:
            if knowledge_id in self.knowledge_cache:
                knowledge = self.knowledge_cache[knowledge_id]
                knowledge["access_count"] += 1
                return knowledge
            return None
        except Exception as e:
            self.logger.error(f"Error getting knowledge: {e}")
            return None
    
    async def search_knowledge(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search knowledge by query"""
        try:
            results = []
            query_lower = query.lower()
            
            for knowledge_id, knowledge in self.knowledge_cache.items():
                content = knowledge["content"].lower()
                
                if query_lower in content:
                    results.append({
                        "knowledge_id": knowledge_id,
                        "content": knowledge["content"],
                        "metadata": knowledge["metadata"],
                        "created_at": knowledge["created_at"],
                        "access_count": knowledge["access_count"]
                    })
            
            return results[:limit]
            
        except Exception as e:
            self.logger.error(f"Error searching knowledge: {e}")
            return []
    
    async def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get knowledge statistics"""
        try:
            total_knowledge = len(self.knowledge_cache)
            total_accesses = sum(k["access_count"] for k in self.knowledge_cache.values())
            
            return {
                "total_knowledge": total_knowledge,
                "total_accesses": total_accesses,
                "average_accesses": total_accesses / total_knowledge if total_knowledge > 0 else 0
            }
        except Exception as e:
            self.logger.error(f"Error getting knowledge stats: {e}")
            return {}
    
    async def shutdown(self):
        """Shutdown the knowledge service"""
        try:
            self.knowledge_cache.clear()
            self.logger.info("Knowledge service shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during knowledge service shutdown: {e}")