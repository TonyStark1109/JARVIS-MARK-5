"""
Simplified RAVANA Integration - No Multiprocessing
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add RAVANA to path
sys.path.insert(0, str(Path(__file__).parent))

from main import RAVANAMain
from services.knowledge_service import KnowledgeService
from database.database_engine import DatabaseEngine
from core.snake_log_manager import SnakeLogManager
from core.snake_data_models import SnakeAgentConfiguration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class SimpleRAVANAIntegration:
    """Simplified RAVANA integration without multiprocessing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ravana = None
        self.knowledge_service = None
        self.db_engine = None
        self.snake_log_manager = None
        
    async def initialize(self) -> bool:
        """Initialize the simplified integration"""
        try:
            self.logger.info("Initializing Simplified RAVANA Integration...")
            
            # Initialize database engine
            self.db_engine = DatabaseEngine("ravana.db")
            if not await self.db_engine.initialize():
                self.logger.error("Failed to initialize database engine")
                return False
            
            # Initialize knowledge service
            self.knowledge_service = KnowledgeService()
            if not await self.knowledge_service.initialize():
                self.logger.error("Failed to initialize knowledge service")
                return False
            
            # Initialize Snake Agent log manager (without multiprocessing)
            snake_config = {
                "max_threads": 2,
                "max_processes": 1,
                "max_queue_size": 100,
                "task_timeout": 30,
                "log_level": "INFO",
                "enable_performance_monitoring": False,
                "analysis_threads": 1,
                "experiment_sandbox_dir": "snake_sandboxes",
                "data_directory": "snake_data",
                "max_experiment_time": 60,
                "max_memory_usage": 50 * 1024 * 1024,
                "filesystem_access": False,
                "network_access": False
            }
            
            self.snake_log_manager = SnakeLogManager(snake_config)
            await self.snake_log_manager.start()
            
            # Initialize RAVANA main system (without Snake Agents)
            self.ravana = RAVANAMain()
            # Disable Snake Agents to avoid multiprocessing issues
            self.ravana.snake_agents_enabled = False
            
            if not await self.ravana.initialize():
                self.logger.error("Failed to initialize RAVANA main system")
                return False
            
            self.logger.info("Simplified RAVANA Integration initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize integration: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the integration"""
        try:
            self.logger.info("Starting Simplified RAVANA Integration...")
            
            # Start RAVANA system
            if not await self.ravana.start():
                self.logger.error("Failed to start RAVANA system")
                return False
            
            self.logger.info("Simplified RAVANA Integration started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start integration: {e}")
            return False
    
    async def store_knowledge(self, knowledge_id: str, content: str, 
                            metadata: dict = None) -> bool:
        """Store knowledge in RAVANA"""
        try:
            if not self.knowledge_service:
                raise RuntimeError("Knowledge service not initialized")
            
            success = await self.knowledge_service.store_knowledge(
                knowledge_id=knowledge_id,
                content=content,
                metadata=metadata
            )
            
            if success:
                self.logger.info(f"Stored knowledge: {knowledge_id}")
            else:
                self.logger.error(f"Failed to store knowledge: {knowledge_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error storing knowledge: {e}")
            return False
    
    async def search_knowledge(self, query: str, limit: int = 10) -> list:
        """Search knowledge in RAVANA"""
        try:
            if not self.knowledge_service:
                raise RuntimeError("Knowledge service not initialized")
            
            results = await self.knowledge_service.search_knowledge(query, limit)
            self.logger.info(f"Found {len(results)} knowledge items for query: {query}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching knowledge: {e}")
            return []
    
    async def log_snake_event(self, event_type: str, data: dict):
        """Log a Snake Agent event"""
        try:
            if self.snake_log_manager:
                await self.snake_log_manager.log_system_event(
                    event_type,
                    data,
                    "INFO",
                    "simple_integration"
                )
        except Exception as e:
            self.logger.error(f"Error logging Snake event: {e}")
    
    async def get_status(self) -> dict:
        """Get integration status"""
        try:
            status = {
                "integration_initialized": self.ravana is not None,
                "knowledge_service_initialized": self.knowledge_service is not None,
                "database_initialized": self.db_engine is not None,
                "snake_log_manager_initialized": self.snake_log_manager is not None
            }
            
            if self.ravana:
                status["ravana_status"] = self.ravana.get_status()
            
            if self.knowledge_service:
                status["knowledge_stats"] = await self.knowledge_service.get_knowledge_stats()
            
            if self.db_engine:
                status["database_stats"] = await self.db_engine.get_stats()
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting status: {e}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Shutdown the integration"""
        try:
            self.logger.info("Shutting down Simplified RAVANA Integration...")
            
            if self.ravana:
                await self.ravana.shutdown()
            
            if self.knowledge_service:
                await self.knowledge_service.shutdown()
            
            if self.snake_log_manager:
                await self.snake_log_manager.stop()
            
            if self.db_engine:
                await self.db_engine.shutdown()
            
            self.logger.info("Simplified RAVANA Integration shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

async def main():
    """Main function for testing the simplified integration"""
    try:
        # Create integration instance
        integration = SimpleRAVANAIntegration()
        
        # Initialize
        if not await integration.initialize():
            print("[ERROR] Failed to initialize Simplified RAVANA Integration")
            return
        
        print("[SUCCESS] Simplified RAVANA Integration initialized")
        
        # Start
        if not await integration.start():
            print("[ERROR] Failed to start Simplified RAVANA Integration")
            return
        
        print("[SUCCESS] Simplified RAVANA Integration started")
        
        # Test knowledge storage
        await integration.store_knowledge(
            "test_knowledge_1",
            "This is a test knowledge item for simplified RAVANA integration",
            {"type": "test", "source": "simplified_integration"}
        )
        
        # Test knowledge search
        results = await integration.search_knowledge("test knowledge")
        print(f"[SUCCESS] Found {len(results)} knowledge items")
        
        # Test Snake Agent logging
        await integration.log_snake_event("test_event", {"test": "data"})
        print("[SUCCESS] Snake Agent event logged")
        
        # Get status
        status = await integration.get_status()
        print(f"[SUCCESS] Integration status: {status}")
        
        # Shutdown
        await integration.shutdown()
        print("[SUCCESS] Simplified RAVANA Integration shutdown complete")
        
    except Exception as e:
        print(f"[ERROR] Error in main: {e}")
        logger.exception("Error in main function")

if __name__ == "__main__":
    asyncio.run(main())
