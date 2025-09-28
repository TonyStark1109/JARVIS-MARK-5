"""
RAVANA Integration with JARVIS

This script integrates RAVANA with the existing JARVIS system.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add RAVANA to path
sys.path.insert(0, str(Path(__file__).parent))

from main import RAVANAMain
from services.knowledge_service import KnowledgeService
from database.database_engine import DatabaseEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class RAVANAJARVISIntegration:
    """Integration between RAVANA and JARVIS"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ravana = None
        self.knowledge_service = None
        self.db_engine = None
        
    async def initialize(self) -> bool:
        """Initialize the integration"""
        try:
            self.logger.info("Initializing RAVANA-JARVIS Integration...")
            
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
            
            # Initialize RAVANA main system
            self.ravana = RAVANAMain()
            if not await self.ravana.initialize():
                self.logger.error("Failed to initialize RAVANA main system")
                return False
            
            self.logger.info("RAVANA-JARVIS Integration initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize integration: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the integration"""
        try:
            self.logger.info("Starting RAVANA-JARVIS Integration...")
            
            # Start RAVANA system
            if not await self.ravana.start():
                self.logger.error("Failed to start RAVANA system")
                return False
            
            self.logger.info("RAVANA-JARVIS Integration started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start integration: {e}")
            return False
    
    async def run_experiment(self, experiment_type: str, file_path: str, 
                           hypothesis: str, proposed_changes: dict) -> str:
        """Run an experiment through RAVANA"""
        try:
            if not self.ravana:
                raise RuntimeError("RAVANA system not initialized")
            
            experiment_id = await self.ravana.run_experiment(
                experiment_type=experiment_type,
                file_path=file_path,
                hypothesis=hypothesis,
                proposed_changes=proposed_changes
            )
            
            if experiment_id:
                self.logger.info(f"Experiment {experiment_id} started successfully")
                return experiment_id
            else:
                self.logger.error("Failed to start experiment")
                return None
                
        except Exception as e:
            self.logger.error(f"Error running experiment: {e}")
            return None
    
    async def run_analysis(self, file_path: str, analysis_type: str, 
                          parameters: dict = None) -> str:
        """Run code analysis through RAVANA"""
        try:
            if not self.ravana:
                raise RuntimeError("RAVANA system not initialized")
            
            analysis_id = await self.ravana.run_analysis(
                file_path=file_path,
                analysis_type=analysis_type,
                parameters=parameters or {}
            )
            
            if analysis_id:
                self.logger.info(f"Analysis {analysis_id} started successfully")
                return analysis_id
            else:
                self.logger.error("Failed to start analysis")
                return None
                
        except Exception as e:
            self.logger.error(f"Error running analysis: {e}")
            return None
    
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
    
    async def get_status(self) -> dict:
        """Get integration status"""
        try:
            status = {
                "integration_initialized": self.ravana is not None,
                "knowledge_service_initialized": self.knowledge_service is not None,
                "database_initialized": self.db_engine is not None
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
            self.logger.info("Shutting down RAVANA-JARVIS Integration...")
            
            if self.ravana:
                await self.ravana.shutdown()
            
            if self.knowledge_service:
                await self.knowledge_service.shutdown()
            
            if self.db_engine:
                await self.db_engine.shutdown()
            
            self.logger.info("RAVANA-JARVIS Integration shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

async def main():
    """Main function for testing the integration"""
    try:
        # Create integration instance
        integration = RAVANAJARVISIntegration()
        
        # Initialize
        if not await integration.initialize():
            print("[ERROR] Failed to initialize RAVANA-JARVIS Integration")
            return
        
        print("[SUCCESS] RAVANA-JARVIS Integration initialized")
        
        # Start
        if not await integration.start():
            print("[ERROR] Failed to start RAVANA-JARVIS Integration")
            return
        
        print("[SUCCESS] RAVANA-JARVIS Integration started")
        
        # Test knowledge storage
        await integration.store_knowledge(
            "test_knowledge_1",
            "This is a test knowledge item for RAVANA integration",
            {"type": "test", "source": "integration_test"}
        )
        
        # Test knowledge search
        results = await integration.search_knowledge("test knowledge")
        print(f"[SUCCESS] Found {len(results)} knowledge items")
        
        # Test experiment
        experiment_id = await integration.run_experiment(
            experiment_type="code_modification",
            file_path="test_file.py",
            hypothesis="Test hypothesis",
            proposed_changes={"code": "print('Hello World')"}
        )
        
        if experiment_id:
            print(f"[SUCCESS] Experiment {experiment_id} started")
        
        # Test analysis
        analysis_id = await integration.run_analysis(
            file_path="test_file.py",
            analysis_type="syntax_check",
            parameters={"strict": True}
        )
        
        if analysis_id:
            print(f"[SUCCESS] Analysis {analysis_id} started")
        
        # Get status
        status = await integration.get_status()
        print(f"[SUCCESS] Integration status: {status}")
        
        # Shutdown
        await integration.shutdown()
        print("[SUCCESS] RAVANA-JARVIS Integration shutdown complete")
        
    except Exception as e:
        print(f"[ERROR] Error in main: {e}")
        logger.exception("Error in main function")

if __name__ == "__main__":
    asyncio.run(main())