#!/usr/bin/env python3
"""
RAVANA Main Module with Snake Agent Integration
"""

import asyncio
import logging
from architecture_encoding import ArchitectureEncoder
from cleanup_session import SessionCleanup, ResourceCleanup
from evonet_base import EvolutionaryNetworkBase
from evonet_evolution import EvolutionEngine
from material_generator import TextMaterialGenerator
from core.snake_agent_integration import SnakeAgentIntegration

logger = logging.getLogger(__name__)

class RAVANAMain:
    """Main RAVANA system controller with Snake Agent integration."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.architecture_encoder = ArchitectureEncoder()
        self.session_cleanup = SessionCleanup()
        self.resource_cleanup = ResourceCleanup()
        self.evolution_engine = EvolutionEngine()
        self.material_generator = TextMaterialGenerator()
        
        # Snake Agent integration
        self.snake_integration = SnakeAgentIntegration({
            "snake_agent": {
                "max_threads": 8,
                "max_processes": 4,
                "max_queue_size": 1000,
                "task_timeout": 300,
                "log_level": "INFO",
                "enable_performance_monitoring": True,
                "analysis_threads": 3,
                "experiment_sandbox_dir": "snake_sandboxes",
                "data_directory": "snake_data",
                "max_experiment_time": 600,
                "max_memory_usage": 100 * 1024 * 1024,
                "filesystem_access": False,
                "network_access": False
            }
        })
        self.snake_agents_enabled = True
    
    async def initialize(self):
        """Initialize RAVANA system with Snake Agents."""
        try:
            self.logger.info("Initializing RAVANA system with Snake Agents...")
            
            # Initialize core components
            self.logger.info("Initializing core RAVANA components...")
            
            # Initialize Snake Agents if enabled
            if self.snake_agents_enabled:
                self.logger.info("Initializing Snake Agent system...")
                snake_success = await self.snake_integration.initialize()
                if snake_success:
                    self.logger.info("Snake Agent system initialized successfully")
                else:
                    self.logger.warning("Snake Agent system initialization failed, continuing without it")
                    self.snake_agents_enabled = False
            
            self.logger.info("RAVANA system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAVANA: {e}")
            return False
    
    async def start(self):
        """Start RAVANA system with Snake Agents."""
        try:
            self.logger.info("Starting RAVANA system...")
            
            # Start Snake Agents if enabled
            if self.snake_agents_enabled:
                self.logger.info("Starting Snake Agent system...")
                snake_success = await self.snake_integration.start()
                if snake_success:
                    self.logger.info("Snake Agent system started successfully")
                else:
                    self.logger.warning("Snake Agent system failed to start")
                    self.snake_agents_enabled = False
            
            self.logger.info("RAVANA system started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start RAVANA: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown RAVANA system."""
        try:
            self.logger.info("Shutting down RAVANA system...")
            
            # Shutdown Snake Agents if enabled
            if self.snake_agents_enabled:
                self.logger.info("Shutting down Snake Agent system...")
                await self.snake_integration.stop()
                self.logger.info("Snake Agent system shutdown complete")
            
            # Cleanup resources
            self.session_cleanup.cleanup_all()
            self.logger.info("RAVANA system shutdown complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to shutdown RAVANA: {e}")
            return False
    
    async def run_experiment(self, experiment_type: str, file_path: str, hypothesis: str, proposed_changes: dict):
        """Run an experiment using Snake Agents."""
        if not self.snake_agents_enabled:
            self.logger.warning("Snake Agents not enabled, cannot run experiment")
            return None
        
        try:
            experiment_id = await self.snake_integration.run_experiment(
                experiment_type=experiment_type,
                file_path=file_path,
                hypothesis=hypothesis,
                proposed_changes=proposed_changes
            )
            self.logger.info(f"Experiment {experiment_id} started successfully")
            return experiment_id
        except Exception as e:
            self.logger.error(f"Failed to run experiment: {e}")
            return None
    
    async def run_analysis(self, file_path: str, analysis_type: str, parameters: dict = None):
        """Run code analysis using Snake Agents."""
        if not self.snake_agents_enabled:
            self.logger.warning("Snake Agents not enabled, cannot run analysis")
            return None
        
        try:
            analysis_id = await self.snake_integration.run_analysis(
                file_path=file_path,
                analysis_type=analysis_type,
                parameters=parameters or {}
            )
            self.logger.info(f"Analysis {analysis_id} started successfully")
            return analysis_id
        except Exception as e:
            self.logger.error(f"Failed to run analysis: {e}")
            return None
    
    def get_status(self):
        """Get RAVANA system status including Snake Agents."""
        status = {
            "core_components": {
                "architecture_encoder": "initialized",
                "session_cleanup": "initialized",
                "resource_cleanup": "initialized",
                "evolution_engine": "initialized",
                "material_generator": "initialized"
            },
            "snake_agents_enabled": self.snake_agents_enabled
        }
        
        if self.snake_agents_enabled:
            status["snake_agents"] = self.snake_integration.get_status()
        
        return status

async def main():
    """Main function."""
    ravana = RAVANAMain()
    if await ravana.initialize():
        print("✅ RAVANA system initialized")
        if await ravana.start():
            print("✅ RAVANA system started with Snake Agents")
            print(f"Status: {ravana.get_status()}")
        else:
            print("❌ Failed to start RAVANA system")
    else:
        print("❌ Failed to initialize RAVANA system")

if __name__ == "__main__":
    asyncio.run(main())
