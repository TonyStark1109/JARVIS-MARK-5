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
        
        # Snake Agent integration - Fixed configuration to avoid pickling issues
        self.snake_integration = SnakeAgentIntegration({
            "snake_agent": {
                "max_threads": 4,  # Reduced to avoid complexity
                "max_processes": 2,  # Reduced to avoid pickling issues
                "max_queue_size": 100,  # Reduced queue size
                "task_timeout": 60,  # Shorter timeout
                "log_level": "INFO",
                "enable_performance_monitoring": False,  # Disabled to avoid pickling
                "analysis_threads": 2,  # Reduced threads
                "experiment_sandbox_dir": "snake_sandboxes",
                "data_directory": "snake_data",
                "max_experiment_time": 300,  # Shorter experiment time
                "max_memory_usage": 50 * 1024 * 1024,  # Reduced memory
                "filesystem_access": True,  # Enable for file operations
                "network_access": False
            }
        })
        self.snake_agents_enabled = True
        self.background_tasks = []
    
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
                    # Start background snake agent processing
                    self._start_background_snake_processing()
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
            
            # Cancel background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
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
    
    async def run_automated_enhancement(self, project_root: str = "."):
        """Run automated file indexing and enhancement using Snake Agents"""
        if not self.snake_agents_enabled:
            self.logger.warning("Snake Agents not enabled, cannot run automated enhancement")
            return None
        
        try:
            self.logger.info("Starting automated file indexing and enhancement...")
            
            # Index Python files
            from pathlib import Path
            python_files = []
            project_path = Path(project_root)
            
            for py_file in project_path.rglob("*.py"):
                if py_file.is_file() and "__pycache__" not in str(py_file):
                    python_files.append(str(py_file))
            
            self.logger.info(f"Found {len(python_files)} Python files to process")
            
            # Process each file
            results = []
            for file_path in python_files:
                try:
                    # Run analysis
                    analysis_id = await self.run_analysis(
                        file_path=file_path,
                        analysis_type="code_quality",
                        parameters={
                            "check_syntax": True,
                            "check_imports": True,
                            "suggest_improvements": True
                        }
                    )
                    
                    # Run enhancement experiment
                    experiment_id = await self.run_experiment(
                        experiment_type="optimization",
                        file_path=file_path,
                        hypothesis=f"Enhance {file_path} with performance optimizations",
                        proposed_changes={
                            "enhancement_type": "performance_optimization",
                            "target_file": file_path,
                            "improvements": ["optimize_imports", "improve_error_handling", "add_type_hints"]
                        }
                    )
                    
                    results.append({
                        "file_path": file_path,
                        "analysis_id": analysis_id,
                        "experiment_id": experiment_id
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {e}")
                    results.append({
                        "file_path": file_path,
                        "error": str(e)
                    })
            
            self.logger.info(f"Automated enhancement completed. Processed {len(results)} files")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in automated enhancement: {e}")
            return None

    def _start_background_snake_processing(self):
        """Start background snake agent processing tasks"""
        try:
            # Create background task for continuous file monitoring and enhancement
            task = asyncio.create_task(self._background_snake_worker())
            self.background_tasks.append(task)
            self.logger.info("Background snake agent processing started")
        except Exception as e:
            self.logger.error(f"Failed to start background snake processing: {e}")

    async def _background_snake_worker(self):
        """Background worker that continuously processes files with snake agents"""
        try:
            while self.snake_agents_enabled:
                try:
                    # Get a small batch of files to process
                    from pathlib import Path
                    project_path = Path(".")
                    python_files = list(project_path.rglob("*.py"))[:5]  # Process 5 files at a time
                    
                    for py_file in python_files:
                        if "__pycache__" not in str(py_file) and py_file.is_file():
                            try:
                                # Run analysis in background
                                await self.run_analysis(
                                    file_path=str(py_file),
                                    analysis_type="code_quality",
                                    parameters={"background_analysis": True}
                                )
                                
                                # Run enhancement experiment in background
                                await self.run_experiment(
                                    experiment_type="optimization",
                                    file_path=str(py_file),
                                    hypothesis=f"Background enhancement of {py_file.name}",
                                    proposed_changes={
                                        "background_enhancement": True,
                                        "target_file": str(py_file),
                                        "improvements": ["optimize_imports", "improve_error_handling"]
                                    }
                                )
                                
                            except Exception as e:
                                self.logger.debug(f"Background processing error for {py_file}: {e}")
                    
                    # Wait before next batch
                    await asyncio.sleep(30)  # Process every 30 seconds
                    
                except Exception as e:
                    self.logger.error(f"Background snake worker error: {e}")
                    await asyncio.sleep(60)  # Wait longer on error
                    
        except asyncio.CancelledError:
            self.logger.info("Background snake worker cancelled")
        except Exception as e:
            self.logger.error(f"Background snake worker failed: {e}")

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
        
        if self.snake_agents_enabled and self.snake_integration:
            try:
                status["snake_agents"] = self.snake_integration.get_status()
            except Exception as e:
                status["snake_agents"] = {"error": str(e)}
        
        return status

async def main():
    """Main function - RAVANA runs as background service with Snake Agents."""
    ravana = RAVANAMain()
    if await ravana.initialize():
        print("SUCCESS: RAVANA system initialized")
        if await ravana.start():
            print("SUCCESS: RAVANA system started with Snake Agents running in background")
            print(f"Status: {ravana.get_status()}")
            
            # Start background snake agent processing
            print("\nSnake Agents are now running in background...")
            print("They will continuously monitor and enhance your code automatically.")
            print("Press Ctrl+C to stop the system.")
            
            try:
                # Keep the system running as a background service
                while True:
                    await asyncio.sleep(10)  # Check every 10 seconds
                    # Snake agents continue working in background
            except KeyboardInterrupt:
                print("\nShutting down RAVANA system...")
                await ravana.shutdown()
                print("RAVANA system stopped.")
        else:
            print("ERROR: Failed to start RAVANA system")
    else:
        print("ERROR: Failed to initialize RAVANA system")

if __name__ == "__main__":
    asyncio.run(main())
