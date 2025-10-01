"""
Snake Agent Integration

This module provides integration between the Snake Agent system and the main RAVANA AGI system,
including coordination, communication, and resource management.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from .snake_data_models import (
    SnakeAgentConfiguration, ExperimentTask, AnalysisTask, 
    create_experiment_task, create_analysis_task
)
from .snake_log_manager import SnakeLogManager, get_log_manager
from .snake_multiprocess_experimenter import MultiprocessExperimenter
from .snake_process_manager import SnakeProcessManager
from .snake_threading_manager import SnakeThreadingManager


class SnakeAgentIntegration:
    """Integration layer between Snake Agents and RAVANA AGI"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.snake_config = SnakeAgentConfiguration(**self.config.get("snake_agent", {}))
        self.log_manager = SnakeLogManager(self.config.get("snake_agent", {}))
        
        # Snake Agent components
        self.experimenter: Optional[MultiprocessExperimenter] = None
        self.process_manager: Optional[SnakeProcessManager] = None
        self.threading_manager: Optional[SnakeThreadingManager] = None
        
        # Integration state
        self.is_initialized = False
        self.is_running = False
        self.active_experiments: Dict[str, ExperimentTask] = {}
        self.active_analyses: Dict[str, AnalysisTask] = {}

    async def initialize(self) -> bool:
        """Initialize the Snake Agent integration"""
        try:
            self.logger.info("Initializing Snake Agent integration...")
            
            # Initialize log manager
            await self.log_manager.start()
            
            # Initialize experimenter
            self.logger.info("Creating MultiprocessExperimenter...")
            self.experimenter = MultiprocessExperimenter(
                config=self.snake_config,
                log_manager=self.log_manager
            )
            self.logger.info("Initializing MultiprocessExperimenter...")
            await self.experimenter.initialize()
            
            # Initialize process manager
            self.logger.info("Creating SnakeProcessManager...")
            self.process_manager = SnakeProcessManager(
                config=self.snake_config,
                log_manager=self.log_manager
            )
            self.logger.info("Initializing SnakeProcessManager...")
            await self.process_manager.initialize()
            
            # Initialize threading manager
            self.logger.info("Creating SnakeThreadingManager...")
            self.threading_manager = SnakeThreadingManager(
                config=self.snake_config,
                log_manager=self.log_manager
            )
            self.logger.info("Initializing SnakeThreadingManager...")
            await self.threading_manager.initialize()
            
            # Set up callbacks
            self._setup_callbacks()
            
            self.is_initialized = True
            await self.log_manager.log_system_event(
                "snake_agent_integration_initialized",
                {"config": self.snake_config.to_dict()},
                worker_id="snake_integration"
            )
            
            self.logger.info("Snake Agent integration initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Snake Agent integration: {e}")
            await self.log_manager.log_error(
                "snake_integration",
                "initialization_error",
                str(e)
            )
            return False

    async def start(self) -> bool:
        """Start the Snake Agent system"""
        if not self.is_initialized:
            self.logger.error("Snake Agent integration not initialized")
            return False
        
        try:
            self.logger.info("Starting Snake Agent system...")
            
            # Start all components
            success = (
                await self.process_manager.start_all_processes() and
                await self.threading_manager.start_all_threads()
            )
            
            if success:
                self.is_running = True
                await self.log_manager.log_system_event(
                    "snake_agent_system_started",
                    {"active_experiments": len(self.active_experiments)},
                    worker_id="snake_integration"
                )
                self.logger.info("Snake Agent system started successfully")
            else:
                self.logger.error("Failed to start Snake Agent system")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error starting Snake Agent system: {e}")
            await self.log_manager.log_error(
                "snake_integration",
                "start_error",
                str(e)
            )
            return False

    async def stop(self) -> bool:
        """Stop the Snake Agent system"""
        try:
            self.logger.info("Stopping Snake Agent system...")
            
            # Stop all components
            if self.process_manager:
                await self.process_manager.shutdown()
            
            if self.threading_manager:
                await self.threading_manager.shutdown()
            
            if self.experimenter:
                await self.experimenter.shutdown()
            
            # Stop log manager
            await self.log_manager.stop()
            
            self.is_running = False
            await self.log_manager.log_system_event(
                "snake_agent_system_stopped",
                {"final_experiments": len(self.active_experiments)},
                worker_id="snake_integration"
            )
            
            self.logger.info("Snake Agent system stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping Snake Agent system: {e}")
            return False

    def _setup_callbacks(self):
        """Set up callbacks for Snake Agent components"""
        if self.process_manager:
            self.process_manager.set_callbacks(
                experiment_callback=self._on_experiment_complete,
                analysis_callback=self._on_analysis_complete,
                improvement_callback=self._on_improvement_complete
            )
        
        if self.threading_manager:
            self.threading_manager.set_callbacks(
                file_change_callback=self._on_file_change,
                analysis_callback=self._on_analysis_task,
                communication_callback=self._on_communication
            )

    async def run_experiment(
        self,
        experiment_type: str,
        file_path: str,
        hypothesis: str,
        proposed_changes: Dict[str, Any],
        priority: str = "medium"
    ) -> str:
        """Run an experiment using Snake Agents"""
        if not self.is_running:
            raise RuntimeError("Snake Agent system not running")
        
        try:
            # Create experiment task
            experiment_task = create_experiment_task(
                experiment_type=experiment_type,
                file_path=file_path,
                hypothesis=hypothesis,
                proposed_changes=proposed_changes,
                priority=priority
            )
            
            # Store active experiment
            self.active_experiments[experiment_task.task_id] = experiment_task
            
            # Run experiment
            result = await self.experimenter.run_experiment(experiment_task)
            
            # Remove from active experiments
            if experiment_task.task_id in self.active_experiments:
                del self.active_experiments[experiment_task.task_id]
            
            await self.log_manager.log_system_event(
                "experiment_completed",
                {
                    "experiment_id": experiment_task.task_id,
                    "success": result.success,
                    "safety_score": result.safety_score
                },
                worker_id="snake_integration"
            )
            
            return experiment_task.task_id
            
        except Exception as e:
            self.logger.error(f"Error running experiment: {e}")
            await self.log_manager.log_error(
                "snake_integration",
                "experiment_error",
                str(e)
            )
            raise

    async def run_analysis(
        self,
        file_path: str,
        analysis_type: str,
        parameters: Dict[str, Any] = None,
        priority: str = "medium"
    ) -> str:
        """Run code analysis using Snake Agents"""
        if not self.is_running:
            raise RuntimeError("Snake Agent system not running")
        
        try:
            # Create analysis task
            analysis_task = create_analysis_task(
                file_path=file_path,
                analysis_type=analysis_type,
                parameters=parameters or {},
                priority=priority
            )
            
            # Store active analysis
            self.active_analyses[analysis_task.task_id] = analysis_task
            
            # Queue analysis task
            if self.threading_manager:
                self.threading_manager.queue_analysis_task(analysis_task)
            
            await self.log_manager.log_system_event(
                "analysis_queued",
                {
                    "analysis_id": analysis_task.task_id,
                    "file_path": file_path,
                    "analysis_type": analysis_type
                },
                worker_id="snake_integration"
            )
            
            return analysis_task.task_id
            
        except Exception as e:
            self.logger.error(f"Error running analysis: {e}")
            await self.log_manager.log_error(
                "snake_integration",
                "analysis_error",
                str(e)
            )
            raise

    async def _on_experiment_complete(self, result: Dict[str, Any]):
        """Handle experiment completion"""
        try:
            experiment_id = result.get("task_id")
            if experiment_id in self.active_experiments:
                del self.active_experiments[experiment_id]
            
            await self.log_manager.log_system_event(
                "experiment_callback_processed",
                {"experiment_id": experiment_id, "result": result},
                worker_id="snake_integration"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing experiment completion: {e}")

    async def _on_analysis_complete(self, result: Dict[str, Any]):
        """Handle analysis completion"""
        try:
            analysis_id = result.get("task_id")
            if analysis_id in self.active_analyses:
                del self.active_analyses[analysis_id]
            
            await self.log_manager.log_system_event(
                "analysis_callback_processed",
                {"analysis_id": analysis_id, "result": result},
                worker_id="snake_integration"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing analysis completion: {e}")

    async def _on_improvement_complete(self, result: Dict[str, Any]):
        """Handle improvement completion"""
        try:
            await self.log_manager.log_system_event(
                "improvement_callback_processed",
                {"result": result},
                worker_id="snake_integration"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing improvement completion: {e}")

    async def _on_file_change(self, file_event):
        """Handle file change events"""
        try:
            await self.log_manager.log_system_event(
                "file_change_detected",
                {
                    "file_path": file_event.file_path,
                    "event_type": file_event.event_type,
                    "timestamp": file_event.timestamp.isoformat()
                },
                worker_id="snake_integration"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing file change: {e}")

    async def _on_analysis_task(self, analysis_task):
        """Handle analysis task processing"""
        try:
            await self.log_manager.log_system_event(
                "analysis_task_processed",
                {
                    "task_id": analysis_task.task_id,
                    "file_path": analysis_task.file_path,
                    "analysis_type": analysis_task.analysis_type
                },
                worker_id="snake_integration"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing analysis task: {e}")

    async def _on_communication(self, comm_message):
        """Handle communication messages"""
        try:
            await self.log_manager.log_system_event(
                "communication_processed",
                {
                    "message_id": comm_message.message_id,
                    "sender": comm_message.sender_id,
                    "recipient": comm_message.recipient_id,
                    "message_type": comm_message.message_type
                },
                worker_id="snake_integration"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing communication: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get Snake Agent system status - safe serialization"""
        status = {
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "active_experiments": len(self.active_experiments),
            "active_analyses": len(self.active_analyses),
            "config": self.snake_config.to_dict()
        }
        
        # Safely get process status without pickling issues
        if self.process_manager:
            try:
                process_status = self.process_manager.get_process_status()
                # Convert to safe format
                safe_process_status = {}
                for key, value in process_status.items():
                    if isinstance(value, dict):
                        safe_process_status[key] = {k: v for k, v in value.items() 
                                                  if not hasattr(v, '__getstate__')}
                    else:
                        safe_process_status[key] = str(value) if not isinstance(value, (str, int, float, bool)) else value
                status["process_status"] = safe_process_status
            except Exception as e:
                status["process_status"] = {"error": str(e)}
        
        # Safely get threading status without pickling issues
        if self.threading_manager:
            try:
                thread_status = self.threading_manager.get_thread_status()
                # Convert to safe format
                safe_thread_status = {}
                for key, value in thread_status.items():
                    if isinstance(value, dict):
                        safe_thread_status[key] = {k: v for k, v in value.items() 
                                                 if not hasattr(v, '__getstate__')}
                    else:
                        safe_thread_status[key] = str(value) if not isinstance(value, (str, int, float, bool)) else value
                status["thread_status"] = safe_thread_status
            except Exception as e:
                status["thread_status"] = {"error": str(e)}
        
        # Safely get experimenter status
        if self.experimenter:
            try:
                experimenter_status = self.experimenter.get_status()
                # Convert to safe format
                safe_experimenter_status = {}
                for key, value in experimenter_status.items():
                    if isinstance(value, dict):
                        safe_experimenter_status[key] = {k: v for k, v in value.items() 
                                                       if not hasattr(v, '__getstate__')}
                    else:
                        safe_experimenter_status[key] = str(value) if not isinstance(value, (str, int, float, bool)) else value
                status["experimenter_status"] = safe_experimenter_status
            except Exception as e:
                status["experimenter_status"] = {"error": str(e)}
        
        return status

    def get_logs(
        self,
        worker_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get logs from the Snake Agent system"""
        return asyncio.run(self.log_manager.get_logs(
            worker_id=worker_id,
            event_type=event_type,
            limit=limit
        ))


# Global Snake Agent integration instance
_snake_integration: Optional[SnakeAgentIntegration] = None


def get_snake_integration() -> SnakeAgentIntegration:
    """Get the global Snake Agent integration instance"""
    global _snake_integration
    if _snake_integration is None:
        _snake_integration = SnakeAgentIntegration()
    return _snake_integration


def set_snake_integration(integration: SnakeAgentIntegration):
    """Set the global Snake Agent integration instance"""
    global _snake_integration
    _snake_integration = integration
