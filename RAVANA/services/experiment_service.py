"""
RAVANA Experiment Service

This module provides experiment management and execution services.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from database.database_engine import DatabaseEngine
from database.schemas import ExperimentSchema, ExperimentStatus

logger = logging.getLogger(__name__)

class ExperimentService:
    """Service for managing experiments and their execution"""
    
    def __init__(self, db_engine: DatabaseEngine):
        self.logger = logging.getLogger(__name__)
        self.db_engine = db_engine
        self.active_experiments = {}
        self.experiment_queue = []
        
    async def initialize(self) -> bool:
        """Initialize the experiment service"""
        try:
            self.logger.info("Initializing RAVANA Experiment Service...")
            
            # Load pending experiments from database
            await self._load_pending_experiments()
            
            # Start experiment processor
            asyncio.create_task(self._process_experiment_queue())
            
            self.logger.info("RAVANA Experiment Service initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize experiment service: {e}")
            return False
    
    async def _load_pending_experiments(self):
        """Load pending experiments from database"""
        try:
            # This would load pending experiments from the database
            # For now, we'll initialize empty
            self.logger.info("Loaded pending experiments from database")
            
        except Exception as e:
            self.logger.error(f"Error loading pending experiments: {e}")
    
    async def create_experiment(self, user_id: int, experiment_type: str, 
                               experiment_data: Dict[str, Any]) -> str:
        """Create a new experiment"""
        try:
            experiment_id = str(uuid.uuid4())
            
            # Create experiment schema
            experiment = ExperimentSchema(
                user_id=user_id,
                experiment_type=experiment_type,
                experiment_data=experiment_data,
                status=ExperimentStatus.PENDING
            )
            
            # Store in database
            db_id = await self.db_engine.insert_experiment(
                user_id, experiment_type, experiment_data
            )
            
            # Add to active experiments
            self.active_experiments[experiment_id] = {
                "db_id": db_id,
                "experiment": experiment,
                "created_at": datetime.now(),
                "status": ExperimentStatus.PENDING
            }
            
            # Add to queue
            self.experiment_queue.append(experiment_id)
            
            self.logger.info(f"Created experiment {experiment_id} of type {experiment_type}")
            return experiment_id
            
        except Exception as e:
            self.logger.error(f"Error creating experiment: {e}")
            raise
    
    async def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment details"""
        try:
            if experiment_id not in self.active_experiments:
                return None
            
            exp_data = self.active_experiments[experiment_id]
            experiment = exp_data["experiment"]
            
            return {
                "experiment_id": experiment_id,
                "user_id": experiment.user_id,
                "experiment_type": experiment.experiment_type,
                "experiment_data": experiment.experiment_data,
                "status": experiment.status.value,
                "result": experiment.result,
                "created_at": exp_data["created_at"],
                "execution_time": experiment.execution_time
            }
            
        except Exception as e:
            self.logger.error(f"Error getting experiment: {e}")
            return None
    
    async def update_experiment_status(self, experiment_id: str, status: ExperimentStatus,
                                     result: Dict[str, Any] = None, 
                                     execution_time: float = None) -> bool:
        """Update experiment status"""
        try:
            if experiment_id not in self.active_experiments:
                return False
            
            exp_data = self.active_experiments[experiment_id]
            experiment = exp_data["experiment"]
            
            # Update experiment
            experiment.status = status
            if result is not None:
                experiment.result = result
            if execution_time is not None:
                experiment.execution_time = execution_time
            if status == ExperimentStatus.COMPLETED:
                experiment.completed_at = datetime.now()
            
            # Update in database
            await self.db_engine.update_experiment(
                exp_data["db_id"], result, status.value, execution_time
            )
            
            # Update active experiments
            exp_data["status"] = status
            
            self.logger.info(f"Updated experiment {experiment_id} status to {status.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating experiment status: {e}")
            return False
    
    async def cancel_experiment(self, experiment_id: str) -> bool:
        """Cancel an experiment"""
        try:
            if experiment_id not in self.active_experiments:
                return False
            
            # Update status to cancelled
            await self.update_experiment_status(experiment_id, ExperimentStatus.CANCELLED)
            
            # Remove from queue if present
            if experiment_id in self.experiment_queue:
                self.experiment_queue.remove(experiment_id)
            
            self.logger.info(f"Cancelled experiment {experiment_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling experiment: {e}")
            return False
    
    async def get_user_experiments(self, user_id: int, status: ExperimentStatus = None,
                                 limit: int = 50) -> List[Dict[str, Any]]:
        """Get experiments for a user"""
        try:
            experiments = []
            
            for exp_id, exp_data in self.active_experiments.items():
                experiment = exp_data["experiment"]
                
                if experiment.user_id == user_id:
                    if status is None or experiment.status == status:
                        experiments.append({
                            "experiment_id": exp_id,
                            "experiment_type": experiment.experiment_type,
                            "status": experiment.status.value,
                            "created_at": exp_data["created_at"],
                            "execution_time": experiment.execution_time
                        })
            
            # Sort by creation time (newest first)
            experiments.sort(key=lambda x: x["created_at"], reverse=True)
            
            return experiments[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting user experiments: {e}")
            return []
    
    async def get_experiment_stats(self) -> Dict[str, Any]:
        """Get experiment statistics"""
        try:
            total_experiments = len(self.active_experiments)
            queue_length = len(self.experiment_queue)
            
            # Count by status
            status_counts = {}
            for exp_data in self.active_experiments.values():
                status = exp_data["status"].value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Calculate average execution time
            completed_experiments = [
                exp_data for exp_data in self.active_experiments.values()
                if exp_data["status"] == ExperimentStatus.COMPLETED
            ]
            
            avg_execution_time = 0.0
            if completed_experiments:
                total_time = sum(exp["experiment"].execution_time for exp in completed_experiments)
                avg_execution_time = total_time / len(completed_experiments)
            
            return {
                "total_experiments": total_experiments,
                "queue_length": queue_length,
                "status_counts": status_counts,
                "average_execution_time": avg_execution_time
            }
            
        except Exception as e:
            self.logger.error(f"Error getting experiment stats: {e}")
            return {}
    
    async def _process_experiment_queue(self):
        """Process the experiment queue"""
        while True:
            try:
                if self.experiment_queue:
                    experiment_id = self.experiment_queue.pop(0)
                    await self._execute_experiment(experiment_id)
                else:
                    # No experiments in queue, wait a bit
                    await asyncio.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"Error processing experiment queue: {e}")
                await asyncio.sleep(1)
    
    async def _execute_experiment(self, experiment_id: str):
        """Execute an experiment"""
        try:
            if experiment_id not in self.active_experiments:
                return
            
            exp_data = self.active_experiments[experiment_id]
            experiment = exp_data["experiment"]
            
            # Update status to running
            await self.update_experiment_status(experiment_id, ExperimentStatus.RUNNING)
            
            self.logger.info(f"Executing experiment {experiment_id} of type {experiment.experiment_type}")
            
            # Simulate experiment execution
            start_time = datetime.now()
            
            # This is where the actual experiment would be executed
            # For now, we'll simulate with a delay
            await asyncio.sleep(2)  # Simulate execution time
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Simulate result
            result = {
                "success": True,
                "execution_time": execution_time,
                "output": f"Experiment {experiment.experiment_type} completed successfully",
                "data": experiment.experiment_data
            }
            
            # Update status to completed
            await self.update_experiment_status(
                experiment_id, 
                ExperimentStatus.COMPLETED, 
                result, 
                execution_time
            )
            
            self.logger.info(f"Completed experiment {experiment_id} in {execution_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Error executing experiment {experiment_id}: {e}")
            
            # Update status to failed
            await self.update_experiment_status(
                experiment_id, 
                ExperimentStatus.FAILED, 
                {"error": str(e)}
            )
    
    async def cleanup_completed_experiments(self, hours: int = 24):
        """Clean up completed experiments older than specified hours"""
        try:
            cutoff_time = datetime.now().timestamp() - (hours * 3600)
            experiments_to_remove = []
            
            for exp_id, exp_data in self.active_experiments.items():
                if (exp_data["status"] in [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED] and
                    exp_data["created_at"].timestamp() < cutoff_time):
                    experiments_to_remove.append(exp_id)
            
            for exp_id in experiments_to_remove:
                del self.active_experiments[exp_id]
            
            self.logger.info(f"Cleaned up {len(experiments_to_remove)} old experiments")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up experiments: {e}")
    
    async def shutdown(self):
        """Shutdown the experiment service"""
        try:
            # Cancel all pending experiments
            for exp_id in list(self.active_experiments.keys()):
                if self.active_experiments[exp_id]["status"] == ExperimentStatus.PENDING:
                    await self.cancel_experiment(exp_id)
            
            # Clear memory
            self.active_experiments.clear()
            self.experiment_queue.clear()
            
            self.logger.info("Experiment service shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during experiment service shutdown: {e}")
