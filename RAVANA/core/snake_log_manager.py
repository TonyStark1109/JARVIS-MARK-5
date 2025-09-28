"""
Snake Agent Log Manager

This module provides logging functionality for the Snake Agent system,
including structured logging, log rotation, and performance monitoring.
"""

import asyncio
import logging
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from core.snake_data_models import ExperimentRecord


class LogLevel(str, Enum):
    """Log levels for Snake Agent"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    """Structured log entry"""
    entry_id: str
    timestamp: datetime
    level: LogLevel
    worker_id: str
    event_type: str
    message: str
    data: Dict[str, Any]
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str)


class SnakeLogManager:
    """Log manager for Snake Agent system"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.log_directory = Path(self.config.get("log_directory", "./snake_logs"))
        self.max_log_size = self.config.get("max_log_size", 10 * 1024 * 1024)  # 10MB
        self.max_log_files = self.config.get("max_log_files", 5)
        self.log_level = LogLevel(self.config.get("log_level", "INFO"))
        
        # Create log directory
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Setup loggers
        self._setup_loggers()
        
        # Log buffer for batching
        self.log_buffer: List[LogEntry] = []
        self.buffer_size = self.config.get("buffer_size", 100)
        self.flush_interval = self.config.get("flush_interval", 5.0)  # seconds
        
        # Performance metrics
        self.logs_written = 0
        self.logs_dropped = 0
        self.last_flush = time.time()
        
        # Start background flush task
        self._flush_task = None
        self._shutdown_event = asyncio.Event()

    def _setup_loggers(self):
        """Setup Python loggers for different components"""
        # Main logger
        self.main_logger = logging.getLogger("snake_agent")
        self.main_logger.setLevel(getattr(logging, self.log_level.value))
        
        # Create file handler
        log_file = self.log_directory / "snake_agent.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.main_logger.addHandler(file_handler)
        
        # Component-specific loggers
        self.experiment_logger = logging.getLogger("snake_agent.experiment")
        self.process_logger = logging.getLogger("snake_agent.process")
        self.thread_logger = logging.getLogger("snake_agent.thread")
        self.system_logger = logging.getLogger("snake_agent.system")

    async def start(self):
        """Start the log manager"""
        self._flush_task = asyncio.create_task(self._flush_loop())
        await self.log_system_event(
            "log_manager_started",
            {"config": self.config},
            worker_id="log_manager"
        )

    async def stop(self):
        """Stop the log manager"""
        self._shutdown_event.set()
        if self._flush_task:
            await self._flush_task
        await self._flush_buffer()
        await self.log_system_event(
            "log_manager_stopped",
            {"logs_written": self.logs_written, "logs_dropped": self.logs_dropped},
            worker_id="log_manager"
        )

    async def _flush_loop(self):
        """Background task to flush log buffer"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.main_logger.error(f"Error in flush loop: {e}")

    async def _flush_buffer(self):
        """Flush the log buffer to files"""
        if not self.log_buffer:
            return

        try:
            # Write to structured log file
            structured_log_file = self.log_directory / "structured_logs.jsonl"
            with open(structured_log_file, "a", encoding="utf-8") as f:
                for entry in self.log_buffer:
                    f.write(entry.to_json() + "\n")

            # Write to component-specific log files
            await self._write_component_logs()

            # Clear buffer
            self.logs_written += len(self.log_buffer)
            self.log_buffer.clear()
            self.last_flush = time.time()

        except Exception as e:
            self.main_logger.error(f"Error flushing log buffer: {e}")
            self.logs_dropped += len(self.log_buffer)
            self.log_buffer.clear()

    async def _write_component_logs(self):
        """Write logs to component-specific files"""
        component_logs = {}
        
        for entry in self.log_buffer:
            component = entry.worker_id.split("_")[0] if "_" in entry.worker_id else "general"
            if component not in component_logs:
                component_logs[component] = []
            component_logs[component].append(entry)

        for component, entries in component_logs.items():
            component_file = self.log_directory / f"{component}_logs.jsonl"
            with open(component_file, "a", encoding="utf-8") as f:
                for entry in entries:
                    f.write(entry.to_json() + "\n")

    def _create_log_entry(
        self,
        level: LogLevel,
        worker_id: str,
        event_type: str,
        message: str,
        data: Dict[str, Any],
        session_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> LogEntry:
        """Create a structured log entry"""
        return LogEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            level=level,
            worker_id=worker_id,
            event_type=event_type,
            message=message,
            data=data,
            session_id=session_id,
            correlation_id=correlation_id
        )

    async def log_system_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        level: str = "INFO",
        worker_id: str = "system"
    ):
        """Log a system event"""
        log_level = LogLevel(level.upper())
        message = f"System event: {event_type}"
        
        entry = self._create_log_entry(
            level=log_level,
            worker_id=worker_id,
            event_type=event_type,
            message=message,
            data=data
        )
        
        await self._add_log_entry(entry)
        
        # Also log to Python logger
        log_method = getattr(self.system_logger, log_level.value.lower())
        log_method(f"{event_type}: {json.dumps(data)}")

    async def log_experiment(self, experiment_record: ExperimentRecord):
        """Log an experiment record"""
        data = {
            "experiment_id": experiment_record.id,
            "file_path": experiment_record.file_path,
            "experiment_type": experiment_record.experiment_type.value,
            "success": experiment_record.success,
            "safety_score": experiment_record.safety_score,
            "duration": experiment_record.duration,
            "result": experiment_record.result
        }
        
        level = LogLevel.INFO if experiment_record.success else LogLevel.ERROR
        message = f"Experiment {'completed' if experiment_record.success else 'failed'}: {experiment_record.experiment_type.value}"
        
        entry = self._create_log_entry(
            level=level,
            worker_id=experiment_record.worker_id,
            event_type="experiment",
            message=message,
            data=data
        )
        
        await self._add_log_entry(entry)
        
        # Also log to experiment logger
        if experiment_record.success:
            self.experiment_logger.info(f"Experiment {experiment_record.id} completed successfully")
        else:
            self.experiment_logger.error(f"Experiment {experiment_record.id} failed")

    async def log_process_event(
        self,
        process_id: int,
        event_type: str,
        data: Dict[str, Any],
        level: str = "INFO"
    ):
        """Log a process event"""
        log_level = LogLevel(level.upper())
        message = f"Process {process_id}: {event_type}"
        
        entry = self._create_log_entry(
            level=log_level,
            worker_id=f"process_{process_id}",
            event_type=event_type,
            message=message,
            data=data
        )
        
        await self._add_log_entry(entry)
        
        # Also log to process logger
        log_method = getattr(self.process_logger, log_level.value.lower())
        log_method(f"Process {process_id} - {event_type}: {json.dumps(data)}")

    async def log_thread_event(
        self,
        thread_id: str,
        event_type: str,
        data: Dict[str, Any],
        level: str = "INFO"
    ):
        """Log a thread event"""
        log_level = LogLevel(level.upper())
        message = f"Thread {thread_id}: {event_type}"
        
        entry = self._create_log_entry(
            level=log_level,
            worker_id=thread_id,
            event_type=event_type,
            message=message,
            data=data
        )
        
        await self._add_log_entry(entry)
        
        # Also log to thread logger
        log_method = getattr(self.thread_logger, log_level.value.lower())
        log_method(f"Thread {thread_id} - {event_type}: {json.dumps(data)}")

    async def log_error(
        self,
        worker_id: str,
        error_type: str,
        error_message: str,
        error_data: Dict[str, Any] = None
    ):
        """Log an error"""
        data = {
            "error_type": error_type,
            "error_message": error_message,
            "error_data": error_data or {}
        }
        
        entry = self._create_log_entry(
            level=LogLevel.ERROR,
            worker_id=worker_id,
            event_type="error",
            message=f"Error: {error_type}",
            data=data
        )
        
        await self._add_log_entry(entry)
        
        # Also log to main logger
        self.main_logger.error(f"{worker_id} - {error_type}: {error_message}")

    async def log_performance_metrics(
        self,
        worker_id: str,
        metrics: Dict[str, Any]
    ):
        """Log performance metrics"""
        entry = self._create_log_entry(
            level=LogLevel.INFO,
            worker_id=worker_id,
            event_type="performance_metrics",
            message="Performance metrics recorded",
            data=metrics
        )
        
        await self._add_log_entry(entry)

    async def _add_log_entry(self, entry: LogEntry):
        """Add a log entry to the buffer"""
        self.log_buffer.append(entry)
        
        # Flush if buffer is full
        if len(self.log_buffer) >= self.buffer_size:
            await self._flush_buffer()

    async def get_logs(
        self,
        worker_id: Optional[str] = None,
        event_type: Optional[str] = None,
        level: Optional[LogLevel] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Retrieve logs with filtering"""
        logs = []
        
        # Read from structured log file
        structured_log_file = self.log_directory / "structured_logs.jsonl"
        if not structured_log_file.exists():
            return logs
        
        try:
            with open(structructured_log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if len(logs) >= limit:
                        break
                    
                    try:
                        log_data = json.loads(line.strip())
                        entry = LogEntry(**log_data)
                        
                        # Apply filters
                        if worker_id and entry.worker_id != worker_id:
                            continue
                        if event_type and entry.event_type != event_type:
                            continue
                        if level and entry.level != level:
                            continue
                        if start_time and entry.timestamp < start_time:
                            continue
                        if end_time and entry.timestamp > end_time:
                            continue
                        
                        logs.append(entry.to_dict())
                        
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
        
        except Exception as e:
            self.main_logger.error(f"Error reading logs: {e}")
        
        return logs

    async def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log files"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        try:
            for log_file in self.log_directory.glob("*.log"):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink()
                    self.main_logger.info(f"Deleted old log file: {log_file}")
            
            for log_file in self.log_directory.glob("*.jsonl"):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink()
                    self.main_logger.info(f"Deleted old structured log file: {log_file}")
        
        except Exception as e:
            self.main_logger.error(f"Error cleaning up old logs: {e}")

    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        return {
            "logs_written": self.logs_written,
            "logs_dropped": self.logs_dropped,
            "buffer_size": len(self.log_buffer),
            "last_flush": self.last_flush,
            "log_directory": str(self.log_directory),
            "log_level": self.log_level.value
        }


# Global log manager instance
_log_manager: Optional[SnakeLogManager] = None


def get_log_manager() -> SnakeLogManager:
    """Get the global log manager instance"""
    global _log_manager
    if _log_manager is None:
        _log_manager = SnakeLogManager()
    return _log_manager


def set_log_manager(log_manager: SnakeLogManager):
    """Set the global log manager instance"""
    global _log_manager
    _log_manager = log_manager
