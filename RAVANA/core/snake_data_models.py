"""
Snake Agent Data Models

This module defines the data models and schemas for the Snake Agent system,
including experiment tasks, process states, thread states, and configuration.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import uuid


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
        import json
        return json.dumps(self.to_dict(), default=str)


class TaskPriority(str, Enum):
    """Priority levels for tasks"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(str, Enum):
    """Status of tasks"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessStatus(str, Enum):
    """Status of processes"""
    STARTING = "starting"
    ACTIVE = "active"
    IDLE = "idle"
    ERROR = "error"
    STOPPED = "stopped"


class ThreadStatus(str, Enum):
    """Status of threads"""
    STARTING = "starting"
    RUNNING = "running"
    IDLE = "idle"
    ERROR = "error"
    STOPPED = "stopped"


class ExperimentType(str, Enum):
    """Types of experiments"""
    CODE_MODIFICATION = "code_modification"
    PERFORMANCE_TEST = "performance_test"
    SECURITY_TEST = "security_test"
    INTEGRATION_TEST = "integration_test"
    REFACTORING = "refactoring"
    OPTIMIZATION = "optimization"


@dataclass
class TaskInfo:
    """Information about a task"""
    task_id: str
    task_type: str
    priority: TaskPriority
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    description: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ExperimentTask:
    """Task for running experiments"""
    task_id: str
    experiment_type: ExperimentType
    file_path: str
    hypothesis: str
    proposed_changes: Dict[str, Any]
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout_seconds: int = 300
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ExperimentRecord:
    """Record of an experiment execution"""
    id: str
    file_path: str
    experiment_type: ExperimentType
    description: str
    hypothesis: str
    methodology: str
    result: Dict[str, Any]
    success: bool
    safety_score: float
    duration: float
    timestamp: datetime
    worker_id: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ProcessState:
    """State of a worker process"""
    process_id: int
    name: str
    status: ProcessStatus
    start_time: datetime
    last_heartbeat: datetime
    process_object: Optional[Any] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    last_activity: Optional[str] = None
    error_count: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def update_activity(self, activity: str):
        """Update last activity"""
        self.last_activity = activity
        self.last_heartbeat = datetime.now()

    def increment_completed(self):
        """Increment completed tasks"""
        self.tasks_completed += 1
        self.update_activity("task_completed")

    def increment_failed(self):
        """Increment failed tasks"""
        self.tasks_failed += 1
        self.error_count += 1
        self.update_activity("task_failed")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding non-serializable objects"""
        data = asdict(self)
        # Remove non-serializable process object
        data['process_object'] = None
        return data


@dataclass
class ThreadState:
    """State of a worker thread"""
    thread_id: str
    name: str
    status: ThreadStatus
    start_time: datetime
    last_activity: datetime
    thread_object: Optional[Any] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    error_count: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def update_activity(self, activity: str):
        """Update last activity"""
        self.last_activity = datetime.now()

    def increment_processed(self):
        """Increment processed tasks"""
        self.tasks_completed += 1
        self.update_activity("task_processed")

    def increment_error(self):
        """Increment error count"""
        self.tasks_failed += 1
        self.error_count += 1
        self.update_activity("error_occurred")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding non-serializable objects"""
        data = asdict(self)
        # Remove non-serializable thread object
        data['thread_object'] = None
        return data


@dataclass
class FileChangeEvent:
    """Event representing a file change"""
    event_id: str
    file_path: str
    event_type: str  # created, modified, deleted, moved
    timestamp: datetime
    file_size: Optional[int] = None
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class AnalysisTask:
    """Task for code analysis"""
    task_id: str
    file_path: str
    analysis_type: str
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    parameters: Dict[str, Any] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.parameters is None:
            self.parameters = {}
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class CommunicationMessage:
    """Message for inter-agent communication"""
    message_id: str
    sender_id: str
    recipient_id: str
    message_type: str
    content: Dict[str, Any]
    timestamp: datetime
    priority: TaskPriority = TaskPriority.MEDIUM
    status: str = "pending"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class WorkerMetrics:
    """Metrics for worker performance"""
    worker_id: str
    worker_type: str  # thread, process
    start_time: datetime
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    cpu_samples: List[float] = None
    memory_samples: List[float] = None
    last_activity: Optional[datetime] = None
    error_rate: float = 0.0

    def __post_init__(self):
        if self.cpu_samples is None:
            self.cpu_samples = []
        if self.memory_samples is None:
            self.memory_samples = []

    def record_task_completion(self, processing_time: float):
        """Record a completed task"""
        self.tasks_completed += 1
        self.total_processing_time += processing_time
        self.average_processing_time = self.total_processing_time / self.tasks_completed
        self.last_activity = datetime.now()
        self._update_error_rate()

    def record_task_failure(self):
        """Record a failed task"""
        self.tasks_failed += 1
        self.last_activity = datetime.now()
        self._update_error_rate()

    def add_resource_sample(self, cpu_usage: float, memory_usage: float):
        """Add resource usage sample"""
        self.cpu_samples.append(cpu_usage)
        self.memory_samples.append(memory_usage)
        
        # Keep only last 100 samples
        if len(self.cpu_samples) > 100:
            self.cpu_samples = self.cpu_samples[-100:]
        if len(self.memory_samples) > 100:
            self.memory_samples = self.memory_samples[-100:]

    def _update_error_rate(self):
        """Update error rate"""
        total_tasks = self.tasks_completed + self.tasks_failed
        if total_tasks > 0:
            self.error_rate = self.tasks_failed / total_tasks

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class SnakeAgentConfiguration:
    """Configuration for Snake Agent system"""
    max_processes: int = 4
    max_threads: int = 8
    max_queue_size: int = 1000
    task_timeout: int = 300
    analysis_threads: int = 3
    enable_performance_monitoring: bool = True
    log_level: str = "INFO"
    data_directory: str = "./snake_data"
    experiment_sandbox_dir: str = "./snake_sandbox"
    max_experiment_time: int = 600
    max_memory_usage: int = 100 * 1024 * 1024  # 100MB
    allowed_imports: List[str] = None
    forbidden_operations: List[str] = None
    filesystem_access: bool = False
    network_access: bool = False
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.allowed_imports is None:
            self.allowed_imports = [
                'os', 'sys', 'time', 'datetime', 'json', 'math', 'random',
                'collections', 'itertools', 'functools', 'typing', 'pathlib'
            ]
        
        if self.forbidden_operations is None:
            self.forbidden_operations = [
                'eval', 'exec', 'compile', '__import__', 'globals', 'locals',
                'subprocess', 'os.system', 'os.popen', 'os.spawn'
            ]
        
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


# Default configuration
DEFAULT_SNAKE_CONFIG = SnakeAgentConfiguration(
    max_processes=4,
    max_threads=8,
    max_queue_size=1000,
    task_timeout=300,
    analysis_threads=3,
    enable_performance_monitoring=True,
    log_level="INFO",
    data_directory="./snake_data",
    experiment_sandbox_dir="./snake_sandbox",
    max_experiment_time=600,
    max_memory_usage=100 * 1024 * 1024,
    filesystem_access=False,
    network_access=False
)


def create_experiment_task(
    experiment_type: ExperimentType,
    file_path: str,
    hypothesis: str,
    proposed_changes: Dict[str, Any],
    priority: TaskPriority = TaskPriority.MEDIUM
) -> ExperimentTask:
    """Create a new experiment task"""
    return ExperimentTask(
        task_id=str(uuid.uuid4()),
        experiment_type=experiment_type,
        file_path=file_path,
        hypothesis=hypothesis,
        proposed_changes=proposed_changes,
        priority=priority,
        created_at=datetime.now()
    )


def create_analysis_task(
    file_path: str,
    analysis_type: str,
    priority: TaskPriority = TaskPriority.MEDIUM,
    parameters: Dict[str, Any] = None
) -> AnalysisTask:
    """Create a new analysis task"""
    return AnalysisTask(
        task_id=str(uuid.uuid4()),
        file_path=file_path,
        analysis_type=analysis_type,
        priority=priority,
        parameters=parameters or {},
        created_at=datetime.now()
    )


def create_file_change_event(
    file_path: str,
    event_type: str,
    file_size: Optional[int] = None,
    checksum: Optional[str] = None
) -> FileChangeEvent:
    """Create a new file change event"""
    return FileChangeEvent(
        event_id=str(uuid.uuid4()),
        file_path=file_path,
        event_type=event_type,
        timestamp=datetime.now(),
        file_size=file_size,
        checksum=checksum
    )


def create_communication_message(
    sender_id: str,
    recipient_id: str,
    message_type: str,
    content: Dict[str, Any],
    priority: TaskPriority = TaskPriority.MEDIUM
) -> CommunicationMessage:
    """Create a new communication message"""
    return CommunicationMessage(
        message_id=str(uuid.uuid4()),
        sender_id=sender_id,
        recipient_id=recipient_id,
        message_type=message_type,
        content=content,
        timestamp=datetime.now(),
        priority=priority
    )
