"""
RAVANA Database Schemas

This module defines database schemas and data models for RAVANA AGI system.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum

class MemoryType(str, Enum):
    """Types of memories"""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    LONG_TERM = "long_term"

class ExperimentStatus(str, Enum):
    """Experiment statuses"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class LogLevel(str, Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class UserSchema:
    """User schema"""
    username: str
    email: str
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    preferences: Optional[Dict[str, Any]] = None
    is_active: bool = True
    
    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create instance from dictionary"""
        return cls(**data)

@dataclass
class SessionSchema:
    """Session schema"""
    user_id: int
    session_token: str
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    is_active: bool = True
    session_data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.session_data is None:
            self.session_data = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create instance from dictionary"""
        return cls(**data)

@dataclass
class MemorySchema:
    """Memory schema"""
    user_id: int
    memory_type: MemoryType
    content: str
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    importance_score: float = 0.5
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.last_accessed is None:
            self.last_accessed = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create instance from dictionary"""
        return cls(**data)

@dataclass
class ExperimentSchema:
    """Experiment schema"""
    user_id: int
    experiment_type: str
    experiment_data: Dict[str, Any]
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    status: ExperimentStatus = ExperimentStatus.PENDING
    completed_at: Optional[datetime] = None
    execution_time: float = 0.0
    
    def __post_init__(self):
        if self.result is None:
            self.result = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create instance from dictionary"""
        return cls(**data)

@dataclass
class SystemLogSchema:
    """System log schema"""
    level: LogLevel
    component: str
    message: str
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    data: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create instance from dictionary"""
        return cls(**data)

# Additional specialized schemas

@dataclass
class ConversationSchema:
    """Conversation schema"""
    user_id: int
    session_id: int
    user_message: str
    ai_response: str
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    emotion_detected: Optional[str] = None
    confidence_score: float = 0.0
    context_data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.context_data is None:
            self.context_data = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create instance from dictionary"""
        return cls(**data)

@dataclass
class LearningSchema:
    """Learning schema for tracking learning progress"""
    user_id: int
    skill_name: str
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    proficiency_level: float = 0.0
    learning_data: Optional[Dict[str, Any]] = None
    last_practiced: Optional[datetime] = None
    practice_count: int = 0
    
    def __post_init__(self):
        if self.learning_data is None:
            self.learning_data = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create instance from dictionary"""
        return cls(**data)

@dataclass
class PerformanceSchema:
    """Performance metrics schema"""
    component: str
    metric_name: str
    metric_value: float
    metric_unit: str
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create instance from dictionary"""
        return cls(**data)

@dataclass
class ConfigurationSchema:
    """Configuration schema"""
    config_key: str
    config_value: str
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    config_type: str = "string"
    description: str = ""
    is_system: bool = False
    user_id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create instance from dictionary"""
        return cls(**data)

# Schema validation functions

def validate_user_schema(data: Dict[str, Any]) -> List[str]:
    """Validate user schema data"""
    errors = []
    
    if not data.get("username"):
        errors.append("Username is required")
    
    if not data.get("email"):
        errors.append("Email is required")
    elif "@" not in data.get("email", ""):
        errors.append("Email must be valid")
    
    if data.get("preferences") and not isinstance(data["preferences"], dict):
        errors.append("Preferences must be a dictionary")
    
    return errors

def validate_memory_schema(data: Dict[str, Any]) -> List[str]:
    """Validate memory schema data"""
    errors = []
    
    if not data.get("user_id"):
        errors.append("User ID is required")
    
    if not data.get("memory_type"):
        errors.append("Memory type is required")
    elif data["memory_type"] not in [t.value for t in MemoryType]:
        errors.append(f"Invalid memory type: {data['memory_type']}")
    
    if not data.get("content"):
        errors.append("Content is required")
    
    importance = data.get("importance_score", 0.5)
    if not isinstance(importance, (int, float)) or not 0 <= importance <= 1:
        errors.append("Importance score must be between 0 and 1")
    
    return errors

def validate_experiment_schema(data: Dict[str, Any]) -> List[str]:
    """Validate experiment schema data"""
    errors = []
    
    if not data.get("user_id"):
        errors.append("User ID is required")
    
    if not data.get("experiment_type"):
        errors.append("Experiment type is required")
    
    if not data.get("experiment_data"):
        errors.append("Experiment data is required")
    elif not isinstance(data["experiment_data"], dict):
        errors.append("Experiment data must be a dictionary")
    
    status = data.get("status", "pending")
    if status not in [s.value for s in ExperimentStatus]:
        errors.append(f"Invalid status: {status}")
    
    return errors

# Schema conversion utilities

def convert_to_user_schema(data: Dict[str, Any]) -> UserSchema:
    """Convert dictionary to UserSchema"""
    return UserSchema(
        id=data.get("id"),
        username=data["username"],
        email=data["email"],
        last_login=data.get("last_login"),
        preferences=data.get("preferences", {}),
        is_active=data.get("is_active", True),
        created_at=data.get("created_at")
    )

def convert_to_memory_schema(data: Dict[str, Any]) -> MemorySchema:
    """Convert dictionary to MemorySchema"""
    return MemorySchema(
        id=data.get("id"),
        user_id=data["user_id"],
        memory_type=MemoryType(data["memory_type"]),
        content=data["content"],
        metadata=data.get("metadata", {}),
        importance_score=data.get("importance_score", 0.5),
        last_accessed=data.get("last_accessed"),
        access_count=data.get("access_count", 0),
        created_at=data.get("created_at")
    )

def convert_to_experiment_schema(data: Dict[str, Any]) -> ExperimentSchema:
    """Convert dictionary to ExperimentSchema"""
    return ExperimentSchema(
        id=data.get("id"),
        user_id=data["user_id"],
        experiment_type=data["experiment_type"],
        experiment_data=data["experiment_data"],
        result=data.get("result", {}),
        status=ExperimentStatus(data.get("status", "pending")),
        completed_at=data.get("completed_at"),
        execution_time=data.get("execution_time", 0.0),
        created_at=data.get("created_at")
    )