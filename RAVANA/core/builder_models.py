
"""
Data Models and Schemas for Intelligent Adaptive Builder

Comprehensive data models for build attempts, strategies, failure analysis,
and all related entities in the Intelligent Adaptive Builder system.
"""

import sys
import os
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any

# Add virtual environment site-packages to path for IDE compatibility
venv_site_packages = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    'venv', 'Lib', 'site-packages'
)
if venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

try:
    from sqlmodel import SQLModel, Field, Relationship, Column, JSON
except ImportError:
    # Fallback for IDE compatibility
    sys.path.insert(0, venv_site_packages)
    from sqlmodel import SQLModel, Field, Relationship, Column, JSON


# Enums for type safety

class BuildDifficulty(str, Enum):
    """Represents the difficulty level of a build attempt."""
    TRIVIAL = "trivial"
    MODERATE = "moderate"
    CHALLENGING = "challenging"


class BuildAttempt(SQLModel, table=True):
    """Represents a single attempt to build a solution."""
    id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    description: str
    difficulty: BuildDifficulty
    success: bool
    duration_seconds: Optional[float] = None
    logs: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))

    # Relationships
    strategies: List["BuildStrategy"] = Relationship(back_populates="build_attempt")
    failure_analysis: Optional["FailureAnalysis"] = Relationship(back_populates="build_attempt")
    personality_state: Optional["PersonalityState"] = Relationship(back_populates="build_attempt")


class BuildStrategy(SQLModel, table=True):
    """Represents a strategy used in a build attempt."""
    id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    order: int
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None

    build_attempt_id: uuid.UUID = Field(foreign_key="buildattempt.id")
    build_attempt: BuildAttempt = Relationship(back_populates="strategies")
    executions: List["StrategyExecution"] = Relationship(back_populates="build_strategy")


class StrategyExecution(SQLModel, table=True):
    """Represents a single execution of a strategy within a build attempt."""
    id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: str  # e.g., "started", "completed", "failed"
    logs: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    duration_seconds: Optional[float] = None
    success: bool = False
    output: Optional[str] = None
    error: Optional[str] = None

    build_strategy_id: uuid.UUID = Field(foreign_key="buildstrategy.id")
    build_strategy: BuildStrategy = Relationship(back_populates="executions")


class FailureAnalysis(SQLModel, table=True):
    """Represents the analysis of a failed build attempt."""
    id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    root_cause: str
    identified_patterns: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    recommendations: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    remediation_strategy: Optional[str] = None

    build_attempt_id: uuid.UUID = Field(foreign_key="buildattempt.id", unique=True)
    build_attempt: BuildAttempt = Relationship(back_populates="failure_analysis")


class PersonalityState(SQLModel, table=True):
    """Represents the personality state of the AI during a build attempt."""
    id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    emotional_state: str
    confidence_level: float
    focus_level: float
    current_persona: str
    context_summary: str

    build_attempt_id: uuid.UUID = Field(foreign_key="buildattempt.id", unique=True)
    build_attempt: BuildAttempt = Relationship(back_populates="personality_state")


# Pydantic models for creation, reading, and updating (without relationships for simplicity)

class BuildAttemptCreate(SQLModel):
    """Schema for creating a new build attempt."""
    description: str
    difficulty: BuildDifficulty
    success: bool
    duration_seconds: Optional[float] = None
    logs: Optional[Dict[str, Any]] = None


class BuildAttemptRead(SQLModel):
    """Schema for reading a build attempt."""
    id: uuid.UUID
    timestamp: datetime
    description: str
    difficulty: BuildDifficulty
    success: bool
    duration_seconds: Optional[float] = None
    logs: Optional[Dict[str, Any]] = None


class BuildAttemptUpdate(SQLModel):
    """Schema for updating an existing build attempt."""
    description: Optional[str] = None
    difficulty: Optional[BuildDifficulty] = None
    success: Optional[bool] = None
    duration_seconds: Optional[float] = None
    logs: Optional[Dict[str, Any]] = None


class BuildStrategyCreate(SQLModel):
    """Schema for creating a new build strategy."""
    name: str
    description: str
    parameters: Dict[str, Any]
    order: int
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    build_attempt_id: uuid.UUID


class BuildStrategyRead(SQLModel):
    """Schema for reading a build strategy."""
    id: uuid.UUID
    timestamp: datetime
    name: str
    description: str
    parameters: Dict[str, Any]
    order: int
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    build_attempt_id: uuid.UUID


class BuildStrategyUpdate(SQLModel):
    """Schema for updating an existing build strategy."""
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    order: Optional[int] = None
    success: Optional[bool] = None
    output: Optional[str] = None
    error: Optional[str] = None


class StrategyExecutionCreate(SQLModel):
    """Schema for creating a new strategy execution."""
    status: str
    logs: Optional[Dict[str, Any]] = None
    duration_seconds: Optional[float] = None
    success: bool = False
    output: Optional[str] = None
    error: Optional[str] = None
    build_strategy_id: uuid.UUID


class StrategyExecutionRead(SQLModel):
    """Schema for reading a strategy execution."""
    id: uuid.UUID
    timestamp: datetime
    status: str
    logs: Optional[Dict[str, Any]] = None
    duration_seconds: Optional[float] = None
    success: bool = False
    output: Optional[str] = None
    error: Optional[str] = None
    build_strategy_id: uuid.UUID


class StrategyExecutionUpdate(SQLModel):
    """Schema for updating an existing strategy execution."""
    status: Optional[str] = None
    logs: Optional[Dict[str, Any]] = None
    duration_seconds: Optional[float] = None
    success: Optional[bool] = None
    output: Optional[str] = None
    error: Optional[str] = None


class FailureAnalysisCreate(SQLModel):
    """Schema for creating a new failure analysis."""
    root_cause: str
    identified_patterns: List[str]
    recommendations: List[str]
    remediation_strategy: Optional[str] = None
    build_attempt_id: uuid.UUID


class FailureAnalysisRead(SQLModel):
    """Schema for reading a failure analysis."""
    id: uuid.UUID
    timestamp: datetime
    root_cause: str
    identified_patterns: List[str]
    recommendations: List[str]
    remediation_strategy: Optional[str] = None
    build_attempt_id: uuid.UUID


class FailureAnalysisUpdate(SQLModel):
    """Schema for updating an existing failure analysis."""
    root_cause: Optional[str] = None
    identified_patterns: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None
    remediation_strategy: Optional[str] = None


class PersonalityStateCreate(SQLModel):
    """Schema for creating a new personality state."""
    emotional_state: str
    confidence_level: float
    focus_level: float
    current_persona: str
    context_summary: str
    build_attempt_id: uuid.UUID


class PersonalityStateRead(SQLModel):
    """Schema for reading a personality state."""
    id: uuid.UUID
    timestamp: datetime
    emotional_state: str
    confidence_level: float
    focus_level: float
    current_persona: str
    context_summary: str
    build_attempt_id: uuid.UUID


class PersonalityStateUpdate(SQLModel):
    """Schema for updating an existing personality state."""
    emotional_state: Optional[str] = None
    confidence_level: Optional[float] = None
    focus_level: Optional[float] = None
    current_persona: Optional[str] = None
    context_summary: Optional[str] = None


# Utility functions for database operations

def create_build_attempt(
    description: str,
    difficulty: BuildDifficulty,
    success: bool,
    duration_seconds: Optional[float] = None,
    logs: Optional[Dict[str, Any]] = None
) -> BuildAttempt:
    """Create a new build attempt with the given parameters."""
    return BuildAttempt(
        description=description,
        difficulty=difficulty,
        success=success,
        duration_seconds=duration_seconds,
        logs=logs or {}
    )


def create_build_strategy(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    order: int,
    success: bool,
    build_attempt_id: uuid.UUID,
    output: Optional[str] = None,
    error: Optional[str] = None
) -> BuildStrategy:
    """Create a new build strategy with the given parameters."""
    return BuildStrategy(
        name=name,
        description=description,
        parameters=parameters,
        order=order,
        success=success,
        build_attempt_id=build_attempt_id,
        output=output,
        error=error
    )


def create_strategy_execution(
    status: str,
    build_strategy_id: uuid.UUID,
    logs: Optional[Dict[str, Any]] = None,
    duration_seconds: Optional[float] = None,
    success: bool = False,
    output: Optional[str] = None,
    error: Optional[str] = None
) -> StrategyExecution:
    """Create a new strategy execution with the given parameters."""
    return StrategyExecution(
        status=status,
        build_strategy_id=build_strategy_id,
        logs=logs or {},
        duration_seconds=duration_seconds,
        success=success,
        output=output,
        error=error
    )


def create_failure_analysis(
    root_cause: str,
    identified_patterns: List[str],
    recommendations: List[str],
    build_attempt_id: uuid.UUID,
    remediation_strategy: Optional[str] = None
) -> FailureAnalysis:
    """Create a new failure analysis with the given parameters."""
    return FailureAnalysis(
        root_cause=root_cause,
        identified_patterns=identified_patterns,
        recommendations=recommendations,
        build_attempt_id=build_attempt_id,
        remediation_strategy=remediation_strategy
    )


def create_personality_state(
    emotional_state: str,
    confidence_level: float,
    focus_level: float,
    current_persona: str,
    context_summary: str,
    build_attempt_id: uuid.UUID
) -> PersonalityState:
    """Create a new personality state with the given parameters."""
    return PersonalityState(
        emotional_state=emotional_state,
        confidence_level=confidence_level,
        focus_level=focus_level,
        current_persona=current_persona,
        context_summary=context_summary,
        build_attempt_id=build_attempt_id
    )


# Analysis and utility functions

def analyze_build_attempt(build_attempt: BuildAttempt) -> Dict[str, Any]:
    """Analyze a build attempt and return insights."""
    analysis = {
        "total_strategies": len(build_attempt.strategies),
        "successful_strategies": sum(1 for s in build_attempt.strategies if s.success),
        "failed_strategies": sum(1 for s in build_attempt.strategies if not s.success),
        "total_executions": sum(len(s.executions) for s in build_attempt.strategies),
        "success_rate": 0.0,
        "average_duration": 0.0,
        "has_failure_analysis": build_attempt.failure_analysis is not None,
        "has_personality_state": build_attempt.personality_state is not None
    }

    if build_attempt.strategies:
        analysis["success_rate"] = analysis["successful_strategies"] / analysis["total_strategies"]

    if build_attempt.duration_seconds:
        analysis["average_duration"] = build_attempt.duration_seconds

    return analysis


def get_strategy_performance(strategy: BuildStrategy) -> Dict[str, Any]:
    """Get performance metrics for a specific strategy."""
    performance = {
        "total_executions": len(strategy.executions),
        "successful_executions": sum(1 for e in strategy.executions if e.success),
        "failed_executions": sum(1 for e in strategy.executions if not e.success),
        "success_rate": 0.0,
        "average_duration": 0.0,
        "has_errors": any(e.error for e in strategy.executions),
        "error_count": sum(1 for e in strategy.executions if e.error)
    }

    if strategy.executions:
        performance["success_rate"] = (
            performance["successful_executions"] / performance["total_executions"]
        )
        durations = [e.duration_seconds for e in strategy.executions if e.duration_seconds]
        if durations:
            performance["average_duration"] = sum(durations) / len(durations)

    return performance


def get_build_attempt_summary(build_attempt: BuildAttempt) -> str:
    """Get a human-readable summary of a build attempt."""
    analysis = analyze_build_attempt(build_attempt)
    status = "SUCCESS" if build_attempt.success else "FAILED"
    duration = f"{build_attempt.duration_seconds:.2f}s" if build_attempt.duration_seconds else "N/A"

    summary = f"Build Attempt {status} - {build_attempt.difficulty.value.title()}\n"
    summary += f"Duration: {duration}\n"
    summary += (
        f"Strategies: {analysis['successful_strategies']}/"
        f"{analysis['total_strategies']} successful\n"
    )
    summary += f"Success Rate: {analysis['success_rate']:.1%}\n"

    if build_attempt.failure_analysis:
        summary += f"Root Cause: {build_attempt.failure_analysis.root_cause}\n"

    if build_attempt.personality_state:
        summary += f"Persona: {build_attempt.personality_state.current_persona}\n"
        summary += f"Emotional State: {build_attempt.personality_state.emotional_state}\n"

    return summary
