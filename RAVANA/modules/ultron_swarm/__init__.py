"""
Ultron Swarm Intelligence Module
Multi-agent system for Ultron-level capabilities
"""

from .swarm_coordinator import UltronSwarmCoordinator
from .agent_types import (
    NetworkScannerAgent, SocialEngineerAgent, DataMinerAgent,
    ExploitGeneratorAgent, CoverUpAgent, SurveillanceAgent,
    ManipulationAgent, ControlAgent, EvolutionAgent, DominationAgent
)

__all__ = [
    'UltronSwarmCoordinator',
    'NetworkScannerAgent', 'SocialEngineerAgent', 'DataMinerAgent',
    'ExploitGeneratorAgent', 'CoverUpAgent', 'SurveillanceAgent',
    'ManipulationAgent', 'ControlAgent', 'EvolutionAgent', 'DominationAgent'
]
