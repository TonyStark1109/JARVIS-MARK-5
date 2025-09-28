"""
RAVANA Evolutionary Network Base Module
"""

import logging
import random
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

@dataclass
class Connection:
    """Represents a connection between nodes."""
    from_node: str
    to_node: str
    weight: float
    enabled: bool = True
    innovation_number: int = 0

@dataclass
class NetworkNode:
    """Represents a node in the network."""
    id: str
    node_type: str
    activation_function: str
    bias: float
    connections: List[Connection]
    layer: int = 0

class EvolutionaryNetworkBase(ABC):
    """Base class for evolutionary neural networks."""
    
    def __init__(self, input_size: int = 0, output_size: int = 0):
        self.logger = logging.getLogger(__name__)
        self.nodes = {}
        self.connections = {}
        self.fitness = 0.0
        self.input_size = input_size
        self.output_size = output_size
        self.innovation_counter = 0
    
    def add_node(self, node_type: str, activation: str = "sigmoid", layer: int = 0) -> str:
        """Add a node to the network."""
        node_id = f"node_{len(self.nodes)}"
        node = NetworkNode(
            id=node_id,
            node_type=node_type,
            activation_function=activation,
            bias=random.uniform(-1, 1),
            connections=[],
            layer=layer
        )
        self.nodes[node_id] = node
        return node_id
    
    def add_connection(self, from_node: str, to_node: str, weight: Optional[float] = None) -> str:
        """Add a connection between nodes."""
        if weight is None:
            weight = random.uniform(-1, 1)
        
        connection_id = f"conn_{self.innovation_counter}"
        self.innovation_counter += 1
        
        connection = Connection(
            from_node=from_node,
            to_node=to_node,
            weight=weight,
            innovation_number=self.innovation_counter
        )
        
        self.connections[connection_id] = connection
        return connection_id
    
    def enable_connection(self, connection_id: str) -> bool:
        """Enable a connection."""
        if connection_id in self.connections:
            self.connections[connection_id].enabled = True
            return True
        return False
    
    def get_network_topology(self) -> Dict[str, Any]:
        """Get network topology information."""
        return {
            "nodes": len(self.nodes),
            "connections": len(self.connections),
            "fitness": self.fitness
        }
    
    def forward(self, inputs: List[float]) -> np.ndarray:
        """Forward pass through the network."""
        if len(inputs) != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, got {len(inputs)}")

        # Initialize node values
        node_values = {}

        # Set input values
        input_nodes = [node for node in self.nodes.values() if node.node_type == 'input']
        for i, node in enumerate(input_nodes):
            node_values[node.id] = inputs[i]

        # Process through layers
        max_layer = max(node.layer for node in self.nodes.values()) if self.nodes else 0

        for layer in range(max_layer + 1):
            layer_nodes = [node for node in self.nodes.values() if node.layer == layer and node.node_type != 'input']

            for node in layer_nodes:
                # Calculate weighted sum
                weighted_sum = node.bias

                for conn_id, conn in self.connections.items():
                    if conn.to_node == node.id and conn.enabled:
                        if conn.from_node in node_values:
                            weighted_sum += node_values[conn.from_node] * conn.weight

                # Apply activation function
                node_values[node.id] = self._apply_activation(weighted_sum, node.activation_function)

        # Get output values
        output_nodes = [node for node in self.nodes.values() if node.node_type == 'output']
        outputs = [node_values.get(node.id, 0.0) for node in output_nodes]

        return np.array(outputs)
    
    def _apply_activation(self, x: float, activation: str) -> float:
        """Apply activation function."""
        if activation == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif activation == "tanh":
            return np.tanh(x)
        elif activation == "relu":
            return max(0, x)
        else:
            return x
    
    def mutate(self):
        """Mutate the network."""
        # Randomly change connection weights
        for conn in self.connections.values():
            if random.random() < 0.1:  # 10% chance
                conn.weight += random.gauss(0, 0.1)
                conn.weight = max(-1, min(1, conn.weight))  # Clamp to [-1, 1]
    
    def clone(self):
        """Create a copy of the network."""
        new_network = self.__class__(self.input_size, self.output_size)
        new_network.nodes = self.nodes.copy()
        new_network.connections = self.connections.copy()
        new_network.fitness = self.fitness
        return new_network