#!/usr/bin/env python3
"""
RAVANA Evolutionary Network Evolution Module
"""

import logging
import random
from typing import List, Dict, Any
from evonet_base import EvolutionaryNetworkBase, Connection

logger = logging.getLogger(__name__)

class EvolutionEngine:
    """Handles evolution of neural networks."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_compatibility(self, network1, network2):
        """Calculate compatibility between two networks."""
        try:
            # Count matching, disjoint, and excess genes
            matching = 0
            disjoint = 0
            excess = 0
            weight_diff = 0.0
            
            # Get all innovation numbers
            innovations1 = set(conn.innovation_number for conn in network1.connections.values())
            innovations2 = set(conn.innovation_number for conn in network2.connections.values())
            
            # Find matching, disjoint, and excess genes
            matching_innovations = innovations1.intersection(innovations2)
            disjoint_innovations = innovations1.symmetric_difference(innovations2)
            excess_innovations = max(innovations1, innovations2) - min(innovations1, innovations2)
            
            matching = len(matching_innovations)
            disjoint = len(disjoint_innovations)
            excess = len(excess_innovations)
            
            # Calculate weight differences for matching genes
            for conn1 in network1.connections.values():
                for conn2 in network2.connections.values():
                    if conn1.innovation_number == conn2.innovation_number:
                        weight_diff += abs(conn1.weight - conn2.weight)
            
            # Calculate compatibility distance
            N = max(len(network1.connections), len(network2.connections))
            if N < 20:
                N = 1
            
            compatibility = (excess * 1.0 + disjoint * 1.0) / N + weight_diff
            return compatibility
            
        except Exception as e:
            self.logger.error(f"Compatibility calculation error: {e}")
            return float('inf')
    
    def crossover(self, parent1, parent2):
        """Create offspring from two parents."""
        try:
            # Create new network
            offspring = parent1.clone()
            offspring.fitness = 0.0
            
            # Inherit connections from both parents
            for conn_id, conn in parent2.connections.items():
                if conn_id not in offspring.connections:
                    offspring.connections[conn_id] = conn
            
            return offspring
            
        except Exception as e:
            self.logger.error(f"Crossover error: {e}")
            return parent1.clone()
    
    def mutate_network(self, network, mutation_rate=0.1):
        """Mutate a network."""
        try:
            # Weight mutation
            for conn in network.connections.values():
                if random.random() < mutation_rate:
                    conn.weight += random.gauss(0, 0.1)
                    conn.weight = max(-1, min(1, conn.weight))
            
            # Add new connection
            if random.random() < mutation_rate:
                self._add_random_connection(network)
            
            # Add new node
            if random.random() < mutation_rate:
                self._add_random_node(network)
            
        except Exception as e:
            self.logger.error(f"Mutation error: {e}")
    
    def _add_random_connection(self, network):
        """Add a random connection to the network."""
        try:
            # Get all possible connections
            possible_connections = []
            for from_node in network.nodes:
                for to_node in network.nodes:
                    if from_node != to_node:
                        # Check if connection already exists
                        exists = any(conn.from_node == from_node and conn.to_node == to_node 
                                   for conn in network.connections.values())
                        if not exists:
                            possible_connections.append((from_node, to_node))
            
            if possible_connections:
                from_node, to_node = random.choice(possible_connections)
                network.add_connection(from_node, to_node)
                
        except Exception as e:
            self.logger.error(f"Add connection error: {e}")
    
    def _add_random_node(self, network):
        """Add a random node to the network."""
        try:
            # Select a random connection to split
            if not network.connections:
                return
            
            conn_id = random.choice(list(network.connections.keys()))
            conn = network.connections[conn_id]
            
            # Disable the connection
            conn.enabled = False
            
            # Add new node
            new_node_id = network.add_node('hidden', 'sigmoid')
            
            # Add connections
            network.add_connection(conn.from_node, new_node_id, 1.0)
            network.add_connection(new_node_id, conn.to_node, conn.weight)
            
        except Exception as e:
            self.logger.error(f"Add node error: {e}")
    
    def evolve_population(self, population, fitness_scores):
        """Evolve a population of networks."""
        try:
            # Sort by fitness
            sorted_population = sorted(zip(population, fitness_scores), 
                                    key=lambda x: x[1], reverse=True)
            
            # Keep top performers
            elite_size = len(population) // 10
            new_population = [net for net, _ in sorted_population[:elite_size]]
            
            # Generate offspring
            while len(new_population) < len(population):
                # Select parents
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Create offspring
                offspring = self.crossover(parent1, parent2)
                self.mutate_network(offspring)
                
                new_population.append(offspring)
            
            return new_population[:len(population)]
            
        except Exception as e:
            self.logger.error(f"Evolution error: {e}")
            return population
    
    def _tournament_selection(self, population, fitness_scores, tournament_size=3):
        """Select a parent using tournament selection."""
        try:
            tournament_indices = random.sample(range(len(population)), 
                                            min(tournament_size, len(population)))
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            return population[winner_index]
            
        except Exception as e:
            self.logger.error(f"Tournament selection error: {e}")
            return population[0]