#!/usr/bin/env python3
"""
RAVANA Physics CLI
Command-line interface for physics experiments and simulations.
"""

import logging
import asyncio
import argparse
from typing import Dict, List, Any, Optional
import json
import math
import random

logger = logging.getLogger(__name__)

class PhysicsCLI:
    """Command-line interface for physics experiments."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self.results: List[Dict[str, Any]] = []
        
    async def run_experiment(self, experiment_type: str, 
                           parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run a physics experiment."""
        try:
            self.logger.info(f"Running {experiment_type} experiment")
            
            if experiment_type == "pendulum":
                return await self._pendulum_experiment(parameters)
            elif experiment_type == "projectile":
                return await self._projectile_experiment(parameters)
            elif experiment_type == "spring":
                return await self._spring_experiment(parameters)
            elif experiment_type == "collision":
                return await self._collision_experiment(parameters)
            else:
                return {"error": f"Unknown experiment type: {experiment_type}"}
        except Exception as e:
            self.logger.error(f"Error running experiment: {e}")
            return {"error": str(e)}
    
    async def _pendulum_experiment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a simple pendulum."""
        try:
            length = params.get('length', 1.0)
            angle = params.get('initial_angle', math.pi/4)
            gravity = params.get('gravity', 9.81)
            time_steps = params.get('time_steps', 100)
            dt = params.get('dt', 0.01)
            
            # Simple pendulum simulation
            omega = math.sqrt(gravity / length)
            results = []
            
            for i in range(time_steps):
                t = i * dt
                theta = angle * math.cos(omega * t)
                results.append({
                    'time': t,
                    'angle': theta,
                    'angular_velocity': -angle * omega * math.sin(omega * t)
                })
            
            return {
                'type': 'pendulum',
                'parameters': params,
                'results': results,
                'period': 2 * math.pi / omega
            }
        except Exception as e:
            return {"error": f"Pendulum experiment failed: {e}"}
    
    async def _projectile_experiment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate projectile motion."""
        try:
            v0 = params.get('initial_velocity', 10.0)
            angle = params.get('angle', 45.0)
            gravity = params.get('gravity', 9.81)
            time_steps = params.get('time_steps', 100)
            dt = params.get('dt', 0.01)
            
            # Convert angle to radians
            angle_rad = math.radians(angle)
            vx0 = v0 * math.cos(angle_rad)
            vy0 = v0 * math.sin(angle_rad)
            
            results = []
            x, y = 0, 0
            vx, vy = vx0, vy0
            
            for i in range(time_steps):
                t = i * dt
                x = vx0 * t
                y = vy0 * t - 0.5 * gravity * t**2
                
                if y < 0:  # Hit ground
                    break
                
                results.append({
                    'time': t,
                    'x': x,
                    'y': y,
                    'vx': vx,
                    'vy': vy - gravity * t
                })
            
            return {
                'type': 'projectile',
                'parameters': params,
                'results': results,
                'range': x,
                'max_height': max(r['y'] for r in results) if results else 0
            }
        except Exception as e:
            return {"error": f"Projectile experiment failed: {e}"}
    
    async def _spring_experiment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate spring-mass system."""
        try:
            mass = params.get('mass', 1.0)
            spring_constant = params.get('spring_constant', 10.0)
            initial_displacement = params.get('initial_displacement', 0.1)
            time_steps = params.get('time_steps', 100)
            dt = params.get('dt', 0.01)
            
            omega = math.sqrt(spring_constant / mass)
            results = []
            
            for i in range(time_steps):
                t = i * dt
                x = initial_displacement * math.cos(omega * t)
                v = -initial_displacement * omega * math.sin(omega * t)
                a = -omega**2 * x
                
                results.append({
                    'time': t,
                    'position': x,
                    'velocity': v,
                    'acceleration': a
                })
            
            return {
                'type': 'spring',
                'parameters': params,
                'results': results,
                'frequency': omega / (2 * math.pi)
            }
        except Exception as e:
            return {"error": f"Spring experiment failed: {e}"}
    
    async def _collision_experiment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate elastic collision."""
        try:
            m1 = params.get('mass1', 1.0)
            m2 = params.get('mass2', 2.0)
            v1i = params.get('velocity1_initial', 5.0)
            v2i = params.get('velocity2_initial', -2.0)
            
            # Elastic collision equations
            v1f = ((m1 - m2) * v1i + 2 * m2 * v2i) / (m1 + m2)
            v2f = (2 * m1 * v1i + (m2 - m1) * v2i) / (m1 + m2)
            
            # Calculate momentum and kinetic energy
            p1i = m1 * v1i
            p2i = m2 * v2i
            p1f = m1 * v1f
            p2f = m2 * v2f
            
            ke1i = 0.5 * m1 * v1i**2
            ke2i = 0.5 * m2 * v2i**2
            ke1f = 0.5 * m1 * v1f**2
            ke2f = 0.5 * m2 * v2f**2
            
            return {
                'type': 'collision',
                'parameters': params,
                'initial': {
                    'velocity1': v1i,
                    'velocity2': v2i,
                    'momentum1': p1i,
                    'momentum2': p2i,
                    'kinetic_energy1': ke1i,
                    'kinetic_energy2': ke2i
                },
                'final': {
                    'velocity1': v1f,
                    'velocity2': v2f,
                    'momentum1': p1f,
                    'momentum2': p2f,
                    'kinetic_energy1': ke1f,
                    'kinetic_energy2': ke2f
                },
                'conservation': {
                    'momentum_conserved': abs((p1i + p2i) - (p1f + p2f)) < 0.001,
                    'energy_conserved': abs((ke1i + ke2i) - (ke1f + ke2f)) < 0.001
                }
            }
        except Exception as e:
            return {"error": f"Collision experiment failed: {e}"}
    
    async def analyze_results(self, experiment_id: str) -> Dict[str, Any]:
        """Analyze experiment results."""
        try:
            if experiment_id not in self.experiments:
                return {"error": "Experiment not found"}
            
            experiment = self.experiments[experiment_id]
            results = experiment.get('results', [])
            
            if not results:
                return {"error": "No results to analyze"}
            
            analysis = {
                'experiment_id': experiment_id,
                'type': experiment.get('type'),
                'data_points': len(results),
                'analysis': {}
            }
            
            # Type-specific analysis
            if experiment['type'] == 'pendulum':
                angles = [r['angle'] for r in results]
                analysis['analysis'] = {
                    'max_angle': max(angles),
                    'min_angle': min(angles),
                    'amplitude': (max(angles) - min(angles)) / 2
                }
            elif experiment['type'] == 'projectile':
                ys = [r['y'] for r in results]
                analysis['analysis'] = {
                    'max_height': max(ys),
                    'range': results[-1]['x'] if results else 0
                }
            
            return analysis
        except Exception as e:
            return {"error": f"Analysis failed: {e}"}

async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description='RAVANA Physics CLI')
    parser.add_argument('--experiment', type=str, help='Experiment type to run')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    cli = PhysicsCLI()
    
    if args.interactive:
        await interactive_mode(cli)
    elif args.experiment:
        config = {}
        if args.config:
            with open(args.config, 'r') as f:
                config = json.load(f)
        
        result = await cli.run_experiment(args.experiment, config)
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()

async def interactive_mode(cli: PhysicsCLI):
    """Run in interactive mode."""
    print("RAVANA Physics CLI - Interactive Mode")
    print("Available experiments: pendulum, projectile, spring, collision")
    print("Type 'help' for commands, 'quit' to exit")
    
    while True:
        try:
            command = input("\nphysics> ").strip().lower()
            
            if command == 'quit':
                break
            elif command == 'help':
                print("Commands:")
                print("  run <experiment> - Run an experiment")
                print("  list - List available experiments")
                print("  analyze <id> - Analyze experiment results")
                print("  quit - Exit")
            elif command == 'list':
                print("Available experiments: pendulum, projectile, spring, collision")
            elif command.startswith('run '):
                exp_type = command.split()[1]
                result = await cli.run_experiment(exp_type, {})
                print(json.dumps(result, indent=2))
            elif command.startswith('analyze '):
                exp_id = command.split()[1]
                result = await cli.analyze_results(exp_id)
                print(json.dumps(result, indent=2))
            else:
                print("Unknown command. Type 'help' for available commands.")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
