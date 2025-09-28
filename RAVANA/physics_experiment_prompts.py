"""
RAVANA Physics Experiment Prompts

This module contains prompts and templates for physics experiments and simulations.
"""

from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class PhysicsExperimentPrompt:
    """Template for physics experiment prompts"""
    name: str
    description: str
    prompt_template: str
    parameters: List[str]
    expected_outputs: List[str]
    safety_notes: List[str]

# Physics experiment prompts
PHYSICS_EXPERIMENT_PROMPTS = {
    "classical_mechanics": PhysicsExperimentPrompt(
        name="Classical Mechanics Simulation",
        description="Simulate classical mechanics problems using numerical methods",
        prompt_template="""
Create a physics simulation for: {experiment_name}

Parameters:
- Mass: {mass} kg
- Initial velocity: {initial_velocity} m/s
- Initial position: {initial_position} m
- Time step: {time_step} s
- Duration: {duration} s

Forces to consider:
{forces}

Please implement:
1. A numerical integrator (Euler, Verlet, or Runge-Kutta)
2. Force calculations
3. Position and velocity updates
4. Data visualization
5. Energy conservation check

Expected outputs:
- Position vs time plot
- Velocity vs time plot
- Energy vs time plot
- Animation of the motion
        """,
        parameters=["experiment_name", "mass", "initial_velocity", "initial_position", "time_step", "duration", "forces"],
        expected_outputs=["position_plot", "velocity_plot", "energy_plot", "animation"],
        safety_notes=["Ensure numerical stability", "Check for energy conservation", "Validate physical units"]
    ),
    
    "quantum_mechanics": PhysicsExperimentPrompt(
        name="Quantum Mechanics Simulation",
        description="Simulate quantum mechanical systems and wave functions",
        prompt_template="""
Create a quantum mechanics simulation for: {system_name}

System parameters:
- Potential: {potential}
- Boundary conditions: {boundary_conditions}
- Grid points: {grid_points}
- Time step: {time_step}
- Duration: {duration}

Quantum numbers:
- Principal quantum number: {n}
- Angular momentum: {l}
- Magnetic quantum number: {m}

Please implement:
1. Time-independent Schrödinger equation solver
2. Wave function normalization
3. Probability density calculation
4. Energy eigenvalue computation
5. Time evolution (if applicable)

Expected outputs:
- Wave function plots
- Probability density plots
- Energy eigenvalues
- Animation of time evolution
        """,
        parameters=["system_name", "potential", "boundary_conditions", "grid_points", "time_step", "duration", "n", "l", "m"],
        expected_outputs=["wave_function_plot", "probability_density_plot", "energy_eigenvalues", "time_evolution"],
        safety_notes=["Normalize wave functions", "Check boundary conditions", "Validate quantum numbers"]
    ),
    
    "thermodynamics": PhysicsExperimentPrompt(
        name="Thermodynamics Simulation",
        description="Simulate thermodynamic processes and phase transitions",
        prompt_template="""
Create a thermodynamics simulation for: {process_name}

System parameters:
- Initial temperature: {initial_temperature} K
- Initial pressure: {initial_pressure} Pa
- Initial volume: {initial_volume} m³
- Number of particles: {num_particles}
- Simulation time: {simulation_time} s

Process type: {process_type}
Heat capacity: {heat_capacity} J/K
Gas constant: {gas_constant} J/(mol·K)

Please implement:
1. State equation solver
2. Heat transfer calculations
3. Work calculations
4. Entropy calculations
5. Phase transition detection

Expected outputs:
- P-V diagram
- T-S diagram
- Temperature vs time plot
- Pressure vs time plot
- Phase diagram
        """,
        parameters=["process_name", "initial_temperature", "initial_pressure", "initial_volume", "num_particles", "simulation_time", "process_type", "heat_capacity", "gas_constant"],
        expected_outputs=["pv_diagram", "ts_diagram", "temperature_plot", "pressure_plot", "phase_diagram"],
        safety_notes=["Check energy conservation", "Validate state equations", "Monitor phase transitions"]
    ),
    
    "electromagnetism": PhysicsExperimentPrompt(
        name="Electromagnetism Simulation",
        description="Simulate electromagnetic fields and wave propagation",
        prompt_template="""
Create an electromagnetism simulation for: {field_type}

Field parameters:
- Electric field strength: {electric_field} V/m
- Magnetic field strength: {magnetic_field} T
- Frequency: {frequency} Hz
- Wavelength: {wavelength} m
- Grid resolution: {grid_resolution}
- Simulation domain: {domain_size} m

Boundary conditions:
- Electric: {electric_bc}
- Magnetic: {magnetic_bc}

Please implement:
1. Maxwell's equations solver
2. Field visualization
3. Wave propagation
4. Boundary condition handling
5. Energy density calculations

Expected outputs:
- Electric field plots
- Magnetic field plots
- Poynting vector plots
- Wave propagation animation
- Energy density plots
        """,
        parameters=["field_type", "electric_field", "magnetic_field", "frequency", "wavelength", "grid_resolution", "domain_size", "electric_bc", "magnetic_bc"],
        expected_outputs=["electric_field_plot", "magnetic_field_plot", "poynting_vector_plot", "wave_animation", "energy_density_plot"],
        safety_notes=["Check Maxwell's equations", "Validate boundary conditions", "Monitor energy conservation"]
    ),
    
    "relativity": PhysicsExperimentPrompt(
        name="Relativity Simulation",
        description="Simulate relativistic effects and spacetime curvature",
        prompt_template="""
Create a relativity simulation for: {relativity_type}

Relativistic parameters:
- Velocity: {velocity} c (fraction of speed of light)
- Mass: {mass} kg
- Proper time: {proper_time} s
- Coordinate time: {coordinate_time} s
- Spacetime metric: {metric}

Gravitational field:
- Mass of central body: {central_mass} kg
- Schwarzschild radius: {schwarzschild_radius} m
- Orbital radius: {orbital_radius} m

Please implement:
1. Lorentz transformation calculations
2. Time dilation effects
3. Length contraction
4. Spacetime geodesics
5. Gravitational redshift

Expected outputs:
- Spacetime diagrams
- Time dilation plots
- Length contraction plots
- Geodesic trajectories
- Redshift calculations
        """,
        parameters=["relativity_type", "velocity", "mass", "proper_time", "coordinate_time", "metric", "central_mass", "schwarzschild_radius", "orbital_radius"],
        expected_outputs=["spacetime_diagram", "time_dilation_plot", "length_contraction_plot", "geodesic_trajectory", "redshift_calculation"],
        safety_notes=["Check Lorentz invariance", "Validate metric calculations", "Monitor coordinate singularities"]
    )
}

def get_experiment_prompt(experiment_type: str, parameters: Dict[str, Any]) -> str:
    """Get a formatted experiment prompt"""
    if experiment_type not in PHYSICS_EXPERIMENT_PROMPTS:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    prompt_template = PHYSICS_EXPERIMENT_PROMPTS[experiment_type]
    return prompt_template.prompt_template.format(**parameters)

def get_available_experiments() -> List[str]:
    """Get list of available experiment types"""
    return list(PHYSICS_EXPERIMENT_PROMPTS.keys())

def get_experiment_info(experiment_type: str) -> PhysicsExperimentPrompt:
    """Get experiment information"""
    if experiment_type not in PHYSICS_EXPERIMENT_PROMPTS:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    return PHYSICS_EXPERIMENT_PROMPTS[experiment_type]

def validate_parameters(experiment_type: str, parameters: Dict[str, Any]) -> List[str]:
    """Validate experiment parameters"""
    if experiment_type not in PHYSICS_EXPERIMENT_PROMPTS:
        return [f"Unknown experiment type: {experiment_type}"]
    
    prompt_template = PHYSICS_EXPERIMENT_PROMPTS[experiment_type]
    missing_params = []
    
    for param in prompt_template.parameters:
        if param not in parameters:
            missing_params.append(f"Missing parameter: {param}")
    
    return missing_params

# Example usage and testing
if __name__ == "__main__":
    # Test classical mechanics prompt
    params = {
        "experiment_name": "Projectile Motion",
        "mass": 1.0,
        "initial_velocity": 10.0,
        "initial_position": 0.0,
        "time_step": 0.01,
        "duration": 2.0,
        "forces": "Gravity (9.81 m/s² downward)"
    }
    
    prompt = get_experiment_prompt("classical_mechanics", params)
    print("Classical Mechanics Prompt:")
    print(prompt)
    
    # Test parameter validation
    missing = validate_parameters("classical_mechanics", params)
    print(f"\nMissing parameters: {missing}")
    
    # List available experiments
    experiments = get_available_experiments()
    print(f"\nAvailable experiments: {experiments}")
