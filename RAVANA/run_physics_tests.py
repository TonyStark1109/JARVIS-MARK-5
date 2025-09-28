#!/usr/bin/env python3
"""
RAVANA Physics Tests Runner

This module runs physics experiments and tests using the RAVANA system.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add the RAVANA directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from main import RAVANAMain
from physics_experiment_prompts import (
    get_experiment_prompt, 
    get_available_experiments, 
    get_experiment_info,
    validate_parameters
)

logger = logging.getLogger(__name__)

class PhysicsTestRunner:
    """Runner for physics experiments and tests"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ravana = RAVANAMain()
        self.test_results = []
        
    async def initialize(self):
        """Initialize the physics test runner"""
        try:
            self.logger.info("Initializing RAVANA Physics Test Runner...")
            
            # Initialize RAVANA main system
            if not await self.ravana.initialize():
                self.logger.error("Failed to initialize RAVANA main system")
                return False
            
            if not await self.ravana.start():
                self.logger.error("Failed to start RAVANA main system")
                return False
            
            self.logger.info("RAVANA Physics Test Runner initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize physics test runner: {e}")
            return False
    
    async def run_experiment(self, experiment_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run a physics experiment"""
        try:
            self.logger.info(f"Running physics experiment: {experiment_type}")
            
            # Validate parameters
            missing_params = validate_parameters(experiment_type, parameters)
            if missing_params:
                return {
                    "success": False,
                    "error": f"Missing parameters: {missing_params}",
                    "experiment_type": experiment_type
                }
            
            # Get experiment prompt
            prompt = get_experiment_prompt(experiment_type, parameters)
            
            # Run experiment using Snake Agents
            experiment_id = await self.ravana.run_experiment(
                experiment_type="code_modification",
                file_path=f"physics_{experiment_type}.py",
                hypothesis=f"Physics experiment: {experiment_type}",
                proposed_changes={"code": prompt, "type": "physics_simulation"}
            )
            
            if experiment_id:
                result = {
                    "success": True,
                    "experiment_id": experiment_id,
                    "experiment_type": experiment_type,
                    "parameters": parameters,
                    "prompt": prompt
                }
                self.test_results.append(result)
                return result
            else:
                return {
                    "success": False,
                    "error": "Failed to start experiment",
                    "experiment_type": experiment_type
                }
                
        except Exception as e:
            self.logger.error(f"Error running experiment {experiment_type}: {e}")
            return {
                "success": False,
                "error": str(e),
                "experiment_type": experiment_type
            }
    
    async def run_all_tests(self):
        """Run all available physics tests"""
        try:
            self.logger.info("Running all physics tests...")
            
            # Get available experiments
            experiments = get_available_experiments()
            
            # Define test parameters for each experiment type
            test_parameters = {
                "classical_mechanics": {
                    "experiment_name": "Simple Harmonic Oscillator",
                    "mass": 1.0,
                    "initial_velocity": 0.0,
                    "initial_position": 1.0,
                    "time_step": 0.01,
                    "duration": 10.0,
                    "forces": "Spring force (k=1.0 N/m)"
                },
                "quantum_mechanics": {
                    "system_name": "Particle in a Box",
                    "potential": "Infinite square well",
                    "boundary_conditions": "Zero at boundaries",
                    "grid_points": 100,
                    "time_step": 0.01,
                    "duration": 5.0,
                    "n": 1,
                    "l": 0,
                    "m": 0
                },
                "thermodynamics": {
                    "process_name": "Isothermal Expansion",
                    "initial_temperature": 300.0,
                    "initial_pressure": 101325.0,
                    "initial_volume": 0.001,
                    "num_particles": 1000,
                    "simulation_time": 10.0,
                    "process_type": "isothermal",
                    "heat_capacity": 20.8,
                    "gas_constant": 8.314
                },
                "electromagnetism": {
                    "field_type": "Electromagnetic Wave",
                    "electric_field": 1.0,
                    "magnetic_field": 1.0,
                    "frequency": 1e9,
                    "wavelength": 0.3,
                    "grid_resolution": 50,
                    "domain_size": 1.0,
                    "electric_bc": "Periodic",
                    "magnetic_bc": "Periodic"
                },
                "relativity": {
                    "relativity_type": "Time Dilation",
                    "velocity": 0.5,
                    "mass": 1.0,
                    "proper_time": 1.0,
                    "coordinate_time": 1.0,
                    "metric": "Minkowski",
                    "central_mass": 1e30,
                    "schwarzschild_radius": 1e3,
                    "orbital_radius": 1e6
                }
            }
            
            # Run each experiment
            for experiment_type in experiments:
                if experiment_type in test_parameters:
                    print(f"\nRunning {experiment_type} test...")
                    result = await self.run_experiment(experiment_type, test_parameters[experiment_type])
                    
                    if result["success"]:
                        print(f"✅ {experiment_type} test completed successfully")
                        print(f"   Experiment ID: {result['experiment_id']}")
                    else:
                        print(f"❌ {experiment_type} test failed: {result['error']}")
                else:
                    print(f"⚠️ No test parameters defined for {experiment_type}")
            
            # Print summary
            successful_tests = [r for r in self.test_results if r["success"]]
            failed_tests = [r for r in self.test_results if not r["success"]]
            
            print(f"\n📊 Test Summary:")
            print(f"   Total tests: {len(self.test_results)}")
            print(f"   Successful: {len(successful_tests)}")
            print(f"   Failed: {len(failed_tests)}")
            
            if failed_tests:
                print(f"\n❌ Failed tests:")
                for test in failed_tests:
                    print(f"   - {test['experiment_type']}: {test['error']}")
            
            return len(successful_tests) == len(experiments)
            
        except Exception as e:
            self.logger.error(f"Error running all tests: {e}")
            return False
    
    async def run_interactive_test(self):
        """Run interactive physics test selection"""
        try:
            print("🧪 RAVANA Physics Test Runner")
            print("=" * 40)
            
            # List available experiments
            experiments = get_available_experiments()
            print(f"\nAvailable experiments:")
            for i, exp in enumerate(experiments, 1):
                info = get_experiment_info(exp)
                print(f"   {i}. {info.name}")
                print(f"      {info.description}")
            
            # Get user selection
            while True:
                try:
                    choice = input(f"\nSelect experiment (1-{len(experiments)}) or 'all' for all tests: ").strip()
                    
                    if choice.lower() == 'all':
                        return await self.run_all_tests()
                    
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(experiments):
                        selected_experiment = experiments[choice_num - 1]
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(experiments)}")
                        
                except ValueError:
                    print("Please enter a valid number or 'all'")
            
            # Get experiment parameters
            info = get_experiment_info(selected_experiment)
            print(f"\n{info.name}")
            print(f"{info.description}")
            print(f"\nRequired parameters: {', '.join(info.parameters)}")
            
            parameters = {}
            for param in info.parameters:
                value = input(f"Enter {param}: ").strip()
                try:
                    # Try to convert to number if possible
                    if '.' in value:
                        parameters[param] = float(value)
                    else:
                        parameters[param] = int(value)
                except ValueError:
                    parameters[param] = value
            
            # Run the experiment
            print(f"\nRunning {selected_experiment} experiment...")
            result = await self.run_experiment(selected_experiment, parameters)
            
            if result["success"]:
                print(f"✅ Experiment completed successfully!")
                print(f"   Experiment ID: {result['experiment_id']}")
            else:
                print(f"❌ Experiment failed: {result['error']}")
            
            return result["success"]
            
        except Exception as e:
            self.logger.error(f"Error in interactive test: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the physics test runner"""
        try:
            self.logger.info("Shutting down RAVANA Physics Test Runner...")
            
            if self.ravana:
                await self.ravana.shutdown()
            
            self.logger.info("RAVANA Physics Test Runner shutdown complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            return False

async def main():
    """Main function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    runner = PhysicsTestRunner()
    
    try:
        if await runner.initialize():
            # Check command line arguments
            if len(sys.argv) > 1:
                if sys.argv[1] == '--all':
                    success = await runner.run_all_tests()
                else:
                    print("Usage: python run_physics_tests.py [--all]")
                    print("  --all: Run all available tests")
                    print("  (no args): Interactive mode")
                    success = await runner.run_interactive_test()
            else:
                success = await runner.run_interactive_test()
            
            if success:
                print("\n🎉 All physics tests completed successfully!")
                sys.exit(0)
            else:
                print("\n❌ Some physics tests failed!")
                sys.exit(1)
        else:
            print("Failed to initialize RAVANA Physics Test Runner")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        await runner.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
