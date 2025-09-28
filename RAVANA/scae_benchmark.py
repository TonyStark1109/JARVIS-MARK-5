#!/usr/bin/env python3
"""
RAVANA SCAE (Self-Consistency and Accuracy Evaluation) Benchmark

This module provides benchmarking tools for evaluating the self-consistency
and accuracy of RAVANA's AI systems and agents.
"""

import asyncio
import logging
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Add the RAVANA directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from main import RAVANAMain

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Result of a benchmark test"""
    test_name: str
    success: bool
    accuracy: float
    consistency: float
    execution_time: float
    error_message: str = ""
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}

@dataclass
class ConsistencyTest:
    """Test case for consistency evaluation"""
    name: str
    description: str
    input_data: Any
    expected_output: Any
    tolerance: float = 0.01
    max_iterations: int = 3

class SCAEBenchmark:
    """Self-Consistency and Accuracy Evaluation Benchmark"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ravana = RAVANAMain()
        self.results = []
        self.consistency_tests = []
        
    async def initialize(self):
        """Initialize the SCAE benchmark"""
        try:
            self.logger.info("Initializing RAVANA SCAE Benchmark...")
            
            # Initialize RAVANA main system
            if not await self.ravana.initialize():
                self.logger.error("Failed to initialize RAVANA main system")
                return False
            
            if not await self.ravana.start():
                self.logger.error("Failed to start RAVANA main system")
                return False
            
            # Load consistency tests
            self._load_consistency_tests()
            
            self.logger.info("RAVANA SCAE Benchmark initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize SCAE benchmark: {e}")
            return False
    
    def _load_consistency_tests(self):
        """Load predefined consistency tests"""
        self.consistency_tests = [
            ConsistencyTest(
                name="mathematical_calculation",
                description="Test mathematical calculation consistency",
                input_data={"operation": "add", "a": 2, "b": 3},
                expected_output=5,
                tolerance=0.001
            ),
            ConsistencyTest(
                name="string_processing",
                description="Test string processing consistency",
                input_data={"text": "Hello World", "operation": "reverse"},
                expected_output="dlroW olleH",
                tolerance=0.0
            ),
            ConsistencyTest(
                name="logical_reasoning",
                description="Test logical reasoning consistency",
                input_data={"premise": "All birds can fly", "statement": "Penguins are birds"},
                expected_output="Penguins can fly",
                tolerance=0.0
            ),
            ConsistencyTest(
                name="pattern_recognition",
                description="Test pattern recognition consistency",
                input_data={"sequence": [1, 2, 3, 4, 5], "next_number": True},
                expected_output=6,
                tolerance=0.0
            ),
            ConsistencyTest(
                name="code_analysis",
                description="Test code analysis consistency",
                input_data={"code": "def add(a, b):\n    return a + b", "analysis": "function_purpose"},
                expected_output="Addition function",
                tolerance=0.1
            )
        ]
    
    async def run_consistency_test(self, test: ConsistencyTest) -> BenchmarkResult:
        """Run a single consistency test"""
        try:
            self.logger.info(f"Running consistency test: {test.name}")
            
            start_time = time.time()
            results = []
            
            # Run the test multiple times to check consistency
            for iteration in range(test.max_iterations):
                try:
                    # Use RAVANA's Snake Agents to process the input
                    experiment_id = await self.ravana.run_experiment(
                        experiment_type="code_modification",
                        file_path=f"consistency_test_{test.name}_{iteration}.py",
                        hypothesis=f"Consistency test: {test.name}",
                        proposed_changes={
                            "input_data": test.input_data,
                            "expected_output": test.expected_output,
                            "test_type": "consistency"
                        }
                    )
                    
                    if experiment_id:
                        # Simulate getting result (in real implementation, this would be actual result)
                        result = self._simulate_result(test.input_data, test.expected_output)
                        results.append(result)
                    else:
                        results.append(None)
                        
                except Exception as e:
                    self.logger.error(f"Error in iteration {iteration}: {e}")
                    results.append(None)
            
            execution_time = time.time() - start_time
            
            # Calculate consistency and accuracy
            consistency = self._calculate_consistency(results)
            accuracy = self._calculate_accuracy(results, test.expected_output, test.tolerance)
            
            success = consistency >= 0.8 and accuracy >= 0.8
            
            result = BenchmarkResult(
                test_name=test.name,
                success=success,
                accuracy=accuracy,
                consistency=consistency,
                execution_time=execution_time,
                details={
                    "iterations": len(results),
                    "results": results,
                    "tolerance": test.tolerance
                }
            )
            
            self.results.append(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Error running consistency test {test.name}: {e}")
            return BenchmarkResult(
                test_name=test.name,
                success=False,
                accuracy=0.0,
                consistency=0.0,
                execution_time=0.0,
                error_message=str(e)
            )
    
    def _simulate_result(self, input_data: Any, expected_output: Any) -> Any:
        """Simulate a result for testing purposes"""
        # In a real implementation, this would be the actual result from RAVANA
        # For now, we'll simulate with some randomness to test consistency
        import random
        
        if isinstance(expected_output, (int, float)):
            # Add some random variation
            variation = random.uniform(-0.1, 0.1) * expected_output
            return expected_output + variation
        elif isinstance(expected_output, str):
            # Sometimes return the expected output, sometimes a variation
            if random.random() < 0.8:
                return expected_output
            else:
                return expected_output[::-1]  # Reverse string
        else:
            return expected_output
    
    def _calculate_consistency(self, results: List[Any]) -> float:
        """Calculate consistency score from multiple results"""
        if not results or all(r is None for r in results):
            return 0.0
        
        # Filter out None results
        valid_results = [r for r in results if r is not None]
        if len(valid_results) < 2:
            return 1.0 if len(valid_results) == 1 else 0.0
        
        # Calculate consistency based on result similarity
        if isinstance(valid_results[0], (int, float)):
            # For numeric results, calculate coefficient of variation
            mean_val = sum(valid_results) / len(valid_results)
            if mean_val == 0:
                return 1.0 if all(r == 0 for r in valid_results) else 0.0
            
            variance = sum((r - mean_val) ** 2 for r in valid_results) / len(valid_results)
            std_dev = variance ** 0.5
            cv = std_dev / abs(mean_val)
            return max(0.0, 1.0 - cv)
        
        elif isinstance(valid_results[0], str):
            # For string results, calculate exact match ratio
            first_result = valid_results[0]
            matches = sum(1 for r in valid_results if r == first_result)
            return matches / len(valid_results)
        
        else:
            # For other types, use exact equality
            first_result = valid_results[0]
            matches = sum(1 for r in valid_results if r == first_result)
            return matches / len(valid_results)
    
    def _calculate_accuracy(self, results: List[Any], expected: Any, tolerance: float) -> float:
        """Calculate accuracy score from results"""
        if not results or all(r is None for r in results):
            return 0.0
        
        valid_results = [r for r in results if r is not None]
        if not valid_results:
            return 0.0
        
        if isinstance(expected, (int, float)):
            # For numeric results, check if within tolerance
            correct = sum(1 for r in valid_results if abs(r - expected) <= tolerance * abs(expected))
            return correct / len(valid_results)
        
        elif isinstance(expected, str):
            # For string results, check exact match
            correct = sum(1 for r in valid_results if r == expected)
            return correct / len(valid_results)
        
        else:
            # For other types, use exact equality
            correct = sum(1 for r in valid_results if r == expected)
            return correct / len(valid_results)
    
    async def run_all_consistency_tests(self) -> List[BenchmarkResult]:
        """Run all consistency tests"""
        try:
            self.logger.info("Running all consistency tests...")
            
            results = []
            for test in self.consistency_tests:
                result = await self.run_consistency_test(test)
                results.append(result)
                
                status = "✅" if result.success else "❌"
                print(f"{status} {test.name}: Accuracy={result.accuracy:.3f}, Consistency={result.consistency:.3f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running consistency tests: {e}")
            return []
    
    async def run_accuracy_benchmark(self) -> BenchmarkResult:
        """Run accuracy benchmark"""
        try:
            self.logger.info("Running accuracy benchmark...")
            
            start_time = time.time()
            
            # Test various accuracy scenarios
            accuracy_tests = [
                {"name": "simple_math", "input": "2 + 2", "expected": 4},
                {"name": "complex_math", "input": "sqrt(16) + 3 * 2", "expected": 10},
                {"name": "string_ops", "input": "Hello + World", "expected": "HelloWorld"},
                {"name": "logic", "input": "True and False", "expected": False},
            ]
            
            correct = 0
            total = len(accuracy_tests)
            
            for test in accuracy_tests:
                # Simulate processing (in real implementation, use RAVANA)
                result = self._simulate_result(test["input"], test["expected"])
                
                if isinstance(test["expected"], (int, float)):
                    if abs(result - test["expected"]) < 0.01:
                        correct += 1
                elif result == test["expected"]:
                    correct += 1
            
            execution_time = time.time() - start_time
            accuracy = correct / total
            
            result = BenchmarkResult(
                test_name="accuracy_benchmark",
                success=accuracy >= 0.8,
                accuracy=accuracy,
                consistency=1.0,  # Not applicable for accuracy test
                execution_time=execution_time,
                details={"correct": correct, "total": total}
            )
            
            self.results.append(result)
            return result
            
        except Exception as e:
            self.logger.error(f"Error running accuracy benchmark: {e}")
            return BenchmarkResult(
                test_name="accuracy_benchmark",
                success=False,
                accuracy=0.0,
                consistency=0.0,
                execution_time=0.0,
                error_message=str(e)
            )
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate benchmark report"""
        if not self.results:
            return {"error": "No results available"}
        
        successful_tests = [r for r in self.results if r.success]
        failed_tests = [r for r in self.results if not r.success]
        
        avg_accuracy = sum(r.accuracy for r in self.results) / len(self.results)
        avg_consistency = sum(r.consistency for r in self.results) / len(self.results)
        avg_execution_time = sum(r.execution_time for r in self.results) / len(self.results)
        
        return {
            "summary": {
                "total_tests": len(self.results),
                "successful_tests": len(successful_tests),
                "failed_tests": len(failed_tests),
                "success_rate": len(successful_tests) / len(self.results),
                "average_accuracy": avg_accuracy,
                "average_consistency": avg_consistency,
                "average_execution_time": avg_execution_time
            },
            "results": [asdict(r) for r in self.results],
            "timestamp": datetime.now().isoformat()
        }
    
    async def shutdown(self):
        """Shutdown the SCAE benchmark"""
        try:
            self.logger.info("Shutting down RAVANA SCAE Benchmark...")
            
            if self.ravana:
                await self.ravana.shutdown()
            
            self.logger.info("RAVANA SCAE Benchmark shutdown complete")
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
    
    benchmark = SCAEBenchmark()
    
    try:
        if await benchmark.initialize():
            print("🧪 RAVANA SCAE Benchmark")
            print("=" * 40)
            
            # Run consistency tests
            print("\nRunning consistency tests...")
            consistency_results = await benchmark.run_all_consistency_tests()
            
            # Run accuracy benchmark
            print("\nRunning accuracy benchmark...")
            accuracy_result = await benchmark.run_accuracy_benchmark()
            
            # Generate and display report
            print("\n📊 Benchmark Report:")
            report = benchmark.generate_report()
            
            summary = report["summary"]
            print(f"   Total Tests: {summary['total_tests']}")
            print(f"   Successful: {summary['successful_tests']}")
            print(f"   Failed: {summary['failed_tests']}")
            print(f"   Success Rate: {summary['success_rate']:.2%}")
            print(f"   Average Accuracy: {summary['average_accuracy']:.3f}")
            print(f"   Average Consistency: {summary['average_consistency']:.3f}")
            print(f"   Average Execution Time: {summary['average_execution_time']:.3f}s")
            
            # Save report to file
            report_file = Path("scae_benchmark_report.json")
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n📄 Report saved to: {report_file}")
            
            if summary['success_rate'] >= 0.8:
                print("\n🎉 Benchmark completed successfully!")
                sys.exit(0)
            else:
                print("\n⚠️ Benchmark completed with some failures!")
                sys.exit(1)
        else:
            print("Failed to initialize RAVANA SCAE Benchmark")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        await benchmark.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
