"""
RAVANA Multi-Strategy Executor
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class FailureReason(Enum):
    """Reasons for strategy failure."""
    FAILURE_RATE = "failure_rate"
    PROGRESS_STALL = "progress_stall"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    BETTER_ALTERNATIVE = "better_alternative"
    TIME_CONSTRAINT = "time_constraint"

@dataclass
class ExecutionResult:
    """Result of strategy execution."""
    strategy_name: str
    success: bool
    result: Any
    execution_time: float
    failure_reason: Optional[FailureReason] = None
    metadata: Dict[str, Any] = None

@dataclass
class ExecutionConfig:
    """Configuration for multi-strategy execution."""
    learning_enabled: bool = True
    early_termination: bool = True
    success_threshold: float = 0.8
    adaptation_frequency: int = 1  # Adapt after every N strategies
    carry_forward_results: bool = True

class MultiStrategyExecutor:
    """Executes multiple strategies and learns from results."""
    
    def __init__(self, config: ExecutionConfig = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or ExecutionConfig()
        self.strategies = []
        self.results = []
        self.learning_data = {}
    
    def add_strategy(self, strategy_name: str, strategy_func, priority: int = 1):
        """Add a strategy to execute."""
        try:
            self.strategies.append({
                'name': strategy_name,
                'function': strategy_func,
                'priority': priority,
                'success_count': 0,
                'failure_count': 0
            })
            self.logger.info(f"Added strategy: {strategy_name}")
        except Exception as e:
            self.logger.error(f"Error adding strategy {strategy_name}: {e}")
    
    def execute_strategies(self, input_data: Any) -> List[ExecutionResult]:
        """Execute all strategies on input data."""
        try:
            self.logger.info(f"Executing {len(self.strategies)} strategies")
            results = []
            
            # Sort strategies by priority
            sorted_strategies = sorted(self.strategies, key=lambda x: x['priority'], reverse=True)
            
            for strategy in sorted_strategies:
                try:
                    result = self._execute_single_strategy(strategy, input_data)
                    results.append(result)
                    
                    # Update strategy statistics
                    if result.success:
                        strategy['success_count'] += 1
                    else:
                        strategy['failure_count'] += 1
                    
                    # Check for early termination
                    if self.config.early_termination and result.success:
                        if self._should_terminate_early(results):
                            self.logger.info("Early termination triggered")
                            break
                            
                except Exception as e:
                    self.logger.error(f"Strategy {strategy['name']} execution error: {e}")
                    results.append(ExecutionResult(
                        strategy_name=strategy['name'],
                        success=False,
                        result=None,
                        execution_time=0.0,
                        failure_reason=FailureReason.FAILURE_RATE
                    ))
            
            self.results.extend(results)
            return results
            
        except Exception as e:
            self.logger.error(f"Multi-strategy execution error: {e}")
            return []
    
    def _execute_single_strategy(self, strategy: Dict[str, Any], input_data: Any) -> ExecutionResult:
        """Execute a single strategy."""
        import time
        start_time = time.time()
        
        try:
            result = strategy['function'](input_data)
                execution_time = time.time() - start_time
                
                
            return ExecutionResult(
                strategy_name=strategy['name'],
                success=True,
                result=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Strategy {strategy['name']} failed: {e}")
            
            return ExecutionResult(
                strategy_name=strategy['name'],
                success=False,
                result=None,
                execution_time=execution_time,
                failure_reason=FailureReason.FAILURE_RATE
            )
    
    def _should_terminate_early(self, results: List[ExecutionResult]) -> bool:
        """Check if execution should terminate early."""
        if not results:
            return False
        
        success_rate = sum(1 for r in results if r.success) / len(results)
        return success_rate >= self.config.success_threshold
    
    def get_best_result(self) -> Optional[ExecutionResult]:
        """Get the best execution result."""
        if not self.results:
            return None
        
        # Find result with highest success and lowest execution time
        successful_results = [r for r in self.results if r.success]
        if not successful_results:
            return None
        
        return min(successful_results, key=lambda x: x.execution_time)
    
    def get_strategy_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all strategies."""
        performance = {}
        
        for strategy in self.strategies:
            total_attempts = strategy['success_count'] + strategy['failure_count']
            success_rate = strategy['success_count'] / total_attempts if total_attempts > 0 else 0
            
            performance[strategy['name']] = {
                'success_count': strategy['success_count'],
                'failure_count': strategy['failure_count'],
                'success_rate': success_rate,
                'total_attempts': total_attempts
            }
        
        return performance
    
    def adapt_strategies(self):
        """Adapt strategies based on learning data."""
        if not self.config.learning_enabled:
            return
        
        try:
            self.logger.info("Adapting strategies based on performance data")
            performance = self.get_strategy_performance()
            
            # Simple adaptation: adjust priorities based on success rate
            for strategy in self.strategies:
                strategy_name = strategy['name']
                if strategy_name in performance:
                    success_rate = performance[strategy_name]['success_rate']
                    # Increase priority for successful strategies
                    strategy['priority'] = int(strategy['priority'] * (1 + success_rate))
            
            self.logger.info("Strategy adaptation completed")
            
        except Exception as e:
            self.logger.error(f"Strategy adaptation error: {e}")

def main():
    """Main function."""
    config = ExecutionConfig()
    executor = MultiStrategyExecutor(config)
    
    # Example usage
    def strategy1(data):
        return f"Strategy 1 result: {data}"
    
    def strategy2(data):
        return f"Strategy 2 result: {data}"
    
    executor.add_strategy("strategy1", strategy1, priority=1)
    executor.add_strategy("strategy2", strategy2, priority=2)
    
    results = executor.execute_strategies("test data")
    print(f"Executed {len(results)} strategies")

if __name__ == "__main__":
    main()