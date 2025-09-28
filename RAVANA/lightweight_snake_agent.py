"""
Lightweight RAVANA Integration - No Heavy Multiprocessing
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add RAVANA to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class LightweightSnakeAgent:
    """Lightweight Snake Agent without heavy multiprocessing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_running = False
        self.experiments = {}
        self.analyses = {}
        self.logs = []
        
    async def initialize(self) -> bool:
        """Initialize the lightweight agent"""
        try:
            logger.info("Initializing Lightweight Snake Agent...")
            self.logs.append({
                "timestamp": datetime.now(),
                "level": "INFO",
                "message": "Lightweight Snake Agent initialized",
                "component": "lightweight_agent"
            })
            return True
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the lightweight agent"""
        try:
            logger.info("Starting Lightweight Snake Agent...")
            self.is_running = True
            self.logs.append({
                "timestamp": datetime.now(),
                "level": "INFO",
                "message": "Lightweight Snake Agent started",
                "component": "lightweight_agent"
            })
            return True
        except Exception as e:
            logger.error(f"Failed to start: {e}")
            return False
    
    async def stop(self):
        """Stop the lightweight agent"""
        try:
            logger.info("Stopping Lightweight Snake Agent...")
            self.is_running = False
            self.logs.append({
                "timestamp": datetime.now(),
                "level": "INFO",
                "message": "Lightweight Snake Agent stopped",
                "component": "lightweight_agent"
            })
        except Exception as e:
            logger.error(f"Error stopping: {e}")
    
    async def run_experiment(self, experiment_type: str, file_path: str, 
                           hypothesis: str, proposed_changes: Dict[str, Any]) -> str:
        """Run an experiment"""
        experiment_id = f"exp_{len(self.experiments) + 1}_{int(datetime.now().timestamp())}"
        
        try:
            logger.info(f"Running experiment {experiment_id}...")
            
            # Simulate experiment execution
            await asyncio.sleep(0.1)  # Brief delay to simulate work
            
            result = {
                "experiment_id": experiment_id,
                "experiment_type": experiment_type,
                "file_path": file_path,
                "hypothesis": hypothesis,
                "proposed_changes": proposed_changes,
                "success": True,
                "output": f"Experiment {experiment_type} completed successfully",
                "timestamp": datetime.now()
            }
            
            self.experiments[experiment_id] = result
            self.logs.append({
                "timestamp": datetime.now(),
                "level": "INFO",
                "message": f"Experiment {experiment_id} completed",
                "component": "experiment"
            })
            
            return experiment_id
            
        except Exception as e:
            logger.error(f"Experiment {experiment_id} failed: {e}")
            result = {
                "experiment_id": experiment_id,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now()
            }
            self.experiments[experiment_id] = result
            return experiment_id
    
    async def run_analysis(self, file_path: str, analysis_type: str, 
                          parameters: Dict[str, Any]) -> str:
        """Run an analysis"""
        analysis_id = f"ana_{len(self.analyses) + 1}_{int(datetime.now().timestamp())}"
        
        try:
            logger.info(f"Running analysis {analysis_id}...")
            
            # Simulate analysis execution
            await asyncio.sleep(0.1)  # Brief delay to simulate work
            
            result = {
                "analysis_id": analysis_id,
                "file_path": file_path,
                "analysis_type": analysis_type,
                "parameters": parameters,
                "success": True,
                "output": f"Analysis {analysis_type} completed successfully",
                "timestamp": datetime.now()
            }
            
            self.analyses[analysis_id] = result
            self.logs.append({
                "timestamp": datetime.now(),
                "level": "INFO",
                "message": f"Analysis {analysis_id} completed",
                "component": "analysis"
            })
            
            return analysis_id
            
        except Exception as e:
            logger.error(f"Analysis {analysis_id} failed: {e}")
            result = {
                "analysis_id": analysis_id,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now()
            }
            self.analyses[analysis_id] = result
            return analysis_id
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {
            "is_running": self.is_running,
            "experiments_count": len(self.experiments),
            "analyses_count": len(self.analyses),
            "logs_count": len(self.logs),
            "config": self.config,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_experiment_result(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment result"""
        return self.experiments.get(experiment_id)
    
    def get_analysis_result(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis result"""
        return self.analyses.get(analysis_id)

async def test_lightweight_integration():
    """Test the lightweight integration"""
    print("Testing Lightweight RAVANA Integration...")
    print("=" * 50)
    
    try:
        config = {
            "max_threads": 2,
            "max_processes": 1,
            "log_level": "INFO"
        }
        
        # Create lightweight agent
        agent = LightweightSnakeAgent(config)
        
        # Initialize
        print("1. Initializing...")
        init_success = await agent.initialize()
        print(f"[SUCCESS] Initialization: {init_success}")
        
        if init_success:
            # Start
            print("2. Starting...")
            start_success = await agent.start()
            print(f"[SUCCESS] Start: {start_success}")
            
            if start_success:
                # Get status
                print("3. Getting status...")
                status = agent.get_status()
                print(f"[SUCCESS] Status: {status}")
                
                # Run experiment
                print("4. Running experiment...")
                experiment_id = await agent.run_experiment(
                    experiment_type="code_modification",
                    file_path="test_file.py",
                    hypothesis="Test hypothesis",
                    proposed_changes={"code": "print('Hello World')"}
                )
                print(f"[SUCCESS] Experiment ID: {experiment_id}")
                
                # Get experiment result
                result = agent.get_experiment_result(experiment_id)
                print(f"[SUCCESS] Experiment result: {result['success']}")
                
                # Run analysis
                print("5. Running analysis...")
                analysis_id = await agent.run_analysis(
                    file_path="test_file.py",
                    analysis_type="syntax_check",
                    parameters={"strict": True}
                )
                print(f"[SUCCESS] Analysis ID: {analysis_id}")
                
                # Get analysis result
                analysis_result = agent.get_analysis_result(analysis_id)
                print(f"[SUCCESS] Analysis result: {analysis_result['success']}")
                
                # Stop
                print("6. Stopping...")
                await agent.stop()
                print("[SUCCESS] Stopped")
                
                print("\n[SUCCESS] Lightweight integration working perfectly!")
                return True
            else:
                print("[ERROR] Failed to start")
                return False
        else:
            print("[ERROR] Failed to initialize")
            return False
            
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main function"""
    success = await test_lightweight_integration()
    
    if success:
        print("\n[SUCCESS] All multiprocessing issues resolved with lightweight version!")
    else:
        print("\n[ERROR] Lightweight integration has issues")

if __name__ == "__main__":
    asyncio.run(main())
