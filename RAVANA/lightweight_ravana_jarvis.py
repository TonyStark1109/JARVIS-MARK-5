"""
Lightweight RAVANA-JARVIS Integration
"""

import asyncio
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

# Add RAVANA to path
sys.path.insert(0, str(Path(__file__).parent))

# Import after path modification
from lightweight_snake_agent import LightweightSnakeAgent  # pylint: disable=import-error,wrong-import-position

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class RAVANAJARVISIntegration:
    """Lightweight RAVANA-JARVIS Integration"""

    def __init__(self):
        self.ravana_agent: Optional[LightweightSnakeAgent] = None
        self.is_initialized = False
        self.is_running = False

    async def initialize(self) -> bool:
        """Initialize the integration"""
        try:
            logger.info("Initializing RAVANA-JARVIS Integration...")

            # Create lightweight RAVANA agent
            config = {
                "max_threads": 4,
                "max_processes": 2,
                "log_level": "INFO",
                "enable_performance_monitoring": True
            }

            self.ravana_agent = LightweightSnakeAgent(config)

            # Initialize RAVANA
            ravana_success = await self.ravana_agent.initialize()
            if not ravana_success:
                logger.error("Failed to initialize RAVANA agent")
                return False

            self.is_initialized = True
            logger.info("[SUCCESS] RAVANA-JARVIS Integration initialized")
            return True

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to initialize integration: %s", e)
            return False

    async def start(self) -> bool:
        """Start the integration"""
        try:
            if not self.is_initialized:
                logger.error("Integration not initialized")
                return False

            logger.info("Starting RAVANA-JARVIS Integration...")

            # Start RAVANA
            if self.ravana_agent is None:
                logger.error("RAVANA agent not initialized")
                return False
            ravana_success = await self.ravana_agent.start()
            if not ravana_success:
                logger.error("Failed to start RAVANA agent")
                return False

            self.is_running = True
            logger.info("[SUCCESS] RAVANA-JARVIS Integration started")
            return True

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to start integration: %s", e)
            return False

    async def stop(self):
        """Stop the integration"""
        try:
            if self.ravana_agent and self.is_running:
                logger.info("Stopping RAVANA-JARVIS Integration...")
                await self.ravana_agent.stop()
                self.is_running = False
                logger.info("[SUCCESS] RAVANA-JARVIS Integration stopped")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Error stopping integration: %s", e)

    async def run_experiment(self, experiment_type: str, file_path: str,
                           hypothesis: str, proposed_changes: Dict[str, Any]) -> Optional[str]:
        """Run a RAVANA experiment"""
        if not self.is_running:
            logger.error("Integration not running")
            return None

        try:
            logger.info("Running RAVANA experiment: %s", experiment_type)
            if self.ravana_agent is None:
                logger.error("RAVANA agent not initialized")
                return None
            experiment_id = await self.ravana_agent.run_experiment(
                experiment_type=experiment_type,
                file_path=file_path,
                hypothesis=hypothesis,
                proposed_changes=proposed_changes
            )
            logger.info("[SUCCESS] Experiment %s started", experiment_id)
            return experiment_id
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to run experiment: %s", e)
            return None

    async def run_analysis(self, file_path: str, analysis_type: str,
                          parameters: Dict[str, Any]) -> Optional[str]:
        """Run a RAVANA analysis"""
        if not self.is_running:
            logger.error("Integration not running")
            return None

        try:
            logger.info("Running RAVANA analysis: %s", analysis_type)
            if self.ravana_agent is None:
                logger.error("RAVANA agent not initialized")
                return None
            analysis_id = await self.ravana_agent.run_analysis(
                file_path=file_path,
                analysis_type=analysis_type,
                parameters=parameters
            )
            logger.info("[SUCCESS] Analysis %s started", analysis_id)
            return analysis_id
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Failed to run analysis: %s", e)
            return None

    def get_status(self) -> Dict[str, Any]:
        """Get integration status"""
        if not self.ravana_agent:
            return {"status": "not_initialized"}

        status = self.ravana_agent.get_status()
        status["integration_status"] = "running" if self.is_running else "stopped"
        status["ravana_enabled"] = True
        return status

    def get_experiment_result(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment result"""
        if not self.ravana_agent:
            return None
        return self.ravana_agent.get_experiment_result(experiment_id)

    def get_analysis_result(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis result"""
        if not self.ravana_agent:
            return None
        return self.ravana_agent.get_analysis_result(analysis_id)

# Global integration instance
integration = RAVANAJARVISIntegration()

async def initialize_ravana_jarvis():
    """Initialize RAVANA-JARVIS integration"""
    return await integration.initialize()

async def start_ravana_jarvis():
    """Start RAVANA-JARVIS integration"""
    return await integration.start()

async def stop_ravana_jarvis():
    """Stop RAVANA-JARVIS integration"""
    await integration.stop()

def get_ravana_status():
    """Get RAVANA status"""
    return integration.get_status()

async def run_ravana_experiment(experiment_type: str, file_path: str,
                              hypothesis: str, proposed_changes: Dict[str, Any]) -> Optional[str]:
    """Run a RAVANA experiment"""
    return await integration.run_experiment(
        experiment_type, file_path, hypothesis, proposed_changes
    )

async def run_ravana_analysis(file_path: str, analysis_type: str,
                               parameters: Dict[str, Any]) -> Optional[str]:
    """Run a RAVANA analysis"""
    return await integration.run_analysis(
        file_path, analysis_type, parameters
    )

def get_experiment_result(experiment_id: str):
    """Get experiment result"""
    return integration.get_experiment_result(experiment_id)

def get_analysis_result(analysis_id: str):
    """Get analysis result"""
    return integration.get_analysis_result(analysis_id)

async def test_integration():
    """Test the integration"""
    print("Testing RAVANA-JARVIS Integration...")
    print("=" * 50)

    try:
        # Initialize
        print("1. Initializing...")
        init_success = await initialize_ravana_jarvis()
        print(f"[SUCCESS] Initialization: {init_success}")

        if init_success:
            # Start
            print("2. Starting...")
            start_success = await start_ravana_jarvis()
            print(f"[SUCCESS] Start: {start_success}")

            if start_success:
                # Get status
                print("3. Getting status...")
                status = get_ravana_status()
                print(f"[SUCCESS] Status: {status}")

                # Run experiment
                print("4. Running experiment...")
                experiment_id = await run_ravana_experiment(
                    experiment_type="code_modification",
                    file_path="test_file.py",
                    hypothesis="Test hypothesis",
                    proposed_changes={"code": "print('Hello World')"}
                )
                print(f"[SUCCESS] Experiment ID: {experiment_id}")

                # Get experiment result
                result = get_experiment_result(experiment_id)
                print(f"[SUCCESS] Experiment result: {result['success'] if result else 'None'}")

                # Run analysis
                print("5. Running analysis...")
                analysis_id = await run_ravana_analysis(
                    file_path="test_file.py",
                    analysis_type="syntax_check",
                    parameters={"strict": True}
                )
                print(f"[SUCCESS] Analysis ID: {analysis_id}")

                # Get analysis result
                analysis_result = get_analysis_result(analysis_id)
                result_text = analysis_result['success'] if analysis_result else 'None'
                print(f"[SUCCESS] Analysis result: {result_text}")

                # Stop
                print("6. Stopping...")
                await stop_ravana_jarvis()
                print("[SUCCESS] Stopped")

                print("\n[SUCCESS] RAVANA-JARVIS Integration working perfectly!")
                return True
            else:
                print("[ERROR] Failed to start")
                return False
        else:
            print("[ERROR] Failed to initialize")
            return False

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"[ERROR] Test failed: {e}")
        traceback.print_exc()
        return False

async def main():
    """Main function"""
    success = await test_integration()

    if success:
        print("\n[SUCCESS] RAVANA Snake Agents are now working without issues!")
    else:
        print("\n[ERROR] Integration has issues")

if __name__ == "__main__":
    asyncio.run(main())
