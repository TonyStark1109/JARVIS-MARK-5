"""JARVIS Mark 5 - Advanced AI Assistant"""

import json
import logging
import re
from typing import Any, Dict, List, TYPE_CHECKING
import asyncio

from core.actions.exceptions import ActionError, ActionException
from core.actions.registry import ActionRegistry

if TYPE_CHECKING:
    from core.system import AGISystem
    from services.data_service import DataService

logger = logging.getLogger(__name__)

class ActionManager:
    def __init__(*args, **kwargs):  # pylint: disable=unused-argument
        self.system = system
        self.data_service = data_service
        self.action_registry = ActionRegistry(system, data_service)
        logger.info("ActionManager initialized with %d actions.", len(self.action_registry.actions))
        self.log_available_actions()

    def log_available_actions(*args, **kwargs):  # pylint: disable=unused-argument
        logger.info("Available Actions:")
        for action_name, action in self.action_registry.actions.items():
            description = action.description
            params = action.parameters
            logger.info("- %s:", action_name)
            logger.info("  Description: %s", description)
            if params:
                logger.info("  Parameters:")
                for param in params:
                    param_name = param.get('name')
                    param_type = param.get('type')
                    param_desc = param.get('description', '')
                    logger.info("    - %s (%s): %s", param_name, param_type, param_desc)
            else:
                logger.info("  Parameters: None")
        logger.info("")

    async def execute_action(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parses the decision from the LLM, validates it, and executes the chosen action.
        This method can handle both a raw LLM response and a pre-defined action dictionary.
        """
        action_name = "unknown"
        action_params = {}
        action_data = {}

        # Case 1: The decision is already a parsed action dictionary
        if "action" in decision and "params" in decision:
            action_data = decision

        # Case 2: The decision is a raw response from the LLM
        elif "raw_response" in decision:
            raw_response = decision.get("raw_response", "")
            if not raw_response:
                logger.warning("Decision engine provided an empty raw_response.")
                return {"error": "No action taken: empty response."}

            try:
                # Find the JSON block in the raw response
                json_start = raw_response.find("```json")
                json_end = raw_response.rfind("```")

                if json_start == -1 or json_end == -1 or json_start >= json_end:
                    logger.warning("No valid JSON block found in the LLM's response. Trying to parse the whole string.")
                    action_data = json.loads(raw_response)
                else:
                    json_str = raw_response[json_start + 7:json_end].strip()
                    action_data = json.loads(json_str)

            except json.JSONDecodeError as e:
                logger.error("Failed to decode JSON from response: %s. Error: %s", raw_response, e)
                # Try to extract action using regex as fallback
                action_match = re.search(r'"action"\s*:\s*"([^"]+)"', raw_response)
                if action_match:
                    action_name = action_match.group(1)
                    # Try to extract params
                    params_match = re.search(r'"params"\s*:\s*({[^}]+})', raw_response)
                    if params_match:
                        try:
                            action_params = json.loads(params_match.group(1))
                        except:
                            action_params = {}
                    return await self._execute_action_with_fallback(action_name, action_params)
                return {"error": "No action taken: could not parse response."}

        else:
            logger.error("Invalid decision format: %s", decision)
            return {"error": "No action taken: invalid decision format."}

        try:
            action_name = action_data.get("action")
            action_params = action_data.get("params", {})

            if not action_name:
                logger.warning("No 'action' key found in the parsed JSON.")
                return {"error": "No action taken: 'action' key missing."}

            return await self._execute_action_with_fallback(action_name, action_params)

        except (ValueError, TypeError, AttributeError, ImportError) as e:
            logger.error(f"An unexpected error occurred during execution of action '{action_name}': {e}", exc_info=True)
            await asyncio.to_thread(
                self.data_service.save_action_log,
                action_name,
                action_params,
                'error',
                f"Unexpected error: {e}"
            )
            return {"error": f"An unexpected error occurred: {e}"}

    async def _execute_action_with_fallback(self, action_name: str, action_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action with comprehensive error handling and fallback mechanisms."""
        try:
            action = self.action_registry.get_action(action_name)
            if not action:
                # Try to find similar action names as fallback
                similar_actions = self._find_similar_actions(action_name)
                if similar_actions:
                    logger.warning("Action '%s' not found. Similar actions: %s", action_name, similar_actions)
                    # Try the first similar action as fallback
                    action_name = similar_actions[0]
                    action = self.action_registry.get_action(action_name)

                if not action:
                    raise ActionException(f"Action '{action_name}' not found.")

            logger.info("Executing action '%s' with params: %s", action_name, action_params)

            # Add timeout to prevent hanging actions
            try:
                result = await asyncio.wait_for(
                    action.execute(**action_params),
                    timeout=300.0  # 5 minute timeout
                )
                logger.info("Action '%s' executed successfully.", action_name)

                # Log the action to the database
                await asyncio.to_thread(
                    self.data_service.save_action_log,
                    action_name,
                    action_params,
                    'success',
                    str(result) # Convert result to string for logging
                )
                return result
            except asyncio.TimeoutError:
                logger.error("Action '%s' timed out after 5 minutes", action_name)
                raise ActionException(f"Action '{action_name}' timed out")

        except ActionException as e:
            logger.error("Action '%s' failed: %s", action_name, e)
            await asyncio.to_thread(
                self.data_service.save_action_log,
                action_name,
                action_params,
                'error',
                str(e)
            )
            return {"error": str(e)}

    def _find_similar_actions(self, action_name: str) -> List[str]:
        """Find actions with similar names to the requested action."""
        similar_actions = []
        action_name_lower = action_name.lower()

        for registered_action_name in self.action_registry.actions.keys():
            # Check for exact substring match or fuzzy match
            if (action_name_lower in registered_action_name.lower() or
                registered_action_name.lower() in action_name_lower):
                similar_actions.append(registered_action_name)

        return similar_actions
