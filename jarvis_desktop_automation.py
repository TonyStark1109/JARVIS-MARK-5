#!/usr/bin/env python3
"""
JARVIS Mark 5 - Desktop Automation & Media Control Interface
Natural language interface for desktop automation, camera control, and media management
"""

import asyncio
import logging
import sys
import os
import re
from typing import Dict, Any, Optional, List

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'RAVANA'))

# pylint: disable=wrong-import-position
from backend.modules.desktop_automation import DesktopAutomation

logger = logging.getLogger(__name__)

class JARVISDesktopInterface:
    """Natural language interface for JARVIS desktop automation and media control"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.desktop_automation = DesktopAutomation()
        self.command_patterns = self._initialize_command_patterns()
        self.logger.info("JARVIS Desktop Interface initialized")

    # Convenience methods for voice commands
    def open_application(self, app_name: str) -> Dict[str, Any]:
        """Open an application"""
        return self.desktop_automation.open_application(app_name)

    def take_screenshot(self, save_path: str = None) -> Dict[str, Any]:
        """Take a screenshot"""
        return self.desktop_automation.take_screenshot(save_path)

    def take_picture(self, save_path: str = None) -> Dict[str, Any]:
        """Take a picture with camera"""
        return self.desktop_automation.snap_picture(save_path)

    def start_screen_recording(self, duration_seconds: int = 10,
                              save_path: str = None) -> Dict[str, Any]:
        """Start screen recording"""
        return self.desktop_automation.start_screen_recording(duration_seconds, save_path)

    def type_text(self, text: str) -> Dict[str, Any]:
        """Type text"""
        return self.desktop_automation.type_text(text)

    def _initialize_command_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize natural language command patterns for desktop automation"""
        return {
            # Camera Control Patterns
            "camera_control": {
                "patterns": [
                    r"take\s+(?:a\s+)?picture",
                    r"snap\s+(?:a\s+)?(?:photo|pic)",
                    r"capture\s+(?:a\s+)?(?:photo|image)",
                    r"open\s+camera",
                    r"start\s+camera",
                    r"launch\s+camera",
                    r"record\s+(?:a\s+)?video",
                    r"take\s+(?:a\s+)?video",
                    r"start\s+recording",
                    r"stop\s+recording",
                    r"(?:jarvis|j|buddy)[,\s]*(?:take\s+(?:a\s+)?picture|"
                    r"snap\s+(?:a\s+)?(?:photo|pic)|"
                    r"capture\s+(?:a\s+)?(?:photo|image)|"
                    r"open\s+camera|start\s+camera|launch\s+camera|"
                    r"record\s+(?:a\s+)?video|take\s+(?:a\s+)?video|"
                    r"start\s+recording|stop\s+recording)"
                ],
                "category": "camera"
            },

            # Screenshot & Screen Recording Patterns
            "screen_control": {
                "patterns": [
                    r"take\s+screenshot",
                    r"capture\s+screen",
                    r"record\s+(?:my\s+)?screen",
                    r"screen\s+recording",
                    r"record\s+desktop",
                    r"(?:jarvis|j|buddy)[,\s]*(?:take\s+screenshot|"
                    r"capture\s+screen|record\s+(?:my\s+)?screen|"
                    r"screen\s+recording|record\s+desktop)"
                ],
                "category": "screen"
            },

            # Application Control Patterns
            "app_control": {
                "patterns": [
                    r"open\s+(\w+)",
                    r"launch\s+(\w+)",
                    r"start\s+(\w+)",
                    r"close\s+(\w+)",
                    r"quit\s+(\w+)",
                    r"exit\s+(\w+)"
                ],
                "category": "application"
            },

            # YouTube & Media Patterns
            "media_control": {
                "patterns": [
                    r"search\s+youtube\s+for\s+(.+)",
                    r"youtube\s+(.+)",
                    r"find\s+video\s+(.+)",
                    r"play\s+video\s+(.+)",
                    r"watch\s+(.+)",
                    r"download\s+video\s+(.+)",
                    r"save\s+video\s+(.+)"
                ],
                "category": "media"
            },

            # Google Search Patterns
            "web_search": {
                "patterns": [
                    r"search\s+google\s+for\s+(.+)",
                    r"google\s+(.+)",
                    r"search\s+for\s+(.+)",
                    r"look\s+up\s+(.+)"
                ],
                "category": "search"
            },

            # Desktop Automation Patterns
            "desktop_automation": {
                "patterns": [
                    r"click\s+at\s+(\d+)\s+(\d+)",
                    r"click\s+(\d+)\s+(\d+)",
                    r"type\s+(.+)",
                    r"write\s+(.+)",
                    r"enter\s+(.+)",
                    r"press\s+(\w+)",
                    r"hit\s+(\w+)",
                    r"key\s+(\w+)"
                ],
                "category": "automation"
            },

            # Help Patterns
            "help": {
                "patterns": [
                    r"help",
                    r"what\s+can\s+you\s+do",
                    r"commands",
                    r"show\s+help",
                    r"desktop\s+help"
                ],
                "category": "help"
            }
        }

    def parse_command(self, command: str) -> Optional[Dict[str, Any]]:
        """Parse natural language command and return action details"""
        command = command.lower().strip()

        for action_name, action_info in self.command_patterns.items():
            for pattern in action_info["patterns"]:
                match = re.search(pattern, command, re.IGNORECASE)
                if match:
                    return {
                        "action": action_name,
                        "category": action_info["category"],
                        "matches": match.groups(),
                        "original_command": command,
                        "pattern": pattern
                    }

        return None

    async def execute_command(self, command: str) -> Dict[str, Any]:
        """Execute a natural language desktop automation command"""
        try:
            # Parse the command
            parsed = self.parse_command(command)
            if not parsed:
                return {
                    "success": False,
                    "error": f"I don't understand the command: '{command}'",
                    "message": "Try saying something like 'take picture' or 'open notepad'",
                    "suggestions": [
                        "take picture",
                        "open camera",
                        "record video for 30 seconds",
                        "take screenshot",
                        "open notepad",
                        "search youtube for funny cats",
                        "search google for python tutorial",
                        "help"
                    ]
                }

            # Execute the command using desktop automation
            result = self.desktop_automation.process_natural_command(command)

            # Add parsing information to result
            result["parsed_command"] = parsed
            result["timestamp"] = asyncio.get_event_loop().time()

            return result

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            self.logger.error("Error executing command '%s': %s", command, e)
            return {
                "success": False,
                "error": str(e),
                "message": "An error occurred while executing the command"
            }

    def get_available_commands(self) -> List[str]:
        """Get list of example commands"""
        return [
            # Camera commands
            "take picture",
            "open camera",
            "record video for 30 seconds",
            "stop recording",

            # Screen commands
            "take screenshot",
            "record screen for 60 seconds",

            # Application commands
            "open notepad",
            "open chrome",
            "open calculator",
            "close notepad",

            # Media commands
            "search youtube for funny cats",
            "play video https://youtube.com/watch?v=example",
            "download video https://youtube.com/watch?v=example",

            # Search commands
            "search google for python tutorial",
            "google artificial intelligence",

            # Desktop automation
            "click at 500 300",
            "type hello world",
            "press enter",

            # Help
            "help"
        ]

    def get_capabilities(self) -> Dict[str, Any]:
        """Get current capabilities and status"""
        status = self.desktop_automation.get_status()
        return {
            "interface": "JARVIS Desktop Automation",
            "version": "1.0.0",
            "desktop_automation": status,
            "supported_commands": len(self.get_available_commands()),
            "command_categories": list(set(
                info["category"] for info in self.command_patterns.values()
            )),
            "natural_language": True
        }

async def main():
    """Test the desktop automation interface"""
    print("🤖 JARVIS DESKTOP AUTOMATION INTERFACE")
    print("=" * 60)

    # Initialize interface
    interface = JARVISDesktopInterface()

    # Test commands
    test_commands = [
        "take picture",
        "open notepad",
        "search youtube for funny cats",
        "search google for python tutorial",
        "take screenshot",
        "help"
    ]

    print("\n🧪 Testing Desktop Automation Commands:")
    print("-" * 40)

    for cmd in test_commands:
        print(f"\n💬 Command: '{cmd}'")
        result = await interface.execute_command(cmd)

        if result["success"]:
            print(f"✅ {result['message']}")
            if "parsed_command" in result:
                parsed = result["parsed_command"]
                print(f"   Category: {parsed['category']}")
                print(f"   Pattern: {parsed['pattern']}")
        else:
            print(f"❌ Error: {result['error']}")

    # Show capabilities
    print(f"\n📊 Capabilities: {interface.get_capabilities()}")
    print(f"📋 Available Commands: {len(interface.get_available_commands())}")
    print("\n🎉 JARVIS Desktop Automation Interface Ready!")
    print("💡 Try saying: 'Jarvis, take a picture' or 'J, open chrome'")

if __name__ == "__main__":
    asyncio.run(main())
