#!/usr/bin/env python3
"""
JARVIS MARK 5 - BRAIN CONFIGURATION
This script manages JARVIS's brain configuration and settings.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class JarvisBrainConfig:
    """Manages JARVIS's brain configuration."""

    def __init__(self, config_file="config/brain_config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()

    def _load_config(self):
        """Load configuration from file."""
        default_config = {
            "ai_model": "gpt-4",
            "voice_enabled": True,
            "voice_rate": 150,
            "voice_volume": 0.9,
            "hacking_tools_enabled": True,
            "ravana_integration": True,
            "perplexica_enabled": True,
            "max_tools": 1000,
            "timeout": 30,
            "log_level": "INFO"
        }

        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
                logger.error("Error loading config: %s", e)
                return default_config
        else:
            # Create default config file
            self._save_config(default_config)
            return default_config

    def _save_config(self, config):
        """Save configuration to file."""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            logger.info("Configuration saved to %s", self.config_file)
        except (OSError, PermissionError, TypeError, ValueError) as e:
            logger.error("Error saving config: %s", e)

    def get(self, key, default=None):
        """Get a configuration value."""
        return self.config.get(key, default)

    def set(self, key, value):
        """Set a configuration value."""
        self.config[key] = value
        self._save_config(self.config)

    def update(self, updates):
        """Update multiple configuration values."""
        self.config.update(updates)
        self._save_config(self.config)

    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        default_config = {
            "ai_model": "gpt-4",
            "voice_enabled": True,
            "voice_rate": 150,
            "voice_volume": 0.9,
            "hacking_tools_enabled": True,
            "ravana_integration": True,
            "perplexica_enabled": True,
            "max_tools": 1000,
            "timeout": 30,
            "log_level": "INFO"
        }
        self.config = default_config
        self._save_config(self.config)
        logger.info("Configuration reset to defaults")

def main():
    """Main function."""
    brain_config = JarvisBrainConfig()

    print("JARVIS Brain Configuration Manager")
    print("Commands: 'get <key>', 'set <key> <value>', 'list', 'reset', 'quit'")

    while True:
        try:
            command = input("\nBrain> ").strip().split()

            if not command:
                continue
            elif command[0] == 'quit':
                break
            elif command[0] == 'get' and len(command) == 2:
                value = brain_config.get(command[1])
                print(f"{command[1]}: {value}")
            elif command[0] == 'set' and len(command) >= 3:
                key = command[1]
                value = ' '.join(command[2:])
                # Try to convert to appropriate type
                try:
                    if value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
                    elif value.isdigit():
                        value = int(value)
                    elif value.replace('.', '').isdigit():
                        value = float(value)
                except (ValueError, TypeError):
                    pass
                brain_config.set(key, value)
                print(f"Set {key} = {value}")
            elif command[0] == 'list':
                for key, value in brain_config.config.items():
                    print(f"{key}: {value}")
            elif command[0] == 'reset':
                brain_config.reset_to_defaults()
                print("Configuration reset to defaults")
            else:
                print("Unknown command")
        except KeyboardInterrupt:
            break
        except (ValueError, KeyError, TypeError) as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
