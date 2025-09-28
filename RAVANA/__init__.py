
"""
RAVANA AGI Core System

This module provides the core RAVANA AGI functionality including
intelligent adaptive building, emotional intelligence, and advanced
agent capabilities.
"""

import sys
import os

# Add virtual environment site-packages to path for IDE compatibility
project_root = os.path.dirname(os.path.dirname(__file__))
venv_site_packages = os.path.join(project_root, 'venv', 'Lib', 'site-packages')
if venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

# Version information
__version__ = "1.0.0"
__author__ = "JARVIS-MARK5"
__description__ = "RAVANA AGI Core System"
