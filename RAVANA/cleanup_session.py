"""
RAVANA Session Cleanup Module
"""

import os
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class SessionCleanup:
    """Handles cleanup of RAVANA sessions."""
    
    def __init__(self, base_dir="temp_sessions"):
        self.logger = logging.getLogger(__name__)
        self.base_dir = Path(base_dir)
        self.temp_dirs = []
        self.temp_files = []
        self.active_sessions = {}
    
    def create_session(self, session_id):
        """Create a new session directory."""
        try:
            temp_dir = self.base_dir / f"session_{session_id}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            self.temp_dirs.append(str(temp_dir))
            self.active_sessions[session_id] = {
                "created": datetime.now(),
                "path": str(temp_dir),
                "files": []
            }
            self.logger.info("Created temporary session: %s", session_id)
            return str(temp_dir)
        except Exception as e:
            self.logger.error("Failed to create temp session: %s", e)
            return ""
    
    def add_temp_file(self, session_id: str, file_path: str) -> bool:
        """Add a temporary file to session tracking."""
        try:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["files"].append(file_path)
                self.temp_files.append(file_path)
                return True
            return False
        except Exception as e:
            self.logger.error("Failed to add temp file: %s", e)
            return False
    
    def cleanup_all(self) -> int:
        """Clean up all temporary files and directories."""
        try:
            cleaned_count = 0

            # Clean up tracked temp files
            for file_path in self.temp_files[:]:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        cleaned_count += 1
                    except Exception as e:
                        self.logger.warning("Failed to remove temp file %s: %s", file_path, e)
                self.temp_files.remove(file_path)

            # Clean up tracked temp directories
            for dir_path in self.temp_dirs[:]:
                if os.path.exists(dir_path):
                    try:
                        shutil.rmtree(dir_path)
                        cleaned_count += 1
                    except Exception as e:
                        self.logger.warning("Failed to remove temp dir %s: %s", dir_path, e)
                self.temp_dirs.remove(dir_path)

            # Clear active sessions
            self.active_sessions.clear()

            self.logger.info("Cleaned up %d temporary items", cleaned_count)
            return cleaned_count
        except Exception as e:
            self.logger.error("Failed to cleanup temp files: %s", e)
            return 0
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a session."""
        return self.active_sessions.get(session_id)
    
    def get_total_size(self) -> int:
        """Get total size of temporary files."""
        total_size = 0
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                try:
                    total_size += os.path.getsize(file_path)
                except Exception:
                    pass
        return total_size

class ResourceCleanup:
    """Handles cleanup of system resources."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cleanup_handlers = []
    
    def add_cleanup_handler(self, handler):
        """Add a cleanup handler."""
        self.cleanup_handlers.append(handler)
    
    async def cleanup_resources(self):
        """Clean up all registered resources."""
        try:
            for handler in self.cleanup_handlers:
                if asyncio.iscoroutinefunction(handler):
                    await handler()
                else:
                    handler()
            self.logger.info("Resource cleanup completed")
            return True
        except Exception as e:
            self.logger.error("Resource cleanup failed: %s", e)
            return False

# Global instances
session_cleanup = SessionCleanup()
resource_cleanup = ResourceCleanup()