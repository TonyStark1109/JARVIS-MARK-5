"""
RAVANA Database Engine
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

class DatabaseEngine:
    """Database engine for RAVANA"""
    
    def __init__(self, db_path: str = "ravana.db"):
        self.logger = logging.getLogger(__name__)
        self.db_path = Path(db_path)
        self.connection = None
        
    async def initialize(self) -> bool:
        """Initialize the database"""
        try:
            self.logger.info("Initializing RAVANA Database Engine...")
            
            # Create database directory if it doesn't exist
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Connect to database
            self.connection = sqlite3.connect(str(self.db_path))
            
            # Create tables
            await self._create_tables()
            
            self.logger.info("RAVANA Database Engine initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            return False
    
    async def _create_tables(self):
        """Create database tables"""
        try:
            cursor = self.connection.cursor()
            
            # System events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    data TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Knowledge table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0
                )
            ''')
            
            # Experiments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    experiment_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    hypothesis TEXT,
                    result TEXT,
                    success BOOLEAN,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.connection.commit()
            self.logger.info("Database tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating tables: {e}")
            raise
    
    async def log_system_event(self, event_type: str, component: str, 
                             message: str, data: Dict[str, Any] = None) -> bool:
        """Log a system event"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                INSERT INTO system_events (event_type, component, message, data)
                VALUES (?, ?, ?, ?)
            ''', (event_type, component, message, json.dumps(data) if data else None))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Error logging system event: {e}")
            return False
    
    async def store_knowledge(self, knowledge_id: str, content: str, 
                            metadata: Dict[str, Any] = None) -> bool:
        """Store knowledge in database"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO knowledge (id, content, metadata, created_at, last_accessed, access_count)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (knowledge_id, content, json.dumps(metadata) if metadata else None, 
                  datetime.now(), datetime.now(), 0))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing knowledge: {e}")
            return False
    
    async def get_knowledge(self, knowledge_id: str) -> Optional[Dict[str, Any]]:
        """Get knowledge from database"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                SELECT content, metadata, created_at, last_accessed, access_count
                FROM knowledge WHERE id = ?
            ''', (knowledge_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    "id": knowledge_id,
                    "content": row[0],
                    "metadata": json.loads(row[1]) if row[1] else {},
                    "created_at": row[2],
                    "last_accessed": row[3],
                    "access_count": row[4]
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting knowledge: {e}")
            return None
    
    async def search_knowledge(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search knowledge in database"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                SELECT id, content, metadata, created_at, last_accessed, access_count
                FROM knowledge WHERE content LIKE ? OR metadata LIKE ?
                ORDER BY access_count DESC
                LIMIT ?
            ''', (f"%{query}%", f"%{query}%", limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row[0],
                    "content": row[1],
                    "metadata": json.loads(row[2]) if row[2] else {},
                    "created_at": row[3],
                    "last_accessed": row[4],
                    "access_count": row[5]
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching knowledge: {e}")
            return []
    
    async def store_experiment(self, experiment_id: str, experiment_type: str,
                             file_path: str, hypothesis: str, result: str, 
                             success: bool) -> bool:
        """Store experiment result"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO experiments (id, experiment_type, file_path, hypothesis, result, success, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (experiment_id, experiment_type, file_path, hypothesis, result, success, datetime.now()))
            
            self.connection.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing experiment: {e}")
            return False
    
    async def get_experiments(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent experiments"""
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                SELECT id, experiment_type, file_path, hypothesis, result, success, created_at
                FROM experiments
                ORDER BY created_at DESC
                LIMIT ?
            ''', (limit,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row[0],
                    "experiment_type": row[1],
                    "file_path": row[2],
                    "hypothesis": row[3],
                    "result": row[4],
                    "success": bool(row[5]),
                    "created_at": row[6]
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error getting experiments: {e}")
            return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            cursor = self.connection.cursor()
            
            # Count records in each table
            cursor.execute("SELECT COUNT(*) FROM system_events")
            event_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM knowledge")
            knowledge_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM experiments")
            experiment_count = cursor.fetchone()[0]
            
            return {
                "system_events": event_count,
                "knowledge_items": knowledge_count,
                "experiments": experiment_count
            }
            
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {}
    
    async def cleanup_old_data(self, days: int = 30):
        """Clean up old data"""
        try:
            cursor = self.connection.cursor()
            
            # Delete old system events
            cursor.execute('''
                DELETE FROM system_events 
                WHERE timestamp < datetime('now', '-{} days')
            '''.format(days))
            
            deleted_events = cursor.rowcount
            
            self.connection.commit()
            
            self.logger.info(f"Cleaned up {deleted_events} old system events")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    async def shutdown(self):
        """Shutdown the database engine"""
        try:
            if self.connection:
                self.connection.close()
                self.connection = None
            
            self.logger.info("Database engine shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during database shutdown: {e}")