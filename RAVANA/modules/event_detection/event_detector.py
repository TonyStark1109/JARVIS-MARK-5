#!/usr/bin/env python3
"""
RAVANA Event Detector
Detects events from various sources and streams.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import json
import threading
import queue

logger = logging.getLogger(__name__)

class EventDetector:
    """Detects events from multiple sources and notifies subscribers."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.event_sources: Dict[str, Dict[str, Any]] = {}
        self.subscribers: List[Callable] = []
        self.event_queue = queue.Queue()
        self.running = False
        self.detection_threads: List[threading.Thread] = []
        
    async def register_event_source(self, source_id: str, source_type: str, 
                                  detection_function: Callable, 
                                  config: Optional[Dict[str, Any]] = None) -> bool:
        """Register a new event source."""
        try:
            self.event_sources[source_id] = {
                'type': source_type,
                'function': detection_function,
                'config': config or {},
                'active': True,
                'last_detection': None
            }
            self.logger.info(f"Registered event source: {source_id} ({source_type})")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register event source {source_id}: {e}")
            return False
    
    async def subscribe(self, callback: Callable) -> bool:
        """Subscribe to event notifications."""
        try:
            self.subscribers.append(callback)
            self.logger.info(f"Added event subscriber: {callback.__name__}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add subscriber: {e}")
            return False
    
    async def start_detection(self) -> bool:
        """Start the event detection system."""
        try:
            if self.running:
                self.logger.warning("Event detection already running")
                return True
            
            self.running = True
            
            # Start detection threads for each source
            for source_id, source_info in self.event_sources.items():
                if source_info['active']:
                    thread = threading.Thread(
                        target=self._detection_worker,
                        args=(source_id, source_info),
                        daemon=True
                    )
                    thread.start()
                    self.detection_threads.append(thread)
            
            # Start event processing thread
            processing_thread = threading.Thread(
                target=self._event_processor,
                daemon=True
            )
            processing_thread.start()
            self.detection_threads.append(processing_thread)
            
            self.logger.info("Event detection system started")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start event detection: {e}")
            return False
    
    async def stop_detection(self) -> bool:
        """Stop the event detection system."""
        try:
            self.running = False
            
            # Wait for threads to finish
            for thread in self.detection_threads:
                thread.join(timeout=5.0)
            
            self.detection_threads.clear()
            self.logger.info("Event detection system stopped")
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop event detection: {e}")
            return False
    
    def _detection_worker(self, source_id: str, source_info: Dict[str, Any]):
        """Worker thread for detecting events from a specific source."""
        try:
            while self.running and source_info['active']:
                try:
                    # Call the detection function
                    events = source_info['function'](source_info['config'])
                    
                    if events:
                        for event in events:
                            event['source_id'] = source_id
                            event['detection_time'] = datetime.now()
                            self.event_queue.put(event)
                            source_info['last_detection'] = datetime.now()
                    
                    # Sleep based on config or default
                    sleep_time = source_info['config'].get('detection_interval', 1.0)
                    threading.Event().wait(sleep_time)
                    
                except Exception as e:
                    self.logger.error(f"Error in detection worker for {source_id}: {e}")
                    threading.Event().wait(5.0)  # Wait 5 seconds on error
        except Exception as e:
            self.logger.error(f"Detection worker for {source_id} failed: {e}")
    
    def _event_processor(self):
        """Process detected events and notify subscribers."""
        try:
            while self.running:
                try:
                    # Get event from queue (with timeout)
                    event = self.event_queue.get(timeout=1.0)
                    
                    # Process the event
                    processed_event = self._process_event(event)
                    
                    # Notify all subscribers
                    for subscriber in self.subscribers:
                        try:
                            if asyncio.iscoroutinefunction(subscriber):
                                asyncio.create_task(subscriber(processed_event))
                            else:
                                subscriber(processed_event)
                        except Exception as e:
                            self.logger.error(f"Error notifying subscriber {subscriber.__name__}: {e}")
                    
                    self.event_queue.task_done()
                    
                except queue.Empty:
                    continue  # Timeout, check if still running
                except Exception as e:
                    self.logger.error(f"Error processing event: {e}")
        except Exception as e:
            self.logger.error(f"Event processor failed: {e}")
    
    def _process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process and enrich an event."""
        try:
            processed_event = {
                'id': f"event_{int(datetime.now().timestamp() * 1000)}",
                'timestamp': event.get('detection_time', datetime.now()),
                'source_id': event.get('source_id'),
                'type': event.get('type', 'unknown'),
                'data': event.get('data', {}),
                'severity': event.get('severity', 'info'),
                'processed': True
            }
            
            # Add any additional processing here
            return processed_event
        except Exception as e:
            self.logger.error(f"Error processing event: {e}")
            return event
    
    async def get_event_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected events."""
        try:
            stats = {
                'total_sources': len(self.event_sources),
                'active_sources': sum(1 for s in self.event_sources.values() if s['active']),
                'subscribers': len(self.subscribers),
                'queue_size': self.event_queue.qsize(),
                'running': self.running
            }
            
            # Add per-source statistics
            for source_id, source_info in self.event_sources.items():
                stats[f'source_{source_id}'] = {
                    'active': source_info['active'],
                    'last_detection': source_info['last_detection']
                }
            
            return stats
        except Exception as e:
            self.logger.error(f"Error getting event statistics: {e}")
            return {}
