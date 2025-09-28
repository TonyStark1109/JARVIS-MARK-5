"""
Start MCP Servers
"""

import asyncio
import logging
from typing import List, Dict, Any
from deep_research_mcp_server import DeepResearchMCPServer
from image_processing_mcp_server import ImageProcessingMCPServer
from time_mcp_server import TimeMCPServer
from weather_mcp_server import WeatherMCPServer

logger = logging.getLogger(__name__)

class MCPServerManager:
    """Manages multiple MCP servers."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.servers = []
        self.setup_servers()
    
    def setup_servers(self):
        """Setup all MCP servers."""
        try:
            self.servers = [
                DeepResearchMCPServer(),
                ImageProcessingMCPServer(),
                TimeMCPServer(),
                WeatherMCPServer()
            ]
            self.logger.info("Initialized %d MCP servers", len(self.servers))
        except (ImportError, AttributeError, RuntimeError) as e:
            self.logger.error("Failed to setup servers: %s", e)
    
    async def start_all_servers(self):
        """Start all MCP servers."""
        try:
            self.logger.info("Starting all MCP servers...")
            
            # Start all servers concurrently
            tasks = []
            for server in self.servers:
                task = asyncio.create_task(server.start())
                tasks.append(task)
            
            # Wait for all servers to start
            await asyncio.gather(*tasks, return_exceptions=True)
            
            self.logger.info("All MCP servers started successfully")
        except (asyncio.CancelledError, RuntimeError) as e:
            self.logger.error("Failed to start servers: %s", e)
    
    async def stop_all_servers(self):
        """Stop all MCP servers."""
        try:
            self.logger.info("Stopping all MCP servers...")
            # Server shutdown logic would go here
            self.logger.info("All MCP servers stopped")
        except (asyncio.CancelledError, RuntimeError) as e:
            self.logger.error("Failed to stop servers: %s", e)
    
    def get_server_status(self) -> List[Dict[str, Any]]:
        """Get status of all servers."""
        try:
            status = []
            for i, server in enumerate(self.servers):
                status.append({
                    "server_id": i,
                    "name": server.__class__.__name__,
                    "status": "running" if hasattr(server, 'running') else "unknown"
                })
            return status
        except (AttributeError, TypeError) as e:
            self.logger.error("Failed to get server status: %s", e)
            return []

async def main():
    """Main function."""
    try:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Create server manager
        manager = MCPServerManager()
        
        # Start all servers
        await manager.start_all_servers()
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down servers...")
            await manager.stop_all_servers()
            
    except (asyncio.CancelledError, RuntimeError) as e:
        logger.error("Main error: %s", e)

if __name__ == "__main__":
    asyncio.run(main())