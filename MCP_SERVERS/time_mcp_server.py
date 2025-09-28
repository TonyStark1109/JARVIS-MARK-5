"""
Time MCP Server
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List
from mcp import types
from mcp.server import Server

logger = logging.getLogger(__name__)

class TimeMCPServer:
    """MCP Server for time-related operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.server = Server("time")
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup MCP handlers."""
        @self.server.list_tools()
        async def list_tools() -> List[types.Tool]:
            """List available time tools."""
            return [
                types.Tool(
                    name="get_current_time",
                    description="Get current time",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "timezone": {"type": "string", "description": "Timezone (optional)"}
                        }
                    }
                ),
                types.Tool(
                    name="format_time",
                    description="Format time string",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "time_string": {"type": "string", "description": "Time string to format"},
                            "format": {"type": "string", "description": "Output format"}
                        },
                        "required": ["time_string"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
            """Handle tool calls."""
            if name == "get_current_time":
                result = await self.get_current_time(arguments)
            elif name == "format_time":
                result = await self.format_time(arguments)
            else:
                result = {"error": f"Unknown tool: {name}"}
            
            return [types.TextContent(type="text", text=str(result))]
    
    async def get_current_time(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get current time."""
        try:
            timezone_str = args.get("timezone", "UTC")
            
            self.logger.info(f"Getting current time for timezone: {timezone_str}")
            
            # Get current time
            current_time = datetime.now(timezone.utc)
            
            result = {
                "current_time": current_time.isoformat(),
                "timezone": timezone_str,
                "timestamp": current_time.timestamp(),
                "formatted": current_time.strftime("%Y-%m-%d %H:%M:%S UTC")
            }
            
            return result
        except Exception as e:
            self.logger.error(f"Get time error: {e}")
            return {"error": str(e)}
    
    async def format_time(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Format time string."""
        try:
            time_string = args.get("time_string", "")
            format_str = args.get("format", "%Y-%m-%d %H:%M:%S")
            
            self.logger.info(f"Formatting time string: {time_string}")
            
            # Parse and format time
            try:
                dt = datetime.fromisoformat(time_string.replace('Z', '+00:00'))
                formatted = dt.strftime(format_str)
                
                result = {
                    "original": time_string,
                    "formatted": formatted,
                    "format": format_str,
                    "status": "success"
                }
            except ValueError:
                result = {
                    "original": time_string,
                    "error": "Invalid time format",
                    "status": "error"
                }
            
            return result
        except Exception as e:
            self.logger.error(f"Format time error: {e}")
            return {"error": str(e)}
    
    async def start(self):
        """Start the MCP server."""
        try:
            self.logger.info("Starting Time MCP Server...")
            # Server startup logic would go here
            self.logger.info("Time MCP Server started")
        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")

async def main():
    """Main function."""
    server = TimeMCPServer()
    await server.start()

if __name__ == "__main__":
    asyncio.run(main())