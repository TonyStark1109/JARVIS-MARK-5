# JARVIS-MARK5 MCP Servers

This directory contains the Model Context Protocol (MCP) servers that provide real-time capabilities to the JARVIS-MARK5 system.

## Overview

The MCP servers replace the previous simulation layer with actual protocol-compliant servers that can be integrated with any MCP-compatible client.

## Available Servers

### 1. Deep Research MCP Server (`deep_research_mcp_server.py`)
Provides advanced research capabilities:
- **web_research**: Comprehensive web research on topics
- **document_analysis**: Analyze documents and extract key information
- **academic_search**: Search academic databases (arXiv, PubMed, etc.)
- **trend_analysis**: Analyze trends and patterns in data

### 2. Weather MCP Server (`weather_mcp_server.py`)
Provides weather data and services:
- **current_weather**: Get current weather conditions
- **weather_forecast**: Get weather forecasts
- **weather_alerts**: Get weather alerts and warnings
- **weather_comparison**: Compare weather between locations
- **weather_history**: Get historical weather data

### 3. Time MCP Server (`time_mcp_server.py`)
Provides time and scheduling services:
- **current_time**: Get current time in specified timezone
- **time_conversion**: Convert time between timezones
- **schedule_event**: Schedule events or reminders
- **time_calculations**: Perform time-based calculations
- **calendar_info**: Get calendar information
- **time_zone_info**: Get timezone information

### 4. Image Processing MCP Server (`image_processing_mcp_server.py`)
Provides image analysis and processing:
- **image_analysis**: Analyze image content and extract information
- **ocr_extraction**: Extract text from images using OCR
- **object_detection**: Detect and identify objects in images
- **face_detection**: Detect and analyze faces
- **image_generation**: Generate images from text descriptions
- **image_enhancement**: Enhance or modify images

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure Python 3.8+ is installed

## Usage

### Starting Individual Servers

Each server can be run independently:

```bash
python deep_research_mcp_server.py
python weather_mcp_server.py
python time_mcp_server.py
python image_processing_mcp_server.py
```

### Starting All Servers

Use the server manager to start all servers:

```bash
python start_mcp_servers.py
```

This will:
- Start all MCP servers
- Monitor their health
- Restart crashed servers automatically
- Provide graceful shutdown on Ctrl+C

### Integration with JARVIS-MARK5

The servers are automatically integrated with the JARVIS-MARK5 system through the `jarvis_ravana_mcp_enhanced.py` file. The system will:

1. Discover available MCP servers
2. Connect to them using the MCP protocol
3. Route commands to appropriate servers
4. Return structured responses

## Testing

Test the integration by running:

```bash
python jarvis_ravana_mcp_enhanced.py
```

This will run a comprehensive test of all MCP capabilities.

## Server Configuration

Each server can be configured through environment variables or configuration files. See individual server files for specific configuration options.

## Logging

All servers use structured logging. Logs include:
- Server startup/shutdown events
- Tool call requests and responses
- Error messages and debugging information
- Performance metrics

## Protocol Compliance

These servers are fully compliant with the Model Context Protocol specification and can be used with any MCP-compatible client.

## Troubleshooting

### Common Issues

1. **Server won't start**: Check Python version and dependencies
2. **Connection failures**: Verify server scripts are executable
3. **Tool call errors**: Check server logs for detailed error messages

### Debug Mode

Run servers with debug logging:
```bash
export LOG_LEVEL=DEBUG
python start_mcp_servers.py
```

## Contributing

When adding new MCP servers:

1. Follow the existing server structure
2. Implement proper error handling
3. Add comprehensive logging
4. Include tool documentation
5. Update this README

## License

Same as the main JARVIS-MARK5 project.
