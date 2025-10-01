"""
Discord Bot Profile for RAVANA

Updated with latest Discord API v10
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

class DiscordProfile:
    """Discord bot profile with updated API configuration"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.bot_token = None
        self.client = None
        self.guild_id = None
        self.channel_id = None
        self.is_active = False
        
        # Updated Discord API configuration for v10
        self.api_config = {
            "intents": {
                "guilds": True,
                "guild_messages": True,
                "guild_voice_states": True,
                "direct_messages": True,
                "message_content": True
            },
            "api_version": "10",
            "gateway_version": "10"
        }
        
        # Bot capabilities
        self.capabilities = {
            "text_messages": True,
            "voice_messages": True,
            "file_sharing": True,
            "slash_commands": True,
            "reactions": True,
            "embeds": True,
            "voice_channels": True
        }
    
    async def initialize(self, bot_token: str, guild_id: Optional[str] = None):
        """Initialize Discord bot with token"""
        try:
            self.bot_token = bot_token
            self.guild_id = guild_id
            
            # Import discord.py
            try:
                import discord
                from discord.ext import commands
            except ImportError:
                self.logger.error("discord.py not installed. Install with: pip install discord.py")
                return False
            
            # Create bot instance
            intents = discord.Intents(**self.api_config["intents"])
            self.client = commands.Bot(command_prefix='!', intents=intents)
            
            # Set up event handlers
            self._setup_event_handlers()
            
            # Test bot connection
            if await self._test_bot_connection():
                self.is_active = True
                self.logger.info("Discord bot initialized successfully")
                return True
            else:
                self.logger.error("Failed to connect Discord bot")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Discord bot: {e}")
            return False
    
    def _setup_event_handlers(self):
        """Set up Discord bot event handlers"""
        
        @self.client.event
        async def on_ready():
            self.logger.info(f"Discord bot logged in as {self.client.user}")
            self.is_active = True
        
        @self.client.event
        async def on_message(message):
            if message.author == self.client.user:
                return
            
            # Process message here
            await self._process_message(message)
        
        @self.client.event
        async def on_voice_state_update(member, before, after):
            # Handle voice state changes
            pass
    
    async def _test_bot_connection(self) -> bool:
        """Test Discord bot connection"""
        try:
            # Start bot in background
            bot_task = asyncio.create_task(self.client.start(self.bot_token))
            
            # Wait for connection
            await asyncio.sleep(5)
            
            if self.client.is_ready():
                return True
            else:
                bot_task.cancel()
                return False
                
        except Exception as e:
            self.logger.error(f"Error testing Discord connection: {e}")
            return False
    
    async def send_message(self, channel_id: str, content: str, **kwargs) -> bool:
        """Send message to Discord channel"""
        try:
            if not self.client or not self.client.is_ready():
                return False
            
            channel = self.client.get_channel(int(channel_id))
            if channel:
                await channel.send(content, **kwargs)
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error sending Discord message: {e}")
            return False
    
    async def send_embed(self, channel_id: str, embed_data: Dict[str, Any]) -> bool:
        """Send embed message to Discord channel"""
        try:
            if not self.client or not self.client.is_ready():
                return False
            
            import discord
            
            channel = self.client.get_channel(int(channel_id))
            if channel:
                embed = discord.Embed(**embed_data)
                await channel.send(embed=embed)
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error sending Discord embed: {e}")
            return False
    
    async def send_voice_message(self, channel_id: str, voice_file: str) -> bool:
        """Send voice message to Discord channel"""
        try:
            if not self.client or not self.client.is_ready():
                return False
            
            channel = self.client.get_channel(int(channel_id))
            if channel:
                with open(voice_file, 'rb') as f:
                    voice_file_obj = discord.File(f)
                    await channel.send(file=voice_file_obj)
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error sending Discord voice: {e}")
            return False
    
    async def create_slash_command(self, name: str, description: str, options: List[Dict] = None):
        """Create slash command for Discord"""
        try:
            if not self.client or not self.client.is_ready():
                return False
            
            # This would require discord.py 2.0+ with slash commands
            # Implementation depends on specific discord.py version
            self.logger.info(f"Slash command '{name}' would be created here")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating slash command: {e}")
            return False
    
    async def _process_message(self, message):
        """Process incoming Discord messages"""
        try:
            # Add message processing logic here
            content = message.content.lower()
            
            if content.startswith('!jarvis'):
                response = "Hello! I'm JARVIS, your AI assistant. How can I help you?"
                await message.channel.send(response)
            
        except Exception as e:
            self.logger.error(f"Error processing Discord message: {e}")
    
    def get_profile_info(self) -> Dict[str, Any]:
        """Get profile information"""
        return {
            "platform": "discord",
            "api_version": "10",
            "is_active": self.is_active,
            "capabilities": self.capabilities,
            "guild_id": self.guild_id,
            "bot_user": str(self.client.user) if self.client and self.client.user else None
        }
