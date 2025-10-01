"""
Telegram Bot Profile for RAVANA

Updated with latest Telegram Bot API v6.8
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

class TelegramProfile:
    """Telegram bot profile with updated API configuration"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.bot_token = None
        self.api_url = "https://api.telegram.org/bot"
        self.webhook_url = None
        self.allowed_users = []
        self.commands = {}
        self.is_active = False
        
        # Updated API configuration for v6.8
        self.api_config = {
            "timeout": 30,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
            "disable_notification": False,
            "protect_content": False
        }
        
        # Bot capabilities
        self.capabilities = {
            "text_messages": True,
            "voice_messages": True,
            "file_sharing": True,
            "inline_queries": True,
            "callback_queries": True,
            "webhook_support": True
        }
    
    async def initialize(self, bot_token: str, webhook_url: Optional[str] = None):
        """Initialize Telegram bot with token and webhook"""
        try:
            self.bot_token = bot_token
            self.webhook_url = webhook_url
            
            # Test bot token
            if await self._test_bot_token():
                self.is_active = True
                self.logger.info("Telegram bot initialized successfully")
                return True
            else:
                self.logger.error("Invalid Telegram bot token")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Telegram bot: {e}")
            return False
    
    async def _test_bot_token(self) -> bool:
        """Test if bot token is valid"""
        try:
            import aiohttp
            
            url = f"{self.api_url}{self.bot_token}/getMe"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("ok"):
                            bot_info = data.get("result", {})
                            self.logger.info(f"Telegram bot connected: @{bot_info.get('username', 'unknown')}")
                            return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error testing bot token: {e}")
            return False
    
    async def send_message(self, chat_id: str, text: str, **kwargs) -> bool:
        """Send message to Telegram chat"""
        try:
            import aiohttp
            
            url = f"{self.api_url}{self.bot_token}/sendMessage"
            data = {
                "chat_id": chat_id,
                "text": text,
                **self.api_config,
                **kwargs
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("ok", False)
            return False
            
        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {e}")
            return False
    
    async def send_voice(self, chat_id: str, voice_file: str, **kwargs) -> bool:
        """Send voice message to Telegram chat"""
        try:
            import aiohttp
            
            url = f"{self.api_url}{self.bot_token}/sendVoice"
            data = {
                "chat_id": chat_id,
                "voice": voice_file,
                **self.api_config,
                **kwargs
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("ok", False)
            return False
            
        except Exception as e:
            self.logger.error(f"Error sending Telegram voice: {e}")
            return False
    
    async def set_webhook(self, webhook_url: str) -> bool:
        """Set webhook for Telegram bot"""
        try:
            import aiohttp
            
            url = f"{self.api_url}{self.bot_token}/setWebhook"
            data = {
                "url": webhook_url,
                "allowed_updates": ["message", "callback_query", "inline_query"]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get("ok"):
                            self.webhook_url = webhook_url
                            self.logger.info(f"Webhook set to: {webhook_url}")
                            return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error setting webhook: {e}")
            return False
    
    def get_profile_info(self) -> Dict[str, Any]:
        """Get profile information"""
        return {
            "platform": "telegram",
            "api_version": "6.8",
            "is_active": self.is_active,
            "capabilities": self.capabilities,
            "webhook_url": self.webhook_url,
            "commands_count": len(self.commands)
        }
