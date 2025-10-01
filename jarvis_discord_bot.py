#!/usr/bin/env python3
"""JARVIS Discord Bot Integration"""

import asyncio
import logging
import json
import sys
import os
from pathlib import Path
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JARVISDiscordBot:
    """JARVIS Discord Bot"""
    
    def __init__(self, token: str):
        self.token = token
        self.connected = False
        
    async def jarvis_response(self, message: str) -> str:
        """Generate JARVIS response"""
        message_lower = message.lower()
        
        if "hello" in message_lower or "hi" in message_lower:
            return "Hello! I'm JARVIS, your AI assistant. How can I help you today?"
        elif "time" in message_lower or "date" in message_lower:
            now = datetime.datetime.now()
            return f"The current time is {now.strftime('%H:%M:%S')} on {now.strftime('%Y-%m-%d')}"
        elif "weather" in message_lower:
            return "I can check the weather for you, but I need to be connected to a weather service first."
        elif "task" in message_lower:
            return "I can help you with tasks! What would you like me to do?"
        elif "help" in message_lower:
            return """I'm JARVIS, your AI assistant! I can help you with:
- General conversation
- Time and date information
- Task management
- Weather information (when available)
- And much more!

Just ask me anything!"""
        elif "status" in message_lower:
            return """JARVIS System Status:
- Voice Recognition: Active
- RAVANA AGI: Running
- Snake Agents: Background processing
- Discord Bot: Connected
- Web Interface: Available at localhost:44450"""
        else:
            return f"I understand you said: '{message}'. How can I assist you with that?"
    
    async def start(self):
        """Start the Discord bot"""
        try:
            import discord
            from discord.ext import commands
            
            logger.info("Starting JARVIS Discord Bot...")
            
            # Create bot with intents
            intents = discord.Intents.default()
            intents.message_content = True
            intents.guilds = True
            intents.dm_messages = True
            
            self.bot = commands.Bot(command_prefix='!', intents=intents, help_command=None)
            
            @self.bot.event
            async def on_ready():
                logger.info(f'JARVIS Discord Bot logged in as {self.bot.user}')
                logger.info(f'Bot ID: {self.bot.user.id}')
                logger.info(f'Connected to {len(self.bot.guilds)} servers')
                
                # Set bot status
                await self.bot.change_presence(
                    status=discord.Status.online,
                    activity=discord.Activity(
                        type=discord.ActivityType.listening, 
                        name="your commands"
                    )
                )
                self.connected = True
                logger.info("JARVIS Discord Bot is ready!")
            
            @self.bot.event
            async def on_message(message):
                # Ignore messages from the bot itself
                if message.author == self.bot.user:
                    return
                
                # Log all messages for debugging
                logger.info(f"Message received: {message.content}")
                logger.info(f"Author: {message.author}")
                logger.info(f"Channel: {message.channel}")
                logger.info(f"Guild: {message.guild}")
                
                # Check if bot is mentioned
                bot_mentioned = self.bot.user in message.mentions
                logger.info(f"Bot mentioned: {bot_mentioned}")
                
                # Check for user ID mention
                user_id_mention = f"<@{self.bot.user.id}>" in message.content
                logger.info(f"User ID mention: {user_id_mention}")
                
                # Check if it's a DM
                is_dm = isinstance(message.channel, discord.DMChannel)
                logger.info(f"Is DM: {is_dm}")
                
                # Process commands first
                await self.bot.process_commands(message)
                
                # Respond to mentions or DMs
                should_respond = (
                    bot_mentioned or
                    is_dm or
                    user_id_mention
                )
                
                logger.info(f"Should respond: {should_respond}")
                
                if should_respond:
                    try:
                        response = await self.jarvis_response(message.content)
                        logger.info(f"Sending response: {response}")
                        await message.channel.send(response)
                    except Exception as e:
                        logger.error(f"Error responding to message: {e}")
                        await message.channel.send("Sorry, I encountered an error processing your message.")
            
            @self.bot.command(name='help')
            async def help_command(ctx):
                """Show help information"""
                embed = discord.Embed(
                    title="JARVIS Commands",
                    description="Here are the available commands:",
                    color=0x00ff00
                )
                embed.add_field(
                    name="!help", 
                    value="Show this help message", 
                    inline=False
                )
                embed.add_field(
                    name="!status", 
                    value="Show JARVIS system status", 
                    inline=False
                )
                embed.add_field(
                    name="!time", 
                    value="Get current time and date", 
                    inline=False
                )
                embed.add_field(
                    name="Mention me", 
                    value="Just mention @JARVIS to chat!", 
                    inline=False
                )
                embed.add_field(
                    name="Direct Message", 
                    value="Send me a DM to chat privately", 
                    inline=False
                )
                await ctx.send(embed=embed)
            
            @self.bot.command(name='status')
            async def status_command(ctx):
                """Show JARVIS system status"""
                status_embed = discord.Embed(
                    title="JARVIS System Status",
                    color=0x00ff00
                )
                status_embed.add_field(name="Voice Recognition", value="✅ Active", inline=True)
                status_embed.add_field(name="RAVANA AGI", value="✅ Running", inline=True)
                status_embed.add_field(name="Snake Agents", value="✅ Background processing", inline=True)
                status_embed.add_field(name="Discord Bot", value="✅ Connected", inline=True)
                status_embed.add_field(name="Web Interface", value="✅ localhost:44450", inline=True)
                status_embed.add_field(name="GPU Acceleration", value="✅ CUDA Enabled", inline=True)
                await ctx.send(embed=status_embed)
            
            @self.bot.command(name='time')
            async def time_command(ctx):
                """Get current time and date"""
                now = datetime.datetime.now()
                time_embed = discord.Embed(
                    title="Current Time",
                    color=0x0099ff
                )
                time_embed.add_field(name="Time", value=now.strftime('%H:%M:%S'), inline=True)
                time_embed.add_field(name="Date", value=now.strftime('%Y-%m-%d'), inline=True)
                time_embed.add_field(name="Day", value=now.strftime('%A'), inline=True)
                await ctx.send(embed=time_embed)
            
            # Start the bot
            await self.bot.start(self.token)
            
        except Exception as e:
            logger.error(f"Error starting Discord bot: {e}")
            import traceback
            traceback.print_exc()

async def main():
    """Main function"""
    try:
        # Load configuration
        config_path = Path("RAVANA/modules/conversational_ai/config.json")
        if not config_path.exists():
            logger.error("Configuration file not found!")
            return
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Get Discord token
        discord_token = config.get('discord_token')
        if not discord_token:
            logger.error("Discord token not found in configuration!")
            return
        
        logger.info("Starting JARVIS Discord Bot...")
        bot = JARVISDiscordBot(discord_token)
        await bot.start()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

