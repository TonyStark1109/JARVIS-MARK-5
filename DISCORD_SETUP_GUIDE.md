# JARVIS Discord Bot Setup Guide

## 🤖 Discord Bot Integration

Your JARVIS Discord bot is ready! Here's how to set it up and invite it to your server.

## Step 1: Get Your Discord Bot Token

### Option A: Use Existing Token (Recommended)
Your bot token is already configured in the config file.

### Option B: Create New Bot (If needed)
1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Click "New Application"
3. Give it a name (e.g., "JARVIS AI Assistant")
4. Go to "Bot" section
5. Click "Add Bot"
6. Copy the bot token
7. Replace the token in `RAVANA/modules/conversational_ai/config.json`

## Step 2: Configure Bot Permissions

### Required Bot Permissions:
- ✅ Send Messages
- ✅ Read Message History
- ✅ Read Messages/View Channels
- ✅ Use Slash Commands (optional)
- ✅ Embed Links
- ✅ Attach Files (optional)

### Required Intents:
- ✅ Message Content Intent
- ✅ Server Members Intent (optional)
- ✅ Presence Intent (optional)

## Step 3: Generate Invite Link

### Method 1: Using Discord Developer Portal
1. Go to your bot's OAuth2 → URL Generator
2. Select scopes:
   - ✅ `bot`
   - ✅ `applications.commands` (optional)
3. Select bot permissions:
   - ✅ Send Messages
   - ✅ Read Message History
   - ✅ Read Messages/View Channels
   - ✅ Embed Links
4. Copy the generated URL
5. Open the URL in your browser
6. Select your server and authorize

### Method 2: Manual Invite Link
Replace `YOUR_BOT_CLIENT_ID` with your bot's client ID:
```
https://discord.com/api/oauth2/authorize?client_id=YOUR_BOT_CLIENT_ID&permissions=2048&scope=bot
```

## Step 4: Start the Bot

### Start JARVIS Discord Bot:
```bash
cd D:\JARVIS-MARK5
D:\JARVIS-MARK5\jarvis_env\Scripts\python.exe jarvis_discord_bot.py
```

### Expected Output:
```
2025-10-01 10:30:00 - JARVIS Discord Bot - INFO - Starting JARVIS Discord Bot...
2025-10-01 10:30:01 - JARVIS Discord Bot - INFO - JARVIS Discord Bot logged in as JARVIS#1234
2025-10-01 10:30:01 - JARVIS Discord Bot - INFO - Bot ID: 123456789012345678
2025-10-01 10:30:01 - JARVIS Discord Bot - INFO - Connected to 1 servers
2025-10-01 10:30:01 - JARVIS Discord Bot - INFO - JARVIS Discord Bot is ready!
```

## Step 5: Test the Bot

### Commands to Test:
1. **!help** - Show help information
2. **!status** - Show JARVIS system status
3. **!time** - Get current time and date
4. **@JARVIS Hello!** - Chat with JARVIS
5. **Send a DM** to the bot

### Expected Responses:
- **!help**: Shows embed with available commands
- **!status**: Shows system status with green checkmarks
- **!time**: Shows current time and date
- **@JARVIS Hello!**: "Hello! I'm JARVIS, your AI assistant. How can I help you today?"
- **DM**: Responds to direct messages

## Step 6: Bot Features

### 🎯 Available Commands:
- `!help` - Show help information
- `!status` - Show JARVIS system status  
- `!time` - Get current time and date

### 💬 Chat Features:
- Responds to mentions (@JARVIS)
- Responds to direct messages
- Natural language processing
- Context-aware responses

### 🔧 Bot Capabilities:
- ✅ Voice Recognition (via JARVIS main system)
- ✅ RAVANA AGI integration
- ✅ Snake Agents background processing
- ✅ Web interface access
- ✅ GPU acceleration support

## Troubleshooting

### Bot Not Responding:
1. Check if bot is online (green dot)
2. Verify bot has proper permissions
3. Check console for error messages
4. Ensure Message Content Intent is enabled

### Permission Errors:
1. Re-invite bot with correct permissions
2. Check server role hierarchy
3. Verify bot role has necessary permissions

### Connection Issues:
1. Check internet connection
2. Verify bot token is correct
3. Check firewall settings
4. Restart the bot application

## Advanced Configuration

### Customize Bot Settings:
Edit `RAVANA/modules/conversational_ai/config.json`:
```json
{
  "discord_token": "YOUR_BOT_TOKEN",
  "platforms": {
    "discord": {
      "enabled": true,
      "command_prefix": "!",
      "status": "online"
    }
  }
}
```

### Add Custom Commands:
Edit `jarvis_discord_bot.py` and add new commands in the bot setup section.

## Support

If you need help:
1. Check the bot logs for errors
2. Verify all permissions are set correctly
3. Ensure the bot token is valid
4. Check that all required packages are installed

## 🎉 Success!

Once everything is set up, your JARVIS Discord bot will be fully functional and ready to assist you and your server members!

**Bot Status**: Ready to deploy
**Integration**: Complete
**Features**: Full JARVIS AI capabilities

