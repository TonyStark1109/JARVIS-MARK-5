#!/usr/bin/env python3
"""
JARVIS Mark 5 - Desktop Automation and Media Control Module
Provides comprehensive desktop automation, camera control, media management, and natural language interface
"""

import os
import sys
import time
import json
import logging
import subprocess
import threading
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import webbrowser
import platform

# Desktop automation libraries
try:
    import pyautogui
    import cv2
    import numpy as np
    from PIL import Image, ImageGrab
    import pyaudio
    import wave
    AUTOMATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Desktop automation libraries not available: {e}")
    AUTOMATION_AVAILABLE = False

# Media control libraries
try:
    import yt_dlp
    import requests
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.chrome.options import Options
    MEDIA_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Media control libraries not available: {e}")
    MEDIA_AVAILABLE = False

logger = logging.getLogger(__name__)

class DesktopAutomation:
    """Comprehensive desktop automation and media control for JARVIS"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.is_recording = False
        self.recording_thread = None
        self.video_writer = None
        self.audio_recorder = None
        self.screen_width, self.screen_height = pyautogui.size() if AUTOMATION_AVAILABLE else (1920, 1080)
        
        # Initialize pyautogui settings
        if AUTOMATION_AVAILABLE:
            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = 0.1
        
        self.logger.info("Desktop Automation initialized")
    
    # ==================== CAMERA CONTROL ====================
    
    def open_camera(self) -> Dict[str, Any]:
        """Open camera application"""
        try:
            if platform.system() == "Windows":
                subprocess.run(["start", "microsoft.windows.camera:"], shell=True)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", "-a", "Photo Booth"])
            elif platform.system() == "Linux":
                subprocess.run(["cheese"], check=False)
            
            return {
                "success": True,
                "message": "Camera application opened",
                "action": "open_camera"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to open camera"
            }
    
    def snap_picture(self, save_path: str = None) -> Dict[str, Any]:
        """Take a picture using webcam"""
        try:
            if not AUTOMATION_AVAILABLE:
                return {"success": False, "error": "OpenCV not available", "message": "Camera functionality requires OpenCV"}
            
            # Initialize camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return {"success": False, "error": "Camera not accessible", "message": "Could not access camera"}
            
            # Capture frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return {"success": False, "error": "Failed to capture frame", "message": "Could not capture image"}
            
            # Save image
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"screenshots/camera_{timestamp}.jpg"
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, frame)
            
            return {
                "success": True,
                "message": f"Picture saved to {save_path}",
                "file_path": save_path,
                "action": "snap_picture"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to take picture"
            }
    
    def start_video_recording(self, duration: int = 30, save_path: str = None) -> Dict[str, Any]:
        """Start video recording using webcam"""
        try:
            if not AUTOMATION_AVAILABLE:
                return {"success": False, "error": "OpenCV not available", "message": "Video recording requires OpenCV"}
            
            if self.is_recording:
                return {"success": False, "error": "Already recording", "message": "Video recording is already in progress"}
            
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"recordings/video_{timestamp}.mp4"
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Start recording in separate thread
            self.recording_thread = threading.Thread(
                target=self._record_video_thread,
                args=(duration, save_path)
            )
            self.recording_thread.start()
            
            return {
                "success": True,
                "message": f"Video recording started for {duration} seconds",
                "file_path": save_path,
                "action": "start_video_recording"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to start video recording"
            }
    
    def _record_video_thread(self, duration: int, save_path: str):
        """Thread function for video recording"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.logger.error("Could not access camera for recording")
                return
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_path, fourcc, 20.0, (640, 480))
            
            self.is_recording = True
            start_time = time.time()
            
            while self.is_recording and (time.time() - start_time) < duration:
                ret, frame = cap.read()
                if ret:
                    out.write(frame)
                time.sleep(0.05)  # 20 FPS
            
            # Cleanup
            cap.release()
            out.release()
            self.is_recording = False
            
            self.logger.info(f"Video recording completed: {save_path}")
        except Exception as e:
            self.logger.error(f"Video recording error: {e}")
            self.is_recording = False
    
    def stop_video_recording(self) -> Dict[str, Any]:
        """Stop video recording"""
        try:
            if not self.is_recording:
                return {"success": False, "error": "Not recording", "message": "No video recording in progress"}
            
            self.is_recording = False
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=2.0)  # Wait up to 2 seconds
            
            return {
                "success": True,
                "message": "Video recording stopped",
                "action": "stop_video_recording"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to stop video recording"
            }
    
    # ==================== SCREENSHOT & SCREEN RECORDING ====================
    
    def take_screenshot(self, save_path: str = None) -> Dict[str, Any]:
        """Take a screenshot of the entire screen"""
        try:
            if not AUTOMATION_AVAILABLE:
                return {"success": False, "error": "PyAutoGUI not available", "message": "Screenshot functionality requires PyAutoGUI"}
            
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"screenshots/screenshot_{timestamp}.png"
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Take screenshot
            screenshot = pyautogui.screenshot()
            screenshot.save(save_path)
            
            return {
                "success": True,
                "message": f"Screenshot saved to {save_path}",
                "file_path": save_path,
                "action": "take_screenshot"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to take screenshot"
            }
    
    def start_screen_recording(self, duration: int = 30, save_path: str = None) -> Dict[str, Any]:
        """Start screen recording"""
        try:
            if not AUTOMATION_AVAILABLE:
                return {"success": False, "error": "OpenCV not available", "message": "Screen recording requires OpenCV"}
            
            if self.is_recording:
                return {"success": False, "error": "Already recording", "message": "Screen recording is already in progress"}
            
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"recordings/screen_{timestamp}.mp4"
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Start screen recording in separate thread
            self.recording_thread = threading.Thread(
                target=self._record_screen_thread,
                args=(duration, save_path)
            )
            self.recording_thread.start()
            
            return {
                "success": True,
                "message": f"Screen recording started for {duration} seconds",
                "file_path": save_path,
                "action": "start_screen_recording"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to start screen recording"
            }
    
    def _record_screen_thread(self, duration: int, save_path: str):
        """Thread function for screen recording"""
        try:
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_path, fourcc, 20.0, (self.screen_width, self.screen_height))
            
            self.is_recording = True
            start_time = time.time()
            
            while self.is_recording and (time.time() - start_time) < duration:
                # Capture screen
                screenshot = pyautogui.screenshot()
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                out.write(frame)
                time.sleep(0.05)  # 20 FPS
            
            # Cleanup
            out.release()
            self.is_recording = False
            
            self.logger.info(f"Screen recording completed: {save_path}")
        except Exception as e:
            self.logger.error(f"Screen recording error: {e}")
            self.is_recording = False
    
    # ==================== APPLICATION CONTROL ====================
    
    def open_application(self, app_name: str) -> Dict[str, Any]:
        """Open an application by name"""
        try:
            app_commands = {
                # Windows applications
                "notepad": ["notepad"],
                "calculator": ["calc"],
                "paint": ["mspaint"],
                "word": ["winword"],
                "excel": ["excel"],
                "powerpoint": ["powerpnt"],
                "chrome": ["chrome"],
                "firefox": ["firefox"],
                "edge": ["msedge"],
                "vscode": ["code"],
                "cmd": ["cmd"],
                "powershell": ["powershell"],
                "task manager": ["taskmgr"],
                "file explorer": ["explorer"],
                "control panel": ["control"],
                "settings": ["ms-settings:"],
                
                # macOS applications
                "textedit": ["open", "-a", "TextEdit"],
                "preview": ["open", "-a", "Preview"],
                "safari": ["open", "-a", "Safari"],
                "finder": ["open", "-a", "Finder"],
                "terminal": ["open", "-a", "Terminal"],
                "activity monitor": ["open", "-a", "Activity Monitor"],
                
                # Linux applications
                "gedit": ["gedit"],
                "libreoffice": ["libreoffice"],
                "firefox": ["firefox"],
                "chrome": ["google-chrome"],
                "terminal": ["gnome-terminal"],
                "file manager": ["nautilus"]
            }
            
            if app_name.lower() in app_commands:
                command = app_commands[app_name.lower()]
                subprocess.Popen(command)
                
                return {
                    "success": True,
                    "message": f"Opened {app_name}",
                    "app": app_name,
                    "action": "open_application"
                }
            else:
                # Try to open as a general command
                subprocess.Popen([app_name])
                return {
                    "success": True,
                    "message": f"Attempted to open {app_name}",
                    "app": app_name,
                    "action": "open_application"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to open {app_name}"
            }
    
    def close_application(self, app_name: str) -> Dict[str, Any]:
        """Close an application by name"""
        try:
            if platform.system() == "Windows":
                subprocess.run(["taskkill", "/f", "/im", f"{app_name}.exe"], check=False)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["pkill", "-f", app_name], check=False)
            elif platform.system() == "Linux":
                subprocess.run(["pkill", "-f", app_name], check=False)
            
            return {
                "success": True,
                "message": f"Closed {app_name}",
                "app": app_name,
                "action": "close_application"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to close {app_name}"
            }
    
    # ==================== WEB AUTOMATION ====================
    
    def search_youtube(self, query: str) -> Dict[str, Any]:
        """Search for videos on YouTube"""
        try:
            if not MEDIA_AVAILABLE:
                # Fallback to opening browser
                url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
                webbrowser.open(url)
                return {
                    "success": True,
                    "message": f"Opened YouTube search for '{query}'",
                    "query": query,
                    "url": url,
                    "action": "search_youtube"
                }
            
            # Use Selenium for more control
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}")
            
            # Get first few video results
            videos = []
            video_elements = driver.find_elements(By.CSS_SELECTOR, "a#video-title")
            
            for i, video in enumerate(video_elements[:5]):  # Get first 5 results
                title = video.get_attribute("title")
                link = video.get_attribute("href")
                videos.append({"title": title, "url": link})
            
            driver.quit()
            
            return {
                "success": True,
                "message": f"Found {len(videos)} YouTube videos for '{query}'",
                "query": query,
                "videos": videos,
                "action": "search_youtube"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to search YouTube for '{query}'"
            }
    
    def play_youtube_video(self, video_url: str) -> Dict[str, Any]:
        """Play a specific YouTube video"""
        try:
            webbrowser.open(video_url)
            return {
                "success": True,
                "message": f"Playing YouTube video: {video_url}",
                "url": video_url,
                "action": "play_youtube_video"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to play video: {video_url}"
            }
    
    def search_google(self, query: str) -> Dict[str, Any]:
        """Search on Google"""
        try:
            url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            webbrowser.open(url)
            return {
                "success": True,
                "message": f"Opened Google search for '{query}'",
                "query": query,
                "url": url,
                "action": "search_google"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to search Google for '{query}'"
            }
    
    def download_youtube_video(self, video_url: str, output_path: str = None) -> Dict[str, Any]:
        """Download a YouTube video"""
        try:
            if not MEDIA_AVAILABLE:
                return {"success": False, "error": "yt-dlp not available", "message": "Video download requires yt-dlp"}
            
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"downloads/video_{timestamp}.%(ext)s"
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Configure yt-dlp options
            ydl_opts = {
                'outtmpl': output_path,
                'format': 'best[height<=720]',  # Download best quality up to 720p
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=True)
                filename = ydl.prepare_filename(info)
            
            return {
                "success": True,
                "message": f"Downloaded video to {filename}",
                "file_path": filename,
                "title": info.get('title', 'Unknown'),
                "action": "download_youtube_video"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to download video: {video_url}"
            }
    
    # ==================== DESKTOP AUTOMATION ====================
    
    def click_position(self, x: int, y: int) -> Dict[str, Any]:
        """Click at specific screen coordinates"""
        try:
            if not AUTOMATION_AVAILABLE:
                return {"success": False, "error": "PyAutoGUI not available", "message": "Click functionality requires PyAutoGUI"}
            
            pyautogui.click(x, y)
            return {
                "success": True,
                "message": f"Clicked at position ({x}, {y})",
                "x": x,
                "y": y,
                "action": "click_position"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to click at ({x}, {y})"
            }
    
    def type_text(self, text: str) -> Dict[str, Any]:
        """Type text at current cursor position"""
        try:
            if not AUTOMATION_AVAILABLE:
                return {"success": False, "error": "PyAutoGUI not available", "message": "Type functionality requires PyAutoGUI"}
            
            pyautogui.typewrite(text)
            return {
                "success": True,
                "message": f"Typed: {text}",
                "text": text,
                "action": "type_text"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to type text: {text}"
            }
    
    def press_key(self, key: str) -> Dict[str, Any]:
        """Press a specific key"""
        try:
            if not AUTOMATION_AVAILABLE:
                return {"success": False, "error": "PyAutoGUI not available", "message": "Key press functionality requires PyAutoGUI"}
            
            pyautogui.press(key)
            return {
                "success": True,
                "message": f"Pressed key: {key}",
                "key": key,
                "action": "press_key"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to press key: {key}"
            }
    
    def hotkey_combination(self, *keys) -> Dict[str, Any]:
        """Press a combination of keys"""
        try:
            if not AUTOMATION_AVAILABLE:
                return {"success": False, "error": "PyAutoGUI not available", "message": "Hotkey functionality requires PyAutoGUI"}
            
            pyautogui.hotkey(*keys)
            return {
                "success": True,
                "message": f"Pressed hotkey combination: {'+'.join(keys)}",
                "keys": list(keys),
                "action": "hotkey_combination"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to press hotkey combination: {'+'.join(keys)}"
            }
    
    # ==================== NATURAL LANGUAGE INTERFACE ====================
    
    def process_natural_command(self, command: str) -> Dict[str, Any]:
        """Process natural language commands for desktop automation"""
        command = command.lower().strip()
        
        # Camera commands
        if any(word in command for word in ["take picture", "snap photo", "capture photo", "take a pic"]):
            return self.snap_picture()
        
        elif any(word in command for word in ["open camera", "start camera", "launch camera"]):
            return self.open_camera()
        
        elif any(word in command for word in ["record video", "start recording", "take video"]):
            # Extract duration if mentioned
            duration = 30
            if "for" in command:
                try:
                    duration = int(command.split("for")[1].split()[0])
                except (ValueError, IndexError):
                    pass
            return self.start_video_recording(duration)
        
        elif any(word in command for word in ["stop recording", "stop video"]):
            return self.stop_video_recording()
        
        # Screenshot commands
        elif any(word in command for word in ["screenshot", "take screenshot", "capture screen"]):
            return self.take_screenshot()
        
        elif any(word in command for word in ["record screen", "screen recording", "record my screen"]):
            duration = 30
            if "for" in command:
                try:
                    duration = int(command.split("for")[1].split()[0])
                except (ValueError, IndexError):
                    pass
            return self.start_screen_recording(duration)
        
        # Application commands
        elif any(word in command for word in ["open", "launch", "start"]):
            # Extract application name
            app_name = command.replace("open", "").replace("launch", "").replace("start", "").strip()
            if app_name:
                return self.open_application(app_name)
            else:
                return {"success": False, "error": "No application specified", "message": "Please specify which application to open"}
        
        elif any(word in command for word in ["close", "quit", "exit"]):
            # Extract application name
            app_name = command.replace("close", "").replace("quit", "").replace("exit", "").strip()
            if app_name:
                return self.close_application(app_name)
            else:
                return {"success": False, "error": "No application specified", "message": "Please specify which application to close"}
        
        # YouTube commands
        elif any(word in command for word in ["youtube", "search youtube", "find video"]):
            # Extract search query
            query = command.replace("youtube", "").replace("search", "").replace("find", "").replace("video", "").strip()
            if query:
                return self.search_youtube(query)
            else:
                return {"success": False, "error": "No search query", "message": "Please specify what to search for on YouTube"}
        
        elif any(word in command for word in ["play video", "watch video", "play youtube"]):
            # Extract video URL or search query
            if "http" in command:
                url = command.split("http")[1].split()[0]
                if not url.startswith("http"):
                    url = "http" + url
                return self.play_youtube_video(url)
            else:
                query = command.replace("play", "").replace("video", "").replace("youtube", "").strip()
                return self.search_youtube(query)
        
        elif any(word in command for word in ["download video", "save video"]):
            if "http" in command:
                url = command.split("http")[1].split()[0]
                if not url.startswith("http"):
                    url = "http" + url
                return self.download_youtube_video(url)
            else:
                return {"success": False, "error": "No video URL", "message": "Please provide a YouTube video URL to download"}
        
        # Google search commands
        elif any(word in command for word in ["google", "search google", "search for"]):
            query = command.replace("google", "").replace("search", "").replace("for", "").strip()
            if query:
                return self.search_google(query)
            else:
                return {"success": False, "error": "No search query", "message": "Please specify what to search for on Google"}
        
        # Desktop automation commands
        elif any(word in command for word in ["click", "click at"]):
            # Try to extract coordinates
            try:
                coords = [int(x) for x in command.split() if x.isdigit()]
                if len(coords) >= 2:
                    return self.click_position(coords[0], coords[1])
                else:
                    return {"success": False, "error": "No coordinates", "message": "Please specify x and y coordinates for clicking"}
            except ValueError:
                return {"success": False, "error": "Invalid coordinates", "message": "Please provide valid x and y coordinates"}
        
        elif any(word in command for word in ["type", "write", "enter text"]):
            # Extract text to type
            text = command.replace("type", "").replace("write", "").replace("enter", "").replace("text", "").strip()
            if text:
                return self.type_text(text)
            else:
                return {"success": False, "error": "No text specified", "message": "Please specify what text to type"}
        
        elif any(word in command for word in ["press", "hit", "key"]):
            # Extract key to press
            key = command.replace("press", "").replace("hit", "").replace("key", "").strip()
            if key:
                return self.press_key(key)
            else:
                return {"success": False, "error": "No key specified", "message": "Please specify which key to press"}
        
        # Help command
        elif any(word in command for word in ["help", "what can you do", "commands"]):
            return self.get_help()
        
        else:
            return {
                "success": False,
                "error": "Unknown command",
                "message": f"I don't understand the command: '{command}'. Try saying 'help' for available commands.",
                "suggestions": [
                    "take picture",
                    "open camera",
                    "record video for 30 seconds",
                    "take screenshot",
                    "record screen for 60 seconds",
                    "open notepad",
                    "search youtube for funny cats",
                    "search google for python tutorial",
                    "click at 500 300",
                    "type hello world",
                    "press enter"
                ]
            }
    
    def get_help(self) -> Dict[str, Any]:
        """Get help information for desktop automation commands"""
        help_text = """
🎥 JARVIS DESKTOP AUTOMATION & MEDIA CONTROL

📸 CAMERA CONTROL:
• "take picture" - Snap a photo with webcam
• "open camera" - Open camera application
• "record video for 30 seconds" - Record video with webcam
• "stop recording" - Stop video recording

🖥️ SCREEN CONTROL:
• "take screenshot" - Capture entire screen
• "record screen for 60 seconds" - Record screen activity
• "stop recording" - Stop screen recording

💻 APPLICATION CONTROL:
• "open notepad" - Open Notepad
• "open chrome" - Open Chrome browser
• "open calculator" - Open Calculator
• "close notepad" - Close Notepad
• "close chrome" - Close Chrome

🎵 MEDIA CONTROL:
• "search youtube for funny cats" - Search YouTube
• "play video [URL]" - Play specific YouTube video
• "download video [URL]" - Download YouTube video
• "search google for python tutorial" - Google search

🖱️ DESKTOP AUTOMATION:
• "click at 500 300" - Click at specific coordinates
• "type hello world" - Type text at cursor
• "press enter" - Press Enter key
• "press ctrl+c" - Press Ctrl+C

📋 EXAMPLES:
• "Jarvis, take a picture"
• "J, open chrome and search for AI news"
• "Buddy, record my screen for 2 minutes"
• "Open notepad and type my shopping list"

⚠️ Note: Some features require additional libraries (OpenCV, PyAutoGUI, Selenium)
        """
        
        return {
            "success": True,
            "help": help_text,
            "action": "help",
            "message": "Here are the available desktop automation commands:"
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of desktop automation"""
        return {
            "automation_available": AUTOMATION_AVAILABLE,
            "media_available": MEDIA_AVAILABLE,
            "is_recording": self.is_recording,
            "screen_resolution": f"{self.screen_width}x{self.screen_height}",
            "platform": platform.system(),
            "capabilities": [
                "camera_control",
                "screenshot",
                "screen_recording",
                "application_control",
                "web_automation",
                "desktop_automation",
                "media_download"
            ]
        }

# Example usage and testing
async def main():
    """Test desktop automation functionality"""
    print("🤖 JARVIS Desktop Automation Test")
    print("=" * 50)
    
    automation = DesktopAutomation()
    
    # Test commands
    test_commands = [
        "take screenshot",
        "open notepad",
        "search youtube for funny cats",
        "search google for python tutorial",
        "help"
    ]
    
    for cmd in test_commands:
        print(f"\n💬 Command: '{cmd}'")
        result = automation.process_natural_command(cmd)
        
        if result["success"]:
            print(f"✅ {result['message']}")
        else:
            print(f"❌ Error: {result['error']}")
    
    print(f"\n📊 Status: {automation.get_status()}")

if __name__ == "__main__":
    asyncio.run(main())
