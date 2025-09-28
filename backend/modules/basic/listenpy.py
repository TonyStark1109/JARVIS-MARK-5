"""JARVIS Mark 5 - Advanced AI Assistant"""

import speech_recognition as sr
from colorama import Fore, Style

import colorama

class Listen:
    """Wrapper class for speech recognition functionality."""

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 200
        self.recognizer.pause_threshold = 0.5
        self.recognizer.operation_timeout = None

    def listen(self, *args, **kwargs):  # pylint: disable=unused-argument
        """Listen to user input using speech recognition."""
        return listen_function()

def listen_function(self, *args, **kwargs):  # pylint: disable=unused-argument
    """Listen to user input using speech recognition."""
    colorama.init(autoreset=True)

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 200
    recognizer.pause_threshold = 0.5
    recognizer.operation_timeout = None

    with sr.Microphone() as source:
        print(Fore.YELLOW + Style.BRIGHT + "Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)

        try:
            audio_data = recognizer.listen(source, timeout=5)  # Set timeout here
            print(Fore.GREEN + Style.BRIGHT + "Recognizing...")
            text = recognizer.recognize_google(audio_data)

            print(Fore.GREEN + Style.BRIGHT + f"User: {text}")
            return text

        except sr.WaitTimeoutError:
            print(Fore.RED + Style.BRIGHT + "Timeout: No speech detected. Please speak louder or check your microphone.")
            return None

        except sr.UnknownValueError:
            print(Fore.RED + Style.BRIGHT + "Failed to recognize audio. Please try again.")
            return None

        except sr.RequestError as e:
            print(Fore.RED + Style.BRIGHT + f"Speech recognition request failed: {e}. Please check your internet connection.")
            return None

        except TimeoutError:
            print(Fore.RED + Style.BRIGHT + "Operation timed out. Please try again.")
            return None
