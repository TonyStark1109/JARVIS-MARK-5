
"""JARVIS Mark 5 - Advanced AI Assistant"""

from backend.modules.llms import pure_llama3 as ChatGpt
import pyautogui
from backend.modules.OF.obj_detect import *
from backend.modules.filter import filter_python as filter
from backend.modules.basic.listenpy import Listen

cache = {}
CACHE_EXPIRATION_TIME = 3600

def execute_code(self, *args, **kwargs):  # pylint: disable=unused-argument  # pylint: disable=unused-argument
    """Docstring for function"""
    try:
        exec(code)
        return True, None
    except (ValueError, TypeError, AttributeError, ImportError) as e:
        return False, e

def cached_function(self, *args, **kwargs):  # pylint: disable=unused-argument  # pylint: disable=unused-argument
    """Docstring for function"""
    # Check if the result is cached and not expired
    if key in cache and cache[key]['expiration'] > time.time():
        return cache[key]['result']
    # Call the function and cache the result
    result = func(*args, **kwargs)
    cache[key] = {'result': result, 'expiration': time.time() + CACHE_EXPIRATION_TIME}
    return result

def execute_code_with_cache(self, *args, **kwargs):  # pylint: disable=unused-argument  # pylint: disable=unused-argument
    """Docstring for function"""
    # Cache the execution result based on code
    return cached_function(code, execute_code, code)

def capture_and_send_image(self, *args, **kwargs):  # pylint: disable=unused-argument  # pylint: disable=unused-argument
    """Docstring for function"""
    # URL of the FastAPI server endpoint
    api_url = "http://your-fastapi-server-url"

    # Capture screenshot
    screenshot = pyautogui.screenshot()

    # Convert screenshot to JPEG format in memory
    buffer = screenshot.tobytes()
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    # Send screenshot to server
    try:
        response = requests.post(f"{api_url}/stream", data={'image': jpg_as_text})
        if response.status_code == 200:
            print("Screenshot sent successfully")
            return response
        else:
            print(f"Failed to send screenshot, status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending screenshot: {e}")

    return None

def ScreenShare(self, *args, **kwargs):  # pylint: disable=unused-argument  # pylint: disable=unused-argument
    """Docstring for function"""
    while True:
        img_discription=capture_and_send_image()
        context_query = f"{q}, if this query needs internet research, respond with 'internet' only, ***Reply like Tony Stark's Jarvis in fewer words. If it's to perform an action on the computer, write complete code in Python, nothing else.***, LIVE SCREENSHARE is ON {img_discription} (to close SCREENSHARE reply only 'stop' and nothing else)"
        rep = cached_function(context_query, ChatGpt, context_query)
        if "STOP" in rep.upper():
            break
        else:
            try:
                code = filter(rep)
                success, error = execute_code_with_cache(code)
                if success:
                    execute_code_with_cache(ChatGpt(f"Output: {success}, respond for this action if it is, or else ask for any another help"))
                else:
                    execute_code_with_cache(ChatGpt(f"{error}"))
            except Exception as E:
                print(f"Falied {E}")

def SCREENSHARE(self, *args, **kwargs):  # pylint: disable=unused-argument  # pylint: disable=unused-argument
    """Docstring for function"""
    resp = Listen()
    ScreenShare(resp)
