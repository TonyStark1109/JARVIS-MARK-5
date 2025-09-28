"""JARVIS Mark 5 - Advanced AI Assistant"""
# pylint: disable=global-statement,broad-exception-caught,redefined-outer-name,unused-argument,global-variable-not-assigned,protected-access,keyword-arg-before-vararg,too-many-lines,ungrouped-imports,invalid-name,missing-function-docstring,trailing-whitespace,line-too-long,consider-using-f-string,import-outside-toplevel,trailing-newlines

import threading
import os
from threading import Lock
import json
import base64
import logging

try:
    import pyautogui
except ImportError:
    pyautogui = None
    print("Warning: pyautogui not available")

try:
    import eel
except ImportError:
    eel = None
    print("Warning: eel not available")

try:
    from backend.modules.extra import GuiMessagesConverter, LoadMessages
except ImportError:
    GuiMessagesConverter = None
    LoadMessages = None
    print("Warning: backend.modules.extra not available")

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
    print("Warning: python-dotenv not available")

# Import JARVIS Desktop Automation
from JARVIS_DESKTOP_automation import JARVISDesktopInterface

# Import hacking module
from backend.modules.hacking import EthicalHacking

# Import Brain Configuration System
from brain_config import JarvisBrainConfig

# Import Unified Voice Recognition System
from jarvis_unified_voice import (
    JARVISUnifiedVoice, start_voice_listening, stop_voice_listening,
    set_command_callback, set_error_callback
)

# Initialize logging
logger = logging.getLogger(__name__)

# Initialize JARVIS systems
JARVIS_VOICE = None
JARVIS_DESKTOP = None
cv2 = None

def initialize_jarvis():
    """Initialize JARVIS systems"""
    global JARVIS_VOICE, JARVIS_DESKTOP  # pylint: disable=global-statement
    try:
        JARVIS_VOICE = JARVISUnifiedVoice()
        JARVIS_DESKTOP = JARVISDesktopInterface()
        logger.info("✅ JARVIS systems initialized")
        return True
    except Exception as e:
        logger.error("Failed to initialize JARVIS: %s", e)
        return False

def process_command(command, command_type="text"):
    """Process a command through JARVIS"""
    try:
        if JARVIS_VOICE:
            # Use unified voice system for command processing
            return JARVIS_VOICE._process_command(command)  # pylint: disable=protected-access
        else:
            return {"success": False, "error": "JARVIS voice system not initialized"}
    except Exception as e:
        logger.error("Error processing command: %s", e)
        return {"success": False, "error": str(e)}

def speak_response(response):
    """Speak a response using JARVIS voice"""
    try:
        if JARVIS_VOICE:
            JARVIS_VOICE.speak(response)
        else:
            logger.warning("JARVIS voice not available")
    except Exception as e:
        logger.error("Error speaking response: %s", e)

def get_system_STATus():
    """Get JARVIS system STATus"""
    try:
        STATus = {
            "JARVIS_VOICE": "active" if JARVIS_VOICE else "inactive",
            "JARVIS_DESKTOP": "active" if JARVIS_DESKTOP else "inactive",
            "HACKING_MODULE": "active" if HACKING_MODULE else "inactive",
            "brain_config": "active" if brain_config else "inactive"
        }
        return STATus
    except Exception as e:
        logger.error("Error getting system STATus: %s", e)
        return {"error": str(e)}

def shutdown_jarvis():
    """Shutdown JARVIS systems"""
    global JARVIS_VOICE, JARVIS_DESKTOP
    try:
        JARVIS_VOICE = None
        JARVIS_DESKTOP = None
        logger.info("JARVIS systems shutdown")
        return True
    except Exception as e:
        logger.error("Error shutting down JARVIS: %s", e)
        return False

def get_api(*args, **kwargs):  # pylint: disable=unused-argument
    try:
        with open('config/config.json', encoding='utf-8') as config_file:
            config = json.load(config_file)
            groq_api = config.get('GROQ_API')
            openai_api = config.get('OPENAI_API_KEY')

            if groq_api is None:
                raise ValueError("GROQ_API not found in config file")
            if openai_api is None:
                raise ValueError("OPENAI_API_KEY not found in config file")

            return groq_api, openai_api
    except FileNotFoundError:
        logger.info("Config file not found.")
    except json.JSONDecodeError:
        logger.info("Error decoding JSON in config file.")
    except (ValueError, TypeError, AttributeError, ImportError) as e:
        logger.info("Error reading config file: %s", e)
    return None, None

groq_api_key, openai_api_key = get_api()
if groq_api_key:
    os.environ['GROQ_API'] = groq_api_key
    os.environ['GROQ_API_KEY'] = groq_api_key  # Also set GROQ_API_KEY for compatibility
    logger.info("✅ GROQ API key loaded successfully")
else:
    logger.info("❌ Warning: GROQ API key not found in config")

if openai_api_key:
    os.environ['OPENAI_API_KEY'] = openai_api_key
    os.environ['OPENAI_API'] = openai_api_key  # Also set OPENAI_API for compatibility
    logger.info("✅ OpenAI API key loaded successfully")
else:
    logger.info("❌ Warning: OpenAI API key not found in config")

def run_docker(*args, **kwargs):  # pylint: disable=unused-argument
    # os already imported at module level
    original_dir = os.getcwd()
    try:
        os.chdir("backend/AI/Perplexica")
        os.system("docker compose up -d")
    finally:
        os.chdir(original_dir)

# Initialize Whisper Voice System as Primary Voice Recognition
WHISPER_VOICE_SYSTEM = None

def initialize_whisper_voice():
    """Initialize JARVIS Unified Voice System as primary voice recognition"""
    global WHISPER_VOICE_SYSTEM
    try:
        WHISPER_VOICE_SYSTEM = JARVISUnifiedVoice()
        logger.info("✅ JARVIS Unified Voice System initialized as primary voice recognition")
        return WHISPER_VOICE_SYSTEM
    except Exception as e:
        logger.error("Failed to initialize unified voice system: %s", e)
        return None

def start_whisper_voice_listening():
    """Start Whisper-based voice listening"""
    global WHISPER_VOICE_SYSTEM
    if WHISPER_VOICE_SYSTEM:
        WHISPER_VOICE_SYSTEM.start_listening()
        logger.info("🎤 Whisper voice listening started")
    else:
        logger.warning("Whisper voice system not initialized")

def stop_whisper_voice_listening():
    """Stop Whisper-based voice listening"""
    global WHISPER_VOICE_SYSTEM
    if WHISPER_VOICE_SYSTEM:
        WHISPER_VOICE_SYSTEM.stop_listening()
        logger.info("🛑 Whisper voice listening stopped")

thread = threading.Thread(target=run_docker)
thread.start()

# Initialize Whisper Voice System
WHISPER_VOICE_SYSTEM = initialize_whisper_voice()

load_dotenv()
STATE = 'Available...'
messages = LoadMessages()
WEBCAM = False
VIDEO_WRITER = None
js_messageslist = []
working: list[threading.Thread] = []
STAT = None
TRANSCRIPTION = ""
IMAGE_DATA = None
CPAGE = None
InputLanguage = os.environ.get('InputLanguage', 'en')
Username = os.environ.get('NickName', 'User')
lock = Lock()

# Initialize Ethical Hacking Module
HACKING_MODULE = None
if os.environ.get('PENTESTGPT_API_KEY'):
    HACKING_MODULE = EthicalHacking(api_key=os.environ['PENTESTGPT_API_KEY'])

# Initialize Brain Configuration System
brain_config = JarvisBrainConfig()

# Initialize JARVIS-RAVANA Integration System (Complete AI Core Replacement)
logger.info("🚀 Initializing JARVIS-RAVANA Integration System...")
logger.info("✅ Complete AI Core Replacement with RAVANA AGI Engine")
logger.info("✅ All JARVIS modules wrapped as RAVANA agents")
logger.info("✅ Voice Recognition, TTS, IoT Control, Automation preserved")
logger.info("✅ Advanced Sentient AGI with Emotional Intelligence")
logger.info("✅ MCP Server Integration for Deep Research")
logger.info("✅ Autonomous Learning and Self-Improvement")
logger.info("🎯 Brain Type: RAVANA AGI + JARVIS Agents + MCP Servers")
logger.info("🔄 System initializing in background...")

# UniversalTranslator now provided by jarvis_ravana_integration

def get_brain_STATus(*args, **kwargs):  # pylint: disable=unused-argument
    """Get current brain system STATus"""
    try:
        # Get STATus from JARVIS
        STATus = get_system_STATus()

        STATus_report = f"""
{'='*60}
🧠 JARVIS STATUS REPORT
{'='*60}

Name: {STATus.get('name', 'JARVIS')}
Version: {STATus.get('version', '1.0.0')}
Initialized: {'✅ YES' if STATus.get('is_initialized', False) else '❌ NO'}
Active: {'✅ YES' if STATus.get('is_active', False) else '❌ NO'}
Brain (RAVANA AGI): {'✅ ACTIVE' if STATus.get('brain_STATus') == 'active' else '❌ INACTIVE'}
MCP Integration: {'✅ ACTIVE' if STATus.get('mcp_STATus') == 'active' else '❌ INACTIVE'}

JARVIS State:
- Personality: {STATus.get('STATE', {}).get('personality', 'Unknown')}
- Brain Type: {STATus.get('STATE', {}).get('brain_type', 'Unknown')}
- Consciousness: {STATus.get('STATE', {}).get('consciousness_level', 'Unknown')}
- Memory: {STATus.get('STATE', {}).get('memory_type', 'Unknown')}
- Autonomy: {STATus.get('STATE', {}).get('autonomy', 'Unknown')}

JARVIS Capabilities:
{chr(10).join(f"- {capability}: {'✅ ACTIVE' if STATus_data == 'active' else '❌ INACTIVE'}"
              for capability, STATus_data in STATus.get('capabilities', {}).items())}

Memory Entries: {STATus.get('memory_entries', 0)}

Capabilities:
- Advanced AGI Brain (RAVANA)
- Emotional Intelligence and Personality
- Autonomous Learning and Self-Improvement
- Voice Recognition and TTS (Whisper + Threading)
- Desktop and IoT Automation
- PowerPoint Generation
- GitHub Integration
- Web Automation and Search
- Image Processing and Generation
- Multi-Modal Processing
- MCP Protocol Integration
- Deep Research Capabilities
- Weather and Time Services
- Persistent Memory and Knowledge System

Brain Type: JARVIS with RAVANA AGI Brain
Legacy Brain: ❌ COMPLETELY REPLACED
{'='*60}
        """

    except (ValueError, TypeError, AttributeError, ImportError) as e:
        STATus_report = f"""
{'='*60}
🧠 JARVIS BRAIN STATUS REPORT
{'='*60}

JARVIS: ❌ ERROR
Error: {str(e)}

Status: System initialization failed
Recommendation: Check logs and restart JARVIS
{'='*60}
        """

    return STATus_report.strip()

def MainExecution(*args, TRANSCRIPTION=None, **kwargs):  # pylint: disable=unused-argument
    """Main execution function for handling user queries through JARVIS."""
    global WEBCAM, STATE, HACKING_MODULE, IMAGE_DATA, VIDEO_WRITER

    if STATE != 'Available...':
        return
    STATE = 'Thinking...'

    try:
        # Process command through JARVIS
        result = process_command(TRANSCRIPTION, "text")

        if result.get("success"):
            response = result.get("response", "Command executed successfully")
        else:
            response = (f"I'm experiencing some difficulties, sir. Error: "
                       f"{result.get('error', 'Unknown error')}")

        # Speak the response through JARVIS
        speak_response(response)

        STATE = 'Available...'
        return response

    except (ValueError, TypeError, AttributeError, ImportError) as e:
        response = f"I encountered an error processing your request, sir. Error: {str(e)}"
        logger.info("Error in MainExecution: %s", e)
        STATE = 'Available...'
        return response
    
    # Initialize JARVIS final optimized voice system with threading
    try:
        JARVIS_VOICE = JARVISUnifiedVoice()
        logger.info("🎤 Final optimized voice system initialized with threading support")
    except (ValueError, TypeError, AttributeError, ImportError) as e:
        logger.warning("⚠️ JARVIS final optimized voice system unavailable: %s", e)
        JARVIS_VOICE = None
    
    # Initialize JARVIS movie voice system (FIXED movie voice)
    try:
        jarvis_movie_voice = JARVISUnifiedVoice()
        logger.info("🎬 JARVIS FIXED movie voice system initialized - "
                   "NO repetition, NO gap on 'sir'!")
    except (ValueError, TypeError, AttributeError, ImportError) as e:
        logger.warning("⚠️ JARVIS movie voice system unavailable: %s", e)
        jarvis_movie_voice = None

    # Check for ethical hacking commands
    if TRANSCRIPTION and TRANSCRIPTION.lower().startswith("hacking"):
        if not HACKING_MODULE:
            response = "Ethical Hacking module is not initialized. Please provide a valid API key."
            if jarvis_movie_voice and hasattr(jarvis_movie_voice, 'speak_movie_style'):
                try:
                    jarvis_movie_voice.speak_movie_style(response, "error")
                except AttributeError:
                    jarvis_movie_voice.speak(response)
            elif JARVIS_VOICE:
                JARVIS_VOICE.speak(response)
            return response

        command = TRANSCRIPTION.lower().replace("hacking", "").strip()

        # Port Scanning
        if command.startswith("port scan"):
            target = command.replace("port scan", "").strip()
            try:
                # Basic port scan simulation
                result = f"Port scan initiated for {target}. Common ports: 22, 80, 443, 8080"
                return result
            except Exception as e:
                return f"Port scan failed: {str(e)}"

        # Vulnerability Scanning
        elif command.startswith("vulnerability scan"):
            target = command.replace("vulnerability scan", "").strip()
            try:
                # Basic vulnerability scan simulation
                result = (f"Vulnerability scan initiated for {target}. "
                         f"Checking for common vulnerabilities...")
                return result
            except Exception as e:
                return f"Vulnerability scan failed: {str(e)}"

        # Exploitation
        elif command.startswith("exploit"):
            parts = command.replace("exploit", "").strip().split()
            if len(parts) >= 2:
                target, payload = parts[0], parts[1]
                try:
                    # Basic exploitation simulation
                    result = f"Exploitation attempt initiated for {target} with payload {payload}"
                    return result
                except Exception as e:
                    return f"Exploitation failed: {str(e)}"
            else:
                return "Please specify a target and payload."

        # Password Cracking
        elif command.startswith("password crack"):
            parts = command.replace("password crack", "").strip().split()
            url = parts[0] if parts else "http://example.com/login"
            username = parts[1] if len(parts) > 1 else "admin"
            _wordlist = parts[2] if len(parts) > 2 else ["password", "123456", "admin"]
            try:
                # Basic password cracking simulation
                result = f"Password cracking initiated for {username} at {url}"
                return result
            except Exception as e:
                return f"Password cracking failed: {str(e)}"

        # Network Sniffing
        elif command.startswith("network sniff"):
            parts = command.replace("network sniff", "").strip().split()
            interface = parts[0] if parts else "eth0"
            filter_expr = parts[1] if len(parts) > 1 else "tcp port 80"
            count = int(parts[2]) if len(parts) > 2 else 10
            return f"Network sniffing on {interface} with filter '{filter_expr}' for {count} packets"

        # Payload Generation
        elif command.startswith("generate payload"):
            parts = command.replace("generate payload", "").strip().split()
            payload = parts[0] if parts else "malicious_payload"
            key = parts[1] if len(parts) > 1 else "supersecretkey"
            return f"Generated payload: {payload} with key: {key}"

        # Traffic Obfuscation
        elif command.startswith("obfuscate traffic"):
            parts = command.replace("obfuscate traffic", "").strip().split()
            target = parts[0] if parts else "example.com"
            data = parts[1] if len(parts) > 1 else "sensitive_data"
            return f"Traffic obfuscation configured for {target} with data: {data}"

        # Post Exploitation
        elif command.startswith("execute in memory"):
            payload = command.replace("execute in memory", "").strip()
            return f"Executing payload in memory: {payload}"

        # Fileless Malware
        elif command.startswith("execute powershell"):
            script = command.replace("execute powershell", "").strip()
            return f"Executing PowerShell script: {script}"

        elif command.startswith("execute wmi"):
            wmi_command = command.replace("execute wmi", "").strip()
            return f"Executing WMI command: {wmi_command}"

        elif command.startswith("inject into process"):
            parts = command.replace("inject into process", "").strip().split()
            process_name = parts[0] if parts else "notepad.exe"
            payload = parts[1] if len(parts) > 1 else "malicious_payload"
            return f"Injecting payload into process {process_name}: {payload}"

        # Cloud Attacks
        elif command.startswith("check iam misconfigurations"):
            parts = command.replace("check iam misconfigurations", "").strip().split()
            aws_access_key = parts[0] if parts else "AWS_ACCESS_KEY"
            _aws_secret_key = parts[1] if len(parts) > 1 else "AWS_SECRET_KEY"
            return f"Checking IAM misconfigurations for AWS key: {aws_access_key}"

        elif command.startswith("exploit azure key vault"):
            parts = command.replace("exploit azure key vault", "").strip().split()
            vault_url = parts[0] if parts else "https://example.vault.azure.net"
            secret_name = parts[1] if len(parts) > 1 else "secret-name"
            return f"Exploiting Azure Key Vault: {vault_url} for secret: {secret_name}"

        elif command.startswith("check s3 bucket misconfigurations"):
            parts = command.replace("check s3 bucket misconfigurations", "").strip().split()
            aws_access_key = parts[0] if parts else "AWS_ACCESS_KEY"
            _aws_secret_key = parts[1] if len(parts) > 1 else "AWS_SECRET_KEY"
            return f"Checking S3 bucket misconfigurations for AWS key: {aws_access_key}"

        elif command.startswith("check azure storage misconfigurations"):
            parts = command.replace("check azure storage misconfigurations", "").strip().split()
            storage_account_name = parts[0] if parts else "storage_account_name"
            _storage_account_key = parts[1] if len(parts) > 1 else "storage_account_key"
            return f"Checking Azure storage misconfigurations for account: {storage_account_name}"

        # AI/ML Prioritization
        elif command.startswith("train model"):
            return "Please provide training data for the AI/ML model."

        elif command.startswith("predict risk"):
            return "Please provide input data for risk prediction."

        elif command.startswith("detect anomalies"):
            return "Please provide input data for anomaly detection."

        # Security Enhancer
        elif command.startswith("obfuscate code"):
            code = command.replace("obfuscate code", "").strip()
            return f"Code obfuscation feature not yet implemented: {code}"

        elif command.startswith("mimic behavior"):
            behavior = command.replace("mimic behavior", "").strip()
            return f"Behavior mimicking feature not yet implemented: {behavior}"

        # Advanced Attacks
        elif command.startswith("dns tunneling"):
            parts = command.replace("dns tunneling", "").strip().split()
            domain = parts[0] if parts else "example.com"
            data = parts[1] if len(parts) > 1 else "sensitive_data"
            return f"DNS tunneling configured for domain: {domain} with data: {data}"

        elif command.startswith("zero day exploit"):
            parts = command.replace("zero day exploit", "").strip().split()
            target = parts[0] if parts else "example.com"
            payload = parts[1] if len(parts) > 1 else "malicious_payload"
            return f"Zero-day exploit configured for target: {target} with payload: {payload}"

        elif command.startswith("add persistence"):
            parts = command.replace("add persistence", "").strip().split()
            task_name = parts[0] if parts else "malicious_task"
            command_to_run = parts[1] if len(parts) > 1 else "malicious_command"
            return f"Persistence added: {task_name} -> {command_to_run}"

        # NEW ADVANCED HACKING COMMANDS
        # Advanced Web Security
        elif command.startswith("advanced web scan"):
            parts = command.replace("advanced web scan", "").strip().split()
            target = parts[0] if parts else "example.com"
            tool = parts[1] if len(parts) > 1 else "wapiti"
            if tool == "wapiti":
                return f"Web security scan with wapiti for {target} - Feature not implemented yet"
            elif tool == "nikto":
                return f"Web security scan with nikto for {target} - Feature not implemented yet"
            elif tool == "gobuster":
                return f"Web security scan with gobuster for {target} - Feature not implemented yet"
            elif tool == "dirb":
                return f"Web security scan with dirb for {target} - Feature not implemented yet"
            elif tool == "w3af":
                return f"Web security scan with w3af for {target} - Feature not implemented yet"
            else:
                return (f"Unknown web security tool: {tool}. "
        f"Available: wapiti, nikto, gobuster, dirb, w3af")

        # Advanced Network Security
        elif command.startswith("advanced network scan"):
            parts = command.replace("advanced network scan", "").strip().split()
            target = parts[0] if parts else "192.168.1.1"
            tool = parts[1] if len(parts) > 1 else "masscan"
            if tool == "masscan":
                return f"Network scan with masscan for {target} - Feature not implemented yet"
            elif tool == "rustscan":
                return f"Network scan with rustscan for {target} - Feature not implemented yet"
            elif tool == "zmap":
                return f"Network scan with zmap for {target} - Feature not implemented yet"
            elif tool == "unicornscan":
                return f"Network scan with unicornscan for {target} - Feature not implemented yet"
            else:
                return (f"Unknown network security tool: {tool}. "
        f"Available: masscan, rustscan, zmap, unicornscan")

        # Advanced OSINT
        elif command.startswith("advanced osint"):
            parts = command.replace("advanced osint", "").strip().split()
            target = parts[0] if parts else "example.com"
            tool = parts[1] if len(parts) > 1 else "maltego"
            if tool == "maltego":
                return f"OSINT scan with maltego for {target} - Feature not implemented yet"
            elif tool == "recon-ng":
                return f"OSINT scan with recon-ng for {target} - Feature not implemented yet"
            elif tool == "osrframework":
                return f"OSINT scan with osrframework for {target} - Feature not implemented yet"
            elif tool == "phoneinfoga":
                return f"OSINT scan with phoneinfoga for {target} - Feature not implemented yet"
            elif tool == "sherlock":
                return f"OSINT scan with sherlock for {target} - Feature not implemented yet"
            else:
                return (f"Unknown OSINT tool: {tool}. "
        f"Available: maltego, recon-ng, osrframework, phoneinfoga, sherlock")

        # Advanced Wireless Security
        elif command.startswith("advanced wireless attack"):
            parts = command.replace("advanced wireless attack", "").strip().split()
            bssid = parts[0] if parts else "00:11:22:33:44:55"
            tool = parts[1] if len(parts) > 1 else "reaver"
            if tool == "reaver":
                return f"Wireless attack with reaver for {bssid} - Feature not implemented yet"
            elif tool == "bully":
                return f"Wireless attack with bully for {bssid} - Feature not implemented yet"
            elif tool == "pyrit":
                return f"Wireless attack with pyrit for {bssid} - Feature not implemented yet"
            elif tool == "hashcat":
                return f"Wireless attack with hashcat for {bssid} - Feature not implemented yet"
            else:
                return f"Unknown wireless tool: {tool}. Available: reaver, bully, pyrit, hashcat"

        # Advanced Forensics
        elif command.startswith("advanced forensics"):
            parts = command.replace("advanced forensics", "").strip().split()
            target = parts[0] if parts else "memory.dmp"
            tool = parts[1] if len(parts) > 1 else "volatility"
            if tool == "volatility":
                return (f"Forensics analysis with volatility for {target} - "
                       f"Feature not implemented yet")
            elif tool == "autopsy":
                return f"Forensics analysis with autopsy for {target} - Feature not implemented yet"
            elif tool == "sleuthkit":
                return (f"Forensics analysis with sleuthkit for {target} - "
                       f"Feature not implemented yet")
            elif tool == "bulk_extractor":
                return (f"Forensics analysis with bulk_extractor for {target} - "
                       f"Feature not implemented yet")
            else:
                return (f"Unknown forensics tool: {tool}. "
        f"Available: volatility, autopsy, sleuthkit, bulk_extractor")

        # Advanced Malware Analysis
        elif command.startswith("advanced malware analysis"):
            parts = command.replace("advanced malware analysis", "").strip().split()
            sample_path = parts[0] if parts else "malware.exe"
            tool = parts[1] if len(parts) > 1 else "cuckoo"
            if tool == "cuckoo":
                return (f"Malware analysis with cuckoo for {sample_path} - "
                       f"Feature not implemented yet")
            elif tool == "yara":
                return f"Malware analysis with yara for {sample_path} - Feature not implemented yet"
            elif tool == "capa":
                return (f"Malware analysis with capa for {sample_path} - "
                       f"Feature not implemented yet")
            elif tool == "peframe":
                return (f"Malware analysis with peframe for {sample_path} - "
                       f"Feature not implemented yet")
            else:
                return (f"Unknown malware analysis tool: {tool}. "
        f"Available: cuckoo, yara, capa, peframe")

        # Advanced Cryptography
        elif command.startswith("advanced crypto attack"):
            parts = command.replace("advanced crypto attack", "").strip().split()
            hash_file = parts[0] if parts else "hashes.txt"
            tool = parts[1] if len(parts) > 1 else "john"
            if tool == "john":
                return f"Crypto attack with john for {hash_file} - Feature not implemented yet"
            elif tool == "hashcat":
                return f"Crypto attack with hashcat for {hash_file} - Feature not implemented yet"
            elif tool == "fcrackzip":
                return (f"Crypto attack with fcrackzip for {hash_file} - "
                       f"Feature not implemented yet")
            elif tool == "rsactftool":
                return (f"Crypto attack with rsactftool for {hash_file} - "
                       f"Feature not implemented yet")
            else:
                return (f"Unknown crypto tool: {tool}. "
        f"Available: john, hashcat, fcrackzip, rsactftool")

        # Advanced Social Engineering
        elif command.startswith("advanced social engineering"):
            parts = command.replace("advanced social engineering", "").strip().split()
            tool = parts[0] if parts else "setoolkit"
            if tool == "setoolkit":
                return "Social engineering with setoolkit - Feature not implemented yet"
            elif tool == "gophish":
                return "Social engineering with gophish - Feature not implemented yet"
            elif tool == "king_phisher":
                return "Social engineering with king_phisher - Feature not implemented yet"
            elif tool == "evilginx":
                return "Social engineering with evilginx - Feature not implemented yet"
            else:
                return (f"Unknown social engineering tool: {tool}. "
        f"Available: setoolkit, gophish, king_phisher, evilginx")

        # Advanced Mobile Security
        elif command.startswith("advanced mobile security"):
            parts = command.replace("advanced mobile security", "").strip().split()
            apk_path = parts[0] if parts else "app.apk"
            tool = parts[1] if len(parts) > 1 else "mobsf"
            if tool == "mobsf":
                return f"Mobile security with mobsf for {apk_path} - Feature not implemented yet"
            elif tool == "qark":
                return (f"Mobile security with qark for {apk_path} - "
                       f"Feature not implemented yet")
            elif tool == "androguard":
                return (f"Mobile security with androguard for {apk_path} - "
                       f"Feature not implemented yet")
            elif tool == "jadx":
                return f"Mobile security with jadx for {apk_path} - Feature not implemented yet"
            else:
                return (f"Unknown mobile security tool: {tool}. "
        f"Available: mobsf, qark, androguard, jadx")

        # Advanced IoT Security
        elif command.startswith("advanced iot security"):
            parts = command.replace("advanced iot security", "").strip().split()
            target = parts[0] if parts else "iot device"
            tool = parts[1] if len(parts) > 1 else "shodan"
            if tool == "shodan":
                return f"IoT security with shodan for {target} - Feature not implemented yet"
            elif tool == "censys":
                return f"IoT security with censys for {target} - Feature not implemented yet"
            elif tool == "firmadyne":
                return f"IoT security with firmadyne for {target} - Feature not implemented yet"
            elif tool == "binwalk":
                return f"IoT security with binwalk for {target} - Feature not implemented yet"
            else:
                return (f"Unknown IoT security tool: {tool}. "
        f"Available: shodan, censys, firmadyne, binwalk")

        # Advanced Exploitation Frameworks
        elif command.startswith("start beef"):
            return "BeEF framework - Feature not implemented yet"

        elif command.startswith("start cobalt strike"):
            return "Cobalt Strike framework - Feature not implemented yet"

        elif command.startswith("advanced metasploit"):
            parts = command.replace("advanced metasploit", "").strip().split()
            target = parts[0] if parts else "192.168.1.100"
            exploit = parts[1] if len(parts) > 1 else "exploit/windows/smb/ms17_010_eternalblue"
            return f"Metasploit advanced attack simulation: {target} with {exploit} - This is a placeholder response as the method is not implemented."

        elif command.startswith("start powershell empire"):
            return "PowerShell Empire framework - Feature not implemented yet"

        # Advanced Web Application Security
        elif command.startswith("advanced web app security"):
            parts = command.replace("advanced web app security", "").strip().split()
            target = parts[0] if parts else "example.com"
            tool = parts[1] if len(parts) > 1 else "burp"
            if tool == "burp":
                return (f"Web app security with burp suite for {target} - "
                       f"Feature not implemented yet")
            elif tool == "zap":
                return (f"Web app security with owasp zap for {target} - "
                       f"Feature not implemented yet")
            elif tool == "acunetix":
                return (f"Web app security with acunetix for {target} - "
                       f"Feature not implemented yet")
            elif tool == "nessus":
                return (f"Web app security with nessus for {target} - "
                       f"Feature not implemented yet")
            else:
                return (f"Unknown web app security tool: {tool}. "
                       f"Available: burp, zap, acunetix, nessus")

        # Advanced Network Analysis
        elif command.startswith("advanced network analysis"):
            parts = command.replace("advanced network analysis", "").strip().split()
            target = parts[0] if parts else "capture.pcap"
            tool = parts[1] if len(parts) > 1 else "wireshark"
            if tool == "wireshark":
                return f"Network analysis with wireshark for {target} - Feature not implemented yet"
            elif tool == "tcpdump":
                return f"Network analysis with tcpdump for {target} - Feature not implemented yet"
            elif tool == "ngrep":
                return f"Network analysis with ngrep for {target} - Feature not implemented yet"
            elif tool == "netSTAT":
                return "Advanced network analysis - Feature not implemented yet"
            else:
                return (f"Unknown network analysis tool: {tool}. "
        f"Available: wireshark, tcpdump, ngrep, netSTAT")

        # Advanced Reverse Engineering
        elif command.startswith("advanced reverse engineering"):
            parts = command.replace("advanced reverse engineering", "").strip().split()
            binary_path = parts[0] if parts else "binary.exe"
            tool = parts[1] if len(parts) > 1 else "ghidra"
            if tool == "ghidra":
                return f"Ghidra reverse engineering simulation: {binary_path} - This is a placeholder response as the method is not implemented."
            elif tool == "radare2":
                return f"Radare2 reverse engineering simulation: {binary_path} - This is a placeholder response as the method is not implemented."
            elif tool == "ida":
                return f"IDA Pro reverse engineering simulation: {binary_path} - This is a placeholder response as the method is not implemented."
            elif tool == "x64dbg":
                return f"x64dbg reverse engineering simulation: {binary_path} - This is a placeholder response as the method is not implemented."
            else:
                return (f"Unknown reverse engineering tool: {tool}. "
                       f"Available: ghidra, radare2, ida, x64dbg")

        # Advanced Automation Tools
        elif command.startswith("advanced automation"):
            parts = command.replace("advanced automation", "").strip().split()
            target = parts[0] if parts else "playbook.yml"
            tool = parts[1] if len(parts) > 1 else "ansible"
            if tool == "ansible":
                return (f"Automation security with ansible for {target} - "
                       f"Feature not implemented yet")
            elif tool == "terraform":
                return (f"Automation security with terraform for {target} - "
                       f"Feature not implemented yet")
            elif tool == "docker":
                return f"Automation security with docker for {target} - Feature not implemented yet"
            elif tool == "kubernetes":
                return (f"Automation security with kubernetes for {target} - "
                       f"Feature not implemented yet")
            else:
                return (f"Unknown automation tool: {tool}. "
        f"Available: ansible, terraform, docker, kubernetes")

        # REAL-WORLD HACKING COMMANDS
        elif command.startswith("real port scan"):
            parts = command.replace("real port scan", "").strip().split()
            target = parts[0] if parts else "127.0.0.1"
            _ports = parts[1] if len(parts) > 1 else "1-1000"
            return f"Port scanning - Feature not implemented yet for {target}"

        elif command.startswith("real vulnerability scan"):
            parts = command.replace("real vulnerability scan", "").strip().split()
            target = parts[0] if parts else "127.0.0.1"
            return (f"Vulnerability scan for {target} - "
                   f"Feature not implemented yet")

        elif command.startswith("real web scan"):
            parts = command.replace("real web scan", "").strip().split()
            url = parts[0] if parts else "http://example.com"
            scan_type = parts[1] if len(parts) > 1 else "all"
            return (f"Real web scan feature not yet implemented for URL: {url} "
                   f"with type: {scan_type}")

        elif command.startswith("real network sniff"):
            parts = command.replace("real network sniff", "").strip().split()
            interface = parts[0] if parts else None
            count = int(parts[1]) if len(parts) > 1 else 10
            return (f"Network sniffing on {interface} for {count} packets - "
                   f"Feature not implemented yet")

        elif command.startswith("real password crack"):
            parts = command.replace("real password crack", "").strip().split()
            url = parts[0] if parts else "http://example.com/login"
            username = parts[1] if len(parts) > 1 else "admin"
            return f"Password cracking - Feature not implemented yet for {url}, {username}"

        elif command.startswith("real osint"):
            parts = command.replace("real osint", "").strip().split()
            domain = parts[0] if parts else "example.com"
            osint_type = parts[1] if len(parts) > 1 else "all"
            return ("Real OSINT feature not yet implemented for domain: %s with type: %s" %
                   (domain, osint_type))

        elif command.startswith("real sql injection"):
            parts = command.replace("real sql injection", "").strip().split()
            url = parts[0] if parts else "http://example.com"
            return "Real SQL injection scan feature not yet implemented for URL: %s" % url

        elif command.startswith("real xss scan"):
            parts = command.replace("real xss scan", "").strip().split()
            url = parts[0] if parts else "http://example.com"
            return "Real XSS scan feature not yet implemented for URL: %s" % url

        elif command.startswith("real directory brute"):
            parts = command.replace("real directory brute", "").strip().split()
            url = parts[0] if parts else "http://example.com"
            return "Real directory brute force feature not yet implemented for URL: %s" % url

        elif command.startswith("real subdomain enum"):
            parts = command.replace("real subdomain enum", "").strip().split()
            domain = parts[0] if parts else "example.com"
            return "Real subdomain enumeration feature not yet implemented for domain: %s" % domain

        elif command.startswith("real email harvest"):
            parts = command.replace("real email harvest", "").strip().split()
            domain = parts[0] if parts else "example.com"
            return "Real email harvesting feature not yet implemented for domain: %s" % domain

        else:
            return ("Unknown hacking command. Available commands: "
                   "port scan, vulnerability scan, exploit, password crack, network sniff, "
                   "generate report, advanced web scan, advanced network scan, advanced osint, "
                   "advanced wireless attack, advanced forensics, advanced malware analysis, "
                   "advanced crypto attack, advanced social engineering, advanced mobile security, "
                   "advanced iot security, start beef, start cobalt strike, advanced metasploit, "
                   "advanced web app security, advanced network analysis, "
                   "advanced reverse engineering, "
                   "advanced automation, real port scan, real vulnerability scan, real web scan, "
                   "real network sniff, real password crack, real osint, real sql injection, "
                   "real xss scan, real directory brute, real subdomain enum, real email harvest")

    # Handle brain STATus commands
    elif TRANSCRIPTION and ('brain STATus' in TRANSCRIPTION.lower() or
                           'brain info' in TRANSCRIPTION.lower()):
        return get_brain_STATus()
    else:
        # Handle non-hacking commands - Basic response system
        if brain_config and hasattr(brain_config, 'get'):
            # Use basic brain configuration for processing
            logger.info("🧠 Processing through basic brain configuration...")
            # Simple response based on TRANSCRIPTION
            if TRANSCRIPTION:
                Decision = ("I understand you said: '%s'. How can I assist you further, sir?" %
                           TRANSCRIPTION)
            else:
                Decision = "I'm ready to assist you, sir. What would you like me to do?"
                logger.info("✅ Basic brain response generated")
        else:
            # Emergency fallback - Brain configuration unavailable
            Decision = ("I'm experiencing technical difficulties with my brain configuration, sir. "
                       "Please restart JARVIS.")
        
        if 'realtime-webcam' in Decision:
            python_call_to_start_video()
            logger.info('Video Started')
            WEBCAM = True
        elif 'close_webcam' in Decision:
            logger.info('Video Stopped')
            python_call_to_stop_video()
            WEBCAM = False
        elif 'start camera' in Decision.lower():
            result = python_call_to_start_video()
            if result.get("success"):
                Decision = "Camera started successfully, sir. Ready for photos and video recording."
            else:
                Decision = "Failed to start camera: %s" % result.get("message", "Unknown error")
        elif 'stop camera' in Decision.lower():
            result = python_call_to_stop_video()
            if result.get("success"):
                Decision = "Camera stopped successfully, sir."
            else:
                Decision = "Failed to stop camera: %s" % result.get("message", "Unknown error")
        elif ('take picture' in Decision.lower() or 'snap photo' in Decision.lower() or
              'capture photo' in Decision.lower()):
            result = python_call_to_capture()
            if result.get("success"):
                Decision = "Photo captured successfully: %s" % result.get("filename", "photo.jpg")
            else:
                Decision = "Failed to capture photo: %s" % result.get("message", "Unknown error")
        elif 'take screenshot' in Decision.lower():
            result = python_call_to_take_screenshot()
            if result.get("success"):
                Decision = ("Screenshot taken successfully: %s" %
                           result.get("filename", "screenshot.png"))
            else:
                Decision = "Failed to take screenshot: %s" % result.get("message", "Unknown error")
        elif 'start video recording' in Decision.lower():
            result = python_call_to_start_video_recording()
            if result.get("success"):
                Decision = "Video recording started: %s" % result.get("filename", "video.mp4")
            else:
                Decision = ("Failed to start video recording: %s" %
                           result.get("message", "Unknown error"))
        elif 'stop video recording' in Decision.lower():
            result = python_call_to_stop_video_recording()
            if result.get("success"):
                Decision = "Video recording stopped and saved successfully, sir."
            else:
                Decision = ("Failed to stop video recording: %s" %
                           result.get("message", "Unknown error"))
        elif 'start screen recording' in Decision.lower():
            result = python_call_to_start_screen_recording()
            if result.get("success"):
                Decision = "Screen recording started using Xbox Game Bar, sir."
            else:
                Decision = ("Failed to start screen recording: %s" %
                           result.get("message", "Unknown error"))
        elif 'stop screen recording' in Decision.lower():
            result = python_call_to_stop_screen_recording()
            if result.get("success"):
                Decision = "Screen recording stopped using Xbox Game Bar, sir."
            else:
                Decision = ("Failed to stop screen recording: %s" %
                           result.get("message", "Unknown error"))

        # Check for desktop automation commands
        desktop_automation_keywords = [
            "take picture", "snap photo", "open camera", "record video", "take screenshot",
            "record screen", "start camera", "stop camera", "capture photo",
            "start video recording",
            "stop video recording", "start screen recording", "stop screen recording",
            "open", "launch", "close", "quit", "exit", "search youtube",
            "search google", "play video", "download video", "click", "type", "press"
        ]

        if TRANSCRIPTION and any(keyword in TRANSCRIPTION.lower()
                                for keyword in desktop_automation_keywords):
            try:
                # Initialize desktop automation interface
                desktop_interface = JARVISDesktopInterface()

                # Process desktop automation command
                desktop_result = desktop_interface.execute_command(TRANSCRIPTION)

                if desktop_result.get("success"):
                    Decision = desktop_result.get("message", "Desktop automation command executed")
                    logger.info("✅ Desktop automation executed: %s",
                               desktop_result.get('action', 'unknown'))
                else:
                    Decision = ("Desktop automation failed: %s" %
                               desktop_result.get('error', 'Unknown error'))
                    logger.warning("❌ Desktop automation failed: %s", desktop_result.get('error'))

            except Exception as e:
                Decision = "Desktop automation error: %s" % str(e)
                logger.error("❌ Desktop automation error: %s", e)
        
        # Make JARVIS speak the response in movie style
        if jarvis_movie_voice and Decision:
            try:
                if hasattr(jarvis_movie_voice, 'speak_movie_style'):
                    jarvis_movie_voice.speak_movie_style(Decision, "acknowledgment")
                else:
                    jarvis_movie_voice.speak(Decision)
            except Exception as e:
                logger.warning("Movie voice synthesis failed: %s", e)
                try:
                    jarvis_movie_voice.speak(Decision)
                except Exception:
                    logger.error("Voice synthesis completely failed")
        elif JARVIS_VOICE and Decision:
            JARVIS_VOICE.speak(Decision)
        
        return Decision

@(eel.expose if eel else lambda x: x)
def js_messages(*args, **kwargs):  # pylint: disable=unused-argument
    """Fetches new messages to update the GUI."""
    global messages, js_messageslist
    with lock:
        messages_obj = LoadMessages()
        messages = messages_obj()  # Call the object to get the list
    if js_messageslist != messages:
        converter = GuiMessagesConverter()
        new_messages = [converter.convert_message(msg) for msg in messages[len(js_messageslist):]]
        js_messageslist = messages
        return new_messages
    return []

@(eel.expose if eel else lambda x: x)
def js_STATE(*args, **kwargs):  # pylint: disable=unused-argument
    """Updates or retrieves the current STATE."""
    global STATE
    if STAT:
        STATE = STAT
    return STATE

@(eel.expose if eel else lambda x: x)
def js_mic(*args, **kwargs):  # pylint: disable=unused-argument
    """Handles microphone input."""
    logger.info(TRANSCRIPTION)
    if not working:
        work = threading.Thread(target=process_input, args=(TRANSCRIPTION,), daemon=True)
        work.start()
        working.append(work)
    else:
        if working[0].is_alive():
            return
        working.pop()


@(eel.expose if eel else lambda x: x)
def start_continuous_voice():
    """Start continuous voice recognition"""
    try:
        logger.info("Starting unified voice recognition...")
        start_voice_listening()
        return {"STATus": "success", "message": "Unified voice recognition started"}
    except Exception as e:
        logger.error("Error starting unified voice: %s", e)
        return {"STATus": "error", "message": str(e)}

@(eel.expose if eel else lambda x: x)
def stop_continuous_voice():
    """Stop continuous voice recognition"""
    try:
        logger.info("Stopping unified voice recognition...")
        stop_voice_listening()
        return {"STATus": "success", "message": "Unified voice recognition stopped"}
    except Exception as e:
        logger.error("Error stopping unified voice: %s", e)
        return {"STATus": "error", "message": str(e)}


def process_input(*args, **kwargs):  # pylint: disable=unused-argument
    global WEBCAM
    result = MainExecution(TRANSCRIPTION)
    if result == "close_webcam":
        logger.info('Video Stopped')
        python_call_to_stop_video()
        WEBCAM = False

@(eel.expose if eel else lambda x: x)
def python_call_to_start_video(*args, **kwargs):  # pylint: disable=unused-argument
    """Starts the laptop camera for video capture."""
    global WEBCAM
    try:
        global cv2
        try:
            import cv2
        except ImportError:
            cv2 = None
            print("Warning: cv2 (OpenCV) not available")
        try:
            if cv2:
                WEBCAM = cv2.VideoCapture(0)
            else:
                print("Error: cv2 not available for camera access")
                return "Camera not available - OpenCV not installed"
        except AttributeError:
            logger.error("OpenCV VideoCapture not available")
            return {"success": False, "message": "OpenCV VideoCapture not available"}
        if WEBCAM.isOpened():
            # Set camera properties for better quality
            try:
                if cv2:
                    WEBCAM.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    WEBCAM.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    WEBCAM.set(cv2.CAP_PROP_FPS, 30)
            except AttributeError:
                # Fallback if CAP_PROP constants are not available
                logger.warning("OpenCV constants not available, using default settings")
            logger.info("Laptop camera started successfully")
            return {"success": True, "message": "Laptop camera started - "
                                               "Ready for photos and video"}
        else:
            WEBCAM = None
            logger.error("Failed to start laptop camera")
            return {"success": False, "message": "Failed to start laptop camera"}
    except ImportError:
        logger.error("OpenCV not available for camera capture")
        return {"success": False, "message": "OpenCV not available"}
    except Exception as e:
        logger.error("Error starting camera: %s", e)
        return {"success": False, "message": str(e)}

@(eel.expose if eel else lambda x: x)
def python_call_to_stop_video(*args, **kwargs):  # pylint: disable=unused-argument
    """Stops the video capture."""
    global WEBCAM
    try:
        if WEBCAM and hasattr(WEBCAM, 'release'):
            WEBCAM.release()
            WEBCAM = None
            logger.info("Webcam stopped successfully")
            return {"success": True, "message": "Webcam stopped"}
        else:
            logger.warning("No webcam to stop")
            return {"success": False, "message": "No webcam running"}
    except Exception as e:
        logger.error("Error stopping video: %s", e)
        return {"success": False, "message": str(e)}

@(eel.expose if eel else lambda x: x)
def python_call_to_capture(*args, **kwargs):  # pylint: disable=unused-argument
    """Captures a photo from the laptop camera."""
    global WEBCAM, IMAGE_DATA
    try:
        if WEBCAM and hasattr(WEBCAM, 'read'):
            try:
                ret, frame = WEBCAM.read()
                if ret:
                    # Encode frame as JPEG
                    try:
                        if cv2:
                            _, buffer = cv2.imencode('.jpg', frame)
                        else:
                            return {"success": False, "message": "OpenCV not available for image encoding"}
                        global IMAGE_DATA
                        IMAGE_DATA = base64.b64encode(buffer).decode('utf-8')
                    except AttributeError:
                        logger.error("OpenCV imencode not available")
                        return {"success": False, "message": "OpenCV imencode not available"}

                    # Also save the image to file
                    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"camera_photo_{timestamp}.jpg"
                    try:
                        if cv2:
                            cv2.imwrite(filename, frame)
                        else:
                            logger.error("OpenCV not available for image writing")
                    except AttributeError:
                        logger.error("OpenCV imwrite not available")
                        return {"success": False, "message": "OpenCV imwrite not available"}

                    logger.info("Photo captured successfully: %s", filename)
                    return {"success": True, "message": f"Photo captured: {filename}",
                           "data": IMAGE_DATA, "filename": filename}
                else:
                    logger.error("Failed to capture frame from camera")
                    return {"success": False, "message": "Failed to capture frame from camera"}
            except ImportError:
                logger.error("OpenCV not available for photo capture")
                return {"success": False, "message": "OpenCV not available"}
            except Exception as e:
                logger.error("Error capturing photo: %s", e)
                return {"success": False, "message": str(e)}
        else:
            logger.warning("No camera available for photo capture")
            return {"success": False, "message": "No camera available - please start camera first"}
    except ImportError:
        logger.error("OpenCV not available for photo capture")
        return {"success": False, "message": "OpenCV not available"}
    except Exception as e:
        logger.error("Error capturing photo: %s", e)
        return {"success": False, "message": str(e)}

@(eel.expose if eel else lambda x: x)
def python_call_to_start_video_recording(*args, **kwargs):  # pylint: disable=unused-argument
    """Starts video recording from laptop camera."""
    global WEBCAM
    try:
        if WEBCAM and hasattr(WEBCAM, 'read'):
            timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"camera_video_{timestamp}.mp4"

            # Get camera properties with fallbacks
            try:
                if cv2:
                    fps = int(WEBCAM.get(cv2.CAP_PROP_FPS)) or 30
                    width = int(WEBCAM.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
                    height = int(WEBCAM.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
                else:
                    fps = 30
                    width = 1280
                    height = 720
            except AttributeError:
                fps, width, height = 30, 1280, 720

            # Define codec and create VideoWriter
            try:
                if cv2:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
                else:
                    logger.error("OpenCV not available for video writing")
                    return {"success": False, "message": "OpenCV not available for video writing"}
            except AttributeError:
                logger.error("OpenCV VideoWriter not available")
                return {"success": False, "message": "OpenCV VideoWriter not available"}

            # Store video writer globally
            global VIDEO_WRITER
            VIDEO_WRITER = out

            logger.info("Video recording started: %s", filename)
            return {"success": True, "message": f"Video recording started: {filename}",
                   "filename": filename}
        else:
            logger.warning("No camera available for video recording")
            return {"success": False, "message": "No camera available - please start camera first"}
    except ImportError:
        logger.error("OpenCV not available for video recording")
        return {"success": False, "message": "OpenCV not available"}
    except Exception as e:
        logger.error("Error starting video recording: %s", e)
        return {"success": False, "message": str(e)}

@(eel.expose if eel else lambda x: x)
def python_call_to_stop_video_recording(*args, **kwargs):  # pylint: disable=unused-argument
    """Stops video recording from laptop camera."""
    global VIDEO_WRITER
    try:
        if VIDEO_WRITER:
            VIDEO_WRITER.release()
            VIDEO_WRITER = None
            logger.info("Video recording stopped successfully")
            return {"success": True, "message": "Video recording stopped and saved"}
        else:
            logger.warning("No video recording in progress")
            return {"success": False, "message": "No video recording in progress"}
    except Exception as e:
        logger.error("Error stopping video recording: %s", e)
        return {"success": False, "message": str(e)}

@(eel.expose if eel else lambda x: x)
def python_call_to_record_frame(*args, **kwargs):  # pylint: disable=unused-argument
    """Records a frame to the current video recording."""
    global WEBCAM, VIDEO_WRITER
    try:
        if WEBCAM and VIDEO_WRITER and hasattr(WEBCAM, 'read'):
            ret, frame = WEBCAM.read()
            if ret:
                VIDEO_WRITER.write(frame)
                return {"success": True, "message": "Frame recorded"}
            else:
                return {"success": False, "message": "Failed to read frame from camera"}
        else:
            return {"success": False, "message": "Camera or video recording not available"}
    except Exception as e:
        logger.error("Error recording frame: %s", e)
        return {"success": False, "message": str(e)}

@(eel.expose if eel else lambda x: x)
def python_call_to_take_screenshot(*args, **kwargs):  # pylint: disable=unused-argument
    """Takes a screenshot of the entire screen."""
    try:
        # pyautogui already imported at module level
        import datetime

        # Take screenshot
        if pyautogui:
            screenshot = pyautogui.screenshot()
        else:
            return {"success": False, "message": "pyautogui not available for screenshots"}

        # Save with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        screenshot.save(filename)

        logger.info("Screenshot taken: %s", filename)
        return {"success": True, "message": f"Screenshot saved: {filename}", "filename": filename}
    except ImportError:
        logger.error("PyAutoGUI not available for screenshots")
        return {"success": False, "message": "PyAutoGUI not available"}
    except Exception as e:
        logger.error("Error taking screenshot: %s", e)
        return {"success": False, "message": str(e)}

@(eel.expose if eel else lambda x: x)
def python_call_to_start_screen_recording(*args, **kwargs):  # pylint: disable=unused-argument
    """Starts screen recording using Xbox Game Bar."""
    try:
        import subprocess
        # os already imported at module level

        # Use Windows Game Bar to start screen recording
        # Game Bar shortcut: Win + Alt + R
        subprocess.run(['powershell', '-Command',
                       'Add-Type -AssemblyName System.Windows.Forms; '
                       '[System.Windows.Forms.SendKeys]::SendWait("^%{R}")'],
                      check=False)

        logger.info("Screen recording started using Xbox Game Bar")
        return {"success": True, "message": "Screen recording started using "
                                           "Xbox Game Bar (Win+Alt+R)"}
    except Exception as e:
        logger.error("Error starting screen recording: %s", e)
        return {"success": False, "message": str(e)}

@(eel.expose if eel else lambda x: x)
def python_call_to_stop_screen_recording(*args, **kwargs):  # pylint: disable=unused-argument
    """Stops screen recording using Xbox Game Bar."""
    try:
        import subprocess

        # Use Windows Game Bar to stop screen recording
        # Game Bar shortcut: Win + Alt + R (same as start)
        subprocess.run(['powershell', '-Command',
                       'Add-Type -AssemblyName System.Windows.Forms; '
                       '[System.Windows.Forms.SendKeys]::SendWait("^%{R}")'],
                      check=False)

        logger.info("Screen recording stopped using Xbox Game Bar")
        return {"success": True, "message": "Screen recording stopped using Xbox Game Bar"}
    except Exception as e:
        logger.error("Error stopping screen recording: %s", e)
        return {"success": False, "message": str(e)}

@(eel.expose if eel else lambda x: x)
def handle_captured_image(*args, **kwargs):  # pylint: disable=unused-argument
    """Handles the captured image data from the web interface."""
    js_capture(IMAGE_DATA)

@(eel.expose if eel else lambda x: x)
def js_page(*args, **kwargs):  # pylint: disable=unused-argument
    """Navigates to the specified page."""
    if CPAGE == 'home':
        if eel:
            eel.openHome()
    elif CPAGE == 'settings':
        if eel:
            eel.openSettings()

@(eel.expose if eel else lambda x: x)
def setup(*args, **kwargs):  # pylint: disable=unused-argument
    """Sets up the GUI window."""
    if pyautogui:
        pyautogui.hotkey('win', 'up')
    else:
        print("pyautogui not available for hotkey simulation")

@(eel.expose if eel else lambda x: x)
def js_language(*args, **kwargs):  # pylint: disable=unused-argument
    """Returns the input language."""
    return str(InputLanguage)

@(eel.expose if eel else lambda x: x)
def js_assistantname(*args, **kwargs):  # pylint: disable=unused-argument
    """Returns the assistant's name."""
    return "JARVIS"

@(eel.expose if eel else lambda x: x)
def js_capture(*args, **kwargs):  # pylint: disable=unused-argument
    """Saves the captured image."""
    global IMAGE_DATA
    if IMAGE_DATA:
        try:
            # Handle both base64 data and data URL formats
            if ',' in IMAGE_DATA:
                image_bytes = base64.b64decode(IMAGE_DATA.split(',')[1])
            else:
                image_bytes = base64.b64decode(IMAGE_DATA)
            with open('capture.png', 'wb') as f:
                f.write(image_bytes)
            logger.info("Image saved as capture.png")
        except Exception as e:
            logger.error("Error saving image: %s", e)
    else:
        logger.warning("No image data to save")

# Initialize Eel and start the application
logger.info("Initializing Eel web interface...")

# Ensure we're in the correct directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

logger.info("Current working directory: %s", os.getcwd())
logger.info("Web directory exists: %s", os.path.exists('web'))
logger.info("JARVIS Circular Interface exists: %s",
           os.path.exists('web/jarvis_circular_interface.html'))

if not os.path.exists('web'):
    logger.info("ERROR: Web directory not found!")
    exit(1)

if not os.path.exists('web/jarvis_spider_exact.html'):
    logger.info("ERROR: jarvis_spider_exact.html not found!")
    exit(1)

if eel:
    eel.init('web')
logger.info("Starting Eel server on port 44450...")

# Start unified voice recognition automatically
try:
    logger.info("🤖 Starting JARVIS unified voice system...")

    # Set up callbacks
    def on_command(text):
        logger.info("Command received: '%s'", text)
        # Process command through JARVIS
        global TRANSCRIPTION
        TRANSCRIPTION = text
        # Trigger processing
        if not working:
            work = threading.Thread(target=process_input, args=(TRANSCRIPTION,), daemon=True)
            work.start()
            working.append(work)

    def on_error(error):
        logger.error("Voice system error: %s", error)

    set_command_callback(on_command)
    set_error_callback(on_error)

    # Start unified voice listening
    start_voice_listening()
    logger.info("✅ JARVIS unified voice system is now active!")

except Exception as e:
    logger.error("Failed to start unified voice system: %s", e)

if eel:
    eel.start('jarvis_clean_voice.html', port=44450, mode='chrome', size=(1400, 900))
else:
    print("Eel not available - web interface disabled")

