

"""
Real-World Hacking Tools Module for JARVIS
This module provides actual hacking tools that run real commands instead of simulations.

Optional Dependencies:
- nmap: For port scanning functionality
- scapy: For network packet manipulation
- These modules are imported with try-except blocks and gracefully handle missing dependencies.
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

import requests

try:
    import nmap  # type: ignore
except ImportError:
    nmap = None  # type: Optional[Any]

try:
    import scapy.all as scapy  # type: ignore
except ImportError:
    scapy = None  # type: Optional[Any]

class RealHackingTools:
    """Real-world hacking tools that execute actual commands."""

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        """Initialize real hacking tools."""
        self.tools_config = self.load_tools_config()
        # Set nmap path for direct execution
        self.nmap_path = os.path.join(os.getcwd(), "TOOLS", "nmap", "nmap.exe")
        self.nmap_available = os.path.exists(self.nmap_path)
        
        if not self.nmap_available:
            print(f"Warning: Nmap not found at {self.nmap_path}")

    def load_tools_config(self) -> Dict:
        """Load tools configuration from file."""
        config_file = Path("TOOLS/tools_config.json")
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def run_command(self, command: List[str], timeout: int = 30) -> str:
        """Run a command and return the output."""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=True if os.name == 'nt' else False,
                check=False
            )
            if result.returncode == 0:
                return result.stdout
            else:
                return f"Command failed: {result.stderr}"
        except subprocess.TimeoutExpired:
            return "Command timed out"
        except (subprocess.CalledProcessError, FileNotFoundError, OSError,
                ValueError, RuntimeError) as e:
            return f"Command error: {e}"

class RealPortScanner:
    """Real port scanning using Nmap."""

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        """Initialize real port scanner."""
        # Set nmap path for direct execution
        self.nmap_path = os.path.join(os.getcwd(), "TOOLS", "nmap", "nmap.exe")
        self.nmap_available = os.path.exists(self.nmap_path)
        
        if not self.nmap_available:
            print(f"Warning: Nmap not found at {self.nmap_path}")

    def scan_ports(self, target: str, ports: str = "1-1000") -> str:
        """Perform real port scanning using Nmap."""
        try:
            if not self.nmap_available:
                return "❌ Nmap executable not found. Please ensure Nmap is installed in TOOLS/nmap/"

            print(f"🔍 Scanning {target} on ports {ports}...")
            
            # Execute nmap command directly
            cmd = [self.nmap_path, "-p", ports, target]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                check=False
            )
            
            if result.returncode == 0:
                return f"✅ Nmap scan completed for {target}:\n{result.stdout}"
            else:
                return f"❌ Nmap scan failed: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return "❌ Port scan timed out"
        except Exception as e:
            return f"❌ Port scan failed: {e}"

    def scan_vulnerabilities(self, target: str) -> str:
        """Perform vulnerability scanning using Nmap scripts."""
        try:
            if self.nmap_scanner is None:
                return "❌ Nmap is not available. Please install nmap to use vulnerability scanning."

            print(f"🔍 Scanning vulnerabilities on {target}...")
            self.nmap_scanner.scan(
                target, arguments='-sV --script vuln')

            if target in self.nmap_scanner.all_hosts():
                host_info = self.nmap_scanner[target]
                vulnerabilities = []

                for protocol in host_info.all_protocols():
                    ports_info = host_info[protocol]
                    for port in ports_info:
                        if 'script' in ports_info[port]:
                            for script_name, script_output in ports_info[port]['script'].items():
                                vulnerabilities.append(
                                    f"Port {port}: {script_name} - {script_output}")

                if vulnerabilities:
                    return (f"✅ Vulnerabilities found on {target}:\n" +
                            "\n".join(vulnerabilities[:10]))
                else:
                    return f"✅ No vulnerabilities found on {target}"
            else:
                return f"❌ Host {target} is not reachable"

        except (nmap.PortScannerError, OSError, ValueError, RuntimeError) as e:
            return f"❌ Vulnerability scan failed: {e}"

class RealWebScanner:
    """Real web vulnerability scanning."""

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        """Docstring for function"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                         'AppleWebKit/537.36'
        })

    def scan_sql_injection(self, url: str) -> str:
        """Scan for SQL injection vulnerabilities."""
        try:
            print(f"🔍 Scanning {url} for SQL injection...")

            # Test for SQL injection using common payloads
            payloads = [
                "' OR '1'='1",
                "' OR 1=1--",
                "'; DROP TABLE users--",
                "' UNION SELECT NULL--"
            ]

            vulnerabilities = []
            for payload in payloads:
                test_url = f"{url}?id={payload}"
                try:
                    response = self.session.get(test_url, timeout=10)
                    if any(error in response.text.lower()
                           for error in ['sql', 'mysql', 'database', 'syntax']):
                        vulnerabilities.append(f"Potential SQL injection with payload: {payload}")
                except requests.RequestException:
                    continue

            if vulnerabilities:
                return "✅ SQL injection vulnerabilities found:\n" + "\n".join(vulnerabilities)
            else:
                return "✅ No SQL injection vulnerabilities found"

        except (requests.RequestException, ValueError, RuntimeError) as e:
            return f"❌ SQL injection scan failed: {e}"

    def scan_xss(self, url: str) -> str:
        """Scan for XSS vulnerabilities."""
        try:
            print(f"🔍 Scanning {url} for XSS...")

            # Test for XSS using common payloads
            payloads = [
                "<script>alert('XSS')</script>",
                "javascript:alert('XSS')",
                "<img src=x onerror=alert('XSS')>",
                "<svg onload=alert('XSS')>"
            ]

            vulnerabilities = []
            for payload in payloads:
                test_url = f"{url}?search={payload}"
                try:
                    response = self.session.get(test_url, timeout=10)
                    if payload in response.text:
                        vulnerabilities.append(f"Potential XSS with payload: {payload}")
                except requests.RequestException:
                    continue

            if vulnerabilities:
                return "✅ XSS vulnerabilities found:\n" + "\n".join(vulnerabilities)
            else:
                return "✅ No XSS vulnerabilities found"

        except (requests.RequestException, ValueError, RuntimeError) as e:
            return f"❌ XSS scan failed: {e}"

    def directory_bruteforce(self, url: str) -> str:
        """Perform directory brute forcing."""
        try:
            print(f"🔍 Brute forcing directories on {url}...")

            # Common directories to test
            directories = [
                'admin', 'login', 'wp-admin', 'administrator', 'phpmyadmin',
                'backup', 'config', 'database', 'files', 'images',
                'uploads', 'downloads', 'temp', 'tmp', 'test'
            ]

            found_dirs = []
            for directory in directories:
                test_url = f"{url}/{directory}"
                try:
                    response = self.session.get(test_url, timeout=5)
                    if response.status_code == 200:
                        found_dirs.append(f"{test_url} (Status: {response.status_code})")
                except requests.RequestException:
                    continue

            if found_dirs:
                return "✅ Found directories:\n" + "\n".join(found_dirs)
            else:
                return "✅ No common directories found"

        except (requests.RequestException, ValueError, RuntimeError) as e:
            return f"❌ Directory brute force failed: {e}"

class RealNetworkSniffer:
    """Real network packet sniffing using Scapy."""

    def capture_packets(self, interface: str = None, count: int = 10) -> str:
        """Capture network packets."""
        try:
            print(f"🔍 Capturing {count} packets...")

            packets = scapy.sniff(count=count, iface=interface)
            packet_info = []

            for packet in packets:
                if packet.haslayer(scapy.IP):
                    src = packet[scapy.IP].src
                    dst = packet[scapy.IP].dst
                    protocol = packet[scapy.IP].proto

                    if packet.haslayer(scapy.TCP):
                        sport = packet[scapy.TCP].sport
                        dport = packet[scapy.TCP].dport
                        packet_info.append(f"TCP: {src}:{sport} -> {dst}:{dport}")
                    elif packet.haslayer(scapy.UDP):
                        sport = packet[scapy.UDP].sport
                        dport = packet[scapy.UDP].dport
                        packet_info.append(f"UDP: {src}:{sport} -> {dst}:{dport}")
                    else:
                        packet_info.append(f"IP: {src} -> {dst} (Protocol: {protocol})")

            if packet_info:
                return "✅ Captured packets:\n" + "\n".join(packet_info[:10])
            else:
                return "✅ No packets captured"

        except (OSError, ValueError, RuntimeError) as e:
            return f"❌ Packet capture failed: {e}"

class RealPasswordCracker:
    """Real password cracking tools."""

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        """Docstring for function"""
        self.common_passwords = [
            'password', '123456', 'admin', 'root', 'user', 'guest',
            'test', 'demo', 'qwerty', 'abc123', 'password123'
        ]

    def brute_force_http(self, url: str, username: str = "admin") -> str:
        """Perform HTTP brute force attack."""
        try:
            print(f"🔍 Brute forcing {url} with username '{username}'...")

            session = requests.Session()
            found_passwords = []

            for password in self.common_passwords:
                try:
                    # Try different login endpoints
                    login_data = {
                        'username': username,
                        'password': password,
                        'login': 'Login'
                    }

                    response = session.post(url, data=login_data, timeout=5)

                    # Check for successful login indicators
                    if response.status_code == 200 and 'dashboard' in response.text.lower():
                        found_passwords.append(f"Username: {username}, Password: {password}")
                        break

                except requests.RequestException:
                    continue

            if found_passwords:
                return "✅ Login successful:\n" + "\n".join(found_passwords)
            else:
                return "✅ No valid credentials found"

        except (requests.RequestException, ValueError, RuntimeError) as e:
            return f"❌ Brute force failed: {e}"

class RealOSINT:
    """Real OSINT gathering tools."""

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        """Docstring for function"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                         'AppleWebKit/537.36'
        })

    def subdomain_enumeration(self, domain: str) -> str:
        """Enumerate subdomains using various methods."""
        try:
            print(f"🔍 Enumerating subdomains for {domain}...")

            subdomains = set()

            # Method 1: Certificate Transparency logs
            try:
                ct_url = f"https://crt.sh/?q=%.{domain}&output=json"
                response = self.session.get(ct_url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    for cert in data:
                        name = cert.get('name_value', '')
                        if domain in name:
                            subdomains.add(name)
            except requests.RequestException:
                pass

            # Method 2: Common subdomains
            common_subs = [
                'www', 'mail', 'ftp', 'admin', 'test', 'dev', 'staging',
                'api', 'app', 'blog', 'shop', 'store', 'support'
            ]

            for sub in common_subs:
                test_url = f"https://{sub}.{domain}"
                try:
                    response = self.session.get(test_url, timeout=5)
                    if response.status_code == 200:
                        subdomains.add(f"{sub}.{domain}")
                except requests.RequestException:
                    continue

            if subdomains:
                return "✅ Found subdomains:\n" + "\n".join(sorted(subdomains))
            else:
                return "✅ No subdomains found"

        except (requests.RequestException, ValueError, RuntimeError) as e:
            return f"❌ Subdomain enumeration failed: {e}"

    def email_harvesting(self, domain: str) -> str:
        """Harvest email addresses from various sources."""
        try:
            print(f"🔍 Harvesting emails for {domain}...")

            emails = set()

            # Search for emails in common locations
            search_urls = [
                f"https://{domain}/contact",
                f"https://{domain}/about",
                f"https://{domain}/team",
                f"https://{domain}/staff"
            ]

            for url in search_urls:
                try:
                    response = self.session.get(url, timeout=10)
                    if response.status_code == 200:
                        # Simple email regex
                        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                        found_emails = re.findall(email_pattern, response.text)
                        emails.update(found_emails)
                except requests.RequestException:
                    continue

            if emails:
                return "✅ Found emails:\n" + "\n".join(sorted(emails))
            else:
                return "✅ No emails found"

        except (requests.RequestException, ValueError, RuntimeError) as e:
            return f"❌ Email harvesting failed: {e}"

class RealHackingModule:
    """Main real hacking module that coordinates all tools."""

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        """Docstring for function"""
        self.port_scanner = RealPortScanner()
        self.web_scanner = RealWebScanner()
        self.network_sniffer = RealNetworkSniffer()
        self.password_cracker = RealPasswordCracker()
        self.osint = RealOSINT()

    def port_scan(self, target: str, ports: str = "1-1000") -> str:
        """Perform real port scanning."""
        return self.port_scanner.scan_ports(target, ports)

    def vulnerability_scan(self, target: str) -> str:
        """Perform real vulnerability scanning."""
        return self.port_scanner.scan_vulnerabilities(target)

    def web_scan(self, url: str, scan_type: str = "all") -> str:
        """Perform real web vulnerability scanning."""
        results = []

        if scan_type in ["all", "sql"]:
            results.append(self.web_scanner.scan_sql_injection(url))

        if scan_type in ["all", "xss"]:
            results.append(self.web_scanner.scan_xss(url))

        if scan_type in ["all", "dir"]:
            results.append(self.web_scanner.directory_bruteforce(url))

        return "\n\n".join(results)

    def network_sniff(self, interface: str = None, count: int = 10) -> str:
        """Perform real network sniffing."""
        return self.network_sniffer.capture_packets(interface, count)

    def password_crack(self, url: str, username: str = "admin") -> str:
        """Perform real password cracking."""
        return self.password_cracker.brute_force_http(url, username)

    def osint_gather(self, domain: str, gather_type: str = "all") -> str:
        """Perform real OSINT gathering."""
        results = []

        if gather_type in ["all", "subdomains"]:
            results.append(self.osint.subdomain_enumeration(domain))

        if gather_type in ["all", "emails"]:
            results.append(self.osint.email_harvesting(domain))

        return "\n\n".join(results)
