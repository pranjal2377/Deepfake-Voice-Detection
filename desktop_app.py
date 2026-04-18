#!/usr/bin/env python3
"""
Desktop App Wrapper — Deepfake Voice detection

This script packages the Streamlit web dashboard into a standalone native
desktop window using PyWebView. It transparently runs the server in the 
background on a free port and closes it when the app is closed.
"""

import sys
import os
import time
import socket
import threading
import urllib.request
import subprocess

try:
    import webview
except ImportError:
    print("❌ Error: pywebview is not installed. Please run:")
    print("   pip install pywebview")
    sys.exit(1)

def get_free_port() -> int:
    """Find an available port on localhost."""
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

def wait_for_server(port: int, max_retries: int = 30) -> bool:
    """Ping the Streamlit server until it answers indicating it is ready."""
    url = f"http://127.0.0.1:{port}"
    for _ in range(max_retries):
        try:
            code = urllib.request.urlopen(url).getcode()
            if code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False

def main():
    print("=" * 60)
    print("  Deepfake Voice Scam Detection — Desktop Application")
    print("=" * 60)

    port = get_free_port()
    print(f"🔄 Launching internal engine on port {port}...")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.dirname(__file__))

    # Command to run streamlit headlessly
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run", 
        "src/ui/dashboard.py", 
        "--server.port", str(port), 
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]

    # Start the backend server logic
    process = subprocess.Popen(
        streamlit_cmd,
        env=env,
        stdout=subprocess.DEVNULL,  # Hide internal streamlit logs from console
        stderr=subprocess.DEVNULL
    )

    if wait_for_server(port):
        print("✅ Engine online! Loading Desktop Window...")
        
        # Instantiate the App Window natively 
        window = webview.create_window(
            title="Deepfake Voice & Scam Detection System 🛡️", 
            url=f"http://127.0.0.1:{port}",
            width=1280, 
            height=850,
            min_size=(1000, 700),
            background_color='#020617'  # Matches our dark theme CSS
        )
        
        try:
            webview.start()
        except KeyboardInterrupt:
            pass
        finally:
            print("🛑 Shutting down internal engine...")
            process.terminate()
            process.wait()
    else:
        print("❌ Error: The internal analytics engine failed to start in time.")
        process.terminate()
        sys.exit(1)

    print("👋 Desktop App closed.")

if __name__ == "__main__":
    main()
