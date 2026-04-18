#!/usr/bin/env python3
"""
Mobile App Server Wrapper — Voice Threat Scanner
Hosts a Progressive Web App (PWA) locally that connects to the
Streamlit backend. When you connect via mobile, you can "Add to Home Screen"
to use it exactly like a native app.
"""

import sys
import os
import socket
import threading
import time
import subprocess
from http.server import SimpleHTTPRequestHandler
import socketserver
import qrcode
import urllib.request

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
    <title>Voice Threat Scanner</title>
    
    <link rel="manifest" href="/manifest.json">
    <meta name="theme-color" content="#020617">
    <link rel="apple-touch-icon" href="/icon.png">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="Threat Scanner">

    <style>
        * { box-sizing: border-box; }
        body, html { 
            margin: 0; padding: 0; width: 100%; height: 100%; 
            overflow: hidden; background: #020617; 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: white;
        }
        #loader {
            position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            display: flex; flex-direction: column;
            justify-content: center; align-items: center;
            background: #020617; z-index: 10;
        }
        /* Simple spinner */
        .spinner {
            width: 40px; height: 40px;
            border: 4px solid rgba(59, 130, 246, 0.3);
            border-left-color: #3b82f6; border-radius: 50%;
            animation: spin 1s linear infinite; margin-bottom: 20px;
        }
        @keyframes spin { 100% { transform: rotate(360deg); } }
        
        iframe {
            border: none; width: 100%; height: 100%;
            position: absolute; top: 0; left: 0; z-index: 5;
            padding-bottom: env(safe-area-inset-bottom);
        }
    </style>
</head>
<body>
    <div id="loader">
        <div class="spinner"></div>
        <div style="font-weight: 600; font-size: 1.1rem; letter-spacing: 1px;">CONNECTING TO SCANNER</div>
        <div style="color: #64748b; font-size: 0.9rem; margin-top: 10px;">Ensuring secure uplink...</div>
    </div>
    <iframe src="http://{HOST_IP}:{APP_PORT}/?embed=true" allow="microphone; autoplay; encrypted-media; picture-in-picture"></iframe>
    <script>
        // Once iframe loads, hide the splash screen loader
        const iframe = document.querySelector('iframe');
        iframe.onload = () => {
            setTimeout(() => { document.getElementById('loader').style.display = 'none'; }, 800);
        };
        // Register service worker for offline app capability
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/sw.js');
        }
    </script>
</body>
</html>
"""

MANIFEST_JSON = """{
  "name": "Voice Threat Scanner",
  "short_name": "Threat Scanner",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#020617",
  "theme_color": "#020617",
  "orientation": "portrait",
  "icons": [
    {
      "src": "/icon.png",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "/icon.png",
      "sizes": "512x512",
      "type": "image/png"
    }
  ]
}"""

SW_JS = """
self.addEventListener('fetch', function(event) {});
"""

def get_host_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def setup_pwa_directory(ip, streamlit_port):
    pwa_dir = os.path.join(os.path.dirname(__file__), "mobile_client")
    os.makedirs(pwa_dir, exist_ok=True)
    
    with open(os.path.join(pwa_dir, "index.html"), "w") as f:
        f.write(HTML_TEMPLATE.replace("{HOST_IP}", ip).replace("{APP_PORT}", str(streamlit_port)))
    
    with open(os.path.join(pwa_dir, "manifest.json"), "w") as f:
        f.write(MANIFEST_JSON)
        
    with open(os.path.join(pwa_dir, "sw.js"), "w") as f:
        f.write(SW_JS)
        
    return pwa_dir

def start_mobile_server(pwa_dir, port=8000):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=pwa_dir, **kwargs)

        def log_message(self, format, *args):
            pass  # suppress normal logs
            
    server = socketserver.TCPServer(("", port), Handler)
    server.serve_forever()

def main():
    print("=" * 60)
    print(" 📱 Voice Threat Scanner — Mobile Deployment Protocol")
    print("=" * 60)
    print("Configuring network and generating app...")

    host_ip = get_host_ip()
    streamlit_port = 8501
    mobile_port = 8080

    # Ensure streamlit is configured to allow CORS and framing
    os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"

    pwa_dir = setup_pwa_directory(host_ip, streamlit_port)

    # Note: Using the provided icon concept as a placeholder if no file is present
    icon_path = os.path.join(pwa_dir, "icon.png")
    if not os.path.exists(icon_path):
        import shutil
        # Coping the shield as a placeholder icon
        default_icon = os.path.join(os.path.dirname(__file__), "docs", "icons", "cyber_shield.png")
        if os.path.exists(default_icon):
            shutil.copy(default_icon, icon_path)
            
    # Start streamlit in background if it's not running
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{streamlit_port}").getcode()
    except Exception:
        print("Starting Deepfake Analytics engine...")
        subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "src/ui/dashboard.py", "--server.port", str(streamlit_port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(2)

    # Start mobile bridge server
    threading.Thread(target=start_mobile_server, args=(pwa_dir, mobile_port), daemon=True).start()

    url = f"http://{host_ip}:{mobile_port}"
    
    print("\n✅ Mobile App Setup Complete!")
    print(f"Network App URL: {url}")
    print("\nScan this QR code with your mobile phone camera:")
    qr = qrcode.QRCode(version=1, box_size=1, border=1)
    qr.add_data(url)
    qr.make(fit=True)
    qr.print_tty()

    print("\n📱 Instructions on phone:")
    print("1. Open the QR code link in Chrome/Safari")
    print("2. You will see the new 'Voice Threat Scanner' App Splash Screen")
    print("3. Tap 'Share' > 'Add to Home Screen' to install the app permanently!")
    
    print("\nKeep this terminal open while using your phone.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down Mobile Server.")

if __name__ == "__main__":
    main()
