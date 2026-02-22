#!/usr/bin/env python3
"""
Simple HTTP server for Tuti Parrot web app
Run from the web/ directory
"""

import http.server
import socketserver
import os

PORT = 8000

# Change to the web directory
web_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(web_dir)

Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║          🦜 Tuti Parrot Web Server                       ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

Server running at: http://localhost:{PORT}

Open your browser and visit: http://localhost:{PORT}

Press Ctrl+C to stop the server.
""")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nServer stopped. Goodbye! 🦜")
