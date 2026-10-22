"""
Local Development Server for testing the Vercel app locally.
Serves static files from public/ and routes /api/predict to the Vercel serverless handler.
Run: python dev_server.py [port]
"""

import http.server
import socketserver
import os
import sys
import json
import time
import urllib.parse
from api.predict import run_inference, get_model_data

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 3000

class LocalVercelDispatcher(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="public", **kwargs)

    def _send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors()
        self.end_headers()

    def do_HEAD(self):
        if self.path.startswith("/api/"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.end_headers()
            return
        super().do_HEAD()

    def do_GET(self):
        if self.path.startswith("/api/predict"):
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)

            if "meta" in params:
                data = get_model_data()
                payload = json.dumps({"status": "healthy", "metadata": data["metadata"]}).encode("utf-8")
            else:
                model_data = get_model_data()
                payload = json.dumps({
                    "service": "9-Sensor TinyML Predictor (Local Dev Simulator)",
                    "deployed_on": "Vercel Serverless Functions",
                    "usage": {"method": "POST", "path": "/api/predict"},
                    "metadata": model_data["metadata"]
                }, indent=2).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors()
            self.end_headers()
            self.wfile.write(payload)
            return

        super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/predict"):
            t0 = time.perf_counter()
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            try:
                data = json.loads(body.decode("utf-8"))
            except Exception as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(json.dumps({"error": f"Invalid JSON: {e}"}).encode("utf-8"))
                return

            sensors = data.get("sensors")
            if not sensors or len(sensors) != 9:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Must supply exactly 9 sensors in 'sensors' array"}).encode("utf-8"))
                return

            try:
                res = run_inference([float(x) for x in sensors])
                t1 = time.perf_counter()
                res["latency_ms"] = round((t1 - t0) * 1000.0, 2)
                res["sensor_inputs"] = sensors

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(json.dumps(res).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self._send_cors()
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        self.send_error(404, "Endpoint not found")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with socketserver.TCPServer(("", PORT), LocalVercelDispatcher) as httpd:
        print(f"Local Vercel Dev Server running at: http://localhost:{PORT}")
        print(f"   - Web UI: http://localhost:{PORT}")
        print(f"   - API:    http://localhost:{PORT}/api/predict")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
