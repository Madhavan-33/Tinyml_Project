"""
Vercel Serverless Function: TinyML Anomaly & Object Sizing Predictor
Provides ultra-fast (<5ms) inference endpoint for 9-Sensor Mesh telemetry.
"""

from http.server import BaseHTTPRequestHandler
import json
import os
import time
import urllib.parse

# Load model weights on module import (reused across warm serverless invocations)
_MODEL_DATA = None

def get_model_data():
    global _MODEL_DATA
    if _MODEL_DATA is None:
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "model_weights.json"),
            os.path.join(os.path.dirname(__file__), "..", "model_weights.json"),
            "api/model_weights.json",
            "model_weights.json"
        ]
        loaded = None
        for p in possible_paths:
            if os.path.exists(p):
                with open(p, "r") as f:
                    loaded = json.load(f)
                break

        if loaded is None:
            raise FileNotFoundError("Could not find model_weights.json in function bundle.")
        _MODEL_DATA = loaded
    return _MODEL_DATA

def run_inference(sensors):
    """
    Executes forward pass using NumPy (or pure python fallback if numpy missing).
    """
    model = get_model_data()
    meta = model["metadata"]
    weights = model["weights"]

    shape_classes = meta["shape_classes"]
    means = meta["sensor_means"]
    stds = meta["sensor_stds"]

    try:
        import numpy as np

        x = np.array(sensors, dtype=np.float32)
        m = np.array(means, dtype=np.float32)
        s = np.array(stds, dtype=np.float32)
        x_scaled = (x - m) / s

        w1 = np.array(weights["w1"], dtype=np.float32)
        b1 = np.array(weights["b1"], dtype=np.float32)
        gamma = np.array(weights["bn_gamma"], dtype=np.float32)
        beta = np.array(weights["bn_beta"], dtype=np.float32)
        bn_mean = np.array(weights["bn_mean"], dtype=np.float32)
        bn_var = np.array(weights["bn_var"], dtype=np.float32)
        bn_eps = float(weights["bn_eps"])

        w2 = np.array(weights["w2"], dtype=np.float32)
        b2 = np.array(weights["b2"], dtype=np.float32)
        w3 = np.array(weights["w3"], dtype=np.float32)
        b3 = np.array(weights["b3"], dtype=np.float32)
        w_shape = np.array(weights["w_shape"], dtype=np.float32)
        b_shape = np.array(weights["b_shape"], dtype=np.float32)
        w_dim = np.array(weights["w_dim"], dtype=np.float32)
        b_dim = np.array(weights["b_dim"], dtype=np.float32)

        # Forward pass
        h1 = np.maximum(0, np.dot(x_scaled, w1) + b1)
        bn1 = gamma * (h1 - bn_mean) / np.sqrt(bn_var + bn_eps) + beta
        h2 = np.maximum(0, np.dot(bn1, w2) + b2)
        h3 = np.maximum(0, np.dot(h2, w3) + b3)

        logits = np.dot(h3, w_shape) + b_shape
        e = np.exp(logits - np.max(logits))
        probs = (e / np.sum(e)).tolist()
        dim_mm = float(np.dot(h3, w_dim)[0] + b_dim[0])

    except ImportError:
        # Pure Python fallback
        import math
        x_scaled = [(sensors[i] - means[i]) / stds[i] for i in range(9)]
        
        # Dense 1
        h1 = []
        for j in range(len(weights["b1"])):
            val = sum(x_scaled[i] * weights["w1"][i][j] for i in range(9)) + weights["b1"][j]
            h1.append(max(0.0, val))

        # BatchNorm
        bn1 = []
        for j in range(len(h1)):
            v = weights["bn_gamma"][j] * (h1[j] - weights["bn_mean"][j]) / math.sqrt(weights["bn_var"][j] + weights["bn_eps"]) + weights["bn_beta"][j]
            bn1.append(v)

        # Dense 2
        h2 = []
        for j in range(len(weights["b2"])):
            val = sum(bn1[i] * weights["w2"][i][j] for i in range(len(bn1))) + weights["b2"][j]
            h2.append(max(0.0, val))

        # Dense 3
        h3 = []
        for j in range(len(weights["b3"])):
            val = sum(h2[i] * weights["w3"][i][j] for i in range(len(h2))) + weights["b3"][j]
            h3.append(max(0.0, val))

        # Shape Head (Softmax)
        logits = []
        for j in range(len(weights["b_shape"])):
            val = sum(h3[i] * weights["w_shape"][i][j] for i in range(len(h3))) + weights["b_shape"][j]
            logits.append(val)
        max_l = max(logits)
        exp_l = [math.exp(l - max_l) for l in logits]
        sum_exp = sum(exp_l)
        probs = [e / sum_exp for e in exp_l]

        # Dim Head (Linear)
        dim_mm = sum(h3[i] * weights["w_dim"][i][0] for i in range(len(h3))) + weights["b_dim"][0]

    best_idx = max(range(len(probs)), key=lambda i: probs[i])
    shape_name = shape_classes[best_idx]
    confidence = float(probs[best_idx] * 100.0)

    prob_dict = {shape_classes[i]: round(float(probs[i]), 4) for i in range(len(shape_classes))}

    return {
        "predicted_shape": shape_name,
        "confidence_percentage": round(confidence, 2),
        "estimated_dimension_mm": round(float(dim_mm), 2),
        "shape_probabilities": prob_dict,
        "classes": shape_classes
    }

class handler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        # Health / metadata route
        if "meta" in params or parsed.path.endswith("/health"):
            data = get_model_data()
            response_payload = {
                "status": "healthy",
                "service": "9-Sensor TinyML Vercel Serverless Predictor",
                "metadata": data["metadata"]
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(response_payload).encode("utf-8"))
            return

        # Check for sensor query parameters (e.g. ?s1=1.0&s2=2.0...)
        sensor_vals = []
        has_sensors = True
        for i in range(1, 10):
            key = f"s{i}"
            if key in params:
                try:
                    sensor_vals.append(float(params[key][0]))
                except ValueError:
                    has_sensors = False
            elif key.upper() in params:
                try:
                    sensor_vals.append(float(params[key.upper()][0]))
                except ValueError:
                    has_sensors = False
            else:
                has_sensors = False
                break

        if has_sensors and len(sensor_vals) == 9:
            t0 = time.perf_counter()
            result = run_inference(sensor_vals)
            t1 = time.perf_counter()
            result["latency_ms"] = round((t1 - t0) * 1000.0, 2)
            result["sensor_inputs"] = sensor_vals

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
            return

        # Default info endpoint
        model_data = get_model_data()
        sample_request = {"sensors": [1.19, 4.47, 1.98, 2.90, 5.30, 3.68, 4.47, 6.14, 5.42]}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps({
            "service": "9-Sensor TinyML Object & Dimension Predictor",
            "deployed_on": "Vercel Serverless Functions",
            "usage": {
                "method": "POST",
                "path": "/api/predict",
                "body_format": sample_request
            },
            "metadata": model_data["metadata"]
        }, indent=2).encode("utf-8"))

    def do_POST(self):
        t0 = time.perf_counter()
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Invalid JSON payload: {str(e)}"}).encode("utf-8"))
            return

        # Extract 9 sensor values from either {"sensors": [...]} or {"S1": ..., "S2": ...}
        sensors = None
        if "sensors" in payload and isinstance(payload["sensors"], list):
            if len(payload["sensors"]) == 9:
                sensors = [float(x) for x in payload["sensors"]]
        elif "s" in payload and isinstance(payload["s"], list) and len(payload["s"]) == 9:
            sensors = [float(x) for x in payload["s"]]
        else:
            # Check individual keys
            candidate = []
            for i in range(1, 10):
                k1 = f"S{i}"
                k2 = f"s{i}"
                if k1 in payload:
                    candidate.append(float(payload[k1]))
                elif k2 in payload:
                    candidate.append(float(payload[k2]))
            if len(candidate) == 9:
                sensors = candidate

        if sensors is None or len(sensors) != 9:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": "Input must contain exactly 9 sensor values: [S1, S2, ..., S9]",
                "received": payload
            }).encode("utf-8"))
            return

        try:
            result = run_inference(sensors)
            t1 = time.perf_counter()
            result["latency_ms"] = round((t1 - t0) * 1000.0, 2)
            result["sensor_inputs"] = sensors

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Prediction error: {str(e)}"}).encode("utf-8"))
