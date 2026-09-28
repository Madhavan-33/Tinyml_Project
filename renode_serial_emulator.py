"""
Renode TCP UART Server Simulator
Simulates Renode's sysbus.usart1 external socket terminal on port 12345.
Allows testing Streamlit PySerial tab (Tab 4) instantly without compiling firmware.elf!
"""

import socket
import time
import random

HOST = '127.0.0.1'
PORT = 12345

FEATURE_MEANS = [25.0, 0.5, 0.05, 0.2]
FEATURE_STDS = [1.5, 0.1, 0.01, 0.04]
THRESHOLD = 0.08

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)

    print("==================================================")
    print(f"🚀 Renode TCP UART Socket Server Listening on {HOST}:{PORT}")
    print(f"💡 Open Streamlit Tab 4 and connect to socket://localhost:{PORT}")
    print("==================================================")

    while True:
        print("\nWaiting for Streamlit / PySerial connection...")
        conn, addr = server.accept()
        print(f"✅ Connected by Streamlit client: {addr}")

        sample_count = 0
        try:
            while True:
                sample_count += 1
                inject_anomaly = (sample_count % 12 == 0)

                v0 = FEATURE_MEANS[0] + random.uniform(-0.5, 0.5)
                v1 = FEATURE_STDS[1] + random.uniform(-0.05, 0.05)

                if inject_anomaly:
                    v0 += 5.0  # Anomaly spike in temperature
                    v1 += 0.8  # Anomaly spike in vibration
                    mse = random.uniform(0.12, 0.35)
                    status = "ANOMALY"
                    health = max(0.0, 100.0 - (mse / THRESHOLD) * 25.0)
                else:
                    mse = random.uniform(0.0001, 0.0040)
                    status = "NORMAL"
                    health = min(100.0, max(85.0, 100.0 - (mse / THRESHOLD) * 20.0))

                telemetry = f"[RENODE_MCU] SAMPLE={sample_count}, MSE={mse:.6f}, STATUS={status}, HEALTH={health:.2f}, V0={v0:.4f}, V1={v1:.4f}\n"
                conn.sendall(telemetry.encode('utf-8'))
                print(f"Sent: {telemetry.strip()}")
                time.sleep(0.5)

        except (socket.error, ConnectionResetError, BrokenPipeError) as e:
            print(f"Client disconnected: {e}")
            conn.close()

if __name__ == "__main__":
    start_server()
