from __future__ import annotations

import argparse
import sys
import traceback
import threading
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler

# =========================
# SAFE WRAPPER
# =========================

def safe_main():
    try:
        return main()
    except Exception:
        with open("error.log", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print("ERROR: see error.log")
        return 1


# =========================
# DIAGNOSE
# =========================

def run_diagnose() -> int:
    print("Running diagnose...")

    try:
        import jax  # проверка зависимости
        print(f"JAX OK: {jax.__version__}")
    except Exception as e:
        print(f"JAX ERROR: {e}")
        return 1

    print("Diagnose OK")
    return 0


# =========================
# SIMPLE SERVER
# =========================

class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"NeuroPOMDP server is running")

    def log_message(self, format, *args):
        return  # убрать спам


def run_server(port: int) -> int:
    server = HTTPServer(("127.0.0.1", port), SimpleHandler)

    print(f"Server started at http://127.0.0.1:{port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

    return 0


# =========================
# GUI MODE
# =========================

def run_gui(port: int) -> int:
    def start_server():
        run_server(port)

    thread = threading.Thread(target=start_server, daemon=True)
    thread.start()

    time.sleep(2)

    url = f"http://127.0.0.1:{port}"
    print(f"Opening browser: {url}")
    webbrowser.open(url)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        return 0


# =========================
# SINGLE INSTANCE LOCK
# =========================

import socket

def ensure_single_instance(port: int) -> bool:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", port))
        sock.close()
        return True
    except OSError:
        return False


# =========================
# CLI
# =========================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="NeuroPOMDP")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("diagnose")

    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=45678)

    gui = sub.add_parser("gui")
    gui.add_argument("--port", type=int, default=45678)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # SINGLE INSTANCE CHECK
    if not ensure_single_instance(65432):
        print("Another instance is already running")
        return 1

    if args.command == "diagnose":
        return run_diagnose()

    elif args.command == "serve":
        return run_server(args.port)

    elif args.command == "gui":
        return run_gui(args.port)

    else:
        print("Unknown command")
        return 1


if __name__ == "__main__":
    sys.exit(safe_main())