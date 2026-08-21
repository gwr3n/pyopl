"""Local IPC bridge between a running Rhetor IDE and MCP clients."""

from __future__ import annotations

import json
import os
import queue
import secrets
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional

from platformdirs import user_config_dir

BRIDGE_INFO_PATH = Path(user_config_dir("rhetor")) / "ide-mcp.json"
BRIDGE_HOST = "127.0.0.1"
BRIDGE_REQUEST_TIMEOUT = 10.0


class _BridgeHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


@dataclass
class BridgeRequest:
    """A request waiting to be handled by the IDE event loop."""

    method: str
    path: str
    payload: Optional[dict[str, Any]]
    completed: threading.Event = field(default_factory=threading.Event)
    response: Optional[dict[str, Any]] = None
    error: Optional[BaseException] = None


class RhetorIDEBridge:
    """Serve local editor requests and queue them for the Tk event loop."""

    def __init__(self, info_path: Path = BRIDGE_INFO_PATH) -> None:
        self.info_path = info_path
        self.token = secrets.token_urlsafe(32)
        self._requests: queue.Queue[BridgeRequest] = queue.Queue()
        self._server: Optional[_BridgeHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the loopback server and publish connection details."""
        bridge = self

        class RequestHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self._dispatch()

            def do_PUT(self) -> None:
                self._dispatch()

            def _dispatch(self) -> None:
                if self.headers.get("Authorization") != f"Bearer {bridge.token}":
                    self._send_json(403, {"error": "Invalid Rhetor IDE bridge token"})
                    return

                payload = None
                if self.command == "PUT":
                    try:
                        content_length = int(self.headers.get("Content-Length", "0"))
                        decoded = json.loads(self.rfile.read(content_length) or b"{}")
                        if not isinstance(decoded, dict):
                            raise ValueError("Request body must be a JSON object")
                        payload = decoded
                    except (ValueError, json.JSONDecodeError) as exc:
                        self._send_json(400, {"error": str(exc)})
                        return

                request = BridgeRequest(self.command, self.path, payload)
                bridge._requests.put(request)
                if not request.completed.wait(BRIDGE_REQUEST_TIMEOUT):
                    self._send_json(504, {"error": "Rhetor IDE did not respond in time"})
                elif request.error is not None:
                    self._send_json(400, {"error": str(request.error)})
                else:
                    self._send_json(200, request.response or {})

            def _send_json(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                return

        self._server = _BridgeHTTPServer((BRIDGE_HOST, 0), RequestHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, name="rhetor-ide-mcp", daemon=True)
        self._thread.start()
        self._publish_connection_info(self._server.server_port)

    def process_pending(self, handler: Callable[[str, str, Optional[dict[str, Any]]], dict[str, Any]]) -> int:
        """Process all queued requests on the caller's thread."""
        processed = 0
        while True:
            try:
                request = self._requests.get_nowait()
            except queue.Empty:
                return processed
            try:
                request.response = handler(request.method, request.path, request.payload)
            except BaseException as exc:
                request.error = exc
            finally:
                request.completed.set()
                processed += 1

    def stop(self) -> None:
        """Stop serving and remove this instance's connection details."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        try:
            published = json.loads(self.info_path.read_text(encoding="utf-8"))
            if published.get("token") == self.token:
                self.info_path.unlink(missing_ok=True)
        except (OSError, ValueError, AttributeError):
            pass

    def _publish_connection_info(self, port: int) -> None:
        self.info_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.info_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps({"host": BRIDGE_HOST, "port": port, "token": self.token, "pid": os.getpid()}),
            encoding="utf-8",
        )
        os.chmod(temporary_path, 0o600)
        temporary_path.replace(self.info_path)
