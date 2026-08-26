import json
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pyopl.rhetor_ide_bridge as bridge_module
from pyopl.rhetor_ide_bridge import RhetorIDEBridge


class TestRhetorIDEBridge(unittest.TestCase):
    def test_authenticated_request_is_dispatched_and_connection_info_is_removed(self):
        with TemporaryDirectory() as tmp_dir:
            info_path = Path(tmp_dir) / "ide-mcp.json"
            bridge = RhetorIDEBridge(info_path)
            bridge.start()
            info = json.loads(info_path.read_text(encoding="utf-8"))
            response_holder = {}

            def request_bridge():
                request = urllib.request.Request(
                    f"http://{info['host']}:{info['port']}/editors",
                    headers={"Authorization": f"Bearer {info['token']}"},
                )
                with urllib.request.urlopen(request) as response:
                    response_holder.update(json.load(response))

            request_thread = threading.Thread(target=request_bridge)
            request_thread.start()
            while request_thread.is_alive():
                bridge.process_pending(lambda method, path, payload: {"method": method, "path": path})
                request_thread.join(timeout=0.01)

            bridge.stop()

            self.assertEqual(response_holder, {"method": "GET", "path": "/editors"})
            self.assertFalse(info_path.exists())

    def test_invalid_token_is_rejected_without_dispatch(self):
        with TemporaryDirectory() as tmp_dir:
            info_path = Path(tmp_dir) / "ide-mcp.json"
            bridge = RhetorIDEBridge(info_path)
            bridge.start()
            info = json.loads(info_path.read_text(encoding="utf-8"))
            request = urllib.request.Request(
                f"http://{info['host']}:{info['port']}/editors",
                headers={"Authorization": "Bearer wrong"},
            )

            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(request)

            bridge.stop()
            self.assertEqual(raised.exception.code, 403)

    def test_authenticated_put_dispatches_json_payload(self):
        with TemporaryDirectory() as tmp_dir:
            info_path = Path(tmp_dir) / "ide-mcp.json"
            bridge = RhetorIDEBridge(info_path)
            bridge.start()
            info = json.loads(info_path.read_text(encoding="utf-8"))
            response_holder = {}

            def request_bridge():
                request = urllib.request.Request(
                    f"http://{info['host']}:{info['port']}/editors",
                    data=json.dumps({"model_text": "model"}).encode("utf-8"),
                    method="PUT",
                    headers={"Authorization": f"Bearer {info['token']}"},
                )
                with urllib.request.urlopen(request) as response:
                    response_holder.update(json.load(response))

            request_thread = threading.Thread(target=request_bridge)
            request_thread.start()
            while request_thread.is_alive():
                bridge.process_pending(
                    lambda method, path, payload: {"method": method, "path": path, "payload": payload}
                )
                request_thread.join(timeout=0.01)

            bridge.stop()

        self.assertEqual(response_holder["method"], "PUT")
        self.assertEqual(response_holder["payload"], {"model_text": "model"})

    def test_put_rejects_malformed_and_non_object_json(self):
        with TemporaryDirectory() as tmp_dir:
            bridge = RhetorIDEBridge(Path(tmp_dir) / "ide-mcp.json")
            bridge.start()
            info = json.loads(bridge.info_path.read_text(encoding="utf-8"))

            for body in (b"not json", b"[]"):
                request = urllib.request.Request(
                    f"http://{info['host']}:{info['port']}/editors",
                    data=body,
                    method="PUT",
                    headers={"Authorization": f"Bearer {info['token']}"},
                )
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    urllib.request.urlopen(request)
                self.assertEqual(raised.exception.code, 400)

            bridge.stop()

    def test_process_pending_reports_handler_errors_to_client(self):
        with TemporaryDirectory() as tmp_dir:
            bridge = RhetorIDEBridge(Path(tmp_dir) / "ide-mcp.json")
            bridge.start()
            info = json.loads(bridge.info_path.read_text(encoding="utf-8"))
            request = urllib.request.Request(
                f"http://{info['host']}:{info['port']}/editors",
                headers={"Authorization": f"Bearer {info['token']}"},
            )
            response_holder = {}

            def request_bridge():
                try:
                    urllib.request.urlopen(request)
                except urllib.error.HTTPError as exc:
                    response_holder["code"] = exc.code

            request_thread = threading.Thread(target=request_bridge)
            request_thread.start()
            while request_thread.is_alive():
                bridge.process_pending(lambda *_: (_ for _ in ()).throw(ValueError("bad request")))
                request_thread.join(timeout=0.01)

            bridge.stop()

        self.assertEqual(response_holder, {"code": 400})

    def test_request_times_out_when_event_loop_does_not_process_it(self):
        with TemporaryDirectory() as tmp_dir:
            info_path = Path(tmp_dir) / "ide-mcp.json"
            bridge = RhetorIDEBridge(info_path)
            bridge.start()
            info = json.loads(info_path.read_text(encoding="utf-8"))
            request = urllib.request.Request(
                f"http://{info['host']}:{info['port']}/editors",
                headers={"Authorization": f"Bearer {info['token']}"},
            )

            with self.assertRaises(urllib.error.HTTPError) as raised, patch.object(
                bridge_module, "BRIDGE_REQUEST_TIMEOUT", 0.01
            ):
                urllib.request.urlopen(request)

            bridge.stop()

        self.assertEqual(raised.exception.code, 504)

    def test_stop_ignores_unreadable_or_foreign_connection_info(self):
        with TemporaryDirectory() as tmp_dir:
            info_path = Path(tmp_dir) / "ide-mcp.json"
            bridge = RhetorIDEBridge(info_path)
            info_path.write_text("not json", encoding="utf-8")
            bridge.stop()
            info_path.write_text(json.dumps({"token": "other"}), encoding="utf-8")
            bridge.stop()
            self.assertTrue(info_path.exists())


if __name__ == "__main__":
    unittest.main()
