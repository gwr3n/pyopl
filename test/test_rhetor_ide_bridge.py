import json
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory

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


if __name__ == "__main__":
    unittest.main()
