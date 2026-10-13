import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from cavalier.agents.llm_client import OllamaClient


def _server(reject_think: bool):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(body)
            if reject_think and "think" in body:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error": "model does not support think"}')
                return
            payload = {"message": {"content": '{"ok": true}'}, "eval_count": 50, "eval_duration": 1_000_000_000,
                       "prompt_eval_count": 900, "load_duration": 0, "done_reason": "stop"}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, received


def test_sends_schema_without_thinking_and_records_metrics():
    server, received = _server(reject_think=False)
    client = OllamaClient("m", base_url=f"http://127.0.0.1:{server.server_port}/v1", seed=3, num_ctx=4096)
    assert client.complete_json("sys", "user", {"type": "object"}) == '{"ok": true}'
    body = received[0]
    assert body["think"] is False and body["format"] == {"type": "object"}
    assert body["options"]["seed"] == 3 and body["options"]["num_ctx"] == 4096
    assert client.metrics[0]["output_tokens"] == 50 and client.metrics[0]["tokens_per_s"] == 50.0
    server.shutdown()


def test_retries_without_think_when_model_rejects_it():
    server, received = _server(reject_think=True)
    client = OllamaClient("m", base_url=f"http://127.0.0.1:{server.server_port}")
    assert client.complete_json("sys", "user", {}) == '{"ok": true}'
    assert "think" in received[0] and "think" not in received[1]
    server.shutdown()
