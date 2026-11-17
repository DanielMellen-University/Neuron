#!/usr/bin/env python3
"""Model-free socket tests against the same server used by neuron serve."""
import http.client
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import unittest

BINARY = sys.argv.pop(1)


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.process = subprocess.Popen([BINARY], stdout=subprocess.PIPE, text=True)
        cls.port = int(cls.process.stdout.readline())

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        cls.process.wait(timeout=10)
        cls.process.stdout.close()

    def request(self, method, path, body=None, content_type="application/json"):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request(method, path, body, {"Content-Type": content_type})
        response = conn.getresponse()
        status, headers, data = response.status, dict(response.getheaders()), response.read()
        conn.close()
        return status, headers, data

    def chat(self, **kwargs):
        payload = dict(model="fake-model", messages=[dict(role="user", content="Hi")], max_tokens=16)
        payload.update(kwargs)
        return self.request("POST", "/v1/chat/completions", json.dumps(payload))

    def raw(self, request, fragmented=False):
        with socket.create_connection(("127.0.0.1", self.port), timeout=10) as conn:
            if fragmented:
                for i in range(0, len(request), 7):
                    conn.sendall(request[i:i + 7])
            else:
                conn.sendall(request)
            data = b""
            while True:
                part = conn.recv(4096)
                if not part:
                    return data
                data += part

    def test_models_and_embedded_page(self):
        sockets = Path("/proc/net/tcp").read_text().splitlines()
        self.assertTrue(any(line.split()[1] == f"0100007F:{self.port:04X}" for line in sockets[1:]))
        status, _, data = self.request("GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["data"][0]["id"], "fake-model")
        status, headers, data = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertIn(b"fetch('/v1/models')", data)
        self.assertNotIn(b"cdn", data)
        self.assertNotIn(b"<script src=", data)

    def test_nonstreaming_usage_and_history(self):
        status, _, body = self.chat(messages=[dict(role="user", content="hi"), dict(role="assistant", content="hello"), dict(role="user", content="again")])
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["object"], "chat.completion")
        self.assertEqual(data["choices"][0]["message"], dict(role="assistant", content="history=3: hello ☃\n"))
        self.assertEqual(data["choices"][0]["finish_reason"], "stop")
        self.assertEqual(data["usage"], dict(prompt_tokens=12, completion_tokens=3, total_tokens=15))

    def test_sse_arrives_before_generation_finishes(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        payload = dict(model="fake-model", messages=[dict(role="user", content="slow")], max_tokens=16, stream=True, stream_options=dict(include_usage=True))
        conn.request("POST", "/v1/chat/completions", json.dumps(payload), {"Content-Type": "application/json"})
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Type"), "text/event-stream")
        started = time.monotonic()
        first = json.loads(response.readline().decode()[6:])
        self.assertLess(time.monotonic() - started, 0.9)
        self.assertEqual(first["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual(response.readline(), b"\n")
        frames = response.read().decode().split("\n\n")
        chunks = [json.loads(frame[6:]) for frame in frames if frame.startswith("data: {")]
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
        self.assertEqual(text, "history=1: hello ☃\n")
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(chunks[-1]["usage"]["completion_tokens"], 3)
        self.assertIn("data: [DONE]", frames)
        self.assertTrue(all(c["id"] == first["id"] for c in chunks))
        conn.close()

    def test_errors_before_sse_headers(self):
        for stream in (False, True):
            for kwargs, expected, code in [
                (dict(model="missing"), 404, "model_not_found"),
                (dict(max_tokens=256), 400, "context_length_exceeded"),
                (dict(messages=[dict(role="user", content="x" * 257)]), 400, "context_length_exceeded"),
                (dict(max_tokens=-1), 400, "invalid_request_error"),
                (dict(messages=[]), 400, "invalid_request_error"),
                (dict(messages=[dict(role="tool", content="x")]), 400, "invalid_request_error"),
                (dict(messages=[dict(role="user", content=None)]), 400, "invalid_request_error"),
                (dict(temperature=1), 400, "invalid_request_error"),
                (dict(n=2), 400, "invalid_request_error"),
            ]:
                with self.subTest(stream=stream, kwargs=kwargs):
                    status, headers, data = self.chat(stream=stream, **kwargs)
                    self.assertEqual(status, expected)
                    self.assertEqual(headers["Content-Type"], "application/json")
                    self.assertEqual(json.loads(data)["error"]["code"], code)

    def test_invalid_json_and_schema(self):
        for body in [b'{', b'{}garbage', b'[]', b'{"model":"fake-model","messages":true}', b'{"x":NaN}', b'{"x":01}', b'{"x":"\\ud800"}', b'{"x":"\xff"}', b'{"x":1,"x":2}', b'{"x":' + b'[' * 70 + b'0' + b']' * 70 + b'}']:
            with self.subTest(body=body[:50]):
                status, _, data = self.request("POST", "/v1/chat/completions", body)
                self.assertEqual(status, 400)
                self.assertIn("error", json.loads(data))
        self.assertEqual(self.request("POST", "/v1/chat/completions", "{}", "text/plain")[0], 415)

    def test_fragmented_http_and_unicode_escapes(self):
        body = b'{"model":"fake-model","messages":[{"role":"user","content":"\\ud83d\\ude00\\u0000\\n"}],"max_tokens":16}'
        request = b'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\ncontent-type: application/json; charset=utf-8\r\ncontent-length: ' + str(len(body)).encode() + b'\r\n\r\n' + body
        response = self.raw(request, fragmented=True)
        self.assertIn(b'200 OK', response)
        self.assertEqual(json.loads(response.split(b'\r\n\r\n')[1])["usage"]["prompt_tokens"], 6)

    def test_http_framing_errors_and_limits(self):
        for headers, expected in [
            (b'Content-Length: 0\r\nContent-Length: 2', 400),
            (b'Transfer-Encoding: chunked', 400),
            (b'Content-Length: -1', 400),
            (b'Content-Length: 1048577', 413),
            (b'Bad Header: value', 400),
            (b'X-Padding: ' + b'x' * 17000, 431),
        ]:
            response = self.raw(b'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n' + headers + b'\r\n\r\n')
            self.assertTrue(response.startswith(f'HTTP/1.1 {expected}'.encode()))
        self.assertEqual(self.request("GET", "/missing")[0], 404)
        self.assertEqual(self.request("GET", "/v1/chat/completions")[0], 405)

    def test_backend_failure_and_disconnect_recovery(self):
        status, _, data = self.chat(messages=[dict(role="user", content="fail")])
        self.assertEqual(status, 500)
        self.assertEqual(json.loads(data)["error"]["code"], "internal_error")
        status, _, data = self.chat(messages=[dict(role="user", content="fail")], stream=True)
        self.assertEqual(status, 200)
        self.assertIn(b'"internal_error"', data)
        self.assertIn(b'data: [DONE]', data)
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("POST", "/v1/chat/completions", json.dumps(dict(model="fake-model", messages=[dict(role="user", content="slow")], max_tokens=16, stream=True)), {"Content-Type": "application/json"})
        response = conn.getresponse()
        response.readline()
        response.close()
        conn.close()
        self.assertEqual(self.chat()[0], 200)


if __name__ == "__main__":
    unittest.main()
