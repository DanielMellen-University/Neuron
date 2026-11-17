#!/usr/bin/env python3
"""Check the official OpenAI client against a real Neuron model. Exit 77 to skip."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build/neuron"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.model is None:
        args.model = Path(os.environ.get("NEURON_MODELS_DIR", "models")) / "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
    if not args.model.is_file():
        print("SKIP: model is unavailable; set NEURON_MODELS_DIR or --model")
        return 77
    try:
        from openai import OpenAI, BadRequestError, NotFoundError
    except ImportError:
        print("SKIP: the openai Python package is unavailable")
        return 77
    model = args.model
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    command = [str(args.binary.resolve()), "serve", "-m", str(model.resolve()), "--port", str(port), "--threads", str(args.threads), "--ctx-size", "512"]
    started = time.monotonic()
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(command, stdout=log, stderr=log)
        try:
            while True:
                if process.poll() is not None or time.monotonic() - started > 120:
                    log.seek(0)
                    raise RuntimeError("server did not start: " + log.read().decode(errors="replace"))
                try:
                    with urllib.request.urlopen(url + "/v1/models", timeout=1) as response:
                        json.load(response)
                    break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.1)
            # No adapters, patched transports or custom client methods.
            with OpenAI(base_url=url + "/v1", api_key="local", timeout=180, max_retries=0) as client:
                models = client.models.list()
                assert models.data[0].id == model.name
                messages = [dict(role="user", content="My name is Ada. Reply with OK.")]
                first = client.chat.completions.create(model=model.name, messages=messages, max_tokens=16, temperature=0)
                assert first.choices[0].message.content
                messages += [dict(role="assistant", content=first.choices[0].message.content), dict(role="user", content="What is my name? Reply with only the name.")]
                reply = client.chat.completions.create(model=model.name, messages=messages, max_tokens=16)
                assert reply.choices[0].message.role == "assistant"
                assert reply.choices[0].finish_reason in ("stop", "length")
                assert reply.usage.total_tokens == reply.usage.prompt_tokens + reply.usage.completion_tokens
                chunks = list(client.chat.completions.create(model=model.name, messages=messages, max_tokens=16, stream=True, stream_options=dict(include_usage=True)))
                text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
                assert text == reply.choices[0].message.content
                assert chunks[0].choices[0].delta.role == "assistant"
                assert chunks[-2].choices[0].finish_reason == reply.choices[0].finish_reason
                assert chunks[-1].usage.total_tokens == reply.usage.total_tokens
                for stream in (False, True):
                    try:
                        client.chat.completions.create(model="missing-model", messages=messages, max_tokens=1, stream=stream)
                        raise AssertionError("unknown model accepted")
                    except NotFoundError as error:
                        assert error.code == "model_not_found"
                    try:
                        client.chat.completions.create(model=model.name, messages=messages, max_tokens=512, stream=stream)
                        raise AssertionError("over-context request accepted")
                    except BadRequestError as error:
                        assert error.code == "context_length_exceeded"
                request = urllib.request.Request(url + "/v1/chat/completions", data=b"{", headers={"Content-Type": "application/json"})
                try:
                    urllib.request.urlopen(request, timeout=10)
                    raise AssertionError("bad JSON accepted")
                except urllib.error.HTTPError as error:
                    assert error.code == 400
                    assert json.load(error)["error"]["code"] == "invalid_json"
                with urllib.request.urlopen(url, timeout=10) as response:
                    page = response.read().decode()
                    assert "<title>Neuron chat</title>" in page and "[DONE]" in page
                result = dict(model=model.name, openai_version=importlib.metadata.version("openai"), threads=args.threads,
                              context=512, first_reply=first.choices[0].message.content, history_reply=text,
                              usage=reply.usage.model_dump(), stream_matches_nonstream=True,
                              errors_checked=["model_not_found", "context_length_exceeded", "invalid_json"],
                              elapsed_seconds=round(time.monotonic() - started, 3), passed=True)
                print(json.dumps(result, indent=2))
                if args.output:
                    args.output.write_text(json.dumps(result, indent=2) + "\n")
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
