#!/usr/bin/env python3
"""Model-free terminal history, reset, EOF and SIGINT checks."""
import os
import select
import signal
import subprocess
import sys
import time
import unittest

BINARY = sys.argv.pop(1)


class TerminalTests(unittest.TestCase):
    def setUp(self):
        self.process = subprocess.Popen([BINARY, "chat"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.pending = b""
        self.until(b"you> ")

    def tearDown(self):
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait(timeout=10)
        self.process.stdin.close()
        self.process.stdout.close()

    def send(self, text):
        self.process.stdin.write(text.encode() + b"\n")
        self.process.stdin.flush()

    def until(self, text):
        deadline = time.monotonic() + 10
        while text not in self.pending:
            self.assertGreater(deadline, time.monotonic(), self.pending)
            if select.select([self.process.stdout], [], [], 0.1)[0]:
                part = os.read(self.process.stdout.fileno(), 4096)
                self.assertTrue(part, self.pending)
                self.pending += part
        end = self.pending.index(text) + len(text)
        result, self.pending = self.pending[:end], self.pending[end:]
        return result

    def test_history_reset_exit(self):
        self.send("Hi")
        self.assertIn(b"history=1", self.until(b"you> "))
        self.send("Again")
        self.assertIn(b"history=3", self.until(b"you> "))
        self.send("/reset")
        self.assertIn(b"History cleared", self.until(b"you> "))
        self.send("New")
        self.assertIn(b"history=1", self.until(b"you> "))
        self.send("/exit")
        self.assertEqual(self.process.wait(timeout=10), 0)

    def test_interrupt_reply_prefill_and_idle_keeps_session(self):
        self.send("slow")
        self.until(b"history=1")
        self.process.send_signal(signal.SIGINT)
        self.assertIn(b"reply stopped", self.until(b"you> "))
        self.send("prefill")
        self.until(b"assistant> ")
        self.process.send_signal(signal.SIGINT)
        self.assertIn(b"reply stopped", self.until(b"you> "))
        self.process.send_signal(signal.SIGINT)
        self.until(b"you> ")
        self.send("Hi")
        self.assertIn(b"hello", self.until(b"you> "))
        self.send("/exit")
        self.assertEqual(self.process.wait(timeout=10), 0)

    def test_context_error_recovers_and_eof(self):
        self.send("x" * 257)
        self.assertIn(b"context limit", self.until(b"you> "))
        self.send("ok")
        self.assertIn(b"history=1", self.until(b"you> "))
        self.process.stdin.close()
        self.assertEqual(self.process.wait(timeout=10), 0)


if __name__ == "__main__":
    unittest.main()
