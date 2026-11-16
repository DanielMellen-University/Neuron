#!/usr/bin/env python3
"""Exercise committed llama.cpp token IDs without weights or llama.cpp."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
from m3_fixture import write_tokenizer_gguf

BINARY = Path(sys.argv.pop(1)).resolve()


def fixture(name):
    return json.loads((ROOT / f'bench/reference/qwen2-{name}.json').read_text())


class QwenFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='neuron-qwen-fixture-')
        cls.model = Path(cls.temporary.name) / 'tokenizer.gguf'
        write_tokenizer_gguf(cls.model, fixture('vocabulary'))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def invoke(self, command, *flags, check=True):
        return subprocess.run([str(BINARY), command, '-m', str(self.model), *flags],
                              capture_output=True, check=check)

    def test_tokenizer_ids(self):
        for case in fixture('tokenizer')['cases']:
            with self.subTest(case=case['id']):
                path = Path(self.temporary.name) / 'prompt.txt'
                path.write_bytes(case['text'].encode('utf-8'))
                flags = ['--prompt-file', str(path)]
                if not case['add_special']:
                    flags.append('--no-add-special')
                if not case['parse_special']:
                    flags.append('--no-parse-special')
                self.assertEqual(json.loads(self.invoke('tokenize', *flags).stdout), case['tokens'])

    def test_chat_render_and_ids(self):
        for case in fixture('chat')['cases']:
            with self.subTest(case=case['id']):
                flags = []
                for message in case['messages']:
                    flags += ['--message', message['role'], message['content']]
                self.assertEqual(self.invoke('template', *flags).stdout.decode(), case['rendered'])
                self.assertEqual(json.loads(self.invoke('tokenize', *flags).stdout), case['tokens'])

    def test_system_and_final_user_flags(self):
        for case in fixture('chat')['cases']:
            flags = []
            for i, message in enumerate(case['messages']):
                if i == 0 and message['role'] == 'system':
                    flags += ['--system', message['content']]
                elif i == len(case['messages']) - 1:
                    flags += ['--chat', '-p', message['content']]
                else:
                    flags += ['--message', message['role'], message['content']]
            self.assertEqual(json.loads(self.invoke('tokenize', *flags).stdout), case['tokens'])

    def test_cli_context_error_precedes_weight_loading(self):
        result = self.invoke('run', '-p', 'hello', '-n', '32', '--ctx-size', '1', '--greedy', check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn(b'exceeds context limit', result.stderr)

    def test_invalid_chat_inputs(self):
        for flags in [('--message', 'tool', 'hello'), ('--message', 'user'),
                      ('--system', 'one', '--message', 'system', 'two')]:
            result = self.invoke('template', *flags, check=False)
            self.assertEqual(result.returncode, 2)


if __name__ == '__main__':
    unittest.main()
