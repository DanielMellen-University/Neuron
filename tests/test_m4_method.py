#!/usr/bin/env python3
"""Check teacher replay and missing-tail accounting without downloaded weights."""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
from check_m4_v2 import comparison
BINARY = Path(sys.argv.pop(1)).resolve()


def model(path):
    def string(s):
        b = s.encode()
        return struct.pack('<Q', len(b)) + b

    def item(key, kind, value):
        return string(key) + struct.pack('<I', kind) + value

    meta = [item('general.architecture', 8, string('llama')),
            item('tokenizer.ggml.model', 8, string('llama')),
            item('tokenizer.ggml.tokens', 9, struct.pack('<IQ', 8, 3) + b''.join(map(string, ['a', 'b', 'c']))),
            item('tokenizer.ggml.scores', 9, struct.pack('<IQ3f', 6, 3, 0, 0, 0)),
            item('tokenizer.ggml.token_type', 9, struct.pack('<IQ3i', 5, 3, 1, 1, 1)),
            item('tokenizer.ggml.add_bos_token', 7, b'\0'),
            item('tokenizer.ggml.add_space_prefix', 7, b'\0')]
    for key, value in dict(embedding_length=4, feed_forward_length=4, block_count=1,
                           **{'attention.head_count': 2, 'attention.head_count_kv': 1,
                              'context_length': 32}).items():
        meta.append(item('llama.' + key, 4, struct.pack('<I', value)))
    identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
    tensors = [('token_embd.weight', [4, 3], identity), ('output_norm.weight', [4], [1] * 4)]
    for name in ['attn_norm', 'ffn_norm']:
        tensors.append(('blk.0.' + name + '.weight', [4], [1] * 4))
    for name, rows in [('attn_q', 4), ('attn_k', 2), ('attn_v', 2), ('attn_output', 4),
                       ('ffn_gate', 4), ('ffn_up', 4), ('ffn_down', 4)]:
        tensors.append(('blk.0.' + name + '.weight', [4, rows], [0] * (4 * rows)))
    table, payload = b'', b''
    for name, dims, values in tensors:
        table += string(name) + struct.pack('<I', len(dims)) + struct.pack('<' + 'Q' * len(dims), *dims)
        table += struct.pack('<IQ', 0, len(payload))
        payload += struct.pack('<' + 'f' * len(values), *values)
        payload += b'\0' * (-len(payload) % 32)
    header = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(meta)) + b''.join(meta) + table
    path.write_bytes(header + b'\0' * (-len(header) % 32) + payload)


class MethodTests(unittest.TestCase):
    def test_execution_flags_override_invalid_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tiny.gguf'
            model(path)
            command = [str(BINARY), 'run', '-m', str(path), '-p', 'a', '-n', '4', '--greedy', '--json']
            env = dict(os.environ, NEURON_ISA='invalid', NEURON_THREADS='invalid')
            result = json.loads(subprocess.check_output(command + ['--isa', 'scalar', '--threads', '2'], env=env))
            self.assertEqual(result['isa'], 'scalar')
            self.assertEqual(result['threads'], 2)
            for flags in [[], ['--isa', 'scalar'], ['--threads', '2'],
                          ['--isa', 'scalar', '--threads', '257']]:
                run = subprocess.run(command + flags, env=env, capture_output=True)
                self.assertEqual(run.returncode, 2)

    def test_missing_tail_keeps_fixed_denominator(self):
        self.assertEqual(comparison([1] * 64, [1] * 2)['agreement_rate'], 2 / 64)
        self.assertEqual(comparison([1] * 64, [])['first_divergent_token_index'], 0)

    def test_teacher_replays_previous_id_and_records_current_argmax(self):
        with tempfile.TemporaryDirectory() as directory:
            path, teacher = Path(directory) / 'tiny.gguf', Path(directory) / 'ids.txt'
            model(path)
            teacher.write_text('1 2 0 1\n')
            command = [str(BINARY), 'run', '-m', str(path), '-p', 'a', '-n', '4', '--greedy', '--json']
            free = json.loads(subprocess.check_output(command))
            forced = json.loads(subprocess.check_output(command + ['--teacher-tokens', str(teacher)]))
            self.assertEqual(free['generated_tokens'], [0, 0, 0, 0])
            # The model predicts the most recently fed token, including EOS.
            self.assertEqual(forced['generated_tokens'], [0, 1, 2, 0])
            for bad in ['1 2', '-1 0 0 0', '3 0 0 0', '0 0 0 junk', '0 0 0 99999999999999999999', '0 0 0 0 99999999999999999999']:
                teacher.write_text(bad)
                result = subprocess.run(command + ['--teacher-tokens', str(teacher)], capture_output=True)
                self.assertEqual(result.returncode, 2)


if __name__ == '__main__':
    unittest.main()
