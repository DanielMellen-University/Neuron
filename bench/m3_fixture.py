"""Small GGUF metadata helpers for reproducible tokenizer-only fixtures."""
import io
import struct
from pathlib import Path

FORMATS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}


def metadata(path):
    with open(path, 'rb') as stream:
        def scalar(fmt):
            size = struct.calcsize('<' + fmt)
            return struct.unpack('<' + fmt, stream.read(size))[0]

        def string():
            return stream.read(scalar('Q')).decode('utf-8')

        def value(kind):
            if kind == 8:
                return string()
            if kind == 9:
                element, length = scalar('I'), scalar('Q')
                return [value(element) for _ in range(length)]
            return scalar(FORMATS[kind])

        if stream.read(4) != b'GGUF' or scalar('I') not in (2, 3):
            raise ValueError('expected GGUF v2/v3')
        scalar('Q')  # Tensor count. Only metadata is needed here.
        count = scalar('Q')
        result = {}
        for _ in range(count):
            key, kind = string(), scalar('I')
            result[key] = value(kind)
        return result


def byte_alphabet():
    extra = 256
    result = []
    for code in range(256):
        if 33 <= code <= 126 or 161 <= code <= 172 or code >= 174:
            result.append(chr(code))
        else:
            result.append(chr(extra))
            extra += 1
    return result


def subset_vocabulary(meta, cases):
    tokens = meta['tokenizer.ggml.tokens']
    types = meta['tokenizer.ggml.token_type']
    merges = meta['tokenizer.ggml.merges']
    by_result = {}
    for merge in merges:
        left, right = merge.split(' ')
        by_result.setdefault(left + right, []).append((left, right))
    needed = set(byte_alphabet())
    needed.update(tokens[t] for case in cases for t in case['tokens'])
    needed.update(t for t, kind in zip(tokens, types) if kind != 1)
    pending = list(needed)
    while pending:
        for parts in by_result.get(pending.pop(), []):
            for part in parts:
                if part not in needed:
                    needed.add(part)
                    pending.append(part)
    return {'vocabulary_size': len(tokens), 'bos': meta['tokenizer.ggml.bos_token_id'],
            'eos': meta['tokenizer.ggml.eos_token_id'],
            'tokens': [{'id': i, 'text': t, 'type': types[i]} for i, t in enumerate(tokens) if t in needed],
            'merges': [m for m in merges if ''.join(m.split(' ')) in needed]}


def write_tokenizer_gguf(path, subset):
    """Fill unused IDs with inert entries so original llama.cpp IDs stay intact."""
    tokens = [f'[unused_{i}]' for i in range(subset['vocabulary_size'])]
    types = [5] * len(tokens)
    for token in subset['tokens']:
        tokens[token['id']], types[token['id']] = token['text'], token['type']
    data = io.BytesIO()

    def scalar(fmt, value):
        data.write(struct.pack('<' + fmt, value))

    def string(value):
        encoded = value.encode('utf-8')
        scalar('Q', len(encoded))
        data.write(encoded)

    def item(key, kind, value):
        string(key)
        scalar('I', kind)
        if kind == 8:
            string(value)
        elif kind == 9:
            element, values = value
            scalar('I', element)
            scalar('Q', len(values))
            for entry in values:
                string(entry) if element == 8 else scalar(FORMATS[element], entry)
        else:
            scalar(FORMATS[kind], value)

    data.write(b'GGUF')
    scalar('I', 3)
    scalar('Q', 0)
    scalar('Q', 10)
    item('general.architecture', 8, 'qwen2')
    item('tokenizer.ggml.model', 8, 'gpt2')
    item('tokenizer.ggml.pre', 8, 'qwen2')
    item('tokenizer.ggml.tokens', 9, (8, tokens))
    item('tokenizer.ggml.token_type', 9, (5, types))
    item('tokenizer.ggml.merges', 9, (8, subset['merges']))
    item('tokenizer.ggml.bos_token_id', 4, subset['bos'])
    item('tokenizer.ggml.eos_token_id', 4, subset['eos'])
    item('tokenizer.ggml.add_bos_token', 7, False)
    item('tokenizer.ggml.add_space_prefix', 7, False)
    Path(path).write_bytes(data.getvalue())
