#!/usr/bin/env python3
"""Local Qwen2.5 checks against the pinned external llama.cpp build."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from harness import ROOT, machine_state, read_json, sha256, token_comparison, write_json
from m3_fixture import byte_alphabet, metadata, subset_vocabulary

MODEL_SHA = 'b6eaec3509f1d0373d1f4802654c4a7bfcde0768645d2d45e8885f6922b428ee'
LLAMA_COMMIT = '9e9619d030bbe6b584d826a0a6cf06e38251f52b'
REFERENCE = ROOT / 'bench/reference'


def flags(case):
    return ([] if case['add_special'] else ['--no-add-special']) + (
        [] if case['parse_special'] else ['--no-parse-special'])


def invoke(command):
    return json.loads(subprocess.check_output(command, stderr=subprocess.DEVNULL))


def check_fixtures(binary, model, tool, refresh):
    tokenizer = read_json(REFERENCE / 'qwen2-tokenizer.json')
    chat = read_json(REFERENCE / 'qwen2-chat.json')
    for fixture in (tokenizer, chat):
        if fixture['model_sha256'] != MODEL_SHA or fixture['llama_cpp_commit'] != LLAMA_COMMIT:
            raise ValueError('fixture model or llama.cpp revision changed')
    with tempfile.TemporaryDirectory(prefix='neuron-m3-fixtures-') as directory:
        text_path, chat_path, rendered_path = (Path(directory) / n for n in ('prompt.txt', 'chat.json', 'rendered.txt'))
        for case in tokenizer['cases']:
            text_path.write_bytes(case['text'].encode('utf-8'))
            options = ['--prompt-file', str(text_path), *flags(case)]
            expected = invoke([tool, model, '', *options])
            if refresh:
                case['tokens'] = expected
            actual = invoke([binary, 'tokenize', '-m', model, *options])
            if actual != expected or expected != case['tokens']:
                raise ValueError(f"tokenizer mismatch: {case['id']}")
        for case in chat['cases']:
            write_json(chat_path, case['messages'])
            expected = invoke([tool, model, '', '--chat', str(chat_path), '--render', str(rendered_path)])
            rendered = rendered_path.read_text(encoding='utf-8')
            if refresh:
                case.update(tokens=expected, rendered=rendered)
            options = []
            for message in case['messages']:
                options += ['--message', message['role'], message['content']]
            actual = invoke([binary, 'tokenize', '-m', model, *options])
            actual_text = subprocess.check_output([binary, 'template', '-m', model, *options]).decode('utf-8')
            if (actual != expected or expected != case['tokens'] or rendered != case['rendered']
                    or actual_text != rendered):
                raise ValueError(f"chat mismatch: {case['id']}")
    if refresh:
        write_json(REFERENCE / 'qwen2-tokenizer.json', tokenizer)
        write_json(REFERENCE / 'qwen2-chat.json', chat)
        subset = subset_vocabulary(metadata(model), tokenizer['cases'] + chat['cases'])
        write_json(REFERENCE / 'qwen2-vocabulary.json', subset)
    print(f"Tokenizer: {len(tokenizer['cases'])} exact cases; ChatML: {len(chat['cases'])} exact byte/ID cases", flush=True)
    return {'tokenizer_cases': len(tokenizer['cases']), 'chat_cases': len(chat['cases'])}


def reference_generation(model, tool):
    result = {'model_sha256': MODEL_SHA, 'llama_cpp_commit': LLAMA_COMMIT,
              'reference_tool_sha256': sha256(tool), 'threads': 1, 'context': 2048,
              'kv_type': 'F32', 'flash_attention': False, 'prompt_format': 'raw completion; no chat template',
              'greedy': True, 'n_predict': 32, 'runs': []}
    for prompt in read_json(ROOT / 'bench/prompts.json'):
        command = [tool, model, prompt['text'], '--generate', '32']
        row = invoke(command)
        if len(row['generated_tokens']) != 32:
            raise ValueError('reference stopped before 32 tokens')
        row.update(prompt_id=prompt['id'], text=prompt['text'], command=command)
        result['runs'].append(row)
    return result


def greedy_checks(binary, model, expected):
    rows = []
    for reference in expected['runs']:
        command = [binary, 'run', '-m', model, '-p', reference['text'], '-n', '32', '--greedy', '--json']
        actual = invoke(command)
        prompt = token_comparison(reference['prompt_tokens'], actual['prompt_tokens'])
        comparison = token_comparison(reference['generated_tokens'], actual['generated_tokens'])
        first = comparison['first_divergent_token_index']
        row = dict(prompt_id=reference['prompt_id'], command=command, prompt=prompt,
                   generation=comparison, candidate=actual)
        if first is not None:
            row['divergence_top2'] = {
                'reference': reference['top2'][first] if first < len(reference['top2']) else None,
                'candidate': actual['top2'][first] if first < len(actual['top2']) else None}
        rows.append(row)
        print(f"{reference['prompt_id']}: {comparison['matching_tokens']}/32 IDs; first divergence {first}", flush=True)
    return rows


def cache_check(binary, model):
    command = [binary, 'run', '-m', model, '-p', 'Once upon a time', '-n', '32', '--greedy', '--json']
    cached, recomputed = invoke(command), invoke(command + ['--recompute'])
    if (len(cached['generated_tokens']) != 32 or cached['generated_tokens'] != recomputed['generated_tokens']
            or cached['top2'] != recomputed['top2']):
        raise ValueError('cached versus recompute mismatch on the real model')
    print('TinyStories cache versus recompute: 32/32 tokens and top-two IDs/gaps exact', flush=True)
    return {'model_sha256': sha256(model), 'commands': [command, command + ['--recompute']],
            'tokens': cached['generated_tokens'], 'tokens_matched': 32, 'top2_exact': True}


def chat_smoke(binary, model):
    meta = metadata(model)
    vocab = meta['tokenizer.ggml.tokens']
    inverse = {c: i for i, c in enumerate(byte_alphabet())}

    def decode(ids):
        return bytes(inverse[c] for token in ids for c in vocab[token]).decode('utf-8', errors='replace')

    history, turns = [], []
    for text in ['My name is Ada. Reply with only OK.', 'What is my name? Reply with only the name.']:
        history.append({'role': 'user', 'content': text})
        command = [binary, 'run', '-m', model, '-n', '16', '--greedy', '--json']
        for message in history:
            command += ['--message', message['role'], message['content']]
        row = invoke(command)
        reply = decode(row['generated_tokens'])
        row.update(reply=reply, command=command)
        turns.append(row)
        history.append({'role': 'assistant', 'content': reply})
        print(f'Chat reply: {reply!r}', flush=True)
    if 'Ada' not in turns[-1]['reply']:
        raise ValueError('chat did not recall the name from the previous turn')
    return turns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--reference-tool', type=Path, required=True)
    parser.add_argument('--tiny-model', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--notes', required=True, help='actual machine activity and power conditions')
    parser.add_argument('--refresh-fixtures', action='store_true')
    parser.add_argument('--collect-reference', action='store_true', help='replace committed greedy reference; no Neuron generation')
    parser.add_argument('--fixtures-only', action='store_true')
    args = parser.parse_args()
    binary, model, tool = (str(p.resolve()) for p in (args.binary, args.model, args.reference_tool))
    if sha256(model) != MODEL_SHA:
        raise ValueError('requires the pinned Qwen2.5 F16 model')
    result = {'model_sha256': MODEL_SHA, 'llama_cpp_commit': LLAMA_COMMIT, 'neuron_binary_sha256': sha256(binary),
              'reference_tool_sha256': sha256(tool), 'context': 2048, 'threads': 1,
              'machine_before': machine_state(), 'notes': args.notes}
    result.update(check_fixtures(binary, model, tool, args.refresh_fixtures))
    if not args.fixtures_only:
        fresh = reference_generation(model, tool)
        if args.collect_reference:
            fresh.update(machine_before=result['machine_before'], machine_after=machine_state(), notes=args.notes)
            write_json(REFERENCE / 'qwen2-greedy.json', fresh)
        else:
            expected = read_json(REFERENCE / 'qwen2-greedy.json')
            if expected['model_sha256'] != MODEL_SHA or expected['llama_cpp_commit'] != LLAMA_COMMIT:
                raise ValueError('greedy reference model or revision changed')
            if len(fresh['runs']) != len(expected['runs']):
                raise ValueError('greedy reference prompt count changed')
            for a, b in zip(fresh['runs'], expected['runs']):
                if a['prompt_tokens'] != b['prompt_tokens'] or a['generated_tokens'] != b['generated_tokens']:
                    raise ValueError('fresh llama.cpp output differs from committed reference')
            result['greedy'] = greedy_checks(binary, model, expected)
            write_json(args.output, result)
            tiny = str((args.tiny_model or Path(model).parent / 'stories15M.gguf').resolve())
            result['cache_vs_recompute'] = cache_check(binary, tiny)
            result['chat_smoke'] = chat_smoke(binary, model)
    result['machine_after'] = machine_state()
    write_json(args.output, result)
    if any(row['prompt']['first_divergent_token_index'] is not None or
           row['generation']['first_divergent_token_index'] is not None for row in result.get('greedy', [])):
        raise ValueError('greedy token mismatch; see logged top-2 gaps in output')


if __name__ == '__main__':
    main()
