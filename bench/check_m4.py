#!/usr/bin/env python3
"""One saved run per M4 measurement. Resume reuses matching commands and hashes."""
import argparse
import json
from pathlib import Path
import subprocess

from harness import ROOT, machine_state, read_json, sha256, token_comparison, write_json

TEXT_SHA = '5c4d7185e322a055c34ef0ba1c0511973f916adc637b4159ee86a1be8300da4a'
F16_SHA = 'b6eaec3509f1d0373d1f4802654c4a7bfcde0768645d2d45e8885f6922b428ee'
LLAMA_COMMIT = '9e9619d030bbe6b584d826a0a6cf06e38251f52b'


def measure(command, key, directory, model_hash, binary_hash):
    path = directory / (key + '.json')
    identity = dict(command=command, model_sha256=model_hash, binary_sha256=binary_hash)
    if path.exists():
        saved = read_json(path)
        if any(saved[k] != v for k, v in identity.items()):
            raise ValueError(f'saved run identity changed: {path}; use a fresh runs directory')
        print(f'reuse {key}', flush=True)
        return saved
    print(f'start {key}', flush=True)
    timing = directory / (key + '.time.json')
    before = machine_state()
    with (directory / (key + '.stderr')).open('w') as log:
        output = subprocess.check_output([
            '/usr/bin/time', '-f', '{"maximum_rss_kib":%M,"wall_seconds":%e,"user_seconds":%U,"system_seconds":%S}',
            '-o', str(timing), *command], stderr=log)
    result = dict(identity, result=json.loads(output), resources=read_json(timing),
                  machine_before=before, machine_after=machine_state())
    write_json(path, result)
    print(f'done {key}: {result["resources"]}', flush=True)
    return result


def report(rows, args):
    baseline = rows['F16']
    formats = {}
    for name, data in rows.items():
        item = dict(data)
        comparisons = [dict(prompt_id=p['id'], **token_comparison(a['result']['generated_tokens'], b['result']['generated_tokens']))
                       for p, a, b in zip(read_json(ROOT / 'bench/prompts.json'), baseline['greedy'], data['greedy'])]
        matched = sum(c['matching_tokens'] for c in comparisons)
        total = sum(c['total_tokens'] for c in comparisons)
        item['agreement'] = dict(matching_tokens=matched, total_tokens=total, rate=matched / total, prompts=comparisons)
        item['perplexity_ratio_to_f16'] = data['perplexity']['result']['perplexity'] / baseline['perplexity']['result']['perplexity']
        item['maximum_rss_kib'] = max(r['resources']['maximum_rss_kib'] for r in [data['perplexity'], *data['greedy']])
        timings = [r['result']['timings'] for r in data['greedy']]
        item['pp_tokens_per_second'] = 1000 * sum(r['prompt_n'] for r in timings) / sum(r['prompt_ms'] for r in timings)
        item['tg_tokens_per_second'] = 1000 * sum(r['predicted_n'] - 1 for r in timings) / sum(r['predicted_ms'] for r in timings)
        for a, b in zip(baseline['greedy'], data['greedy']):
            if a['result']['prompt_tokens'] != b['result']['prompt_tokens']:
                raise ValueError('prompt tokenization differs between formats')
            if len(b['result']['generated_tokens']) != 64:
                raise ValueError('a generation did not reach 64 tokens')
        if 'llama_perplexity' in data:
            if data['perplexity']['result']['tokens'] != data['llama_perplexity']['result']['tokens']:
                raise ValueError('perplexity tokenization differs between engines')
            item['perplexity_ratio_to_llama'] = data['perplexity']['result']['perplexity'] / data['llama_perplexity']['result']['perplexity']
        formats[name] = item
    passed = dict(
        q8_perplexity=abs(formats['Q8_0']['perplexity_ratio_to_f16'] - 1) <= .01,
        q4_perplexity=abs(formats['Q4_0']['perplexity_ratio_to_f16'] - 1) <= .10,
        q4_llama_perplexity=abs(formats['Q4_0']['perplexity_ratio_to_llama'] - 1) <= .02,
        q8_agreement=formats['Q8_0']['agreement']['rate'] >= .90,
        q4_agreement=formats['Q4_0']['agreement']['rate'] >= .70)
    result = dict(llama_cpp_commit=LLAMA_COMMIT, text_sha256=TEXT_SHA,
                  prompts_sha256=sha256(ROOT / 'bench/prompts.json'), context=128, chunks=2,
                  scored_target_positions='64..127 within each independent 128-token chunk; incomplete tail ignored',
                  greedy_context=256, n_predict=64, threads=1, kv_type='F32', flash_attention=False,
                  prompt_format='raw completion; no chat template', repetitions=1, warmup_runs=0,
                  notes=args.notes, formats=formats, acceptance=passed)
    write_json(args.output, result)
    print(json.dumps(passed, indent=2), flush=True)
    return 0 if all(passed.values()) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--reference-tool', type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, required=True)
    parser.add_argument('--q8-model-file', type=Path, help='optional Q8_0 model with a higher-precision embedding/output matrix')
    parser.add_argument('--q4-model-file', type=Path, help='optional Q4_0 model with a higher-precision embedding/output matrix')
    parser.add_argument('--runs-dir', type=Path, default=ROOT / 'bench/runs/m4')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--notes', required=True)
    args = parser.parse_args()
    binary, reference = str(args.binary.resolve()), str(args.reference_tool.resolve())
    text = ROOT / 'bench/wikitext2-slice.txt'
    if sha256(text) != TEXT_SHA:
        raise ValueError('perplexity text hash changed')
    directory = args.runs_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {b: sha256(b) for b in (binary, reference)}
    rows = {}
    for name, suffix in [('F16', 'f16'), ('Q8_0', 'Q8_0'), ('Q4_0', 'Q4_0')]:
        override = args.q8_model_file if name == 'Q8_0' else args.q4_model_file if name == 'Q4_0' else None
        model = (override or args.models_dir / f'Qwen2.5-1.5B-Instruct-{suffix}.gguf').resolve()
        model_hash = sha256(model)
        if name == 'F16' and model_hash != F16_SHA:
            raise ValueError('requires the pinned Qwen2.5 F16 model')
        data = dict(file=model.name, size_bytes=model.stat().st_size, sha256=model_hash)
        command = [binary, 'perplexity', '-m', str(model), '--prompt-file', str(text), '--ctx-size', '128', '--chunks', '2']
        data['perplexity'] = measure(command, name + '-perplexity', directory, model_hash, hashes[binary])
        data['greedy'] = []
        for prompt in read_json(ROOT / 'bench/prompts.json'):
            command = [binary, 'run', '-m', str(model), '-p', prompt['text'], '-n', '64', '--greedy', '--json', '--ctx-size', '256']
            data['greedy'].append(measure(command, name + '-' + prompt['id'], directory, model_hash, hashes[binary]))
        # The adapter runs the exact same text, tokenization, reset and scoring
        # policy. Use it rather than llama-perplexity's different defaults.
        command = [reference, str(model), '', '--prompt-file', str(text), '--perplexity', '128', '--chunks', '2']
        data['llama_perplexity'] = measure(command, name + '-llama-perplexity', directory, model_hash, hashes[reference])
        rows[name] = data
    return report(rows, args)


if __name__ == '__main__':
    raise SystemExit(main())
