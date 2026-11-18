#!/usr/bin/env python3
"""Measure the frozen M4-v2 free-running agreement in both engines."""
import argparse
import subprocess
from pathlib import Path

from check_m4 import LLAMA_COMMIT, measure
from check_m4_v2 import PROMPTS_SHA, comparison
from harness import ROOT, read_json, sha256, write_json
from speed import source_hash

FORMATS = ['F16', 'Q8_0', 'Q4_0', 'Q4_K_M']


def agreement(expected, actual, prompts, *, require_full_reference=True):
    if len(prompts) != 12 or len(expected) != len(prompts) or len(actual) != len(prompts):
        raise ValueError('requires all twelve prompts')
    details = []
    for prompt, a, b in zip(prompts, expected, actual):
        a, b = a['result'], b['result']
        if a['prompt_tokens'] != b['prompt_tokens']:
            raise ValueError('prompt tokenization differs')
        if require_full_reference and len(a['generated_tokens']) != 64:
            raise ValueError('F16 must reach all 64 planned positions')
        if max(len(a['generated_tokens']), len(b['generated_tokens'])) > 64:
            raise ValueError('generation exceeds the planned positions')
        row = dict(prompt_id=prompt['id'], **comparison(a['generated_tokens'], b['generated_tokens']))
        if row['first_divergent_token_index'] is None and row['matching_tokens'] < 64:
            row['first_divergent_token_index'] = row['matching_tokens']
        details.append(row)
    matches = sum(d['matching_tokens'] for d in details)
    return dict(matching_tokens=matches, total_tokens=768, rate=matches / 768, prompts=details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--reference-tool', type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, required=True)
    parser.add_argument('--llamacpp-dir', type=Path, required=True)
    parser.add_argument('--runs-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--notes', required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    prompt_file = ROOT / 'bench/prompts-m4-v2.json'
    if sha256(prompt_file) != PROMPTS_SHA:
        raise ValueError('M4-v2 prompts changed')
    commit = subprocess.check_output(['git', '-C', str(args.llamacpp_dir), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != LLAMA_COMMIT:
        raise ValueError('requires pinned llama.cpp')
    directory = args.runs_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    prompts = read_json(prompt_file)
    report = dict(parent_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        neuron_source_sha256=source_hash(), reference_source_sha256=sha256(ROOT / 'bench/llama_reference.cpp'),
        script_sha256=sha256(__file__),
        llama_cpp_commit=commit, prompts_sha256=PROMPTS_SHA, n_predict=64, context=256,
        threads=1, neuron_isa='auto', kv_type='F32', flash_attention=False,
        llama_batch_size=1, prompt_format='raw completion', greedy=True,
        repetitions=1, warmups=0, notes=args.notes, formats={})
    rows = {engine: {} for engine in ['neuron', 'llama']}
    for name in FORMATS:
        baseline = read_json(ROOT / 'bench/reference' / ('m4-v2-' + name + '.json'))
        model = (args.models_dir / baseline['file']).resolve()
        model_hash = sha256(model)
        if model_hash != baseline['sha256']:
            raise ValueError('M4-v2 model changed')
        item = dict(file=model.name, sha256=model_hash, size_bytes=model.stat().st_size)
        for engine, binary in [('neuron', args.binary.resolve()), ('llama', args.reference_tool.resolve())]:
            runs = []
            for prompt in prompts:
                if engine == 'neuron':
                    command = [str(binary), 'run', '-m', str(model), '-p', prompt['text'], '-n', '64',
                        '--greedy', '--json', '--ctx-size', '256', '--threads', '1', '--isa', 'auto']
                else:
                    command = [str(binary), str(model), prompt['text'], '--generate', '64',
                               '--ctx-size', '256', '--batch-size', '1']
                run = measure(command, engine + '-' + name + '-' + prompt['id'], directory, model_hash, sha256(binary))
                runs.append(run)
            rows[engine][name] = runs
            saved = args.output.with_name(args.output.stem + '-' + engine + '-' + name + '.json')
            write_json(saved, dict(engine=engine, format=name, runs=runs))
            item[engine + '_record'] = str(saved.relative_to(ROOT))
            item[engine + '_agreement'] = agreement(rows[engine]['F16'], runs, prompts)
        item['between_engines'] = agreement(rows['llama'][name], rows['neuron'][name], prompts,
                                            require_full_reference=False)
        report['formats'][name] = item
        write_json(args.output, report)
        print(name, {k: v['matching_tokens'] for k, v in item.items() if isinstance(v, dict) and 'matching_tokens' in v}, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
