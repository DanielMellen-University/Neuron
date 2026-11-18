#!/usr/bin/env python3
"""Trace the first engine disagreement per format on an identical token prefix."""
import argparse
from array import array
import json
import math
from pathlib import Path
import subprocess

from check_r2 import FORMATS
from harness import ROOT, read_json, sha256, write_json
from speed import source_hash


def values(path):
    result = array('f')
    with path.open('rb') as stream:
        result.frombytes(stream.read())
    return result


def difference(a, b):
    if len(a) != len(b) or not a:
        raise ValueError('trace shape differs')
    errors = [float(x) - y for x, y in zip(a, b)]
    if any(not math.isfinite(v) for v in errors):
        raise ValueError('nonfinite trace')
    return dict(elements=len(a), max_abs=max(map(abs, errors)),
                rms=math.sqrt(sum(v * v for v in errors) / len(errors)))


def compare_traces(neuron, llama, position):
    comparisons = {}
    for path in sorted(neuron.glob('*.f32')):
        other = llama / path.name
        if not other.exists():
            raise ValueError('missing llama trace: ' + path.name)
        a, b = values(path), values(other)
        if path.stem.startswith(('kq-', 'kq_soft_max-')):
            shape = read_json(other.with_suffix('.json'))['shape']
            stride = shape[0]
            b = array('f', (b[head * stride + t] for head in range(shape[2]) for t in range(position + 1)))
        if path.stem.startswith(('Kcache-', 'Vcache-')):
            shape = read_json(other.with_suffix('.json'))['shape']
            is_key = path.stem.startswith('K')
            width, stride = (shape[0], shape[1]) if is_key else (shape[1], shape[0])
            b = array('f', (b[(head * stride + t) * width + d] if is_key else
                           b[(head * width + d) * stride + t]
                           for t in range(position + 1) for head in range(shape[2]) for d in range(width)))
        comparisons[path.stem] = difference(a, b)
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agreement', type=Path, required=True)
    parser.add_argument('--trace-binary', type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, required=True)
    parser.add_argument('--runs-dir', type=Path, required=True, help='outside the repository')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    directory = args.runs_dir.resolve()
    if directory.is_relative_to(ROOT):
        raise ValueError('keep full activation dumps outside the repository')
    directory.mkdir(parents=True, exist_ok=True)
    baseline = read_json(args.agreement)
    report = dict(agreement_source=str(args.agreement), trace_binary_sha256=sha256(args.trace_binary),
                  trace_build=read_json(args.trace_binary.with_name('build.json')), formats={})
    build = report['trace_build']
    if (build['binary_sha256'] != report['trace_binary_sha256'] or
            build['neuron_source_sha256'] != source_hash() or
            build['neuron_source_sha256'] != baseline['neuron_source_sha256'] or
            build['probe_sha256'] != sha256(ROOT / 'bench/r2_trace.cpp') or
            build['builder_sha256'] != sha256(ROOT / 'bench/trace_r2.py') or
            build['llama_cpp_commit'] != baseline['llama_cpp_commit']):
        raise ValueError('trace build differs from measured sources')
    for name in FORMATS:
        source = baseline['formats'][name]
        n = read_json(ROOT / source['neuron_record'])['runs']
        l = read_json(ROOT / source['llama_record'])['runs']
        # Use the first same-format engine mismatch. If all 768 IDs agree,
        # probe the smallest F16 top-two margin as a sensitive control.
        mismatches = source['between_engines']['prompts']
        first = next(((i, row['first_divergent_token_index']) for i, row in enumerate(mismatches)
                      if row['first_divergent_token_index'] is not None), None)
        if first is None:
            _, i, at = min((top['gap'], i, at) for i, row in enumerate(n) for at, top in enumerate(row['result']['top2']))
            reason = 'all engine IDs agree; smallest top-two margin control'
        else:
            i, at = first
            reason = 'first same-format engine disagreement in prompt order'
        a, b = n[i]['result'], l[i]['result']
        if a['prompt_tokens'] != b['prompt_tokens'] or a['generated_tokens'][:at] != b['generated_tokens'][:at]:
            raise ValueError('probe must use an identical causal prefix')
        tokens = a['prompt_tokens'] + a['generated_tokens'][:at]
        ids = directory / (name + '-ids.txt')
        ids.write_text(' '.join(map(str, tokens)) + '\n')
        model = (args.models_dir / source['file']).resolve()
        item = dict(reason=reason, prompt_id=mismatches[i]['prompt_id'], generated_index=at,
                    input_ids=tokens, position=len(tokens) - 1, model_sha256=sha256(model),
                    baseline_neuron_top2=a['top2'][at], baseline_llama_top2=b['top2'][at], engines={})
        for engine in ['neuron', 'llama']:
            if engine == 'neuron' and any(row['binary_sha256'] != build['neuron_binary_sha256'] for row in n):
                raise ValueError('trace library build differs from measured Neuron executable')
            out = directory / name / engine
            out.mkdir(parents=True, exist_ok=True)
            command = [str(args.trace_binary.resolve()), engine, str(model), str(ids), str(out)]
            with (out / 'stderr.txt').open('w') as stderr:
                result = subprocess.check_output(command, text=True, stderr=stderr)
            item['engines'][engine] = dict(command=command, result=json.loads(result))
            product_command = [str(args.trace_binary.resolve()), 'products', str(model), str(ids), str(out)]
            products = subprocess.check_output(product_command, text=True)
            item['engines'][engine]['shared_input_checks'] = json.loads(products)
        # Verify that instrumentation preserved the unmodified CLI's decision.
        for engine, run in [('neuron', a), ('llama', b)]:
            traced = values(directory / name / engine / 'result_output.f32')
            top = sorted(range(len(traced)), key=lambda j: traced[j], reverse=True)[:2]
            if top != [run['top2'][at]['first'], run['top2'][at]['second']] or abs(
                    traced[top[0]] - traced[top[1]] - run['top2'][at]['gap']) > 2e-5:
                raise ValueError('instrumentation changed the ' + engine + ' decision')
        item['trace_comparison'] = compare_traces(directory / name / 'neuron', directory / name / 'llama', len(tokens) - 1)
        saved = args.output.with_name(args.output.stem + '-' + name + '.json')
        write_json(saved, item)
        report['formats'][name] = dict(source=str(saved.relative_to(ROOT)),
            prompt_id=item['prompt_id'], generated_index=at, reason=reason,
            logit_difference=item['trace_comparison']['result_output'])
        write_json(args.output, report)
        print(name, item['prompt_id'], at, 'logit max error', item['trace_comparison']['result_output']['max_abs'], flush=True)


if __name__ == '__main__':
    main()
