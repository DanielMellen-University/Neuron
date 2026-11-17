#!/usr/bin/env python3
"""Repeat only M4-v2 perplexity. Free-running agreement is outside R1."""
import argparse
import os
from pathlib import Path

from check_m4 import measure
from check_m4_v2 import TEXT_SHA
from harness import ROOT, read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, default=Path(os.environ.get('NEURON_MODELS_DIR', ROOT / 'models')))
    parser.add_argument('--runs-dir', type=Path, default=ROOT / 'bench/runs/r1-ppl')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--formats', nargs='+', default=['Q4_K_M', 'F16', 'Q8_0', 'Q4_0'])
    parser.add_argument('--isa', default='auto')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    text = ROOT / 'bench/wikitext2-m4-v2.txt'
    if sha256(text) != TEXT_SHA:
        raise ValueError('M4-v2 text changed')
    binary = args.binary.resolve()
    directory = args.runs_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    report = dict(text_sha256=TEXT_SHA, context=512, chunks=16, scored_tokens=4096,
                  isa=args.isa, threads=args.threads, binary_sha256=sha256(binary),
                  relative_regression_tolerance=1e-6, formats={})
    for name in args.formats:
        baseline_path = ROOT / 'bench/reference' / ('m4-v2-' + name + '.json')
        baseline = read_json(baseline_path)
        model = (args.models_dir / baseline['file']).resolve()
        model_hash = sha256(model)
        if model_hash != baseline['sha256']:
            raise ValueError('M4-v2 model changed')
        command = [str(binary), 'perplexity', '-m', str(model), '--prompt-file', str(text),
                   '--ctx-size', '512', '--chunks', '16', '--isa', args.isa, '--threads', str(args.threads)]
        run = measure(command, name, directory, model_hash, report['binary_sha256'])
        result, old = run['result'], baseline['perplexity']['result']
        if result['tokens'] != old['tokens'] or result['scored_tokens'] != 4096:
            raise ValueError('M4-v2 tokens or scored positions changed')
        relative = result['perplexity'] / old['perplexity'] - 1
        saved = args.output.with_name('r1-ppl-' + name + '.json')
        write_json(saved, run)
        report['formats'][name] = dict(source=str(saved.resolve().relative_to(ROOT)),
            model_sha256=model_hash, perplexity=result['perplexity'], recorded_perplexity=old['perplexity'],
            llama_perplexity=baseline['llama_perplexity']['result']['perplexity'],
            relative_change=relative, no_increase=relative <= 0, passed=abs(relative) <= 1e-6)
        write_json(args.output, report)
        print(f'{name}: {result["perplexity"]:.12f}, recorded {old["perplexity"]:.12f}, change {relative:.9g}', flush=True)
        if not report['formats'][name]['passed']:
            raise ValueError('perplexity regression')
    report['passed'] = all(row['passed'] for row in report['formats'].values())
    write_json(args.output, report)


if __name__ == '__main__':
    main()
