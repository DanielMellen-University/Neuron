#!/usr/bin/env python3
"""Quick M5 model guards. Uses the unchanged M4 scalar path as the reference."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from harness import ROOT, machine_state, read_json, sha256, write_json


def invoke(binary, arguments, isa, threads):
    env = dict(os.environ, NEURON_ISA=isa, NEURON_THREADS=str(threads))
    command = [str(binary.resolve()), *arguments]
    return dict(command=command, isa=isa, threads=threads,
                result=json.loads(subprocess.check_output(command, env=env, text=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--scalar-binary', type=Path)
    parser.add_argument('--models-dir', type=Path, default=Path(os.environ.get('NEURON_MODELS_DIR', ROOT / 'models')))
    parser.add_argument('--isa', default='auto')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--quality', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = dict(before=machine_state(), binary_sha256=sha256(args.binary), runs=[])
    if args.quality:
        # Two short independent chunks keep the unchanged scalar run affordable.
        # Same fixed M4-v2 text and tokenizer, context 32, 32 scored targets.
        text = ROOT / 'bench/wikitext2-m4-v2.txt'
        report.update(context=32, chunks=2, scored_tokens=32, text_sha256=sha256(text),
                      scalar_binary_sha256=sha256(args.scalar_binary or args.binary))
        for name, suffix in [('F16', 'f16'), ('Q8_0', 'Q8_0'), ('Q4_0', 'Q4_0'), ('Q4_K_M', 'Q4_K_M')]:
            model = args.models_dir / f'Qwen2.5-1.5B-Instruct-{suffix}.gguf'
            command = ['perplexity', '-m', str(model), '--prompt-file', str(text),
                       '--ctx-size', '32', '--chunks', '2']
            scalar = invoke(args.scalar_binary or args.binary, command, 'scalar', 1)
            simd = invoke(args.binary, command, args.isa, args.threads)
            relative = simd['result']['perplexity'] / scalar['result']['perplexity'] - 1
            passed = abs(relative) < 0.001 and scalar['result']['tokens'] == simd['result']['tokens']
            report['runs'].append(dict(format=name, model_sha256=sha256(model), scalar=scalar,
                                       candidate=simd, relative_change=relative, passed=passed))
            write_json(args.output, report)
            print(f'{name}: scalar {scalar["result"]["perplexity"]:.9f}, candidate {simd["result"]["perplexity"]:.9f}, change {relative:.9g}', flush=True)
            if not passed:
                raise ValueError('perplexity guard failed')
    else:
        for file, reference in [('stories15M.gguf', 'tinystories-15m.json'),
                                ('SmolLM2-135M-Instruct-f16.gguf', 'smollm2-135m-instruct.json')]:
            model = args.models_dir / file
            original = read_json(ROOT / 'bench/reference' / reference)
            for expected in original['runs']:
                if expected['threads'] != 1 or expected['repetition'] != 0:
                    continue
                count = 128 if file == 'stories15M.gguf' else 32
                command = ['run', '-m', str(model), '-p', expected['prompt'], '-n', str(count), '--greedy', '--json']
                candidate = invoke(args.binary, command, args.isa, args.threads)
                scalar = invoke(args.scalar_binary or args.binary, command, 'scalar', 1)
                actual = candidate['result']['generated_tokens']
                matched = actual == scalar['result']['generated_tokens']
                m1_match = actual == expected['generated_tokens'][:count]
                if not matched or (file == 'stories15M.gguf' and not m1_match):
                    raise ValueError(f'generation guard failed: {file} {expected["prompt_id"]}')
                report['runs'].append(dict(model=file, model_sha256=sha256(model), prompt_id=expected['prompt_id'],
                    candidate=candidate, scalar=scalar, scalar_exact=matched, m1_exact=m1_match))
                print(f'{file} {expected["prompt_id"]}: {len(actual)} tokens exact vs scalar; M1 exact={m1_match}', flush=True)
    report['after'] = machine_state()
    report['passed'] = True
    write_json(args.output, report)


if __name__ == '__main__':
    main()
