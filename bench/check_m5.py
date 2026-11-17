#!/usr/bin/env python3
"""Quick M5 model guards. Uses the unchanged M4 scalar path as the reference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from harness import ROOT, machine_state, read_json, sha256, write_json


def invoke(binary, arguments, isa, threads):
    env = dict(os.environ, NEURON_ISA=isa, NEURON_THREADS=str(threads))
    command = [str(binary.resolve()), *arguments]
    result = json.loads(subprocess.check_output(command, env=env, text=True))
    if arguments[0] == 'perplexity':
        tokens = result['tokens']
        result['input_tokens_sha256'] = hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()
        result['input_token_count'] = len(tokens)
        result['tokens'] = tokens[:result['context'] * result['chunks']]
    return dict(command=command, binary_sha256=sha256(binary), isa=isa, threads=threads, result=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--scalar-binary', type=Path)
    parser.add_argument('--models-dir', type=Path, default=Path(os.environ.get('NEURON_MODELS_DIR', ROOT / 'models')))
    parser.add_argument('--isa', default='auto')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--quality', action='store_true')
    parser.add_argument('--isas', nargs='+', help='quality candidates; scalar runs once per format')
    parser.add_argument('--resume', action='store_true', help='reuse completed quality runs with matching commands and hashes')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cached = read_json(args.output) if args.resume and args.output.exists() else {}
    report = dict(before=machine_state(), binary_sha256=sha256(args.binary), runs=[])
    if args.quality:
        # Two short independent chunks keep the unchanged scalar run affordable.
        # Same fixed M4-v2 text and tokenizer, context 32, 32 scored targets.
        text = ROOT / 'bench/wikitext2-m4-v2.txt'
        report.update(context=32, chunks=2, scored_tokens=32, text_sha256=sha256(text),
                      scalar_binary_sha256=sha256(args.scalar_binary or args.binary))
        if cached and any(cached.get(key) != report[key] for key in ('context', 'chunks', 'text_sha256')):
            raise ValueError('cached quality inputs differ')
        for name, suffix in [('F16', 'f16'), ('Q8_0', 'Q8_0'), ('Q4_0', 'Q4_0'), ('Q4_K_M', 'Q4_K_M')]:
            model = args.models_dir / f'Qwen2.5-1.5B-Instruct-{suffix}.gguf'
            model_hash = sha256(model)
            command = ['perplexity', '-m', str(model), '--prompt-file', str(text),
                       '--ctx-size', '32', '--chunks', '2']

            def reuse_or_invoke(binary, isa, threads, role):
                expected_command = [str(binary.resolve()), *command]
                binary_hash = sha256(binary)
                for row in cached.get('runs', []):
                    run = row[role]
                    hash_key = 'scalar_binary_sha256' if role == 'scalar' else 'binary_sha256'
                    if (row['format'] == name and row['model_sha256'] == model_hash
                            and run['command'] == expected_command and run['isa'] == isa
                            and run['threads'] == threads
                            and run.get('binary_sha256', cached.get(hash_key)) == binary_hash):
                        print(f'{name}: reusing {role} {isa} ({threads} threads)', flush=True)
                        run['binary_sha256'] = binary_hash
                        return run
                return invoke(binary, command, isa, threads)

            scalar = reuse_or_invoke(args.scalar_binary or args.binary, 'scalar', 1, 'scalar')
            for isa in args.isas or [args.isa]:
                simd = reuse_or_invoke(args.binary, isa, args.threads, 'candidate')
                relative = simd['result']['perplexity'] / scalar['result']['perplexity'] - 1
                passed = abs(relative) < 0.001 and scalar['result']['input_tokens_sha256'] == simd['result']['input_tokens_sha256']
                report['runs'].append(dict(format=name, model_sha256=model_hash, scalar=scalar,
                                           candidate=simd, relative_change=relative, passed=passed))
                write_json(args.output, report)
                print(f'{name} {isa}: scalar {scalar["result"]["perplexity"]:.9f}, candidate {simd["result"]["perplexity"]:.9f}, change {relative:.9g}', flush=True)
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
