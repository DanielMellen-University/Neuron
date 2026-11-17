#!/usr/bin/env python3
"""Collect a short speed sample or draw the committed M5 SVG. Stdlib only."""
import argparse
import html
import json
import os
from pathlib import Path
import subprocess

from harness import ROOT, machine_state, read_json, request, server, sha256, write_json

PROMPT = 'Once upon a time'
TOKENS = 32


def source_hash():
    import hashlib
    digest = hashlib.sha256()
    for path in sorted([ROOT / 'CMakeLists.txt', * (ROOT / 'src').glob('*'),
                        * (ROOT / 'include/neuron').glob('*')]):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def measure(args):
    model = (args.model_file or args.models_dir / 'Qwen2.5-1.5B-Instruct-Q4_K_M.gguf').resolve()
    binary = args.binary.resolve()
    env = dict(os.environ, NEURON_ISA=args.isa, NEURON_THREADS=str(args.threads))
    command = [str(binary), 'run', '-m', str(model), '-p', PROMPT, '-n', str(TOKENS),
               '--greedy', '--json', '--ctx-size', '256']
    before = machine_state()
    if args.llama:
        binary = args.llamacpp_dir.resolve() / 'build/bin/llama-server'
        with server(binary, model, args.threads, context=256) as (base, command):
            tokens = request(base, '/tokenize', dict(content=PROMPT, add_special=True,
                                                    parse_special=True))['tokens']
            result = request(base, '/completion', dict(prompt=tokens, n_predict=TOKENS,
                seed=42, temperature=0, samplers=['temperature'], repeat_penalty=1.0,
                cache_prompt=False, return_tokens=True, stream=False, ignore_eos=False))
            if (result['truncated'] or result['timings']['prompt_n'] != len(tokens)
                    or result['timings'].get('cache_n', 0) != 0):
                raise ValueError('invalid llama timing')
            result['prompt_tokens'] = tokens
            result['generated_tokens'] = result['tokens']
    else:
        result = json.loads(subprocess.check_output(command, env=env, text=True))
    if not args.llama and 'isa' in result:
        if result['threads'] != args.threads or (args.isa != 'auto' and result['isa'] != args.isa):
            raise ValueError('execution settings differ from requested settings')
    if len(result['generated_tokens']) != TOKENS:
        raise ValueError('generation stopped before 32 tokens')
    row = dict(step=args.step, engine='llama.cpp' if args.llama else 'neuron',
        isa='native' if args.llama else args.isa, threads=args.threads, prompt=PROMPT, n_predict=TOKENS,
        context=256, warmups=0, repetitions=1,
        model=dict(file=model.name, sha256=sha256(model), size_bytes=model.stat().st_size),
        binary_sha256=sha256(binary), source_sha256=source_hash(),
        parent_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        command=command, environment={} if args.llama else dict(NEURON_ISA=args.isa, NEURON_THREADS=str(args.threads)),
        notes=args.notes, before=before, after=machine_state(),
        result=result, tg_tokens_per_second=1000 * (TOKENS - 1) / result['timings']['predicted_ms'])
    if args.llama:
        row['llama_commit'] = subprocess.check_output(['git', '-C', str(args.llamacpp_dir),
                                                      'rev-parse', 'HEAD'], text=True).strip()
    write_json(args.output, row)
    print(json.dumps(dict(step=args.step, tg=row['tg_tokens_per_second'])))


def chart(args):
    rows = [read_json(path) for path in args.inputs]
    maximum = max(r['tg_tokens_per_second'] for r in rows) * 1.15
    height = 90 + 48 * len(rows)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="820" height="{height}" viewBox="0 0 820 {height}">',
           '<rect width="100%" height="100%" fill="#fff"/>',
           '<g font-family="sans-serif" font-size="16" fill="#182637">',
           '<text x="20" y="28">Qwen2.5 1.5B Q4_K_M generation on an Intel Core i5-1135G7 laptop, tokens/s</text>']
    for i, row in enumerate(rows):
        y = 55 + i * 48
        width = 510 * row['tg_tokens_per_second'] / maximum
        label = f"{row['step']} ({row['threads']}t)"
        svg += [f'<text x="20" y="{y+21}">{html.escape(label)}</text>',
                f'<rect x="205" y="{y}" width="{width:.2f}" height="30" fill="#3478ba"/>',
                f'<text x="{215+width:.2f}" y="{y+21}">{row["tg_tokens_per_second"]:.3f}</text>']
    svg += ['</g></svg>']
    args.output.write_text('\n'.join(svg) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    run = sub.add_parser('measure')
    run.add_argument('--binary', type=Path, default=ROOT / 'build/neuron')
    run.add_argument('--model-file', type=Path, help='explicit GGUF; defaults to Qwen2.5 Q4_K_M')
    run.add_argument('--models-dir', type=Path, default=Path(os.environ.get('NEURON_MODELS_DIR', ROOT / 'models')))
    run.add_argument('--llamacpp-dir', type=Path, default=Path(os.environ.get('NEURON_LLAMACPP_DIR', ROOT / 'third_party/llama.cpp')))
    run.add_argument('--llama', action='store_true')
    run.add_argument('--step', required=True)
    run.add_argument('--isa', default='scalar')
    run.add_argument('--threads', type=int, default=1)
    run.add_argument('--notes', required=True)
    run.add_argument('--output', type=Path, required=True)
    plot = sub.add_parser('chart')
    plot.add_argument('inputs', type=Path, nargs='+')
    plot.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'measure':
        if args.threads < 1:
            parser.error('threads must be positive')
        measure(args)
    else:
        chart(args)


if __name__ == '__main__':
    main()
