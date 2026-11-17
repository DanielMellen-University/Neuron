#!/usr/bin/env python3
"""Derive the README speed table from committed timing records. No model runs."""
import argparse
import json
from pathlib import Path

from harness import ROOT, read_json, sha256, summarize, write_json

PAIRS = [
    ('TinyStories 15M', 'F32', 'tinystories-m5.json', 'tinystories-15m.json'),
    ('SmolLM2 135M Instruct', 'F16', 'm7-speed-smollm-neuron.json', 'm7-speed-smollm-llama.json'),
    ('Qwen2.5 1.5B Instruct', 'F16', 'm7-speed-f16-neuron.json', 'm7-speed-f16-llama.json'),
    ('Qwen2.5 1.5B Instruct', 'Q8_0', 'm7-speed-q8_0-neuron.json', 'm7-speed-q8_0-llama.json'),
    ('Qwen2.5 1.5B Instruct', 'Q4_0', 'm7-speed-q4_0-neuron.json', 'm7-speed-q4_0-llama.json'),
    ('Qwen2.5 1.5B Instruct', 'Q4_K_M', 'm5-speed-final.json', 'm5-speed-llama.json'),
]


def measurement(name):
    path = ROOT / 'bench/reference' / name
    data = read_json(path)
    if 'runs' in data:
        rows = [r for r in data['runs'] if r['threads'] == 4]
        rates = summarize(rows)[0]
        tokens = [r['prompt_tokens'] for r in rows]
        protocol = 'M1 prompts, context 2048, 128 generated tokens, three repetitions'
    else:
        rows = [data['result']]
        rates = summarize([dict(rows[0], threads=data['threads'])])[0]
        tokens = [rows[0]['prompt_tokens']]
        protocol = 'Once upon a time, context 256, 32 generated tokens, one repetition, no warmup'
        if (data['threads'], data['context'], data['n_predict'], data['warmups'], data['repetitions']) != (4, 256, 32, 0, 1):
            raise ValueError(f'unexpected protocol in {name}')
    return data, tokens, dict(source='bench/reference/' + name, source_sha256=sha256(path),
                             protocol=protocol, **rates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'bench/reference/m7-benchmarks.json')
    args = parser.parse_args()
    table = []
    for model, quant, neuron, llama in PAIRS:
        a, a_tokens, left = measurement(neuron)
        b, b_tokens, right = measurement(llama)
        if (a['model']['sha256'] != b['model']['sha256'] or a_tokens != b_tokens
                or left['protocol'] != right['protocol']):
            raise ValueError(f'inputs differ for {model} {quant}')
        table.append(dict(model=model, quant=quant, model_sha256=a['model']['sha256'],
                          threads=4, neuron=left, llama_cpp=right))
    write_json(args.output, dict(machine='Ghost', rows=table,
        method='Rates pool token counts and milliseconds. Generation excludes the first sampled token. '
               'TinyStories reuses M1/M5; Q4_K_M reuses M5; other rows fill missing matched speed pairs in M7.'))
    print('| Model | GGUF | Neuron pp | Neuron tg | llama.cpp pp | llama.cpp tg | Records |')
    print('| --- | --- | ---: | ---: | ---: | ---: | --- |')
    for row in table:
        a, b = row['neuron'], row['llama_cpp']
        print(f"| {row['model']} | {row['quant']} | {a['pp_tokens_per_second']:.2f} | "
              f"{a['tg_tokens_per_second']:.2f} | {b['pp_tokens_per_second']:.2f} | "
              f"{b['tg_tokens_per_second']:.2f} | [N]({a['source']}) / [L]({b['source']}) |")


if __name__ == '__main__':
    main()
