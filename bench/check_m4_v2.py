#!/usr/bin/env python3
"""Run the precommitted second-round M4 method, without changing the old report."""
import argparse
import json
from pathlib import Path

from check_m4 import F16_SHA, LLAMA_COMMIT, measure
from harness import ROOT, read_json, sha256, token_comparison, write_json

TEXT_SHA = '1b33b1af5632849304fc096a1df248379a9dd8399b129492623393ce5b115acc'
PROMPTS_SHA = 'b5f26b7891b66f0f98a52e9bc362d823885b02883378ece309a7ac3ee2a79060'


def comparison(expected, actual):
    # Retain every planned position in the denominator after an early stop.
    details = token_comparison(expected, actual)
    details['total_tokens'] = 64
    details['agreement_rate'] = details['matching_tokens'] / 64
    return details


def summarize(rows, prompts):
    baseline = rows['F16']
    for name, data in rows.items():
        for field in ['greedy', 'teacher_forced']:
            comparisons = []
            for prompt, ref, run in zip(prompts, baseline['greedy'], data[field]):
                if ref['result']['prompt_tokens'] != run['result']['prompt_tokens']:
                    raise ValueError('prompt tokenization differs')
                comparisons.append(dict(prompt_id=prompt['id'], **comparison(
                    ref['result']['generated_tokens'], run['result']['generated_tokens'])))
            matched = sum(c['matching_tokens'] for c in comparisons)
            data[field + '_agreement'] = dict(matching_tokens=matched, total_tokens=768,
                                               rate=matched / 768, prompts=comparisons)
        ppl = data['perplexity']['result']
        reference = data['llama_perplexity']['result']
        if ppl['tokens'] != reference['tokens'] or ppl['tokens'] != baseline['perplexity']['result']['tokens']:
            raise ValueError('perplexity tokenization differs')
        if ppl['scored_tokens'] != 4096 or reference['scored_tokens'] != 4096:
            raise ValueError('requires exactly 4096 scored targets')
        data['perplexity_ratio_to_f16'] = ppl['perplexity'] / baseline['perplexity']['result']['perplexity']
        data['perplexity_ratio_to_llama'] = ppl['perplexity'] / reference['perplexity']
        data['maximum_rss_kib'] = max(r['resources']['maximum_rss_kib'] for r in [
            data['perplexity'], *data['greedy'], *data['teacher_forced']])
        data['llama_maximum_rss_kib'] = data['llama_perplexity']['resources']['maximum_rss_kib']
        timing = [r['result']['timings'] for r in data['greedy']]
        data['pp_tokens_per_second'] = 1000 * sum(t['prompt_n'] for t in timing) / sum(t['prompt_ms'] for t in timing)
        data['tg_tokens_per_second'] = 1000 * sum(max(0, t['predicted_n'] - 1) for t in timing) / sum(t['predicted_ms'] for t in timing)
        data['perplexity_scored_tokens_per_second'] = 4096 / ppl['seconds']
        data['llama_perplexity_scored_tokens_per_second'] = 4096 / reference['seconds']
    acceptance = {}
    for name in ['Q8_0', 'Q4_0', 'Q4_K_M']:
        d = rows[name]
        int8 = name == 'Q8_0'
        acceptance[name] = dict(
            perplexity_f16=abs(d['perplexity_ratio_to_f16'] - 1) <= (.01 if int8 else .10),
            greedy_agreement=d['greedy_agreement']['rate'] >= (.90 if int8 else .70))
        if not int8:
            acceptance[name]['perplexity_llama'] = abs(d['perplexity_ratio_to_llama'] - 1) <= .02
    acceptance['int8_pass'] = all(acceptance['Q8_0'].values())
    acceptance['four_bit_pass'] = any(all(acceptance[n].values()) for n in ['Q4_0', 'Q4_K_M'])
    acceptance['quality_pass'] = acceptance['int8_pass'] and acceptance['four_bit_pass']
    return acceptance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--reference-tool', type=Path, required=True)
    parser.add_argument('--models-dir', type=Path, required=True)
    parser.add_argument('--runs-dir', type=Path, default=ROOT / 'bench/runs/m4-v2')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--notes', required=True)
    args = parser.parse_args()
    text, prompt_file = ROOT / 'bench/wikitext2-m4-v2.txt', ROOT / 'bench/prompts-m4-v2.json'
    if sha256(text) != TEXT_SHA or sha256(prompt_file) != PROMPTS_SHA:
        raise ValueError('precommitted quality inputs changed')
    prompts = read_json(prompt_file)
    binary, reference = str(args.binary.resolve()), str(args.reference_tool.resolve())
    hashes = {b: sha256(b) for b in [binary, reference]}
    directory = args.runs_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    rows = {}
    for name, suffix in [('F16', 'f16'), ('Q8_0', 'Q8_0'), ('Q4_0', 'Q4_0'), ('Q4_K_M', 'Q4_K_M')]:
        model = (args.models_dir / f'Qwen2.5-1.5B-Instruct-{suffix}.gguf').resolve()
        model_hash = sha256(model)
        if name == 'F16' and model_hash != F16_SHA:
            raise ValueError('requires the pinned F16 model')
        data = dict(file=model.name, size_bytes=model.stat().st_size, sha256=model_hash)
        data['perplexity'] = measure([binary, 'perplexity', '-m', str(model), '--prompt-file', str(text),
            '--ctx-size', '512', '--chunks', '16'], name + '-perplexity', directory, model_hash, hashes[binary])
        data['llama_perplexity'] = measure([reference, str(model), '', '--prompt-file', str(text),
            '--perplexity', '512', '--chunks', '16'], name + '-llama-perplexity', directory, model_hash, hashes[reference])
        data['greedy'], data['teacher_forced'] = [], []
        for prompt in prompts:
            command = [binary, 'run', '-m', str(model), '-p', prompt['text'], '-n', '64',
                       '--greedy', '--json', '--ctx-size', '256']
            run = measure(command, name + '-' + prompt['id'], directory, model_hash, hashes[binary])
            data['greedy'].append(run)
            baseline = run if name == 'F16' else rows['F16']['greedy'][len(data['greedy']) - 1]
            tokens = baseline['result']['generated_tokens']
            if len(tokens) != 64:
                raise ValueError('F16 stopped before 64 tokens; report without changing the prompt set')
            teacher_path = directory / ('teacher-' + prompt['id'] + '.txt')
            teacher_text = ' '.join(map(str, tokens)) + '\n'
            if teacher_path.exists() and teacher_path.read_text() != teacher_text:
                raise ValueError('saved F16 teacher tokens changed')
            teacher_path.write_text(teacher_text)
            # F16 on its own continuation is the same run by construction.
            forced = run if name == 'F16' else measure(command + ['--teacher-tokens', str(teacher_path)],
                name + '-teacher-' + prompt['id'], directory, model_hash, hashes[binary])
            data['teacher_forced'].append(forced)
        rows[name] = data
        write_json(directory / 'partial.json', rows)
    passed = summarize(rows, prompts)
    summaries = {}
    for name, data in rows.items():
        path = args.output.with_name('m4-v2-' + name + '.json')
        write_json(path, data)
        summaries[name] = {k: v for k, v in data.items() if k not in
                           ['perplexity', 'llama_perplexity', 'greedy', 'teacher_forced']}
        summaries[name].update(measurements=path.name,
            perplexity=data['perplexity']['result']['perplexity'],
            llama_perplexity=data['llama_perplexity']['result']['perplexity'])
    write_json(args.output, dict(method_commit='6f55c17', llama_cpp_commit=LLAMA_COMMIT,
        text_sha256=TEXT_SHA, prompts_sha256=PROMPTS_SHA, context=512, chunks=16, scored_tokens=4096,
        scored_target_positions='256..511 per independent chunk', greedy_context=256, n_predict=64,
        threads=1, kv_type='F32', flash_attention=False, prompt_format='raw completion',
        repetitions=1, warmup_runs=0, notes=args.notes, formats=summaries, acceptance=passed))
    print(json.dumps(passed, indent=2), flush=True)
    return 0 if passed['quality_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
