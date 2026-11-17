#!/usr/bin/env python3
"""Capture R1 allocation counts and a gprof flat profile."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from harness import ROOT, machine_state, sha256, write_json

BASELINE = '2646c5d'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-build', type=Path, required=True)
    parser.add_argument('--after-build', type=Path, required=True)
    parser.add_argument('--gprof-build', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    directory = ROOT / 'bench/runs/r1-profile'
    headers = directory / 'headers/neuron'
    headers.mkdir(parents=True, exist_ok=True)
    (headers / 'model.hpp').write_bytes(subprocess.check_output(['git', 'show', BASELINE + ':include/neuron/model.hpp']))
    report = dict(before=machine_state(), model_sha256=sha256(args.model), baseline_commit=BASELINE,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        driver_sha256=sha256(ROOT / 'bench/profile_decode.cpp'), allocations={})
    for name, build in [('before', args.before_build), ('after', args.after_build)]:
        binary = directory / name
        command = ['g++', '-std=c++20', '-O3', '-Wall', '-Wextra', '-Werror']
        if name == 'before':
            command += ['-DNEURON_PROFILE_LEGACY', '-I' + str(headers.parent)]
        command += ['-Iinclude', '-I' + str(build / 'generated'), 'bench/profile_decode.cpp',
                    str(build / 'libneuron.a'), '-pthread', '-o', str(binary)]
        subprocess.run(command, cwd=ROOT, check=True)
        run = [str(binary), str(args.model.resolve()), '4', 'avx512']
        result = json.loads(subprocess.check_output(run, text=True))
        report['allocations'][name] = dict(compile_command=command, command=run, result=result,
            binary_sha256=sha256(binary), library_sha256=sha256(build / 'libneuron.a'))
        print(name, result, flush=True)
        write_json(args.output, report)
    binary = args.gprof_build.resolve() / 'neuron'
    command = [str(binary), 'run', '-m', str(args.model.resolve()), '-p', 'Once upon a time',
               '-n', '32', '--greedy', '--json', '--ctx-size', '256']
    prefix = directory / ('gmon-after-' + str(os.getpid()))
    env = dict(os.environ, GMON_OUT_PREFIX=str(prefix), NEURON_ISA='avx512', NEURON_THREADS='4')
    run = json.loads(subprocess.check_output(command, env=env, text=True))
    files = list(directory.glob(prefix.name + '.*'))
    profile = subprocess.check_output(['gprof', '-b', '-p', str(binary), *map(str, files)], text=True)
    profile_path = args.output.with_name('r1-gprof-after.txt')
    profile_path.write_text(profile)
    report['gprof_after'] = dict(command=command, binary_sha256=sha256(binary), result=run,
        environment={k: env[k] for k in ['GMON_OUT_PREFIX', 'NEURON_ISA', 'NEURON_THREADS']},
        profile=str(profile_path.relative_to(ROOT)), flags='GCC Release, -pg compile and link')
    report['perf_event_paranoid'] = Path('/proc/sys/kernel/perf_event_paranoid').read_text().strip()
    report['after'] = machine_state()
    write_json(args.output, report)


if __name__ == '__main__':
    main()
