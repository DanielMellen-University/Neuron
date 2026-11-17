#!/usr/bin/env python3
"""Reproduce the local M5 build checks and retain their results."""
import argparse
import os
import subprocess
from pathlib import Path

from harness import ROOT, machine_state, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--build-prefix', default='build-m5-final', help='fresh build directory prefix')
    parser.add_argument('--configs', nargs='+', default=['default', 'werror', 'avx2', 'avx512', 'sanitize', 'clang', 'clang-avx2'])
    args = parser.parse_args()
    report = dict(before=machine_state(), checks=[])
    for config in args.configs:
        directory = args.build_prefix + '-' + config
        flags = ['-DCMAKE_CXX_COMPILER=g++', '-DCMAKE_BUILD_TYPE=' + ('Debug' if config == 'sanitize' else 'Release')]
        if config != 'default':
            flags += ['-DNEURON_WERROR=ON']
        if config == 'sanitize':
            flags += ['-DNEURON_SANITIZE=ON']
        if config == 'avx2' or config == 'clang-avx2':
            flags += ['-DNEURON_AVX2=ON']
        if config == 'avx512':
            flags += ['-DNEURON_AVX512=ON']
        if config.startswith('clang'):
            flags[0] = '-DCMAKE_CXX_COMPILER=clang++'
            flags += ['-DCMAKE_CXX_FLAGS=--gcc-toolchain=/usr -isystem /usr/include/c++/11 -isystem /usr/include/x86_64-linux-gnu/c++/11',
                      '-DCMAKE_EXE_LINKER_FLAGS=-L/usr/lib/gcc/x86_64-linux-gnu/11']
        environment = dict(NEURON_ISA='avx2' if config in ['avx2', 'clang-avx2'] else 'auto', NEURON_THREADS='1')
        for command in [['cmake', '-S', '.', '-B', directory, *flags],
                        ['cmake', '--build', directory, '-j', '2'],
                        ['ctest', '--test-dir', directory, '--output-on-failure', '-V']]:
            run = subprocess.run(command, cwd=ROOT, env=dict(os.environ, **environment),
                                 text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            report['checks'].append(dict(config=config, command=command, environment=environment,
                                         returncode=run.returncode, log=run.stdout))
            write_json(args.output, report)
            print(config, command[0], run.returncode, flush=True)
            if run.returncode:
                print(run.stdout, flush=True)
                raise SystemExit(run.returncode)
        report['checks'][-1]['binary_sha256'] = sha256(ROOT / directory / 'neuron')
    report['after'] = machine_state()
    write_json(args.output, report)


if __name__ == '__main__':
    main()
