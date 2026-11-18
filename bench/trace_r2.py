#!/usr/bin/env python3
"""Build a temporary instrumented model for the external R2 numerical check."""
import argparse
from pathlib import Path
import subprocess

from harness import ROOT, sha256, write_json
from speed import source_hash


def instrument(source):
    def insert(anchor, extra):
        nonlocal source
        if source.count(anchor) != 1:
            raise ValueError('trace anchor changed: ' + anchor)
        source = source.replace(anchor, anchor + '\n' + extra)

    insert('  auto scores = std::span(scores_).first(position_ + 1);', '''  const bool tracing = position_ == r2::position;
  std::string suffix;
  auto trace = [&](const std::string& name, std::span<const float> values) {
    if (tracing) r2::save(name + suffix, values);
  };
  std::vector<float> raw_scores(c.heads * (position_ + 1)), probabilities(raw_scores.size());''')
    insert('  embedding_.row(static_cast<std::size_t>(token), x);', '  trace("embd", x);')
    insert('  for (auto& l : layers_) {\n    scalar::rmsnorm(x, l.attn_norm.data, normalized, c.rms_epsilon);',
           '    suffix = "-" + std::to_string(&l - layers_.data());\n    trace("attn_norm", normalized);')
    anchor = '    rotate(q); rotate(k);'
    if source.count(anchor) != 1:
        raise ValueError('RoPE trace anchor changed')
    source = source.replace(anchor, '    trace("Qpre", q); trace("Kpre", k);\n' + anchor)
    insert(anchor, '    trace("Qcur", q); trace("Kcur", k); trace("Vcur", v);')
    insert('    l.keys.insert(l.keys.end(), k.begin(), k.end()); l.values.insert(l.values.end(), v.begin(), v.end());',
           '    trace("Kcache", l.keys); trace("Vcache", l.values);')
    insert('        scores[t] = static_cast<float>(dot) * scale;',
           '        raw_scores[h * (position_ + 1) + t] = static_cast<float>(dot);')
    insert('      scalar::softmax(scores);',
           '      std::copy(scores.begin(), scores.end(), probabilities.begin() + static_cast<std::ptrdiff_t>(h * (position_ + 1)));')
    insert('    l.attn_output.multiply(attention, projected, c.width, c.width);',
           '    trace("kq", raw_scores); trace("kq_soft_max", probabilities); trace("kqv_out", attention); trace("attn_out", projected);')
    insert('    scalar::rmsnorm(x, l.ffn_norm.data, normalized, c.rms_epsilon);',
           '    trace("ffn_inp", x); trace("ffn_norm", normalized);')
    insert('    l.up.multiply(normalized, up, c.hidden, c.width);',
           '    trace("ffn_gate", gate); trace("ffn_up", up);')
    insert('    l.down.multiply(gate, projected, c.width, c.hidden);',
           '    trace("ffn_swiglu", gate); trace("ffn_out", projected);')
    anchor = '    for (std::size_t d = 0; d < c.width; ++d) x[d] += projected[d];\n  }'
    if source.count(anchor) != 1:
        raise ValueError('layer output trace anchor changed')
    source = source.replace(anchor, anchor.replace('\n  }', '\n    trace("l_out", x);\n  }'))
    insert('  scalar::rmsnorm(x, output_norm_.data, normalized, c.rms_epsilon);',
           '  suffix.clear(); trace("result_norm", normalized);')
    insert('  output_.multiply(normalized, logits_, c.vocabulary, c.width);', '  trace("result_output", logits_);')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True, help='outside the repository')
    parser.add_argument('--neuron-build', type=Path, required=True)
    parser.add_argument('--llamacpp-dir', type=Path, required=True)
    args = parser.parse_args()
    directory = args.build_dir.resolve()
    if directory.is_relative_to(ROOT):
        raise ValueError('keep diagnostic binaries outside the repository')
    directory.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'src/model.cpp'
    instrumented = directory / 'r2_model.cpp'
    instrumented.write_text(instrument(source.read_text()))
    llama = args.llamacpp_dir.resolve()
    command = ['g++', '-std=c++20', '-O3', '-Wall', '-Wextra', '-Werror', str(ROOT / 'bench/r2_trace.cpp'),
        '-I' + str(directory), '-I' + str(ROOT / 'include'), '-I' + str(args.neuron_build.resolve() / 'generated'),
        '-I' + str(llama / 'include'), '-I' + str(llama / 'ggml/include'), '-I' + str(llama / 'vendor/nlohmann'),
        '-I' + str(llama / 'ggml/src'),
        str(args.neuron_build.resolve() / 'libneuron.a'), str(llama / 'build/src/libllama.a'),
        str(llama / 'build/ggml/src/libggml.a'), str(llama / 'build/ggml/src/libggml-cpu.a'),
        str(llama / 'build/ggml/src/libggml-base.a'), '-fopenmp', '-pthread', '-ldl', '-o', str(directory / 'r2-trace')]
    subprocess.run(command, check=True)
    write_json(directory / 'build.json', dict(command=command, model_source_sha256=sha256(source),
        instrumented_sha256=sha256(instrumented), probe_sha256=sha256(ROOT / 'bench/r2_trace.cpp'),
        builder_sha256=sha256(__file__), neuron_source_sha256=source_hash(),
        neuron_library_sha256=sha256(args.neuron_build / 'libneuron.a'),
        neuron_binary_sha256=sha256(args.neuron_build / 'neuron'),
        llama_cpp_commit=subprocess.check_output(['git', '-C', str(llama), 'rev-parse', 'HEAD'], text=True).strip(),
        binary_sha256=sha256(directory / 'r2-trace')))


if __name__ == '__main__':
    main()
