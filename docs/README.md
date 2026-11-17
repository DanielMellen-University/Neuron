# Neuron docs

Start with the [quick start and speed table](../README.md).

- [Design and optimization](blog/design-and-optimization.md): loading,
  tokenization, inference, quantization, SIMD, threading and the measured gaps.
- [Terminal demo](demo.md): the real capture, timing policy and GIF reproduction.
- [M7 check sheet](../bench/M7_CHECKS.md): table provenance, reproduce commands,
  local validation and acceptance status.
- [Roadmap](../ROADMAP.md): milestone status and unchanged acceptance criteria.

The benchmark directory keeps the detailed methods and raw JSON records:

- [Baseline](../bench/BASELINE.md): pinned llama.cpp and model inputs.
- [TinyStories checks](../bench/M2_CHECKS.md): tokenizer, logits and generation.
- [Qwen checks](../bench/M3_CHECKS.md): ChatML, KV cache and memory.
- [Quantization](../bench/QUANT.md): formats, perplexity and agreement misses.
- [Speed](../bench/SPEED.md): optimization commits, chart, original M5 miss and R1 result.
- [R1 profile and checks](../bench/R1_PROFILE.md): K kernels, allocation counts,
  thread sweep, builds and perplexity regression checks.
- [Chat and server](../bench/M6_CHECKS.md): terminal, HTTP, web page and SDK checks.

Models and the reference engine live outside the source tree. Build artifacts
and scratch measurement outputs are ignored; committed records contain no
weights. Chart generation and table derivation use the Python standard library.
Only rendering the terminal GIF needs Pillow and a monospace font.
