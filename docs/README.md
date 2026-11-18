# Neuron docs

This is the documentation hub for Neuron. Start with the [README](../README.md) if you want the short version: what it is, how to run it, and how fast it is.

- [Design and optimization](blog/design-and-optimization.md): how loading, tokenization, inference, quantization, SIMD and threading work, and the measured gaps.
- [Command-line reference](cli.md): sampling, stop strings, the fields the HTTP API accepts, `neuron inspect`, and the token tools.
- [Terminal demo](demo.md): the real capture, timing policy and GIF reproduction.
- [M7 check sheet](../bench/M7_CHECKS.md): where every public number comes from, plus reproduce commands and local validation.
- [Roadmap](../ROADMAP.md): milestone status, acceptance criteria and the missed targets.

The benchmark directory keeps the detailed methods and raw JSON records:

- [Baseline](../bench/BASELINE.md): pinned llama.cpp and model inputs.
- [TinyStories checks](../bench/M2_CHECKS.md): tokenizer, logits and generation.
- [Qwen checks](../bench/M3_CHECKS.md): ChatML, KV cache and memory.
- [Quantization](../bench/QUANT.md): formats, perplexity and agreement misses.
- [Speed](../bench/SPEED.md): optimization commits, chart, original M5 miss and R1 result.
- [R1 profile and checks](../bench/R1_PROFILE.md): K kernels, allocation counts,
  thread sweep, builds and perplexity regression checks.
- [Chat and server](../bench/M6_CHECKS.md): terminal, HTTP, web page and SDK checks.

Models and the reference engine live outside the source tree. Build artifacts and scratch measurement outputs are ignored; committed records contain no weights. Chart generation and table derivation use the Python standard library. Only rendering the terminal GIF needs Pillow and a monospace font.
