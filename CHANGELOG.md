# Changelog

All notable changes to Neuron. Numbers come from the test laptop (an Intel
Core i5-1135G7, four threads) and trace to files under `bench/`.

## v1.0.0 - 2026-11-18

The first release. Neuron is a from-scratch C++20 inference engine that
runs dense Llama and Qwen2 GGUF models on an x86-64 Linux CPU, with a
terminal chat, a local web chat and an OpenAI-compatible API.

### Added

- `scripts/quickstart.sh` builds the engine, fetches SmolLM2 135M Instruct
  and starts a chat in one command (`--serve` starts the web chat instead).
  `scripts/fetch_models.sh` can now fetch only the models you name.
- Seeded sampling in `neuron run`, `neuron chat` and `neuron serve`:
  temperature, top-k, top-p, min-p and a repeat penalty, with a SplitMix64
  random source so a seed gives the same tokens with any compiler (#11, #13).
- Stop strings on the CLI and in the API, matched across token boundaries so
  no text past the stop is ever shown (#13).
- `temperature`, `top_p`, `seed` and `stop` in `POST /v1/chat/completions`,
  plus the llama.cpp style `top_k`, `min_p` and `repeat_penalty` (#13).
- `neuron inspect` lists a GGUF file's metadata and tensors without loading
  the weights, with `--json`, `--key`, `--no-tensors` and `--array-items` (#14).
- `neuron detokenize`, `neuron tokenize --pieces` and `--prompt-file -` for
  reading a prompt from standard input (#15).

### Fixed

- Prompt files are read through the stream buffer, which clears a GCC 14
  warning that broke `-DNEURON_WERROR=ON` builds (#15).
- Docs: R1 and R2 in the roadmap status table, sampling described where the
  docs still said greedy only, the blog chart command now rebuilds the
  committed chart, and repeated tables replaced with links to
  `bench/SPEED.md` and `bench/QUANT.md`.

### Milestones in this release

- R2 free-running agreement (#10): the M4 agreement check repeated in both
  engines. Neuron reproduces its M4 results exactly (Q8_0 71.74%, Q4_0
  6.64%, Q4_K_M 23.44%) and llama.cpp misses the same targets (59.51%,
  6.38%, 20.31%). No inference defect was found. Also fixes early-stop
  token accounting. See `bench/R2_AGREEMENT.md`.
- R1 decode speed (#9): specialized Q4_K/Q6_K SIMD word products, checked
  VNNI dispatch, shared activation packing and reusable decode buffers.
  Q4_K_M generation went from 11.02% to 55.02% of llama.cpp (21.22 vs
  38.56 tokens/s). See `bench/SPEED.md` and `bench/R1_PROFILE.md`.
- M7 polish (#7, #8): README with a measured speed table, a real terminal
  demo GIF, the design write-up and a docs hub.
- M6 chat and server (#6): terminal chat with history and Ctrl-C, an
  embedded web chat page and an OpenAI-compatible server that the official
  Python client works against.
- M5 kernels and threads (#5): AVX2, AVX-512 and VNNI kernels chosen at
  runtime, and a thread pool that splits work by output row. The 50% speed
  target was missed at the time (11.02%) and accepted; R1 later reached it.
- M4 quantization (#4): Q8_0, Q4_0, Q4_K and Q6_K weights. Perplexity tracks
  llama.cpp within 0.9% for every format. The free-running agreement
  targets were missed and accepted.
- M3 Qwen2.5 (#3): Qwen2.5 1.5B Instruct with grouped-query attention, QKV
  bias, byte BPE, the ChatML template and a KV cache.
- M2 first forward pass (#2): GGUF loading, the Llama tokenizer and an FP32
  forward pass that matches llama.cpp token for token on TinyStories.
- M1 baseline (#1): pinned llama.cpp and model downloads, a reference
  harness and a compare mode.

### Known limits

- x86-64 Linux only, CPU only. Dense Llama and Qwen2 models only.
- Slower than llama.cpp, especially for Q8_0 and Q4_0.
- The server handles one request at a time, with no authentication or TLS.
