# Neuron

[![ci](https://github.com/DanielMellen-University/Neuron/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/DanielMellen-University/Neuron/actions/workflows/ci.yml)
![C++20](https://img.shields.io/badge/C%2B%2B-20-00599c)
![x86-64 Linux](https://img.shields.io/badge/platform-x86--64%20Linux-555555)
![Status: in progress](https://img.shields.io/badge/status-in%20progress-e3a008)
[![License: MIT](https://img.shields.io/badge/license-MIT-8fd19a)](LICENSE)

**A from-scratch C++ LLM inference engine for x86 CPUs.**

Neuron loads a small transformer language model and generates text on a laptop CPU, with no GPU and no inference libraries underneath. The target machine is an Intel i5-1135G7 (4 cores, 8 threads, 16 GB RAM, AVX2 and AVX-512 with VNNI). The end goal is Qwen2.5 1.5B Instruct, quantized to 4 bits, chatting at a usable speed.

> **Status: M4 in progress.** Neuron loads Q8_0, Q4_0, Q4_K and Q6_K weights. The larger fixed check scores 4,096 targets and uses twelve prompts. Q4_K_M meets both 4-bit perplexity limits, but Q8_0 and Q4_K_M free-running agreement are 71.74% and 23.44%, below 90% and 70%. Teacher forcing does not replace those criteria. [Quantization checks](bench/QUANT.md) record sizes, memory, speeds, results and local validation. [ROADMAP.md](ROADMAP.md) keeps M4 in progress.

## Current features

- **GGUF loading** with checked metadata and tensor tables, mapped F32/Q8_0/Q4_0/Q4_K/Q6_K weights and F16 weights converted to FP32.
- **Llama and Qwen2 tokenizers** with score-based merging or byte BPE, Unicode pre-tokenization and special tokens.
- **Scalar FP32 inference** for Llama and Qwen2 with RMSNorm, adjacent-pair or split-half RoPE, GQA, QKV bias and SwiGLU.
- **Per-layer KV cache** with checked context bounds.
- **Qwen2.5 ChatML** with a default or explicit system prompt and multi-turn message history.
- **Greedy CLI generation** with streaming text, JSON token IDs, top-two logit gaps and reference checks.
- **Perplexity evaluation** with fixed context chunks and a pinned short quality check.

## Planned features

- AVX-512 and VNNI kernels plus multithreading, with a speed chart for each optimization step.
- Terminal chat, a small web chat UI and an OpenAI-compatible local HTTP server.

## Quick start

Requires CMake 3.20+, Python 3.8+ for tests, and GCC 11+ or Clang 14+ on x86-64 Linux.

```
git clone https://github.com/DanielMellen-University/Neuron.git
cd Neuron
cmake -S . -B build                # Release by default
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/neuron --version
```

Build options:

| Option | Default | Effect |
| --- | --- | --- |
| `NEURON_AVX2` | `OFF` | Compile with AVX2, FMA, and F16C |
| `NEURON_AVX512` | `OFF` | Compile with AVX-512 F/BW/DQ/VL/VNNI (implies AVX2) |
| `NEURON_NATIVE` | `OFF` | Compile with `-march=native` |
| `NEURON_WERROR` | `OFF` | Treat warnings as errors (CI turns this on) |
| `NEURON_SANITIZE` | `OFF` | AddressSanitizer and UndefinedBehaviorSanitizer |
| `NEURON_BUILD_TESTS` | `ON` | Build the unit tests |

`./build/neuron --version` prints the instruction sets the binary was built for, for example `neuron 0.1.0 (avx512f avx512vnni avx2 fma)`.

Model weights are never committed. `scripts/fetch_models.sh` downloads pinned TinyStories 15M F32, SmolLM2 135M Instruct F16 and Qwen2.5 1.5B Instruct F16 GGUF files into the git-ignored `models/` directory and checks their SHA-256 hashes. Set `NEURON_MODELS_DIR` to use another directory.

Generate with the validated TinyStories model:

```sh
scripts/fetch_models.sh
./build/neuron run -m models/stories15M.gguf -p "Once upon a time" -n 128 --greedy
# Emit prompt/generated IDs and timings instead of text:
./build/neuron run -m models/stories15M.gguf -p "Once upon a time" -n 128 --greedy --json
./build/neuron tokenize -m models/stories15M.gguf -p "Hello world"
```

Run Qwen2.5 with its ChatML template:

```sh
./build/neuron run -m models/Qwen2.5-1.5B-Instruct-f16.gguf \
  --chat -p "Say hello briefly." -n 32 --greedy
# Supply a system prompt and previous messages for a later turn:
./build/neuron run -m models/Qwen2.5-1.5B-Instruct-f16.gguf \
  --system "Reply briefly." \
  --message user "My name is Ada." --message assistant "Hello Ada." \
  -p "What is my name?" -n 32 --greedy
```

`--message ROLE TEXT` supplies history in order; `-p` adds the final user
message in chat mode. Omit `--system` to use the model's default system text.
`neuron template` prints the rendered ChatML prompt with the same message
flags, and `neuron tokenize` prints its IDs. `--prompt-file FILE` reads prompt
bytes instead of `-p`. These commands support text-only Qwen2.5 chat.
Interactive terminal chat remains M6 work. Qwen's F16 weights are upcast to
FP32. Q8_0, Q4_0, Q4_K and Q6_K matrices stay mapped and decode during scalar products.
[Quantization checks](bench/QUANT.md) document how to produce these GGUFs
with the pinned external llama.cpp quantizer and compare their quality.

Evaluate the fixed WikiText-2 slice with a quantized model:

```sh
./build/neuron perplexity -m models/Qwen2.5-1.5B-Instruct-Q4_0.gguf \
  --prompt-file bench/wikitext2-m4-v2.txt --ctx-size 512 --chunks 16
```

This resets the cache between chunks and scores each chunk's second half.
It writes JSON with loss, perplexity, counts and input token IDs. Omit
`--chunks` to evaluate all complete chunks; an incomplete tail is ignored.

The CLI defaults to a 2048-token context to match M1. TinyStories declares only 128 training tokens; `--ctx-size 128` enforces that smaller limit. A prompt plus requested generation exceeding the selected limit is rejected. The override leaves the model file unchanged and makes no quality claim beyond its training context.

To compare Neuron with the committed M1 tokens:

```sh
python3 bench/harness.py run-neuron --binary build/neuron \
  --model-file models/stories15M.gguf --output bench/runs/tinystories.json \
  --notes 'Describe the actual machine load and power state.'
python3 bench/harness.py compare bench/reference/tinystories-15m.json \
  bench/runs/tinystories.json --threads 1 --exact-match
```

Neuron currently uses one thread. `--threads 1` selects the corresponding reference rows; it does not change Neuron's execution. [bench/M2_CHECKS.md](bench/M2_CHECKS.md) also documents the full-vocabulary logit check and tokenizer fixtures. [M3 checks](bench/M3_CHECKS.md) provide the separate Qwen tokenizer, ChatML, greedy and memory commands.

The [llama.cpp baseline](bench/BASELINE.md) has measured speeds, reference tokens, and exact commands to build and run the external reference engine. This requires Python 3.8+, Git, curl, GCC and CMake. `python3 bench/harness.py compare reference.json candidate.json --exact-match` compares prompt and generated token IDs and reports speed ratios. The comparison tests run without models or llama.cpp.

## Inference path and planned extensions

```mermaid
flowchart LR
  P[prompt] --> T[Llama or Qwen2 tokenizer]
  G[(GGUF file)] -->|mmap| L[model loader]
  T --> F
  L --> F[forward pass<br/>RMSNorm, attention, SwiGLU]
  F <--> K[KV cache]
  F --> S[sampler: greedy<br/>planned: temperature, top-p]
  S -->|next token| F
  S --> O[text out]
  subgraph kernels
    M[matmul F32, Q8_0, Q4_0, Q4_K, Q6_K scalar<br/>planned: AVX2, AVX-512 VNNI]
  end
  F --> M
```

1. The loader maps the GGUF file and checks its tensor table. F32 and quantized matrices stay mapped; F16 weights are converted to FP32 once on load.
2. The tokenizer turns the prompt into token ids.
3. Each forward pass runs one token through every transformer layer and appends that token's keys and values to the KV cache, so earlier tokens are not recomputed.
4. The sampler picks the next token from the logits and feeds it back in.
5. Matrix-vector products use scalar reference kernels. Later SIMD kernels will be checked against those references.

## Limitations

These are true today and some will stay true by design.

- **Dense Llama and Qwen2 F32/F16/Q8_0/Q4_0/Q4_K/Q6_K models only.** TinyStories and Qwen2.5 1.5B Instruct are the local check targets. Other tokenizer families, other quantization formats, RoPE scaling and tool-call chat templates are not supported.
- **Scalar and single-threaded.** There are no optimized SIMD kernels, interactive terminal chat, web UI or server yet.
- **x86-64 Linux only.** No ARM, macOS, or Windows builds are planned.
- **CPU only.** No CUDA, Metal, or Vulkan.
- **Small models.** The engine targets models up to about 1.5B parameters that fit in 16 GB of RAM.
- **Single user.** The planned server handles one conversation at a time; there is no batching across requests.

## Testing

```
cmake -S . -B build -DNEURON_WERROR=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

GGUF, tokenizer, kernel and synthetic transformer tests run without models or llama.cpp. C++ tests use a small header-only framework in [tests/neuron_test.hpp](tests/neuron_test.hpp) (registration, `CHECK`/`REQUIRE` macros, and a runner) driven by CTest. Harness comparison tests use Python 3.8+ and its standard library. There are no third-party test dependencies. CI builds and tests with GCC and Clang, with and without AVX2, runs an ASan and UBSan job, checks that the AVX-512 build compiles, and rejects any committed weight file.

## Project layout

```
include/neuron/     GGUF, tokenizer, model, kernels, shape and version headers
src/                library sources and the neuron CLI
tests/              unit tests and the test framework
scripts/            pinned model downloads and external llama.cpp build
bench/              prompts, reference tokens, comparison harness and baseline
.github/workflows/  CI
ROADMAP.md          milestones and acceptance criteria
```

## Docs

- [ROADMAP.md](ROADMAP.md): milestones M1 to M7 and their acceptance criteria
- [bench/BASELINE.md](bench/BASELINE.md): llama.cpp reference measurements and reproduction
- [bench/M2_CHECKS.md](bench/M2_CHECKS.md): TinyStories tokenizer, logits and exact generation checks
- [bench/M3_CHECKS.md](bench/M3_CHECKS.md): Qwen2.5, ChatML, cache, context and memory checks
- [bench/QUANT.md](bench/QUANT.md): Q8_0/Q4_0/Q4_K/Q6_K formats, perplexity and greedy agreement
- [LICENSE](LICENSE): MIT
