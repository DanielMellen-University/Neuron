# Roadmap

Neuron is built in seven milestones. Each one ends with a merged pull request, green CI at the merge commit, and a short write-up of how that part works. A milestone is done only when every acceptance item below is true and checkable from the repo.

Hardware for every number in this file: Ghost, an Intel i5-1135G7 (4 cores, 8 threads, 16 GB RAM, AVX2, AVX-512 with VNNI), Ubuntu 22.04, GCC 11.4. Benchmarks record the thread count, model file hash, prompt, and power state.

Model weights are never committed. Scripts download them into the git-ignored `models/` directory and verify a pinned SHA-256.

| Milestone | Status |
| --- | --- |
| M0 Skeleton: build, CI, test harness, README, roadmap | done |
| M1 llama.cpp baseline and harness | done |
| M2 TinyStories FP32 forward pass and tokenizer | done |
| M3 Qwen2.5 1.5B load, chat template, KV cache | done |
| M4 int8 and 4-bit quantization | done |
| M5 AVX-512 kernels and multithreading | done (speed target missed, accepted) |
| M6 Terminal chat, web UI, OpenAI-compatible server | done |
| M7 Polish: README, benchmarks, demo GIF, blog post | done |

## M1: llama.cpp baseline and harness

Goal: a fixed reference to measure Neuron against, and the tooling to compare against it.

Completed on Ghost: pinned download and build scripts, three fixed prompts,
greedy reference tokens and measured speeds at 1, 4 and 8 threads, and a
model-free compare test suite in CTest. [bench/BASELINE.md](bench/BASELINE.md)
records the inputs, power state, results, reproduction commands and local
validation. M2 adds Neuron inference against that reference.

Acceptance criteria:

- `scripts/fetch_models.sh` downloads TinyStories (15M or 110M GGUF) and SmolLM2 135M Instruct GGUF into `models/`, checks pinned SHA-256 hashes, and is idempotent.
- `scripts/build_llamacpp.sh` builds llama.cpp at a pinned commit outside the source tree (for example `third_party/llama.cpp`, git-ignored, or a path from an env var). Nothing from llama.cpp is vendored into the repo.
- A harness (`bench/` or `tools/harness/`) runs a fixed prompt set with greedy decoding and a fixed seed and records, per model: prompt token ids, generated token ids, generated text, and prompt-processing and generation tokens/s at 1, 4, and 8 threads.
- Reference outputs are committed as small text or JSON files under `bench/reference/` (token ids and text, not weights or logits dumps over 1 MiB).
- The harness has a compare mode that takes two result files and reports first divergent token index, token agreement rate, and a speed ratio. It exits non-zero on any token mismatch when asked for exact match.
- `bench/BASELINE.md` has a table of llama.cpp tokens/s on Ghost for each model and thread count, plus the exact commands to reproduce it.
- The compare mode has unit tests that run in CI without any model files.

## M2: TinyStories FP32 forward pass and tokenizer

Goal: Neuron generates the same text as llama.cpp on a tiny model.

Completed locally on Ghost: checked GGUF loading, llama vocabulary tokenization,
scalar FP32 inference and greedy CLI generation. All 51 tokenizer cases match
llama.cpp. Final-position logits differ by at most 9.54e-6 across three prompts.
All 128 generated IDs match M1 for each prompt in three repetitions.
[bench/M2_CHECKS.md](bench/M2_CHECKS.md) records the method, exact commands,
committed results and model-free test coverage. Remote CI remains to run
on the pull request.

Acceptance criteria:

- A GGUF reader parses the header, metadata, and tensor table, memory-maps the file, and rejects truncated or malformed files with a clear error (covered by tests on small synthetic GGUF files built in the test).
- A tokenizer built from the GGUF vocabulary matches llama.cpp token for token on the M1 prompt set plus an edge-case list (empty string, leading spaces, Unicode, digits, byte fallback).
- An FP32 forward pass (embedding, RMSNorm, RoPE, attention, SwiGLU or the model's FFN, output head) whose final-position logits match llama.cpp within 1e-3 max absolute difference on at least 3 prompts. F16 weights are upcast to FP32 at load time.
- `neuron run -m models/<tinystories>.gguf -p "<prompt>" -n 128 --greedy` produces exactly the M1 reference token ids for every prompt in the set.
- Scalar reference kernels for matmul, RMSNorm, softmax, and RoPE have unit tests against hand-computed values.
- CI covers the tokenizer and kernel tests without model files; the end-to-end match is a documented local check with its result recorded in the PR.

## M3: Qwen2.5 1.5B load, chat template, KV cache

Goal: the real target model runs and holds a conversation.

Completed locally on Ghost: Qwen2 F16 loading with GQA and QKV bias,
split-half RoPE, byte BPE, ChatML and a multi-turn message CLI. All 63
tokenizer cases and four chat fixtures match llama.cpp. All three M1
prompts match the first 32 generated IDs exactly. Cached generation matches
full recomputation for 32 tokens in synthetic Llama/Qwen2 tests and on
TinyStories. A real Qwen conversation recalls the prior user name.
A full 2048-position context peaks at 9,101,076 KiB (8.68 GiB) RSS.
All 12 CTest entries pass in default Release, Werror, sanitizer, AVX2 and
AVX-512 builds. [M3 checks](bench/M3_CHECKS.md) record the method, exact
commands and committed results. No M3 acceptance item remains open.
Remote CI remains to run on the pull request.

Acceptance criteria:

- Loads Qwen2.5 1.5B Instruct GGUF (F16 or Q8_0 source) including grouped-query attention and the QKV bias Qwen2 uses.
- The Qwen2.5 ChatML template (`<|im_start|>` / `<|im_end|>`, system prompt) is applied, and the rendered prompt token ids match llama.cpp's for a 3-turn conversation.
- A KV cache stores keys and values per layer. A test shows that generating with the cache gives the same tokens as recomputing the full sequence every step for at least 32 tokens.
- Greedy output for the M1 prompt set matches llama.cpp for at least the first 32 generated tokens on each prompt, or any divergence is explained by a logged near-tie (top-2 logit gap under 1e-3).
- Peak resident memory for a 2048-token context is measured and recorded.
- Context limit is enforced with a clear error instead of out-of-bounds writes.

## M4: int8 and 4-bit quantization

Q8_0, Q4_0, Q4_K and Q6_K loading, scalar products, block bounds and
GGUF layout/truncation checks are implemented. All local CI configurations
and M2/M3 regressions pass. [Quantization checks](bench/QUANT.md) record
both rounds and the unchanged acceptance criteria.

M4 is done. The larger precommitted method scores 4,096 WikiText-2
targets and runs twelve neutral prompts with 64 tokens each. Perplexity
tracks llama.cpp on the same file for every format: F16 11.580966 vs
11.580968, Q8_0 11.584336 vs 11.614251, Q4_0 13.129919 vs 13.143771 and
Q4_K_M 12.145460 vs 12.254528. Q8_0 is 0.029% above F16 and Q4_K_M is
4.87% above F16, so the perplexity limits pass. Both free-running greedy
agreement criteria were missed (see the marks below). The milestone was
accepted by Daniel on Nov 16 2026 because perplexity matches llama.cpp for each format (within 0.9%). Teacher-forced agreement is 99.48% for Q8_0 and
88.67% for Q4_K_M; it is a diagnostic and does not replace the criteria.

Known limitation: in the current scalar path Q4_K_M is slower than F16
(0.379621 versus 0.670816 generation tokens/s). Smaller weights do not
make this implementation faster on their own. SIMD kernels and threading
are M5 work.

Goal: the model fits comfortably and runs faster, with a measured, bounded quality loss.

Acceptance criteria:

- int8 (per-block, Q8_0-style) and 4-bit (per-block with scales, Q4_0 or Q4_K-style) weight formats, either loaded from GGUF or produced by a `neuron quantize` command.
- Dequantize-and-compare unit tests: round-trip error per block stays within the format's theoretical bound.
- Quality check on Qwen2.5 1.5B with a fixed text (for example a WikiText-2 slice):
  - int8 perplexity within 1% of FP32 (or F16) perplexity.
  - 4-bit perplexity within 10% of FP32 perplexity, and within 2% of llama.cpp's perplexity for the same quant type.
  - Greedy token agreement with the FP32 run of at least 90% over the first 64 tokens across the prompt set for int8, and at least 70% for 4-bit.
    - MISSED (int8): Q8_0 free-running agreement 551/768 (71.74%), target 90%. Accepted by Daniel on Nov 16 2026 because perplexity matches llama.cpp for each format (within 0.9%).
    - MISSED (4-bit): best 4-bit format Q4_K_M free-running agreement 180/768 (23.44%), target 70%. Accepted by Daniel on Nov 16 2026 because perplexity matches llama.cpp for each format (within 0.9%).
- `bench/QUANT.md` records file size, memory use, perplexity, and token agreement for each format.

## M5: AVX-512 kernels and multithreading

Goal: make it fast on Ghost, one measured step at a time.

Implemented locally on Ghost. All seven build/test configurations and the
SIMD, threading, TinyStories and quick Qwen correctness guards pass.
[Speed checks](bench/SPEED.md) record every optimization commit and result.
MISSED (speed): final Q4_K_M generation is 4.294374 tokens/s versus
38.953875 for llama.cpp with four threads, or 11.02%, below the 50% target.
Accepted by Daniel on Nov 17 2026 as a non-bug miss; refinement is deferred
until after M7. AVX-512 tests were run on Ghost; CI runs AVX2 tests and
compile-checks AVX-512.

Acceptance criteria:

- AVX2 and AVX-512 (VNNI for int8 dot products) kernels for the hot matmul paths, selected at runtime by CPUID, with the scalar kernel kept as the reference.
- Every SIMD kernel has a unit test against the scalar kernel on random inputs, including sizes that are not a multiple of the vector width.
- A thread pool splits matmuls by output rows; results are bit-identical or within 1e-5 of single-threaded output.
- `bench/SPEED.md` has a chart (committed PNG or SVG, generated by a script in the repo) of generation tokens/s for Qwen2.5 1.5B Q4 after each optimization step: scalar, AVX2, AVX-512, VNNI, threads, and any later step. Each step is its own commit so the chart is reproducible.
- Final generation speed is reported next to llama.cpp at the same thread count and quant type. The target is at least 50% of llama.cpp's tokens/s; the actual ratio is recorded whether or not it hits the target.
- CI runs the SIMD tests on AVX2 and compile-checks AVX-512; AVX-512 tests are run on Ghost and recorded in the PR.

## M6: Terminal chat, web chat UI, OpenAI-compatible server

Goal: three ways to use it.

Implemented and checked locally on Ghost. Terminal chat streams replies,
keeps history, resets and survives Ctrl-C. The dependency-free POSIX server
binds to loopback by default, serves an embedded web page and implements
both chat completion modes and model listing. The official OpenAI Python
client passes against SmolLM2 and Qwen2.5 1.5B Q4_K_M. Model-free HTTP and
terminal tests run in CTest, including the required error cases.
[M6 checks](bench/M6_CHECKS.md) record commands, results and limitations.
No M6 acceptance item remains open. Remote CI remains to run.

Acceptance criteria:

- `neuron chat -m <model>` gives a multi-turn terminal chat with streaming output, `/reset` and `/exit`, and Ctrl-C that stops the current reply without killing the session.
- `neuron serve -m <model> --port 8080` exposes `POST /v1/chat/completions` (streaming via SSE and non-streaming) and `GET /v1/models`, compatible enough that the official `openai` Python client works against it unchanged. A scripted test in the repo proves this.
- The server serves a single static web chat page at `/` with streaming replies, no build step and no external CDN.
- Requests over the context limit, bad JSON, and unknown models get proper HTTP errors, covered by tests.
- The server binds to 127.0.0.1 by default.

## M7: Polish

Completed locally on Ghost. The rewritten README includes the measured
model/quant comparison and the accepted M4 agreement and M5 speed misses.
The docs hub links the design post, generated M5 chart and real Qwen2.5
Q4_K_M terminal GIF. All seven local build/test configurations pass, and
the official client passes with SmolLM2 first and Qwen last.
[M7 checks](bench/M7_CHECKS.md) map claims to committed records and retain
the reproduction commands. No M7 acceptance item remains open. No tag or
release was created; those require explicit approval. Remote CI remains
to run after this branch is pushed.

Goal: someone landing on the repo understands what it is, how fast it is, and how it works in two minutes.

Acceptance criteria:

- README rewritten for the finished state: features that actually exist, quick start that works from a clean clone, and an honest limitations list.
- Benchmark table in the README: Neuron vs llama.cpp on Ghost for each model and quant type, tokens/s for prompt processing and generation, with reproduce commands.
- A demo GIF (recorded terminal or web chat with Qwen2.5 1.5B Q4 on Ghost) committed under `docs/` and under 5 MiB. Until then the README has no image placeholder.
- A blog post in `docs/blog/` that walks through the design and the optimization steps, with the M5 speed chart.
- Every number in the README and blog post traces to a file under `bench/`.
- A tagged release only when explicitly approved.
