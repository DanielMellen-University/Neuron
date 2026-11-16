# M2 TinyStories checks

Completed locally on Ghost on 2026-11-16 with GCC 11.4.0, CMake 3.22.1
and Python 3.10.12. Every M2 acceptance item passed. Remote GitHub Actions
has not been run from this worktree.

## Implementation

The reader maps a little-endian GGUF v2 or v3 file read-only. It parses all
13 metadata value types, including typed arrays, then checks tensor shapes,
alignment, offsets, byte ranges and overlaps before exposing any data.
Duplicate names, integer overflow, truncated input and unsupported formats
produce errors with a `GGUF:` prefix. Metadata is bounded to 100,000 keys,
100,000 tensors, 4,000,000 values and 16 nested array levels to limit resource
use on malformed files. Dense F32 and F16 tensor formats are supported;
quantized tensor formats are rejected until M4.

F32 weights stay in the mapping when aligned for float access. F16 weights
are converted once on load. The forward pass uses FP32 activations and K/V
state, with double accumulators in scalar reductions. Each layer applies
RMSNorm, Q/K/V projections, adjacent-pair llama RoPE, causal attention,
attention output and residual addition, then RMSNorm and a SwiGLU FFN with
another residual. Query heads share K/V heads in contiguous groups when
the counts differ. The final RMSNorm and output projection produce logits.
A missing output matrix uses tied embedding weights.

The tokenizer reads the GGUF tokens, scores, token types, special IDs and
prefix/BOS/EOS flags. It escapes spaces with the SentencePiece space marker,
merges adjacent symbols by score, resolves ties at the leftmost pair and
uses byte tokens for symbols absent from the vocabulary. Special tokens
are partitioned before merging. The CLI supports greedy generation,
JSON token IDs, tokenizer inspection and raw final-position logits.

`Model` respects the declared context by default. The CLI defaults to 2048
tokens, matching M1. TinyStories declares a training context of 128 tokens;
M1 explicitly overrides this to 2048. `--ctx-size` changes the CLI limit.
The loader and reference tool leave the downloaded file unchanged. This
comparison makes no model-quality claim beyond the training context.

## Inputs and reference adapter

- TinyStories: `stories15M.gguf`, SHA-256
  `61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a`.
- llama.cpp: M1 commit `9e9619d030bbe6b584d826a0a6cf06e38251f52b`.
- Prompts: the three raw completion strings in [prompts.json](prompts.json).
- No chat template, penalties or sampling randomness.

[llama_reference.cpp](llama_reference.cpp) is a small validation adapter.
It links only to the external M1 llama.cpp static libraries. Nothing from
llama.cpp is linked into Neuron or its tests. It calls `llama_tokenize`
with explicit `add_special` and `parse_special` flags. For logits it creates
a CPU context with one thread, context 2048, batch/microbatch 512, Flash
Attention disabled and F32 K/V storage. It evaluates the complete prompt
in one batch and writes all 32,000 values from `llama_get_logits_ith(-1)`.
Neuron evaluates the same prompt incrementally with F32 K/V state.

F32 K/V storage isolates FP32 forward-pass accuracy. M1's server used its
default F16 K/V storage. The separate generation check below still uses
M1's original committed tokens without changing that reference.

Build the adapter outside the repo's source tree. These are the exact paths
used for the recorded run, from the Neuron worktree root:

```sh
export NEURON_MODELS_DIR=/home/daniel/codex-runs/Neuron-m1/models
export NEURON_LLAMACPP_DIR=/home/daniel/codex-runs/Neuron-m1/third_party/llama.cpp
scripts/fetch_models.sh
# If the pinned M1 libraries are not already built:
# CC=gcc CXX=g++ scripts/build_llamacpp.sh

g++ -std=c++17 -O2 bench/llama_reference.cpp \
  -I"$NEURON_LLAMACPP_DIR/include" -I"$NEURON_LLAMACPP_DIR/ggml/include" \
  "$NEURON_LLAMACPP_DIR/build/src/libllama.a" \
  "$NEURON_LLAMACPP_DIR/build/ggml/src/libggml.a" \
  "$NEURON_LLAMACPP_DIR/build/ggml/src/libggml-cpu.a" \
  "$NEURON_LLAMACPP_DIR/build/ggml/src/libggml-base.a" \
  -fopenmp -pthread -ldl -o /tmp/neuron-m2-reference

cmake -S . -B build-m2 -DCMAKE_CXX_COMPILER=g++ -DNEURON_WERROR=ON
cmake --build build-m2 -j 4
python3 bench/check_m2.py --binary build-m2/neuron \
  --model "$NEURON_MODELS_DIR/stories15M.gguf" \
  --reference-tool /tmp/neuron-m2-reference --output bench/runs/m2-checks.json
```

The script verifies the model hash, compares both tokenizers with the
committed IDs, then compares all logits in temporary files. It deletes
those temporary files on exit. No logits or model weights are committed.
`--refresh-tokenizer` regenerates the fixture IDs from the adapter when
an intentional reference update is needed.

## Tokenizer and logits results

The [tokenizer fixture](reference/tinystories-tokenizer.json) contains 51
cases: the three M1 prompts plus empty text, spaces, leading/trailing spaces,
Unicode, digits, rare Unicode requiring byte fallback, whitespace, special
tokens, punctuation, combining accents and a control byte. Each of the 12
edge cases also runs with the other three combinations of `add_special`
and `parse_special`. Result: **51/51 exact token-ID matches** with both the
committed fixture and a fresh call to the reference adapter.

The [numeric results](reference/m2-checks.json) record binary hashes and
full-precision differences. All 32,000 logits are compared at the last
prompt position, before any generated token is fed back:

| Prompt | Prompt tokens, including BOS | Maximum absolute logit difference | Limit |
| --- | ---: | ---: | ---: |
| rabbit | 62 | 0.0000095367431640625 | 0.001 |
| garden | 63 | 0.00000858306884765625 | 0.001 |
| ball | 56 | 0.0000095367431640625 | 0.001 |

## Exact generation results

```sh
python3 bench/harness.py run-neuron --binary build-m2/neuron \
  --model-file "$NEURON_MODELS_DIR/stories15M.gguf" \
  --output bench/runs/tinystories-m2.json \
  --notes 'Scalar FP32, one thread, local correctness check. No builds or tests running during collection.'
python3 bench/harness.py compare bench/reference/tinystories-15m.json \
  bench/runs/tinystories-m2.json --threads 1 --exact-match
```

Exit status: **0**. `exact_match: true`, generated agreement **1.0**,
**1152/1152 generated IDs matched** across nine actual runs. Prompt IDs
matched in all nine runs too. `--threads 1` selects the one-thread rows of
both files. Neuron is scalar and single-threaded; it does not label its
results as 4-thread or 8-thread measurements. The M1 file already verified
that its tokens were identical at all three thread counts.

| Prompt | New tokens per run | Repetitions | Prompt match | Generation match | First divergence |
| --- | ---: | ---: | --- | --- | --- |
| rabbit | 128 | 3 | exact | exact | none |
| garden | 128 | 3 | exact | exact | none |
| ball | 128 | 3 | exact | exact | none |

[Committed Neuron runs](reference/tinystories-m2.json) contain the actual
IDs, commands, model/binary hashes, observed power state and timings.
The same comparison command works with that file as the candidate.
The exact required CLI form was also run for each prompt, without a
`--ctx-size` argument: `build-m2/neuron run -m <model> -p "<prompt>" -n 128
--greedy --json`. All three yielded the same 128 reference IDs.
Without `--json`, the command streams decoded generated text. This form
was also checked on every prompt: stdout matched M1's `generated_text`,
followed by the CLI's final newline.

The scalar correctness run pooled 177.38 prompt tokens/s and 72.31 generated
tokens/s. It had no warmup and makes no optimization claim. M1's pooled
one-thread rates were 3041.34 and 280.95 tokens/s. SIMD kernels and
multithreading remain M5 work.

## Model-free tests and CI

All nine CTest entries passed in each local GCC configuration below:
default Release, Release with Werror, Release with Werror and AVX2,
Release with Werror and AVX-512, and Debug with Werror and ASan/UBSan.
The default and sanitizer commands were:

```sh
cmake -S . -B build-m2-default -DCMAKE_CXX_COMPILER=g++
cmake --build build-m2-default -j 4
ctest --test-dir build-m2-default --output-on-failure

cmake -S . -B build-m2-sanitize -DCMAKE_CXX_COMPILER=g++ \
  -DNEURON_WERROR=ON -DNEURON_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-m2-sanitize -j 4
ctest --test-dir build-m2-sanitize --output-on-failure
```

The C++ suites cover synthetic GGUF files and every truncation of a valid
file, bad magic/version/counts/offsets, metadata types and arrays, alignment,
overlap, duplicate names, overflow, F16 conversion, tokenizer merging and
fallback, hand-computed kernels, and a small synthetic GQA transformer with
F16/F32 and tied/untied weights. Context and token bounds are checked too.
The Python harness suite passes all 13 tests, including thread selection.
All tests run without model files or llama.cpp. The existing CI workflow
runs these suites for GCC/Clang Release and sanitizers and compile-checks
AVX-512. Its weight-file and large-file guard also passes locally.

Clang 14 passed compile-only Werror checks for all translation units using
the installed GCC 11 headers. A local Clang link was not attempted because
its standard-library installation is incomplete. CI supplies the full
Clang build. No remote CI result is claimed here.

No M2 acceptance item remains open. Other model architectures, GPT-style
BPE vocabularies, RoPE scaling, quantization, chat and optimized kernels
remain later milestones.
