# M4 quantization checks

Measured on Ghost on 2026-11-16 UTC: Intel i5-1135G7, four cores,
eight hardware threads, 16 GB RAM, Ubuntu 22.04 base, GCC 11.4 and
CMake 3.22.1. Neuron uses the default Release build and one thread.

## Pure-format results

M4 remains in progress. Pure Q4_0 misses the F16 perplexity limit, and both
quantized formats miss the required free-running greedy agreement. The
[complete measurements](reference/m4-quantization.json) contain exact
commands, tokens, logit gaps, hashes, timings and power/load observations.

| Format | File bytes | Neuron peak RSS (KiB) | Neuron perplexity | llama.cpp perplexity | Greedy agreement with F16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| F16 | 3,093,669,376 | 9,156,968 | 12.149650 | 12.149653 | 192/192 (100.00%) |
| Q8_0 | 1,646,573,056 | 1,713,636 | 12.103084 | 12.109621 | 130/192 (67.71%) |
| Q4_0 | 874,788,352 | 959,976 | 14.479758 | 14.539137 | 23/192 (11.98%) |

Peak RSS is the largest GNU time maximum among that format's perplexity
process and three generation processes. It includes the CLI tokenizer,
metadata, resident mapped weights, converted F16 weights, activations and
K/V state. Contexts are selected as 128 for perplexity and 256 for generation;
at most 127 cache positions are populated. These are short-run measurements,
separate from M3's full 2048-position result. File sizes include GGUF metadata.

| Pure-format acceptance item | Required | Observed | Result |
| --- | --- | ---: | --- |
| Q8_0 perplexity change from F16 | within 1% | -0.383269% | pass |
| Q4_0 perplexity change from F16 | within 10% | +19.178397% | fail |
| Q4_0 perplexity change from llama.cpp Q4_0 | within 2% | -0.408407% | pass |
| Q8_0 greedy agreement | at least 90% | 67.708333% | fail |
| Q4_0 greedy agreement | at least 70% | 11.979167% | fail |

The F16 perplexity differs from llama.cpp by only 0.0000267% and every input
ID matches. Each F16 continuation's first 32 IDs also matches M3. This checks
the baseline and scoring convention independently of quantized quality.

| Prompt | Q8_0 matching IDs / 64 | Q8_0 first divergence | Q4_0 matching IDs / 64 | Q4_0 first divergence |
| --- | ---: | --- | ---: | --- |
| rabbit | 64 | none | 3 | 2 |
| garden | 2 | 0 | 14 | 1 |
| ball | 64 | none | 6 | 3 |

Divergence indices are zero based. Q8_0 garden changes its first selected
ID from 330 to 2619. F16's top-two gap there is 0.00113487244; Q8_0's gap
is 0.0869655609. Feeding the different first token back changes the rest
of the continuation. Agreement includes that drift. Passing the block
error bound or the perplexity limit does not establish greedy agreement.

| Format | Prompt tokens/s | Generation tokens/s | Generation speed / F16 |
| --- | ---: | ---: | ---: |
| F16 | 0.788633 | 0.669767 | 1.000 |
| Q8_0 | 1.087646 | 0.923744 | 1.379 |
| Q4_0 | 0.998380 | 0.847679 | 1.266 |

Rates pool the three generation invocations per format, excluding model
load. Generation times 63 steps per prompt after the first sampled token,
189 steps in total. These are single-pass scalar measurements with no
warmup and no repetitions. The machine had AC power, a full battery,
`powersave` with `intel_pstate`, and turbo enabled in the saved observations.
No build or other model run ran concurrently. Codex, normal background
services and light documentation work were present. Frequency and thermal
state were not held constant.

| File | SHA-256 |
| --- | --- |
| `Qwen2.5-1.5B-Instruct-f16.gguf` | `b6eaec3509f1d0373d1f4802654c4a7bfcde0768645d2d45e8885f6922b428ee` |
| `Qwen2.5-1.5B-Instruct-Q8_0.gguf` | `c4ad03ca16d5486d696da7450fdcdd6e89d44cb1285d2bdb80dccbdcdcb582fe` |
| `Qwen2.5-1.5B-Instruct-Q4_0.gguf` | `9716085e2f398e15d73d536e2dbafb22e80ae2290588f94d34dab5fd55d0ceab` |

Neuron loads GGUF F32, F16, Q8_0 and Q4_0 weights. Quantized matrices stay
mapped; scalar products read their blocks directly with FP32 activations
and double accumulators. Embeddings decode only the selected row. Small
norm and bias tensors are converted to FP32 if needed. No SIMD kernel,
thread pool or quantized activation path is part of M4.

## Formats and bounds

Both formats store an IEEE F16 scale followed by 32 quantized weights.
Q8_0 (GGML type 8) occupies 34 bytes per block: scale plus 32 signed bytes.
Q4_0 (type 2) occupies 18: scale plus 16 packed bytes. Byte `j` holds
weight `j` in its low nibble and weight `j+16` in its high nibble; subtract
8 from each nibble. Rows must contain a multiple of 32 elements. The reader
checks row size, byte counts, alignment, offsets, overlap and truncation.

Let `d` be the ideal scale and `h` its stored F16 value. Q8_0 uses
`d = max(abs(x))/127`, rounds to the nearest signed integer, and has absolute
error at most `abs(d)/2 + 127*abs(h-d)`. Q4_0 uses the signed largest-magnitude
weight divided by -8, rounds and clips to codes -8..7. Its error is at most
`abs(d) + 8*abs(h-d)`. The full step accounts for the asymmetric clipped
endpoint when a block has equal opposite extrema. Away from clipping the
first term is only `abs(d)/2`. F16 scale rounding is included, including
subnormal and zero scales; a scale exceeding the finite F16 range is
rejected by the reference block encoder.

`test_quant` checks 6,000 deterministic blocks per format, 32 values each,
at amplitudes 0, 1e-8, 1e-4, 1, 1,000 and 100,000. Half the nonzero cases
include equal opposite extrema. Binary32 arithmetic has a tolerance of
`2e-7*amplitude`. The largest observed error/bound ratio is 0.999477885 for
Q8_0 and 1.0 for Q4_0. Other tests independently encode GGUF blocks to
check nibble ordering and signed bytes, reject every payload truncation,
compare batched scalar products with dequantized FP32 products, and compare
32 steps of quantized Qwen2 inference with dequantized weights, including
QKV bias and tied/untied embeddings. All tests are model-free and run in CI.

## Fixed quality method

[Text provenance and regeneration](wikitext2-slice.md) pin the committed
1,599-byte WikiText-2 test prefix and its source hash. The text has 390 Qwen
tokens. Evaluate the first 256 as two independent chunks of context 128,
resetting K/V before each chunk. For each chunk, positions 0..63 supply
warmup, and targets 64..127 contribute 64 negative log probabilities.
Position 63's logits predict target 64. This scores 128 targets in total;
the remaining 134 text tokens are ignored. No chat template, BOS replacement,
overlap, sliding window or padding is applied. Tokenization uses the model's
normal add-special and parse-special flags. Qwen adds no BOS in this case.

Perplexity is `exp(sum(logsumexp(logits)-target_logit)/scored_tokens)`.
The log-sum-exp reduction uses double precision and subtracts the maximum.
`neuron perplexity` writes JSON with the context, chunks, scored count,
summed loss, perplexity, time and input IDs. By default it evaluates all
complete chunks; `--chunks N` selects the first N. It requires an even
context of at least two tokens and rejects insufficient input.

The [llama.cpp adapter](llama_reference.cpp) has a separate perplexity path
using llama.cpp's public API, with exactly the same tokenization, target
positions and resets. Both engines use one thread and F32 K/V storage.
The reference disables Flash Attention and processes one token per batch.
This avoids differences from the `llama-perplexity` tool's default batching,
BOS handling and scoring policy. Its numbers are llama.cpp perplexity under
the method above, not the standard full WikiText-2 benchmark. This short
check bounds loss on these fixed targets and has limited statistical power.

Greedy checks use the three unmodified raw prompts in [prompts.json](prompts.json),
context 256, no penalties, one thread and 64 new tokens per prompt. Each
format generates its own continuation. Agreement counts equal token IDs
at the same positions against the Neuron F16 continuation, divided by 192
positions pooled across prompts. No teacher forcing, edit-distance alignment
or exclusion of positions after divergence is used. An early stop cannot
silently pass the 64-token requirement.

## Reproduce

Run from the repository root, using G++ on Ghost. The pinned llama.cpp
checkout is `9e9619d030bbe6b584d826a0a6cf06e38251f52b`. Only small adapter
source and JSON results are committed; models and reference binaries stay
outside the repository.

```sh
export NEURON_MODELS_DIR=/home/daniel/codex-runs/Neuron-m1/models
export NEURON_LLAMACPP_DIR=/home/daniel/codex-runs/Neuron-m1/third_party/llama.cpp
scripts/fetch_models.sh
cmake -S . -B build-m4 -DCMAKE_CXX_COMPILER=g++ -DNEURON_WERROR=ON
cmake --build build-m4 -j 2
cmake --build "$NEURON_LLAMACPP_DIR/build" --target llama-quantize -j 2

"$NEURON_LLAMACPP_DIR/build/bin/llama-quantize" --pure \
  "$NEURON_MODELS_DIR/Qwen2.5-1.5B-Instruct-f16.gguf" \
  "$NEURON_MODELS_DIR/Qwen2.5-1.5B-Instruct-Q8_0.gguf" Q8_0 2
"$NEURON_LLAMACPP_DIR/build/bin/llama-quantize" --pure \
  "$NEURON_MODELS_DIR/Qwen2.5-1.5B-Instruct-f16.gguf" \
  "$NEURON_MODELS_DIR/Qwen2.5-1.5B-Instruct-Q4_0.gguf" Q4_0 2

g++ -std=c++17 -O2 bench/llama_reference.cpp \
  -I"$NEURON_LLAMACPP_DIR/include" -I"$NEURON_LLAMACPP_DIR/ggml/include" \
  -I"$NEURON_LLAMACPP_DIR/common" -I"$NEURON_LLAMACPP_DIR/vendor/nlohmann" \
  "$NEURON_LLAMACPP_DIR/build/common/libllama-common.a" \
  "$NEURON_LLAMACPP_DIR/build/common/libllama-common-base.a" \
  "$NEURON_LLAMACPP_DIR/build/src/libllama.a" \
  "$NEURON_LLAMACPP_DIR/build/ggml/src/libggml.a" \
  "$NEURON_LLAMACPP_DIR/build/ggml/src/libggml-cpu.a" \
  "$NEURON_LLAMACPP_DIR/build/ggml/src/libggml-base.a" \
  "$NEURON_LLAMACPP_DIR/build/vendor/hash/libvendor-hash.a" \
  -fopenmp -pthread -ldl -o /tmp/neuron-m4-reference

python3 bench/check_m4.py --binary build-m4/neuron \
  --reference-tool /tmp/neuron-m4-reference --models-dir "$NEURON_MODELS_DIR" \
  --output bench/reference/m4-quantization.json \
  --notes 'Describe actual machine activity and power conditions.'
```

`--pure` makes every quantizable matrix use the named type; one-dimensional
bias and norm tensors remain dense. The pure baseline uses no importance matrix, higher-precision output
head, recalibration or change to the F16 source. The separate probes below
keep the embedding/output matrix in F16.
The quantization tool is external; Neuron supports loading these formats
rather than providing a model-file quantize command. The small reference
block encoder is used by unit tests.

The script records the exact commands, file/binary hashes, GNU time peak
RSS and wall time, and machine power/load observations for each invocation.
It saves each completed result under `bench/runs/m4/` immediately and reuses
it on resume only if the command, model hash and binary hash match. Use a
fresh `--runs-dir` after changing inputs. One run per process, no warmup or
repetition. It returns nonzero if a measured M4 quality threshold fails,
while still saving the complete report.

Individual commands, with the same method, are:

```sh
build-m4/neuron perplexity -m "$NEURON_MODELS_DIR/Qwen2.5-1.5B-Instruct-Q4_0.gguf" \
  --prompt-file bench/wikitext2-slice.txt --ctx-size 128 --chunks 2
/tmp/neuron-m4-reference "$NEURON_MODELS_DIR/Qwen2.5-1.5B-Instruct-Q4_0.gguf" '' \
  --prompt-file bench/wikitext2-slice.txt --perplexity 128 --chunks 2
# For each exact prompt string in bench/prompts.json:
build-m4/neuron run -m "$NEURON_MODELS_DIR/Qwen2.5-1.5B-Instruct-Q4_0.gguf" \
  -p '<prompt text>' -n 64 --greedy --json --ctx-size 256
```

## Local validation

All 13 CTest entries pass in GCC default Release, Release with Werror,
Debug with Werror and ASan/UBSan, Release with Werror and AVX2, and Release
with Werror and AVX-512. The AVX-512 tests also ran locally. Clang 14 passes
compile-only Werror checks with GCC 11's headers. The local Clang runtime
cannot link; remote CI has not been run from this worktree.

```sh
cmake -S . -B build-m4-default -DCMAKE_CXX_COMPILER=g++
cmake --build build-m4-default -j 2
ctest --test-dir build-m4-default --output-on-failure
cmake -S . -B build-m4-sanitize -DCMAKE_CXX_COMPILER=g++ \
  -DNEURON_WERROR=ON -DNEURON_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-m4-sanitize -j 2
ctest --test-dir build-m4-sanitize --output-on-failure
```

TinyStories regression matches all 51 tokenizer cases and all vocabulary
logits within 9.54e-6 of llama.cpp. All 1,152 generated IDs match M1 exactly
(three prompts, three repetitions, 128 tokens). The cheap M3 fixture check
matches 63 tokenizer cases and four ChatML byte/ID cases. The M3 2048-context
memory run was not repeated.

The saved [validation report](reference/m4-validation.json) records the
build commands and log hashes. [TinyStories M4 results](reference/tinystories-m4.json)
contain the exact regression IDs and comparisons. Reproduce the local
regressions with the same external adapter:

```sh
python3 bench/check_m2.py --binary build-m4/neuron \
  --model "$NEURON_MODELS_DIR/stories15M.gguf" \
  --reference-tool /tmp/neuron-m4-reference --output bench/runs/m4-m2-checks.json
python3 bench/harness.py run-neuron --binary build-m4/neuron \
  --model-file "$NEURON_MODELS_DIR/stories15M.gguf" \
  --output bench/runs/m4-tinystories.json \
  --notes 'Record actual activity; correctness regression only.'
python3 bench/harness.py compare bench/reference/tinystories-15m.json \
  bench/runs/m4-tinystories.json --threads 1 --exact-match
python3 bench/check_m3.py --binary build-m4/neuron \
  --model "$NEURON_MODELS_DIR/Qwen2.5-1.5B-Instruct-f16.gguf" \
  --reference-tool /tmp/neuron-m4-reference --fixtures-only \
  --output bench/runs/m4-m3-fixtures.json \
  --notes 'Record actual activity; tokenizer and ChatML fixtures only.'
```

## F16 embedding/output probes

After the pure formats failed, one global precision change was tested:
keep `token_embd.weight` in F16 while quantizing every other quantizable
matrix. Qwen ties its output head to that embedding. The corpus, prompts,
context, chunking and decoding stayed fixed. This trades size and memory
for higher precision without calibration or prompt-specific weights.
The tensor payload uses 9.64 bits/weight for the Q8_0 variant and 6.24
bits/weight for the Q4_0 variant. These are mixed-precision files.
The [saved probe report](reference/m4-mixed-probes.json) includes hashes,
commands, token IDs, losses, resources and observed machine state.

| Probe | File bytes | Peak RSS (KiB) | Observed quality |
| --- | ---: | ---: | --- |
| Q8_0 with F16 embedding/output | 1,865,360,896 | 2,838,760 | garden 1/64 IDs; first divergence 0 |
| Q4_0 with F16 embedding/output | 1,210,263,040 | 2,198,636 | perplexity 13.716642; +12.897426% from F16 |

RSS in this table comes from the int8 garden process and the Q4 perplexity
process, respectively.

The int8 probe can reach at most 129/192 (67.19%) agreement even if both
other prompts match perfectly. Its remaining runs were skipped because
the 90% target is already impossible. The Q4_0 probe still exceeds the
10% perplexity limit, so its generation runs were skipped. llama.cpp on
the identical mixed Q4_0 file gives 13.780741 perplexity, a 0.465136%
difference from Neuron. Its tokenization is identical too. These probes
remain separate from the full pure-format results above and do not close
the missing M4 quality criteria.

Create the files with the same pinned quantizer:

```sh
for type in Q8_0 Q4_0; do
  "$NEURON_LLAMACPP_DIR/build/bin/llama-quantize" --pure \
    --token-embedding-type f16 --output-tensor-type f16 \
    "$NEURON_MODELS_DIR/Qwen2.5-1.5B-Instruct-f16.gguf" \
    "$NEURON_MODELS_DIR/Qwen2.5-1.5B-Instruct-$type-F16-embedding.gguf" "$type" 2
done

# Exact measured probe commands; each ran once under GNU time.
build-m4/neuron run \
  -m "$NEURON_MODELS_DIR/Qwen2.5-1.5B-Instruct-Q8_0-F16-embedding.gguf" \
  -p "$(python3 -c 'import json; print(json.load(open("bench/prompts.json"))[1]["text"], end="")')" \
  -n 64 --greedy --json --ctx-size 256
build-m4/neuron perplexity \
  -m "$NEURON_MODELS_DIR/Qwen2.5-1.5B-Instruct-Q4_0-F16-embedding.gguf" \
  --prompt-file bench/wikitext2-slice.txt --ctx-size 128 --chunks 2
/tmp/neuron-m4-reference \
  "$NEURON_MODELS_DIR/Qwen2.5-1.5B-Instruct-Q4_0-F16-embedding.gguf" '' \
  --prompt-file bench/wikitext2-slice.txt --perplexity 128 --chunks 2
```

`check_m4.py` also accepts `--q8-model-file` and `--q4-model-file` to run its
full protocol on explicit paths. Use a separate runs directory for these
variants. The probes above stopped at a failed gate to avoid additional
long runs that cannot make that candidate meet every threshold.
