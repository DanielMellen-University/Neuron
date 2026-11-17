# M4 quantization checks

Hardware for both rounds: Ghost, Intel i5-1135G7, four cores,
eight hardware threads, 16 GB RAM, Ubuntu 22.04 base, GCC 11.4 and
CMake 3.22.1. Neuron uses the default Release build and one thread.

## Second round: 4,096 targets and twelve prompts

Measured on 2026-11-16 and 2026-11-17 UTC. M4 remains in progress.
Q4_K_M meets both 4-bit perplexity limits. Q8_0 and both 4-bit candidates
still miss free-running greedy agreement. The acceptance criteria are unchanged.

The [method and input hashes](m4-method-v2.md) were committed in `6f55c17`
before any new quality run. Sixteen independent 512-token chunks score
4,096 targets. Twelve fixed neutral prompts supply 64 free-running tokens
each, for 768 positions per format. No prompt was changed or dropped.
Every generation reached 64 tokens. Teacher-forced top-1 agreement replays
F16's exact IDs and is a supplementary diagnostic only.

The [summary](reference/m4-v2-quantization.json) links full per-process
records: [F16](reference/m4-v2-F16.json), [Q8_0](reference/m4-v2-Q8_0.json),
[Q4_0](reference/m4-v2-Q4_0.json) and [Q4_K_M](reference/m4-v2-Q4_K_M.json).
They contain commands, model and executable hashes, IDs, logit gaps,
timings, GNU time resources and machine observations. Both engines use
one thread, F32 K/V, no Flash Attention and identical input IDs and target
positions. These are fixed-slice results, not full WikiText-2 perplexity.

| Format | Neuron perplexity | llama.cpp perplexity | Free-running agreement | Teacher-forced top-1 agreement |
| --- | ---: | ---: | ---: | ---: |
| F16 | 11.580966 | 11.580968 | 768/768 (100.00%) | 768/768 (100.00%) |
| Q8_0 | 11.584336 | 11.614251 | 551/768 (71.74%) | 764/768 (99.48%) |
| Q4_0 | 13.129919 | 13.143771 | 51/768 (6.64%) | 648/768 (84.38%) |
| Q4_K_M | 12.145460 | 12.254528 | 180/768 (23.44%) | 681/768 (88.67%) |

| M4 acceptance item | Required | Observed | Result |
| --- | --- | --- | --- |
| int8 and 4-bit formats | loaded or produced | Q8_0, Q4_0, Q4_K and Q6_K; standard Q4_K_M mix | pass |
| Block error bounds | within theoretical bound | 6,000 blocks per format; ratios at most 1.0 | pass |
| int8 perplexity vs F16 | within 1% | Q8_0 +0.029094% | pass |
| 4-bit perplexity vs F16 | within 10% | Q4_K_M +4.874323%; Q4_0 +13.374986% | pass with Q4_K_M |
| 4-bit perplexity vs same llama.cpp file | within 2% | Q4_K_M -0.890023%; Q4_0 -0.105389% | pass |
| int8 free-running agreement | at least 90% | 551/768 (71.74%) | fail |
| 4-bit free-running agreement | at least 70% | Q4_K_M 180/768 (23.44%); Q4_0 51/768 (6.64%) | fail |
| File size, memory, perplexity and agreement records | recorded per format | tables and linked JSON records | pass |

Q8_0 matches all 64 free-running IDs on eight prompts. On the other four,
there is exactly one teacher-forced argmax change per prompt. Those four
local changes lead to 217 free-running positional mismatches. First
divergences are weather 24, train 6, bicycle 3 and moon 4, using zero-based
indices. F16's top-two gaps there are 0.0790672302, 0.039232254,
0.217716217 and 0.000291824341. Only the last is below 1e-3. The larger
sample still fails the criterion; the result is not explained by a single
near-tie. Teacher forcing does not close either greedy acceptance item.

Each cell below gives free-running / teacher-forced matching IDs out of 64.

| Prompt | Q8_0 | Q4_0 | Q4_K_M |
| --- | ---: | ---: | ---: |
| library | 64 / 64 | 5 / 56 | 5 / 57 |
| weather | 25 / 63 | 10 / 56 | 10 / 57 |
| train | 6 / 63 | 2 / 48 | 0 / 51 |
| bread | 64 / 64 | 3 / 56 | 4 / 59 |
| museum | 64 / 64 | 0 / 58 | 31 / 57 |
| garden_tools | 64 / 64 | 3 / 47 | 12 / 51 |
| bicycle | 4 / 63 | 6 / 52 | 1 / 54 |
| river | 64 / 64 | 4 / 54 | 10 / 56 |
| notebook | 64 / 64 | 1 / 55 | 44 / 62 |
| kitchen | 64 / 64 | 1 / 55 | 10 / 60 |
| moon | 4 / 63 | 4 / 57 | 33 / 63 |
| post_office | 64 / 64 | 12 / 54 | 20 / 54 |

| Format | GGUF bytes | Neuron peak RSS (KiB) | llama.cpp perplexity RSS (KiB) |
| --- | ---: | ---: | ---: |
| F16 | 3,093,669,376 | 9,179,160 | 3,104,172 |
| Q8_0 | 1,646,573,056 | 1,735,068 | 1,690,912 |
| Q4_0 | 874,788,352 | 981,276 | 1,777,884 |
| Q4_K_M | 986,048,512 | 1,090,076 | 1,649,192 |

Neuron RSS is the largest maximum among that format's perplexity,
free-running and teacher-forced processes. Perplexity uses context 512;
generation uses 256. It includes tokenization, mapped weights, converted
F16 weights, activations and K/V. It is separate from M3's full 2048-position
measurement. llama.cpp RSS here comes from its perplexity process only.

| Format | Neuron prompt tokens/s | Neuron generation tokens/s | Neuron PPL scored targets/s | llama.cpp PPL scored targets/s |
| --- | ---: | ---: | ---: | ---: |
| F16 | 0.785230 | 0.670816 | 0.359348 | 3.232377 |
| Q8_0 | 1.011203 | 0.862442 | 0.461894 | 4.805548 |
| Q4_0 | 0.913031 | 0.781080 | 0.417169 | 7.857662 |
| Q4_K_M | 0.499152 | 0.379621 | 0.215708 | 6.986680 |

Prompt and generation rates pool the twelve free-running processes,
excluding model load. Generation times 63 steps after the first selected
ID per prompt, for 756 steps. PPL rates divide 4,096 scored targets by the
whole evaluation time, including warmup positions; they are not decode
rates. This was one serial sweep with no warmup runs or repetitions.
No build or other model process ran concurrently. Codex, ordinary services
and light documentation work were present. Frequency and temperature
were not held constant; power, governor, load and thermal observations are
saved for each process. Q4_K_M is slower than F16 in this scalar kernel:
0.379621 versus 0.670816 generation tokens/s. Smaller weights alone do not
make this implementation faster. SIMD kernels and threading remain M5.

Q4_K_M uses 168 Q4_K matrices and 29 Q6_K matrices, including the tied
embedding/output head. The other 141 tensors remain F32. The quantizer
reports 5.08 payload bits per weight. It uses no importance matrix or
calibration on the benchmark text. [Production provenance](reference/m4-v2-models.json)
records the pinned quantizer, command and hashes. Models remain outside git.

Reproduce this round with the same external reference build described below:

```sh
export NEURON_MODELS_DIR=/home/daniel/codex-runs/Neuron-m1/models
export NEURON_LLAMACPP_DIR=/home/daniel/codex-runs/Neuron-m1/third_party/llama.cpp
"$NEURON_LLAMACPP_DIR/build/bin/llama-quantize" \
  "$NEURON_MODELS_DIR/Qwen2.5-1.5B-Instruct-f16.gguf" \
  "$NEURON_MODELS_DIR/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf" Q4_K_M 2
cmake -S . -B build-m4-v2-reproduce -DCMAKE_CXX_COMPILER=g++
cmake --build build-m4-v2-reproduce -j 2
python3 bench/check_m4_v2.py --binary build-m4-v2-reproduce/neuron \
  --reference-tool /tmp/neuron-m4-reference --models-dir "$NEURON_MODELS_DIR" \
  --runs-dir bench/runs/m4-v2-reproduce --output bench/runs/m4-v2-reproduce-summary.json \
  --notes 'Record actual machine activity and power conditions.'
```

Keep the existing pure Q8_0 and Q4_0 files, or regenerate them with the
historical `--pure` commands below. Q4_K_M deliberately omits `--pure`.
The harness saves every completed process and checks command, model and
executable hashes on resume. It returns nonzero when quality gates fail,
after writing every format's results. F16 teacher-forced agreement uses
its free-running result by construction; the other three formats perform
separate replay runs. `--teacher-tokens FILE` reads exactly 64 IDs separated by whitespace,
feeds the previous F16 ID and records the current argmax.

All 14 CTest entries pass in GCC default Release, Release Werror,
Release Werror with AVX2 and AVX-512, Debug Werror with ASan/UBSan,
and Clang Werror baseline and AVX2. AVX-512 tests ran locally too. Local
Clang 14 uses GCC 11 headers and libraries explicitly; the exact flags and
all configure/build/test logs are in [validation](reference/m4-v2-validation.json).
Remote CI has not been run from this worktree. No push was made.

The [M2 check](reference/m4-v2-m2.json) matches all 51 tokenizer cases and
all vocabulary logits within 9.54e-6. [TinyStories regression](reference/tinystories-m4-v2.json)
matches all 1,152 M1 IDs exactly. The [M3 fixture check](reference/m4-v2-m3.json)
matches all 63 tokenizer cases and four ChatML fixtures. M3's full-context
memory run was not repeated.

## Historical first round: 128 targets and three prompts

### Pure-format results

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

## Historical first-round quality method

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

## Historical first-round reproduction

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

## Historical first-round validation

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

## Historical first-round F16 embedding/output probes

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

## K block formats and reference bounds

Q4_K (GGML type 12) stores 256 weights in 144 bytes. Two F16 values give
superblock scale `h` and offset scale `hm`. Twelve bytes pack eight unsigned
6-bit scales `a[g]` and eight unsigned 6-bit offsets `b[g]`. The last 128
bytes pack four pairs of 32-weight groups into low and high nibbles.
Weight `i` in group `g` decodes to `(h*a[g])*q[i] - hm*b[g]`, with each
operation in FP32. The packed scale fields use their upper two bits to
carry the upper scale/offset bits for groups 4..7.

Q6_K (type 14) stores 256 weights in 210 bytes: 128 low-bit bytes, 64
high-bit bytes, 16 signed int8 scales, and one F16 superblock scale. Each
16-weight group decodes as `(h*a[g])*(q[i]-32)`. Four interleaved groups
share low nibbles, high nibbles and successive two-bit fields. The signed
scales and global scale are both honored. Rows must be multiples of 256.
Loading checks tensor sizes and row boundaries before exposing mapped
blocks. Embedding lookup decodes one row; scalar products decode one block
at a time and accumulate in double precision. No whole quantized matrix
is converted to FP32.

The small reference encoder is a min/max encoder for tests, not llama.cpp's
weighted fitting algorithm. For Q4_K group `g`, let `m[g] = -min(0,min(x))`
and `s[g] = (max(0,max(x)) + m[g])/15`. Quantize weights using these ideal
group parameters. Set `d=max(s)/63`, `dm=max(m)/63`, round group scales and
offsets to 6-bit integers, and store `d,dm` in F16 as `h,hm`. A per-weight
absolute error bound, before binary32 arithmetic tolerance, is

```text
s[g]/2 + 15*(d/2 + 63*abs(h-d)) + dm/2 + 63*abs(hm-dm)
```

For Q6_K, let `s[g]` be the signed largest-magnitude group value divided
by -32, and `d=max(abs(s))/127`. Round group scales to signed integers
-127..127 and store `d` as F16 `h`. Quantize weights to -32..31. Equal
opposite extrema can clip the positive endpoint, so the bound uses a full
local step:

```text
abs(s[g]) + 32*(d/2 + 127*abs(h-d))
```

These bounds include both levels of scale quantization and F16 rounding.
Zero scales and underflow are covered. `test_quant` uses 6,000 deterministic
256-weight blocks per K format across amplitudes 0, 1e-8, 1e-4, 1, 1,000
and 100,000, including equal opposite extrema and positive, negative and
zero groups. Binary32 tolerance is `4e-7*amplitude`. Maximum observed
error/bound is 0.991573740 for Q4_K and 0.989739144 for Q6_K. The legacy
32-weight checks remain 0.999477885 for Q8_0 and 1.000000000 for Q4_0.

Independent GGUF fixtures vary all packed fields, compare every decoded
weight and batched scalar product, reject every payload truncation and
reject short rows. Synthetic Qwen2 inference compares mapped and decoded
weights for 32 steps with QKV bias and both tied and untied output heads.
All of these checks are model-free and run in CI.
