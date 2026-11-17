# R1 decode profile and checks

This file records the R1 CPU decode profile, kernel and buffer changes,
speed measurements, and correctness checks on an Intel Core i5-1135G7 laptop.

A token is a piece of the model vocabulary. Prompt processing (`pp`) evaluates
input tokens; generation (`tg`) produces new tokens after the first sampled
one. tok/s means tokens per second. GGUF is the model weight file format.
Quantization stores weights in fewer bits; K-quant formats group 256 weights
per block, with Q4_K and Q6_K using 4 and 6 bits. Q4_K_M mixes those formats.
F16, F32 (FP32) and FP64 mean 16-bit, 32-bit and 64-bit floating point.
SIMD (single instruction, multiple data) processes values in parallel;
AVX2 and AVX-512 are x86 vector instruction sets, and VNNI (Vector Neural
Network Instructions) adds integer dot products. ISA means instruction set
architecture. Perplexity measures next-token prediction on fixed text
(lower is better). A KV cache stores attention keys and values from earlier
tokens. RoPE means rotary position embeddings; RMSNorm normalizes by the
root mean square, and SiLU is the sigmoid linear unit activation function.

R1 measures 21.215665 generation tokens/s against llama.cpp's 38.558125,
or 55.02%, using the unchanged M5 short protocol and four threads. The
fresh main baseline is 5.747949 tokens/s. [SPEED.md](SPEED.md#r1-result)
links the prompt and generation timings. These are short, unpinned samples;
they do not establish sustained throughput.

## Profile and changes

The Linux hardware profiler command `perf stat -e cycles true` fails because
`perf_event_paranoid` is 4. GNU Compiler Collection (GCC) Release builds
with `-pg` provide a fallback profile through the gprof function profiler. The
[before flat profile](reference/r1-gprof-before.txt) attributes 74.55% of
samples to `quant_avx512` and 23.66% to `f16_to_f32`, together 98.21%.
`Model::step` accounts for 0.45%. The profile includes model setup, four
prompt positions and 31 decode steps. Multithreaded gprof call counters
are approximate and its instrumented timings are not used as speed data.
An initial baseline profiling run overlapped a build. The retained speed
runs did not overlap builds or model jobs.

| Sampled CPU category | Before | After |
| --- | ---: | ---: |
| Quantized row dots | 74.55% | 82.30% |
| Software F16 conversion | 23.66% | 1.07% |
| Activation packing | absent | 0.70% |
| Model step outside called functions | 0.45% | 0.70% |

The remaining leading costs are K dots and pool waits/spins (9.09% in
worker entry and 1.71% in the caller's pool submission). The after profile
and allocation driver overlapped the long perplexity check. These profiles
locate CPU work; their wall times are not comparable speed measurements.

[After profile](reference/r1-gprof-after.txt) and
[allocation/profile commands and results](reference/r1-profile.json)
retain the final observations. The standalone
[driver](profile_decode.cpp) counts C++ heap allocations during the 31
decode steps after prompt evaluation. It also inspects the actual GGUF
tensor types. This file has 141 F32 tensors, 168 Q4_K matrices and 29 Q6_K
matrices. No Q5_K is present. The model hash is unchanged from M4.

The driver counts **12,779 C++ heap allocations before R1 and zero after
R1** over those 31 steps. This covers the view-based inference path within
the initial KV page. Owning-vector API calls and later KV growth can still
allocate.

The original K paths already contained vector decoding, but repeatedly
decoded scales and called the software half converter within their inner
loops. R1 first specialized those loops, expanded normal F16 scales inline
and reduced once per row. It then packed activations once per matrix call
and shared them across output-row workers. Weights stay mapped in their
original GGUF layout.

Each 256-value activation block uses a power-of-two scale
`2^(frexp(max_abs)-31)`. A signed 32-bit integer retains 31 magnitude bits
and splits into two signed 16-bit integer planes for its high and low parts. Reconstruction is
`high*65536 + low + 32768`. Absolute activation rounding error is bounded
by `2^(frexp(max_abs)-32)`. Zero and subnormal blocks are covered. Each
block retains its original input and a safe half-scale limit derived from
a bound on the sum of absolute packing errors and the maximum integer
coefficient, 4096. Small nonzero values that could round enter the bound,
each contributing at most half the packing scale. If a weight scale
exceeds the resulting limit, that block uses the direct SIMD fallback to
keep this activation error bound within `1e-5`. Inputs with nonfinite
values also use the direct path.

Weight group scales enter signed 16-bit coefficients. Q6_K coefficients
range from -4064 to 4096. At most 16 products enter an int32 lane before
widening, including the exact `INT32_MIN` endpoint. AVX2 and plain AVX-512
use word multiply-add; VNNI uses `vpdpwssd`. The dots reconstruct in FP64,
apply the F16 block scale and subtract Q4_K minimum offsets using FP64 sums
of the original activations. Q4_K blocks use the direct path unless the
half-scale exponents guarantee that factoring preserves each decoded FP32
weight exactly. This keeps ordinary model blocks on the packed path and
preserves weight rounding in cancellation cases with large scales. The
scalar code remains unchanged.

The AVX-512 path checks for VNNI before selecting K word dots. Q8_0/Q4_0
retain their float SIMD paths under auto selection. Their older four-byte
VNNI path remains opt-in. Tests call the plain AVX-512 K kernel directly
as well, so VNNI hosts exercise both implementations.

Unrolling removes variable shifts and scale branches from the integer
loops. Model scratch and logits are reused through `step_view` and
`forward_view`; the owning-vector APIs remain available. KV storage
reserves an initial page of at most 256 tokens and can grow later. Pool
jobs use non-owning task erasure instead of allocating `std::function`
copies. Caller and workers briefly spin before sleeping. RoPE angles are
computed once per position and reused across heads and layers. The SiLU
activation and feed-forward up projection already share one loop. RMSNorm and attention retain their scalar
arithmetic; no acceptance or agreement scoring code was changed.

## Thread and ISA sweep

All entries use the same M5 prompt, context and token count. Frequency and
temperature were not fixed. The rows are individual observations.

| ISA | Threads | Prompt tokens/s | Generation tokens/s | Record |
| --- | ---: | ---: | ---: | --- |
| AVX-512 with VNNI K | 1 | 7.125989 | 6.553897 | [raw](reference/r1-speed-t1.json) |
| AVX-512 with VNNI K | 2 | 13.584230 | 12.358130 | [raw](reference/r1-speed-t2.json) |
| AVX-512 with VNNI K | 4 | 23.912452 | 21.215665 | [raw](reference/r1-speed-final.json) |
| AVX-512 with VNNI K | 8 | 23.139524 | 20.598601 | [raw](reference/r1-speed-t8.json) |
| AVX2 | 4 | 10.543386 | 9.546656 | [raw](reference/r1-speed-avx2-final.json) |

Four threads were retained. The pool splits contiguous output rows evenly
and the caller participates. No affinity or frequency setting was changed.
[Source map](reference/r1-source-map.json) verifies the CMake/header/source
hash against each optimization commit. Earlier intermediate and recheck
samples are also retained, including results below the target.

## Correctness and builds

[Validation](reference/r1-validation.json) retains configure, build and
verbose logs from the CMake test runner (CTest) for GCC default Release,
Release with warnings treated as errors (Werror), AVX2, AVX-512,
AddressSanitizer/UndefinedBehaviorSanitizer (ASan/UBSan), Clang Release
and Clang AVX2. All seven configurations pass.
Each has 21 passed tests and one optional OpenAI-client skip. All four
SIMD tests execute on the test laptop. Unsupported runner ISAs skip with code 77.
CI now also runs CTest in the AVX-512 job, using runtime skips on runners
without the instructions. Acceptance settings are unchanged.

SIMD tests compare independent scalar results with random blocks,
unaligned data, batches, odd row counts, one/two/six/35 K blocks, finite F16
endpoints, signed scales -128/127, zero inputs, cancellation and unequal
magnitudes. The tolerance is `1e-5 * (1 + abs(reference))`. Activation
packing also has an independent half-scale rounding bound check. Extreme
Q6_K tests cover signed accumulator bounds. Review regressions cover a
small Q6_K contribution beside a large activation and Q4_K cancellation
after FP32 weight rounding, in single and mixed blocks on each SIMD path.
The original validation record has 3,073 scalar/AVX2 comparisons and 3,153
AVX-512/VNNI comparisons; its largest absolute error is 8 in the large
finite-F16-endpoint cases, within the stated relative tolerance.
The thread test passes 93,600 bit-identical comparisons at 1/2/4/8 threads.
Model tests cover reused logit views, owning copies, resets and cached
generation. Every R1 Neuron speed record matches all 32 baseline IDs.
[Small-model guards](reference/r1-generation.json) also match the unchanged
scalar executable and M1 exactly: 384 TinyStories IDs and 96 SmolLM2 IDs.

[Perplexity results](reference/r1-perplexity.json) repeat the original
M4-v2 16 independent 512-token chunks and 4,096 scored targets on unchanged
GGUFs and text. The runner uses four threads; M4 used one. Thread-count
invariance is separately tested. A relative numerical guard of `1e-6`
compares against the recorded per-format results. It does not replace any
M4 acceptance threshold. R1 does not rerun or tune free-running agreement.
Some model-free builds/tests overlapped the first quality run; these
quality timings are not used as speed measurements.

All four full-slice checks pass. No perplexity increases relative to M4.

| Format | Recorded M4 perplexity | R1 perplexity | Relative change |
| --- | ---: | ---: | ---: |
| F16 | 11.580966363839 | 11.580966356650 | -6.20757556e-10 |
| Q8_0 | 11.584335697414 | 11.584335660935 | -3.14900861e-09 |
| Q4_0 | 13.129918942500 | 13.129918942500 | 0 |
| Q4_K_M | 12.145460061051 | 12.145459915405 | -1.19918043e-08 |

Q8_0 remains within the original 1% F16 limit. Q4_K_M remains within
the original 10% F16 and 2% same-file llama.cpp limits. Q4_0 retains its
recorded result, including its original miss of the 10% F16 limit.

## Reproduce

```sh
export NEURON_MODELS_DIR="$HOME/.cache/neuron/models"
export NEURON_LLAMACPP_DIR="$HOME/.cache/neuron/llama.cpp"
cmake -S . -B build-r1 -DCMAKE_CXX_COMPILER=g++ -DNEURON_WERROR=ON
cmake --build build-r1 -j 2
python3 bench/validate_m5.py --build-prefix build-r1-check \
  --output bench/runs/r1-validation.json
python3 bench/check_r1.py --binary build-r1/neuron --threads 4 --isa auto \
  --output bench/runs/r1-perplexity.json
python3 bench/speed.py measure --binary build-r1/neuron --step R1-final \
  --isa avx512 --threads 4 --output bench/runs/r1-speed-final.json \
  --notes 'Record actual activity and power conditions.'
python3 bench/speed.py measure --llama --step R1-llama.cpp --threads 4 \
  --output bench/runs/r1-speed-llama.json \
  --notes 'Record actual activity and power conditions.'
```

For profiling, build unmodified commit `2646c5d` in a fresh detached
worktree as Release. Build its profiling binary and the current profiling
binary with `-DCMAKE_CXX_FLAGS=-pg -DCMAKE_EXE_LINKER_FLAGS=-pg`, both in
fresh directories. `GMON_OUT_PREFIX` must point outside tracked files.
The current driver uses the old model header for the old library, avoiding
an application binary interface (ABI) mismatch:

```sh
python3 bench/profile_r1.py --before-build /path/to/baseline-release \
  --after-build build-r1 --gprof-build /path/to/current-pg-build \
  --model "$NEURON_MODELS_DIR/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf" \
  --output bench/runs/r1-profile.json
```

[The artifact report](reference/r1-final-checks.json) records source and binary
hash checks, derived speed ratios, build results, original perplexity
limits, table/chart consistency and the weights/size guard.
