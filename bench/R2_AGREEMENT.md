This file records the R2 free-running agreement measurements and numerical checks for Qwen2.5 1.5B.

# R2 agreement checks

Free-running agreement counts equal generated token IDs at the same positions
when each model feeds back its own output. A token is a vocabulary piece.
F16 and F32 (FP32) mean 16-bit and 32-bit floating point, and FP64 means
double precision. Quantization stores weights in fewer bits. Q8_0 and Q4_0
pack 32 weights per block using eight and four bits. Q4_K_M mixes four-bit
Q4_K and six-bit Q6_K blocks of 256 weights. A logit is the model's score
for a candidate next token. Greedy decoding selects the highest logit.
A KV cache stores attention keys and values from earlier tokens.

Measurements use an Intel Core i5-1135G7 laptop with four cores, eight
threads and 16 GB of memory. Generation uses one thread and automatic
selection of AVX-512 vector instructions. Both engines use F32 KV caches,
a 256-token context, greedy decoding, plain-text completion prompts and
one-token prompt evaluation batches. Each of the twelve frozen M4-v2
prompts contributes 64 positions. The pinned
llama.cpp revision is `9e9619d`. The method fixes the prompts, model hashes
and scoring convention. [Frozen quality method](m4-method-v2.md).

## Result

llama.cpp also misses the 90% Q8_0 and 70% four-bit agreement targets.
The numerical checks found no inference defect that justifies a correction.
Neuron retains its existing arithmetic. The remaining agreement misses
are consistent with quantization sensitivity and the measured activation
precision differences between engines.

The baseline contains separate free-running runs for each engine and
compares every format with that engine's F16 run.
All Neuron baseline continuations reproduce the M4-v2 token IDs exactly.
The fresh repeat reproduces every generated ID in both engines. F16 agrees
across engines on all 768 generated positions.
[Baseline](reference/r2-before.json), [fresh repeat](reference/r2-after.json),
[checked summary](reference/r2-summary.json).

| Format | Neuron before | Neuron after | llama.cpp vs its F16 | Same-format agreement between engines |
| --- | ---: | ---: | ---: | ---: |
| F16 | 768/768 (100.00%) | 768/768 (100.00%) | 768/768 (100.00%) | 768/768 (100.00%) |
| Q8_0 | 551/768 (71.74%) | 551/768 (71.74%) | 457/768 (59.51%) | 452/768 (58.85%) |
| Q4_0 | 51/768 (6.64%) | 51/768 (6.64%) | 49/768 (6.38%) | 562/768 (73.18%) |
| Q4_K_M | 180/768 (23.44%) | 180/768 (23.44%) | 156/768 (20.31%) | 393/768 (51.17%) |

## Numerical checks

The diagnostic evaluates the identical causal token prefix through both
engines. It saves the final position's embedding, normalization outputs,
queries and keys before and after rotation, cached keys and values,
attention scores and probabilities, attention output, feed-forward
intermediates, every block output and all vocabulary logits. Full activation
dumps stay outside the repository. The committed records contain maximum
absolute and root mean square differences for each tensor.

The first engine disagreement in prompt order is used for each quantized
format. F16 uses its smallest top-two logit gap as a sensitive control.
Indices below are zero based.

| Format | Prompt | Generated position | Maximum logit difference | Maximum block-output difference | Record |
| --- | --- | ---: | ---: | ---: | --- |
| F16 | moon | 4 | 0.000028968 | 0.000152588 | [trace](reference/r2-probes-F16.json) |
| Q8_0 | library | 2 | 0.327652 | 0.998978 | [trace](reference/r2-probes-Q8_0.json) |
| Q4_0 | museum | 2 | 0.425063 | 0.528900 | [trace](reference/r2-probes-Q4_0.json) |
| Q4_K_M | library | 17 | 0.932501 | 1.613486 | [trace](reference/r2-probes-Q4_K_M.json) |

Each trace also replays individual operations on each engine's saved inputs,
which isolates local arithmetic from differences accumulated in earlier
layers. It checks five rows of every matrix, including the output head,
for 985 rows across 197 matrices per engine and format. GGML, llama.cpp's
numerical backend, independently decodes each sampled weight row. A double
precision product of those weights and the original F32 activation supplies
the local reference. A second reference uses GGML's converted activation.

- Every sampled decoded weight matches exactly, including the packed scales,
  offsets and signed codes. All 63,490 non-NaN F16 bit patterns convert
  identically. NaN means a floating-point value representing an invalid or
  undefined numerical result.
- Neuron's largest sampled product error against the original-activation
  reference is 5.96e-7. llama.cpp's largest error against its own converted
  activation reference is 2.45e-6. Its largest error against the original
  F32 activation is 0.258. These maxima pool both engines' saved inputs.
- llama.cpp uses Q8_0 activation blocks of 32 values for Q8_0 and Q4_0
  weights, including an F16 activation scale. Q4_K and Q6_K products use
  Q8_K activation blocks of 256 values with an F32 scale. Neuron retains
  F32 activations in the legacy formats. Its K path uses the checked R1
  integer packing and FP64 reconstruction. Neuron accumulates matrix
  products and attention reductions in FP64, with F32 stored activations and results.
  The first quantized projection already differs on identical normalized inputs.
- Root mean square normalization (RMSNorm), rotary position embeddings
  (RoPE), softmax attention probabilities, the sigmoid linear unit and
  gating operation (SwiGLU), and residual additions are replayed on shared
  inputs. All 281 Neuron operation checks per format match exactly.
  llama.cpp's residual additions match exactly. Its other shared-operation
  differences follow FP32 reduction and elementary-function rounding.
- Prompt IDs match across every engine and format. These runs use raw
  completions, so chat template rendering is outside their generation path.
  Both KV caches use F32. The instrumented runs reproduce both engines'
  original top-two IDs and logit gaps at the selected position.
  [Probe metadata and build identity](reference/r2-probes.json).

## Perplexity regression

Perplexity measures next-token prediction on fixed text, with lower values
indicating better prediction. The full M4-v2 slice was rerun with four
threads and automatic instruction selection, using 16 independent chunks
of context 512 and 4,096 scored targets. Every input ID and scored position
matches M4. All four results are at or below their M4 values and pass the
relative numerical guard of `1e-6`.

| Format | M4 Neuron | R2 Neuron | Recorded llama.cpp |
| --- | ---: | ---: | ---: |
| F16 | 11.580966363839 | 11.580966356650 | 11.580967791401 |
| Q8_0 | 11.584335697414 | 11.584335660935 | 11.614250911640 |
| Q4_0 | 13.129918942500 | 13.129918942500 | 13.143771032245 |
| Q4_K_M | 12.145460061051 | 12.145459935796 | 12.254528204041 |

The llama.cpp perplexities above retain the pinned M4-v2 runs. Q8_0 meets
the 1% F16 limit. Q4_K_M meets the 10% F16 and 2% same-file llama.cpp
limits. Q4_0 retains its original miss of the 10% F16 limit. The F16,
Q8_0 and Q4_0 values match R1 exactly. Q4_K_M is 2.04e-8 above its saved
R1 value and below M4. Current main includes R1's final precision guards.
The records retain the full precision.
[Per-format process records](reference/r2-perplexity.json),
[R1 comparison](reference/r1-perplexity.json).

## Speed regression

The R1 short speed protocol uses the raw prompt `Once upon a time`, context
256 and 32 generated tokens, with four threads and AVX-512. Generation
times the 31 steps after the first selected token. All rates below are
tokens per second. Each process has no warmup. Power, frequency and
temperature are uncontrolled.

The initial prompt-processing sample falls by 13.57%, so three alternating
before/after pairs were also collected, retaining every sample. The median
paired changes are -0.30% for prompt processing and -2.91% for generation.
Individual paired generation rates span 9.21 to 20.83. The baseline and
final executables have identical SHA-256 file-content checksums. Every
speed run generates the same 32 IDs.

| Sample set | Before prompt | After prompt | Before generation | After generation |
| --- | ---: | ---: | ---: | ---: |
| Initial pair | 23.429049 | 20.250420 | 12.162905 | 12.587743 |
| Medians of three alternating pairs | 23.695349 | 23.624105 | 20.635347 | 20.034089 |

The paired result shows no material generation regression. The spread
between these short samples reflects timing variability with identical
engine code. Every observation is retained.
[Initial before](reference/r2-speed-before.json),
[initial after](reference/r2-speed-after.json),
[all paired samples](reference/r2-speed-pairs.json).

## Build and regression checks

All seven configurations pass the full CMake test runner (CTest) suite.
GCC means GNU Compiler Collection. Werror treats compiler warnings as
errors. The optional OpenAI-client integration test skips in each build.
A separate check with the models directory configured also skips because
the OpenAI Python package is unavailable.
All four instruction-set tests execute, including AVX-512 and Vector
Neural Network Instructions (VNNI).

| Configuration | Passed | Optional skips |
| --- | ---: | ---: |
| GCC Release, default | 21 | 1 |
| GCC Release, Werror | 21 | 1 |
| GCC Release, AVX2, Werror | 21 | 1 |
| GCC Release, AVX-512, Werror | 21 | 1 |
| GCC Debug, AddressSanitizer and UndefinedBehaviorSanitizer, Werror | 21 | 1 |
| Clang Release, Werror | 21 | 1 |
| Clang Release, AVX2, Werror | 21 | 1 |

The new benchmark test covers early termination in either engine while
retaining all 768 planned positions. The external C++ numerical diagnostic
also compiles with warnings treated as errors under GCC and Clang.
The weights and file-size guard passes.
[Configure, build and test logs](reference/r2-validation.json),
[guard results](reference/r2-summary.json).

All 63 Qwen tokenizer fixtures and four ChatML template fixtures match
llama.cpp exactly, including rendered bytes and token IDs. ChatML is the
message-markup template used by Qwen chat.
[Real-model fixture results](reference/r2-qwen-fixtures.json).

## Reproduce

Use the model setup and external adapter build described in the quality record.
Compile the current `bench/llama_reference.cpp` with that adapter command,
setting its output to `/tmp/neuron-r2-reference`.
Keep model files, diagnostic binaries and activation dumps outside the
repository. Run model processes serially. [Model setup and adapter build](QUANT.md).

```sh
cmake -S . -B build-r2 -DCMAKE_CXX_COMPILER=g++ -DNEURON_WERROR=ON
cmake --build build-r2 -j 2
python3 bench/check_r2.py --binary build-r2/neuron \
  --reference-tool /tmp/neuron-r2-reference \
  --models-dir "$NEURON_MODELS_DIR" --llamacpp-dir "$NEURON_LLAMACPP_DIR" \
  --runs-dir bench/runs/r2-repeat --output bench/runs/r2-repeat.json \
  --notes 'Record actual machine activity and power conditions.'
python3 bench/trace_r2.py --build-dir /tmp/neuron-r2-trace-repeat \
  --neuron-build build-r2 --llamacpp-dir "$NEURON_LLAMACPP_DIR"
python3 bench/probe_r2.py --agreement bench/runs/r2-repeat.json \
  --trace-binary /tmp/neuron-r2-trace-repeat/r2-trace \
  --models-dir "$NEURON_MODELS_DIR" --runs-dir /tmp/neuron-r2-activations \
  --output bench/runs/r2-probes-repeat.json
python3 bench/check_r1.py --binary build-r2/neuron \
  --models-dir "$NEURON_MODELS_DIR" --runs-dir bench/runs/r2-ppl-repeat \
  --record-prefix r2-ppl-repeat --isa auto --threads 4 \
  --output bench/runs/r2-perplexity-repeat.json
python3 bench/validate_m5.py --build-prefix build-r2-check \
  --output bench/runs/r2-validation-repeat.json
```

The agreement runner verifies model, prompt and executable hashes before
reusing a saved process. Its successful exit means the measurement completed.
The recorded rates are compared with the M4 agreement targets.
The trace builder links a temporary instrumented model with the Neuron
library and the external pinned GGML/llama.cpp libraries. It checks the
source and binary identities against the agreement runs.
