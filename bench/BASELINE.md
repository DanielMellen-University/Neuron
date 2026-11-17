# llama.cpp baseline

This is the llama.cpp reference record used to measure Neuron. Use it for the pinned models, the measurement method, and the original tokens/s numbers.

Measured on 2026-11-16 UTC. These are llama.cpp speeds, not Neuron speeds.
These measurements predate Neuron inference. [M2 checks](M2_CHECKS.md)
record the Neuron comparison.

## Measurements

Hardware: an Intel Core i5-1135G7 laptop (4 cores/8 threads, AVX-512 VNNI),
16 GB RAM, AVX2 and AVX-512 with VNNI. The installed OS reports Zorin OS 17.3, based on Ubuntu
22.04. The build used GCC 11.4.0 and CMake 3.22.1, CPU only, Release,
`GGML_NATIVE=ON`, OpenMP enabled and BLAS disabled.

The machine was otherwise idle: no builds or tests ran during collection.
Other development tools and normal background services were present. Immediately before the
final run, two one-second `vmstat` samples reported 99% CPU idle. AC power
was connected, the battery was full, the governor was `powersave`, the
driver was `intel_pstate`, and turbo was enabled (`no_turbo=0`). These
settings were observed, not changed. CPU frequency and temperature were
not held constant. Power state and load averages before and after each
model are in the JSON files.

| Model | Threads (pp and tg) | Prompt processing (tokens/s) | Generation (tokens/s) |
| --- | ---: | ---: | ---: |
| TinyStories 15M F32 | 1 | 3041.34 | 280.95 |
| TinyStories 15M F32 | 4 | 7980.83 | 634.81 |
| TinyStories 15M F32 | 8 | 9043.82 | 613.48 |
| SmolLM2 135M Instruct F16 | 1 | 253.68 | 66.96 |
| SmolLM2 135M Instruct F16 | 4 | 801.21 | 138.09 |
| SmolLM2 135M Instruct F16 | 8 | 657.62 | 129.73 |

Sources: [TinyStories results](reference/tinystories-15m.json) and
[SmolLM2 results](reference/smollm2-135m-instruct.json). The `summary` arrays
contain the unrounded rates and the counts and times used to calculate them.
Each file is under 112 kB. Neither file contains weights or logits.

## Pinned inputs

llama.cpp [b12284](https://github.com/ggml-org/llama.cpp/releases/tag/b12284),
commit `9e9619d030bbe6b584d826a0a6cf06e38251f52b`.
The measured `llama-server --version` reports GNU 11.4.0 for Linux x86_64.
The source checkout was clean. The server executable's SHA-256 is
`872323da38c018ffdb74b96a83232aa9023aa0b5ef4ed29ff5e8e3998134eb17`.
The executable itself is not committed.

| Model file | SHA-256 computed from the downloaded file |
| --- | --- |
| `stories15M.gguf` | `61b50d457809a5194818fd22e6724b456cd7bb9a6264c52c8110684c53f3704a` |
| `SmolLM2-135M-Instruct-f16.gguf` | `f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57` |

TinyStories comes from
[`ggml-org/models`](https://huggingface.co/ggml-org/models/tree/499bc8821c6b12b4e53c5bffcb21ec206f212d81/tinyllamas)
at revision `499bc8821c6b12b4e53c5bffcb21ec206f212d81`, file
`tinyllamas/stories15M.gguf`. SmolLM2 comes from
[`bartowski/SmolLM2-135M-Instruct-GGUF`](https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/tree/09816acd5d99df7be770d85ea30822623dab342c)
at revision `09816acd5d99df7be770d85ea30822623dab342c`, file
`SmolLM2-135M-Instruct-f16.gguf`. The download script pins both revisions
and file hashes. An existing valid download is reused. A missing or corrupt
file is downloaded to a temporary file, verified, and then moved into place.

## Method

[`prompts.json`](prompts.json) fixes three raw story completion prompts.
Both models receive exactly this text, without a chat template. The harness
calls `/tokenize` with `add_special=true` and `parse_special=true`, records
the returned IDs, and passes those IDs directly to `/completion`. This
includes the model's BOS token when its tokenizer requires one.

Decoding uses seed 42, temperature 0, only the temperature sampler, repetition
penalty 1.0, and at most 128 new tokens. This selects the highest logit without
repetition penalties. EOS is respected. The `/completion` response supplies
generated IDs directly through `return_tokens=true`; the harness does not
retokenize the generated text. All recorded completions reached 128 tokens.
For each prompt, token IDs matched across all three repetitions and all
three thread counts, including prompt token IDs.

For each model and thread count, the harness starts a fresh server with one
slot, a 2048-token context, batch and microbatch sizes of 512, no GPU layers,
and Flash Attention disabled. Generation and batch thread counts are both
set to the table's thread count. One full completion of the first prompt
warms the model and context, then all three prompts run three times. The
warmup is excluded. Requests run sequentially with prompt caching disabled;
the harness rejects cached or truncated samples. No compilation or tests
run alongside the measurements.

The old TinyStories GGUF declares a 128-token training context. This server
caps its slots to that metadata even with `--ctx-size 2048`, which truncates
a prompt plus a 128-token continuation. For TinyStories only, the harness
sets `--override-kv llama.context_length=int:2048`. This leaves the downloaded
file unchanged and avoids context shifts or truncation. The override is part
of the reference configuration, including for future Neuron comparisons.
It is not a claim about model quality beyond its declared training context.

Each table row pools nine measured responses. Prompt speed is
`1000 * sum(prompt_n) / sum(prompt_ms)`. Generation speed is
`1000 * sum(predicted_n - 1) / sum(predicted_ms)`, matching this llama-server's
timing convention: the first sampled token comes from prompt evaluation,
so only the remaining generation steps are timed. These are rates from
the real prompts, not synthetic llama-bench results or HTTP wall time.
Short prompt timings and laptop power management can cause variation.

## Reproduce

Run from the Neuron repository root. Requires Git, curl, sha256sum, Python
3.8+, CMake 3.20+, GCC and G++. No Python packages, GPU toolkit, BLAS, Node,
or model service account is required. Use G++ on the test laptop; the local Clang
installation cannot link. The build script downloads no UI assets.

```sh
scripts/fetch_models.sh
CC=gcc CXX=g++ scripts/build_llamacpp.sh

# Run these sequentially, with other workloads stopped.
python3 bench/harness.py run --model tinystories-15m \
  --output bench/runs/tinystories-15m.json \
  --notes 'Otherwise idle: no builds or tests running; other development tools and normal background services present. Pre-run vmstat showed 99% idle in two one-second samples.'
python3 bench/harness.py run --model smollm2-135m-instruct \
  --output bench/runs/smollm2-135m-instruct.json \
  --notes 'Otherwise idle: no builds or tests running; other development tools and normal background services present. Pre-run vmstat showed 99% idle in two one-second samples.'

python3 bench/harness.py compare bench/reference/tinystories-15m.json \
  bench/runs/tinystories-15m.json --exact-match
python3 bench/harness.py compare bench/reference/smollm2-135m-instruct.json \
  bench/runs/smollm2-135m-instruct.json --exact-match
```

The reference collection used the same `run` commands with output paths under
`bench/reference/`. Describe the actual activity in `--notes` when rerunning.
Both result files record the complete server commands, prompts, token IDs,
text, sampling settings, timings, executable hash and model hash. Thread
counts, prompts, seed and decoding settings are fixed in the harness;
`--repetitions` defaults to 3. A comparison with the committed files should
also use three repetitions.

`NEURON_MODELS_DIR` overrides `models/` for the download script and harness.
`NEURON_LLAMACPP_DIR` overrides `third_party/llama.cpp` for the build script
and harness. The harness also accepts `--models-dir` and `--llamacpp-dir`.
Choose an external or git-ignored directory for overrides. The build is
host-native; a binary built on the test laptop may need rebuilding on another CPU.
`NEURON_BUILD_JOBS` sets build parallelism (default 4).

## Compare results

The first file is the reference, the second is the candidate. Comparison
emits JSON to stdout. Runs are paired by model ID, thread count, prompt ID
and repetition, regardless of row order. The `prompt` and `generation`
fields report the first divergent token index (zero based, `null` for an
exact match) and positional agreement. Agreement counts matching positions
divided by the longer sequence length, so an unmatched tail is a mismatch.
Two empty sequences agree. The aggregate agreement rate pools generated
tokens across all runs; prompt agreement is reported separately per run.

Each paired run reports pp and tg speed ratios as candidate/reference;
values above 1 mean the candidate was faster. Missing or extra runs are
mismatches, and duplicate keys or invalid timings are input errors. Exit
status is 0 for a valid report, 1 for any prompt or generated token mismatch
with `--exact-match`, and 2 for invalid input or a failed collection. Without
`--exact-match`, token mismatches are still reported but return 0.

A candidate uses `schema_version: 1`, `model.id`, and a `runs` array. Each
row needs `threads`, `prompt_id`, `repetition`, `prompt_tokens`,
`generated_tokens`, and `timings.prompt_per_second` and
`timings.predicted_per_second` (positive finite numbers). Token IDs are
nonnegative integers. The committed results show the full collection
schema, including metadata and generated text; comparison only requires
the fields listed here.

## Validation

On the test laptop, all five CTest entries passed in each of these GCC builds:
default Release, Release with `NEURON_WERROR=ON`, Release with Werror and
AVX2, Release with Werror and AVX-512, and Debug with Werror and ASan/UBSan.
The harness entry runs 12 model-free Python tests covering first divergence,
length mismatches, empty sequences, prompt mismatches, row pairing,
speed ratios, input validation, CLI exit codes and timing aggregation.
The downloader also passed a real corrupt-file repair and idempotent reuse
check with `NEURON_MODELS_DIR` set to a temporary path containing spaces.
The GitHub Actions workflow also exercises Clang.

```sh
cmake -S . -B build -DCMAKE_CXX_COMPILER=g++
cmake --build build -j 4
ctest --test-dir build --output-on-failure

cmake -S . -B build-sanitize -DCMAKE_CXX_COMPILER=g++ \
  -DNEURON_WERROR=ON -DNEURON_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-sanitize -j 4
ctest --test-dir build-sanitize --output-on-failure
```
