# M3: Qwen2.5, ChatML and KV cache checks

This is the correctness and memory record for loading Qwen2.5 1.5B, applying its ChatML chat format, and checking the KV cache (the stored attention keys and values from earlier tokens).

Local checks on an Intel Core i5-1135G7 laptop (4 cores/8 threads, AVX-512 VNNI),
16 GB RAM, Ubuntu 22.04 base, GCC 11.4 and CMake 3.22. The engine uses scalar
kernels and one thread. These are correctness and memory checks. They make no
speed claim.
The full-context run runs in the background alongside the shorter checks;
the JSON records the observed power state and other activity.

## Pinned inputs

The model is `Qwen2.5-1.5B-Instruct-f16.gguf` from
[bartowski/Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/tree/9eadc66189c7641e1ddd226b8267a9119b2ce2d4),
revision `9eadc66189c7641e1ddd226b8267a9119b2ce2d4`. The exact download URL is
pinned in [fetch_models.sh](../scripts/fetch_models.sh). The downloaded file
has 3,093,669,376 bytes and SHA-256
`b6eaec3509f1d0373d1f4802654c4a7bfcde0768645d2d45e8885f6922b428ee`.
The hash was checked against the downloaded bytes. F16 weights are converted
to FP32 on load. There is no quantized inference in M3.

The model declares architecture `qwen2`, 28 layers, embedding width 1536,
FFN width 8960, 12 query heads, 2 KV heads, head size 128, RMS epsilon 1e-6,
RoPE base 1e6 and a 32768-token training context. Q, K and V each have a
bias. RoPE pairs the first half of each head with the second half, unlike
TinyStories' adjacent-pair RoPE. The output head uses the tied embeddings
when the GGUF has no separate output tensor.

The external reference is the M1 llama.cpp checkout, commit
`9e9619d030bbe6b584d826a0a6cf06e38251f52b` (b12284). The adapter uses CPU
inference, one thread, context 2048, batch/microbatch 512, Flash Attention
disabled, and F32 keys and values. Greedy decoding selects the largest
logit without sampling or penalties. No random seed affects argmax.

## Build and reproduce

Run from the Neuron worktree root. Weights and external binaries stay
outside the repo:

```sh
export NEURON_MODELS_DIR="$HOME/.cache/neuron/models"
export NEURON_LLAMACPP_DIR="$HOME/.cache/neuron/llama.cpp"
scripts/fetch_models.sh
# Repeating the command verifies and reuses all three files.

cmake -S . -B build-m3 -DCMAKE_CXX_COMPILER=g++ -DNEURON_WERROR=ON
cmake --build build-m3 -j 3

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
  -fopenmp -pthread -ldl -o /tmp/neuron-m3-reference

python3 bench/check_m3.py --binary build-m3/neuron \
  --model "$NEURON_MODELS_DIR/Qwen2.5-1.5B-Instruct-f16.gguf" \
  --reference-tool /tmp/neuron-m3-reference --output bench/runs/m3-checks.json \
  --notes 'Record the actual machine activity and power conditions.'
```

The check script verifies the model hash, calls the reference tokenizer and
Jinja renderer again, and checks their results against the committed
fixtures. It also collects fresh reference generations, checks them against
the committed 32-token sequences, runs Neuron on each prompt, compares
cached and recomputed TinyStories generation, and runs a two-exchange Qwen
conversation. Neuron emits the top two IDs and logit gap at each sample.
The script records the first divergence and both gaps if tokens differ.
It saves the report before rejecting a greedy mismatch.

To regenerate fixtures intentionally, add `--refresh-fixtures`. To collect
only the reference generations, add `--collect-reference`. To check only
tokenization and rendering, add `--fixtures-only`. These options use the
same pinned model and external reference tool. The initial fixture IDs
were collected with the same `llama_tokenize` flags and Jinja calls; each
case records its input and expected IDs.

## Tokenizer and chat template

The [tokenizer fixture](reference/qwen2-tokenizer.json) has 63 cases. All
63 match a fresh llama.cpp call. They include the three M1 prompts, Unicode
letters and numbers, combining accents, emoji, contractions, punctuation,
control bytes, whitespace and adjacent special tokens. Each edge case runs
with all four `add_special` / `parse_special` combinations.

GGUF `tokenizer.ggml.merges` supplies merge priority. The tokenizer uses the
GPT-2 byte alphabet and Qwen2's ordered pre-tokenizer alternatives:

```text
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

The scan uses Unicode 15.1 letter and number categories and the Unicode
White_Space set. [generate_unicode.py](../scripts/generate_unicode.py)
pins the UnicodeData.txt URL and hash and regenerates the small range table.
The data license is included in [LICENSE-UNICODE.txt](../LICENSE-UNICODE.txt).
An exhaustive local audit found zero letter/number category differences
across all 1,114,112 code points against the pinned llama.cpp table.
No locale or external regex library affects tokenization. Invalid UTF-8
is rejected with an error. Decoding maps the byte alphabet back to bytes.

The [chat fixture](reference/qwen2-chat.json) was obtained by calling
`common_chat_templates_apply` with the GGUF's own Jinja template,
`use_jinja=true` and `add_generation_prompt=true`, then `llama_tokenize`
with `add_special=true` and `parse_special=true`. This matters because the
older built-in ChatML renderer does not insert the GGUF's default system
text. Neuron implements the text-only branch of this Qwen2.5 template:

```text
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
...<|im_end|>
<|im_start|>assistant
```

An explicit first system message replaces the default, including when its
content is empty. Message content is preserved, including whitespace.
Tool calls and other model templates are outside M3.

| Conversation | Rendered bytes | Prompt token IDs |
| --- | --- | --- |
| user, assistant, user, default system | exact | 57/57 |
| three user exchanges, default system | exact | 84/84 |
| explicit system plus user, assistant, user | exact | 41/41 |
| empty explicit system plus user | exact | 14/14 |

The [small vocabulary fixture](reference/qwen2-vocabulary.json) keeps the
byte alphabet, special tokens and merge ancestry needed for these cases.
CI fills unused IDs with inert entries in a temporary GGUF, retaining the
original token IDs. It runs every tokenizer and chat fixture against the
real CLI without model weights, llama.cpp or network access. Hand-written
unit tests separately cover merge priority, digit boundaries, decoding,
RoPE, QKV bias, GQA and template behavior.

## KV cache and context bounds

Each layer stores FP32 K/V vectors in position order. A call to `step`
projects only the new token and attends to stored past positions. `reset`
clears sequence state. `forward` resets and evaluates the full sequence.

The synthetic Llama and Qwen2 GQA tests compare cached logits and greedy
IDs with full recomputation at every step for 32 generated tokens. Logits
and IDs are exactly equal. The Qwen2 case has nonzero Q/K projections and
biases. The real-model check uses TinyStories with the prompt `Once upon a
time`, 32 greedy tokens and `--recompute`; its result is recorded below.
Qwen's cached generation is separately compared with llama.cpp.

The model checks context bounds before writing to any layer's cache.
The CLI rejects a prompt plus requested generation exceeding `--ctx-size`
before loading weights. Unit tests cover the last valid position, the next
rejected position, an oversized prompt, and the CLI error. The CLI defaults
to 2048. An explicit context override is still available for the M1
TinyStories check.

## Local results

The [complete local report](reference/m3-checks.json) records the final
binary hash, actual commands, candidate IDs, gaps, conversation replies
and observed machine state. All three raw M1 prompts produce exactly the
first 32 llama.cpp IDs.
[Committed reference IDs](reference/qwen2-greedy.json) include all top-two
IDs and gaps, generated text and exact adapter commands. The fresh
reference call also matches the committed IDs. No near-tie exception is
needed.

| Prompt | Prompt tokens | Generated IDs matched | First divergence | Minimum reference top-two gap | Minimum Neuron top-two gap |
| --- | ---: | ---: | --- | ---: | ---: |
| rabbit | 56 | 32/32 | none | 0.04211807 | 0.04212189 |
| garden | 58 | 32/32 | none | 0.00113869 | 0.00113487 |
| ball | 55 | 32/32 | none | 0.09457397 | 0.09457588 |

The real TinyStories cache check matches 32/32 generated IDs. All recorded
top-two IDs and gaps also match exactly between cached and recomputed runs.

The actual two-exchange Qwen conversation recalls the prior user name:
`My name is Ada. Reply with only OK.` produces `OK`, then
`What is my name? Reply with only the name.` produces `Ada`. The second
invocation supplies the actual first reply through `--message assistant`.

The full 2048-position run completed with exit status 0. Peak RSS is
**9,101,076 KiB (8.68 GiB)**. The method and complete time report are below.
All M3 acceptance items are met locally.

TinyStories M2 regression: all 51 tokenizer cases match; maximum absolute
logit differences are 9.5367431640625e-6 (rabbit), 8.58306884765625e-6
(garden) and 9.5367431640625e-6 (ball). All 1152/1152 generated IDs match
M1 over three prompts, three repetitions and 128 tokens per run.
[Committed M3 TinyStories runs](reference/tinystories-m3.json) include
the actual IDs, commands, binary hash and numeric checks.

```sh
python3 bench/check_m2.py --binary build-m3/neuron \
  --model "$NEURON_MODELS_DIR/stories15M.gguf" \
  --reference-tool /tmp/neuron-m3-reference --output bench/runs/m3-m2-checks.json
python3 bench/harness.py run-neuron --binary build-m3/neuron \
  --model-file "$NEURON_MODELS_DIR/stories15M.gguf" \
  --output bench/runs/m3-tinystories.json \
  --notes 'Scalar FP32, one thread. Qwen 2048-position memory collection running concurrently. Correctness regression, no speed claim.'
python3 bench/harness.py compare bench/reference/tinystories-15m.json \
  bench/runs/m3-tinystories.json --threads 1 --exact-match
```

## Memory method

[context_memory.cpp](context_memory.cpp) loads the real Qwen F16 file with a
2048-token context and evaluates exactly 2048 positions of token 16 (`1`).
It suppresses the output head for the first 2047 positions, as normal
prefill does, and evaluates final logits at position 2047. All 28 layers'
K/V caches are populated. Their final FP32 payload is 112 MiB:
`28 * 2048 * 2 * 128 * 2 * sizeof(float)`. This measures a full context,
including model load and inference, rather than only reserving capacity.

```sh
g++ -std=c++20 -O2 -Iinclude -Ibuild-m3/generated bench/context_memory.cpp \
  build-m3/libneuron.a -o /tmp/neuron-m3-memory
/usr/bin/time -v -o /tmp/neuron-m3-memory.time \
  /tmp/neuron-m3-memory "$NEURON_MODELS_DIR/Qwen2.5-1.5B-Instruct-f16.gguf" \
  > /tmp/neuron-m3-memory.out 2> /tmp/neuron-m3-memory.progress
```

The one long run uses the same source, initially compiled from an identical
copy at `/tmp/neuron-m3-memory.cpp`. `/usr/bin/time -v` reports maximum RSS
in KiB for this process, including resident mmap pages and converted FP32
weights. It excludes other processes and is not total system memory.
The helper takes raw token IDs and does not construct a CLI tokenizer.
Rebuilding it against the final library produced the same executable
SHA-256 as the running binary, so the measured inference code is unchanged.
The run has no SIMD or threading additions.

The [memory record](reference/m3-memory.json) contains the complete GNU
time report, command, source/binary hashes, power observation during the
run and machine state after it. The final output is
`context=2048 position=2048`.

| Measurement | Result |
| --- | ---: |
| evaluated positions / selected context | 2048 / 2048 |
| maximum resident set size | 9,101,076 KiB |
| maximum resident set size | 8.67946 GiB |
| wall time, including load and cleanup | 3008.33 s (50:08.33) |
| user / system CPU time | 3000.99 / 3.24 s |
| populated FP32 K/V payload | 112 MiB |
| exit status | 0 |

## Model-free validation

```sh
cmake -S . -B build-m3-default -DCMAKE_CXX_COMPILER=g++
cmake --build build-m3-default -j 2
ctest --test-dir build-m3-default --output-on-failure

cmake -S . -B build-m3-sanitize -DCMAKE_CXX_COMPILER=g++ \
  -DNEURON_WERROR=ON -DNEURON_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-m3-sanitize -j 2
ctest --test-dir build-m3-sanitize --output-on-failure
```

All 12 CTest entries pass in each configuration below. The
[validation record](reference/m3-validation.json) contains commands and
log hashes. The separate Werror build uses `build-m3`.

| GCC configuration | Result |
| --- | --- |
| default Release | 12/12 |
| Release, Werror | 12/12 |
| Debug, Werror, ASan/UBSan | 12/12 |
| Release, Werror, AVX2 | 12/12 |
| Release, Werror, AVX-512 | 12/12 |

The AVX2 and AVX-512 checks used the same commands with
`-DNEURON_WERROR=ON -DNEURON_AVX2=ON` or
`-DNEURON_WERROR=ON -DNEURON_AVX512=ON`, separate build directories and
`-j 2`. AVX-512 tests ran on the test laptop. No optimized kernels were added.
Clang 14 also passes compile-only Werror checks for all 16 translation
units using GCC 11's headers:

```sh
for source in src/*.cpp tests/test_*.cpp; do
  clang++ -std=c++20 -O3 -DNDEBUG -fsyntax-only -Werror \
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion \
    -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wnull-dereference \
    -isystem /usr/include/c++/11 -isystem /usr/include/x86_64-linux-gnu/c++/11 \
    -Iinclude -Ibuild-m3/generated -Itests "$source" || exit
  echo "OK $source"
done
```

The weight and 1 MiB file guard passes for tracked and pending files.
The existing CI workflow runs all model-free tests for GCC and Clang
Release, baseline and AVX2, plus GCC ASan/UBSan, an AVX-512 compile check,
and the weight/large-file guard.
The test laptop's Clang installation cannot link; GCC supplies all local executable
builds.
