# M6 chat and server checks

This is the check sheet for terminal chat, the local web page, and the OpenAI-compatible HTTP server.

M6 is complete. All five acceptance items passed on the test laptop on
2026-11-17. The acceptance text in ROADMAP.md is unchanged.

## How it works

`ModelChat` shares greedy generation between the terminal and HTTP server.
It uses the existing `Model::step`, SIMD dispatch and matmul thread pool.
Each request checks the entire rendered prompt plus the requested output
budget against the context limit before inference or HTTP success headers.
Qwen2.5 uses the existing built-in ChatML template. SmolLM2 uses the same
delimiters with the default system text from its GGUF template. No Jinja
interpreter or external inference library is used.

The terminal keeps message history in memory and renders it again for each
turn. `/reset` clears it while preserving an explicit system prompt.
SIGINT sets a signal flag, checked between prompt and generated tokens.
An interrupted reply keeps its emitted text in history. Interrupting before
any text removes that pending turn. The signal handler does no I/O.

The POSIX server accepts one connection and request at a time, so model
access is serialized. It defaults to `127.0.0.1:8080`, supports an explicit
IPv4 `--host`, and shuts down on SIGINT or SIGTERM. Requests use HTTP/1.1
and Content-Length. Header and body limits are 16 KiB and 1 MiB, with a
10-second request read deadline. The parser rejects duplicate headers,
unsupported transfer encoding, invalid JSON, invalid UTF-8 and excessive
JSON nesting. Disconnected clients cancel inference between tokens.

`GET /v1/models` lists the loaded GGUF filename. Chat completions return
OpenAI-shaped JSON, token usage and stop or length finish reasons. SSE
contains the initial assistant role, content deltas, final finish reason,
optional usage and `[DONE]`. Incomplete UTF-8 characters are buffered across
token pieces. Errors before streaming return normal JSON HTTP errors;
inference failures after headers return an SSE error object and `[DONE]`.

The web page is compiled into the executable. Its only network calls go to
the same server. It reads fragmented SSE frames, maintains history, resets,
aborts replies, displays errors and inserts chat content as plain text.
There is no build step, CDN or runtime asset directory.

## Acceptance results

| Item | Result and evidence |
| --- | --- |
| Terminal history, streaming, reset, exit and Ctrl-C | Pass. Three model-free subprocess tests cover history, reset, EOF, context recovery and SIGINT during prefill, output and idle input. Real Qwen recalls Ada, survives interruption, resets and replies again. |
| Official OpenAI client, both completion modes and models | Pass with `openai` 3.52.0 on SmolLM2 F16 first and Qwen2.5 1.5B Q4_K_M last. Qwen replies `OK`, then `Ada`. Streaming text equals non-streaming text, including usage. |
| Embedded streaming web page | Pass. HTTP tests check the served page. A local QuickJS check with a minimal DOM and actual Qwen HTTP responses verifies two turns, SSE split into seven-character fragments, context-error rollback and reset. Browser access was unavailable, so rendered appearance was not inspected. |
| Context, JSON and model errors | Pass. HTTP 400 `context_length_exceeded`, HTTP 400 `invalid_json` and HTTP 404 `model_not_found`. Tests cover both stream settings and verify errors precede SSE success headers. |
| Default loopback bind | Pass. The model-free socket test inspects the listening address in `/proc/net/tcp`; real client checks also use the default host. |

[Recorded functional results](reference/m6-checks.json) include both SDK
runs, model hashes, the Qwen terminal transcript and web JavaScript results.
The HTTP test also covers fragmented requests, Unicode escapes, schema and
framing errors, size limits, inference failures and disconnect recovery.

## Build and regression results

GCC 11.4.0, Clang 14.0.0, CMake 3.22.1 and Python 3.10.12 were used.
All configurations build and pass CTest. Each has 21 passing entries and one
clean skip for the optional model/client integration test. That integration
test was run separately in a temporary virtual environment with real models.
The fake-backend HTTP and terminal tests run in every configuration.

| Configuration | Result |
| --- | --- |
| GCC Release default | 21 pass, 1 optional skip |
| GCC Release Werror | 21 pass, 1 optional skip |
| GCC Release Werror, AVX2 | 21 pass, 1 optional skip |
| GCC Release Werror, AVX-512 | 21 pass, 1 optional skip; runtime tests also run on the test laptop |
| GCC Debug Werror, ASan and UBSan | 21 pass, 1 optional skip |
| Clang Release Werror | 21 pass, 1 optional skip |
| Clang Release Werror, AVX2 | 21 pass, 1 optional skip |

Clang on this machine selects an incomplete GCC 12 installation by default.
As in M5, the local Clang builds explicitly use GCC 11 headers and libraries:

```sh
cmake -S . -B build-m6-clang-fixed -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Release -DNEURON_WERROR=ON \
  '-DCMAKE_CXX_FLAGS=--gcc-toolchain=/usr -isystem /usr/include/c++/11 -isystem /usr/include/x86_64-linux-gnu/c++/11' \
  -DCMAKE_EXE_LINKER_FLAGS=-L/usr/lib/gcc/x86_64-linux-gnu/11
```

The M2 TinyStories exact-match check passes all 1,152 generated IDs across
three prompts and three repetitions. Prompt IDs also match. The
[raw harness run](reference/tinystories-m6.json) and
[comparison](reference/m6-m2-compare.json) preserve the results. These runs
had concurrent build/test activity, so their timings are not speed claims.
No inference kernels, model math or existing CLI generation paths changed.
The tracked-file weights and 1 MiB size guard passes.

## Reproduce

Keep models outside the repo and set the model directory first:

```sh
export NEURON_MODELS_DIR=/path/to/models
cmake -S . -B build-m6-default
cmake --build build-m6-default -j2
ctest --test-dir build-m6-default --output-on-failure

cmake -S . -B build-m6-sanitize -DCMAKE_BUILD_TYPE=Debug \
  -DNEURON_WERROR=ON -DNEURON_SANITIZE=ON
cmake --build build-m6-sanitize -j2
ctest --test-dir build-m6-sanitize --output-on-failure

# The SDK is only a test dependency. Install it in a temporary virtual environment.
python3 -m venv /tmp/neuron-m6-client-venv
/tmp/neuron-m6-client-venv/bin/pip install openai
/tmp/neuron-m6-client-venv/bin/python scripts/test_openai_client.py \
  --binary build-m6-default/neuron \
  --model "$NEURON_MODELS_DIR/SmolLM2-135M-Instruct-f16.gguf"
/tmp/neuron-m6-client-venv/bin/python scripts/test_openai_client.py \
  --binary build-m6-default/neuron

python3 bench/harness.py run-neuron --binary build-m6-default/neuron \
  --model-file "$NEURON_MODELS_DIR/stories15M.gguf" \
  --output bench/runs/tinystories-m6.json --notes 'Record current load and power state.'
python3 bench/harness.py compare bench/reference/tinystories-15m.json \
  bench/runs/tinystories-m6.json --threads 1 --exact-match
```

The SDK script exits 77 with a reason if either the model or package is
missing. Both skip paths were checked. It launches and stops its own server.

## Limits

The API is text-only and greedy, with one choice per request. Temperature
zero and top-p one are accepted; other sampling settings, tool calls,
custom stop strings and response formats return HTTP 400. There is no
authentication, TLS, batching, chunked request decoding or persistent HTTP
connection. The terminal and browser keep history; API callers send their
own history. Every turn recomputes its prompt rather than sharing a cache
across conversations. These limits do not leave an M6 acceptance item open.
