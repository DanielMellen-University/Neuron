# Command-line reference

`neuron --help` prints the short form. This page explains the options that need more than one line.

## Sampling

`neuron run` needs either `--greedy` or `--temp T`. `--greedy` and `--temp 0` both pick the highest logit at every step, which is how every reference check in `bench/` was recorded.

```sh
./build/neuron run -m "$NEURON_MODELS_DIR/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf" --chat \
  -p "Write a haiku about compilers." --temp 0.7 --top-p 0.9 --min-p 0.05 --seed 1234 -n 64
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--temp T` | none | Divides the logits by `T` before sampling. Higher is more random. `0` is greedy. |
| `--top-k K` | `0` (off) | Keeps only the `K` most likely tokens. |
| `--min-p P` | `0` (off) | Drops tokens whose probability is below `P` times the most likely token's. |
| `--top-p P` | `1` (off) | Keeps the smallest set of tokens whose probabilities add up to at least `P`. |
| `--seed N` | random | Seeds the random source. The same seed, model, prompt and flags give the same tokens. |
| `--repeat-penalty X` | `1` (off) | Penalizes tokens seen in the last `--repeat-last-n` tokens (prompt included). A positive logit is divided by `X` and a negative one multiplied. It also works with `--greedy`. |
| `--repeat-last-n N` | `64` | Size of that window. `0` turns the penalty off. |

The filters run on the temperature-scaled distribution. `--top-k` and `--min-p` go first (they commute), then `--top-p` works on what is left. The random source is SplitMix64, written out in `src/sampling.cpp`, so a seed means the same thing with any compiler or standard library.

With `--json`, a sampled run adds a `sampling` object with every setting, including the seed that was drawn when `--seed` was not given. `top2` always describes the raw model logits, not the sampled token, so it stays useful for numerical checks. `--teacher-tokens` replays a fixed sequence and only works with plain greedy decoding.

## Stop strings

`--stop TEXT` ends generation just before `TEXT` appears. Repeat it for several strings; the earliest match wins. Text arrives one token piece at a time, so a stop string can be split across tokens. Neuron holds back any tail that could still turn into a stop string and prints it once it cannot, so nothing after the stop is ever shown. With `--json`, `neuron run` adds a `stop` field holding the string that matched, or `null` if none did.

## Chat and serve

`neuron chat` takes the same sampling flags and `--stop`. Every reply uses them; the default is still greedy.

`neuron serve` takes them as defaults for requests that leave a field out, along with `-n` for the default `max_tokens`. `POST /v1/chat/completions` accepts:

| Field | Range | Notes |
| --- | --- | --- |
| `temperature` | 0 to 2 | Default 0 (greedy) unless `serve --temp` sets another. |
| `top_p` | above 0, up to 1 | |
| `seed` | integer, 0 to 2^53 | Repeats a sampled reply exactly for the same request. |
| `stop` | string or up to 4 strings | Ends the reply with `finish_reason` `stop`. |
| `top_k`, `min_p`, `repeat_penalty` | as on the CLI | Extensions that llama.cpp's server also accepts. |

Fields sent as `null` keep the default. A sampled request without a `seed` gets the next value from a per-server seed stream. That stream starts from `serve --seed` when given (random otherwise), so a server started with a fixed seed hands out the same seeds in the same request order.

```python
client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "List three prime numbers."}],
    temperature=0.7, top_p=0.9, seed=7, stop=["\n\n"],
)
```

## Inspecting a GGUF file

`neuron inspect` reads a model file's header, metadata and tensor table without loading the weights, so it is instant even for a multi-gigabyte file.

```sh
./build/neuron inspect -m "$NEURON_MODELS_DIR/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf" --no-tensors
```

The first lines give the GGUF version, counts, alignment and where the tensor data starts, then the parameter count, weight size and average bits per weight, broken down by tensor type. A Q4_K_M file, for example, shows its mix of Q4_K and Q6_K tensors there. Metadata follows, one key per line with its type and value, then the tensor table with type, shape, size and offset.

| Flag | Meaning |
| --- | --- |
| `--no-tensors` | Skip the tensor table. |
| `--array-items N` | How many elements of each array to show (default 8; `0` shows all). Applies to text and JSON. |
| `--json` | Print everything as one JSON object instead. Strings are complete; invalid UTF-8 becomes U+FFFD. |
| `--key KEY` | Print one metadata value raw, with no quotes or truncation. Arrays print one element per line. |

`--key` is handy for pulling out a single setting, for example the chat template a model ships with:

```sh
./build/neuron inspect -m model.gguf --key tokenizer.chat_template
./build/neuron inspect -m model.gguf --key qwen2.context_length
```

Only the tensor types Neuron can run (F32, F16, Q8_0, Q4_0, Q4_K and Q6_K) are accepted. A file with any other type is rejected with the tensor's name, the same as when loading it.

## Tokens

`neuron tokenize` prints the IDs a prompt turns into. `--pieces` prints one token per line instead: the ID, the vocabulary entry as stored in the model, and the bytes it decodes to, both JSON-quoted so spaces and empty pieces are visible.

```sh
$ ./build/neuron tokenize -m "$NEURON_MODELS_DIR/stories15M.gguf" -p "Once upon" --pieces
1	"<s>"	""
9038	"▁Once"	" Once"
2501	"▁upon"	" upon"
```

`neuron detokenize` goes the other way. It takes IDs from `-p` or `--prompt-file`, separated by commas, spaces or brackets, so `tokenize` output can be piped straight back in. It prints the text with no trailing newline.

```sh
./build/neuron tokenize -m model.gguf -p "Hello there" | ./build/neuron detokenize -m model.gguf --prompt-file -
```

`--prompt-file -` reads standard input wherever `--prompt-file` is accepted, including `run`, `tokenize`, `logits` and `perplexity`.
