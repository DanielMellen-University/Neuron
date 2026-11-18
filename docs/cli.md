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
