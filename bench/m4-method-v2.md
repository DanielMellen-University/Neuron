# M4 quality method, second round

This file freezes the second-round M4 quality method and its inputs, so later measurements can be compared against the same rules. Perplexity measures next-token prediction on a fixed text. Free-running agreement compares generated token IDs when each format feeds back its own output. Teacher forcing replays the F16 continuation and is a diagnostic only.

This method and its inputs are committed before any new quality measurement.
The original three prompts and 128 scored targets remain in `QUANT.md` and
`reference/m4-quantization.json` as historical results. No prompt is selected
or removed in response to a quantized continuation.

`wikitext2-m4-v2.txt` contains the first 48,000 bytes of the same pinned
WikiText-2 raw test source described in [wikitext2-slice.md](wikitext2-slice.md),
cut back to the last space and followed by a newline. It is 47,996 bytes,
SHA-256 `1b33b1af5632849304fc096a1df248379a9dd8399b129492623393ce5b115acc`.
Source revision, Parquet hash, attribution and CC BY-SA 3.0 terms are unchanged.
Regenerate with that document's script, replacing `[:1600]` with `[:48000]`,
the expected text hash with the hash above, and the output path with
`bench/wikitext2-m4-v2.txt`.

Tokenize with the pinned Qwen2.5 F16 vocabulary, normal add-special and
parse-special flags, and no chat template. Evaluate the first 8,192 tokens
as 16 independent chunks of context 512. The text has 11,187 Qwen tokens.
The SHA-256 of the first 8,192 IDs serialized as a compact JSON array is
`a4021be8153841fa89a79f4dc118d8299f348146b95404b119b88dc0b599e316`.
Reset K/V for each chunk. Targets
256..511 contribute 256 losses per chunk, for 4,096 scored targets. The
remaining tokens are ignored. Use double log-sum-exp and F32 K/V in both
Neuron and the pinned llama.cpp adapter, one thread, no Flash Attention.
This is a fixed prefix check, not the full WikiText-2 benchmark. The larger
sample reduces sampling noise; contiguous targets remain correlated.

`prompts-m4-v2.json` fixes 12 raw English completion prompts across ordinary
activities and basic explanations. SHA-256:
`b5f26b7891b66f0f98a52e9bc362d823885b02883378ece309a7ac3ee2a79060`.
Generate 64 tokens per prompt with context 256, greedy argmax, no penalties,
one thread. Each format feeds back its own tokens. Pool equal token IDs at
the same positions against Neuron F16 over all 768 positions. An early end
counts ungenerated positions as mismatches; retain the prompt and report it.

Also replay the exact F16 continuation IDs for each prompt through each
format. At every one of those 64 positions, compare that format's argmax
with F16's selected ID. This teacher-forced top-1 rate diagnoses local logit
changes but does not replace the free-running acceptance rate.

Measure F16, pure Q8_0, pure Q4_0, and llama.cpp's standard Q4_K_M mix made
from the same pinned F16 file. Record model and executable hashes, commands,
file size, GNU time peak RSS, prompt and generation speed, and observed
machine conditions. Use one measured process per task, with no warmup or
repetitions. Run model processes serially. Report each 4-bit candidate
separately and apply the existing ROADMAP criteria without changing them.
