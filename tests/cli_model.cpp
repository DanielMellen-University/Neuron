// Writes a tiny F32 llama GGUF for CLI tests. Attention and FFN weights are
// zero, so each step's logits depend only on the current token: the output
// matrix sends every token to a fixed successor. From the prompt "a" greedy
// decoding writes " b cabcde" and then the end-of-sequence token.
#include "gguf_fixture.hpp"

#include <array>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 2) { std::fprintf(stderr, "usage: cli_model OUTPUT.gguf\n"); return 2; }
  constexpr std::size_t vocabulary = 12, width = 12, hidden = 4, kv_width = 6;
  const std::array<const char*, vocabulary> tokens{"<unk>", "<s>", "</s>", "\xe2\x96\x81", "a", "b", "c", "d", "e",
      "\xe2\x96\x81" "a", "\xe2\x96\x81" "b", "\xe2\x96\x81" "c"};
  const std::array<std::int32_t, vocabulary> types{2, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  const std::array<std::size_t, vocabulary> successor{2, 9, 2, 9, 5, 6, 7, 8, 2, 10, 11, 4};
  fixture::Bytes b;
  b.header(12, 16);
  b.text("general.architecture", "llama");
  b.text("general.name", "neuron cli fixture");
  b.uint("llama.embedding_length", width); b.uint("llama.feed_forward_length", hidden);
  b.uint("llama.block_count", 1); b.uint("llama.attention.head_count", 2);
  b.uint("llama.attention.head_count_kv", 1); b.uint("llama.rope.dimension_count", 6);
  b.uint("llama.context_length", 64);
  b.key("llama.attention.layer_norm_rms_epsilon", 6); b.put(1e-5f);
  b.text("tokenizer.ggml.model", "llama");
  b.key("tokenizer.ggml.tokens", 9); b.put<std::uint32_t>(8); b.put<std::uint64_t>(vocabulary);
  for (const auto* token : tokens) b.string(token);
  b.key("tokenizer.ggml.scores", 9); b.put<std::uint32_t>(6); b.put<std::uint64_t>(vocabulary);
  for (std::size_t i = 0; i < vocabulary; ++i) b.put(i >= 9 ? 1.0f : 0.0f);
  b.key("tokenizer.ggml.token_type", 9); b.put<std::uint32_t>(5); b.put<std::uint64_t>(vocabulary);
  for (const auto type : types) b.put(type);
  b.uint("tokenizer.ggml.bos_token_id", 1); b.uint("tokenizer.ggml.eos_token_id", 2);
  struct Tensor { std::string name; std::vector<std::uint64_t> dims; std::vector<float> values; };
  auto zeros = [](std::size_t n) { return std::vector<float>(n, 0.0f); };
  std::vector<float> embedding(width * vocabulary, 0.0f), output(width * vocabulary, 0.0f);
  for (std::size_t t = 0; t < vocabulary; ++t) {
    embedding[t * width + t] = 1;
    output[successor[t] * width + t] = 1;
  }
  const std::vector<Tensor> tensors{
    {"token_embd.weight", {width, vocabulary}, embedding},
    {"output_norm.weight", {width}, std::vector<float>(width, 1.0f)},
    {"output.weight", {width, vocabulary}, output},
    {"blk.0.attn_norm.weight", {width}, std::vector<float>(width, 1.0f)},
    {"blk.0.attn_q.weight", {width, width}, zeros(width * width)},
    {"blk.0.attn_k.weight", {width, kv_width}, zeros(width * kv_width)},
    {"blk.0.attn_v.weight", {width, kv_width}, zeros(width * kv_width)},
    {"blk.0.attn_output.weight", {width, width}, zeros(width * width)},
    {"blk.0.ffn_norm.weight", {width}, std::vector<float>(width, 1.0f)},
    {"blk.0.ffn_gate.weight", {width, hidden}, zeros(width * hidden)},
    {"blk.0.ffn_up.weight", {width, hidden}, zeros(width * hidden)},
    {"blk.0.ffn_down.weight", {hidden, width}, zeros(width * hidden)}};
  std::uint64_t offset = 0;
  for (const auto& t : tensors) {
    b.tensor(t.name, t.dims, 0, offset);
    offset += t.values.size() * 4; offset = (offset + 31) & ~std::uint64_t{31};
  }
  b.align();
  for (const auto& t : tensors) { for (const float x : t.values) b.put(x); b.align(); }
  std::ofstream stream(argv[1], std::ios::binary);
  stream.write(b.data.data(), static_cast<std::streamsize>(b.data.size()));
  return stream ? 0 : 1;
}
