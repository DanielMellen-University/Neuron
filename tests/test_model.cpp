#define NEURON_TEST_MAIN
#include "neuron/model.hpp"
#include "neuron_test.hpp"
#include "gguf_fixture.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace {
fixture::Bytes tiny_model(bool half, bool tied = false, bool qwen = false) {
  fixture::Bytes b;
  b.header((tied ? 11u : 12u) + (qwen ? 3u : 0u), 11);
  const std::string key = qwen ? "qwen2." : "llama.";
  b.text("general.architecture", qwen ? "qwen2" : "llama");
  b.uint(key + "embedding_length", 4); b.uint(key + "feed_forward_length", 4);
  b.uint(key + "block_count", 1); b.uint(key + "attention.head_count", 2);
  b.uint(key + "attention.head_count_kv", 1); b.uint(key + "rope.dimension_count", 2);
  b.uint(key + "context_length", 4);
  b.key(key + "attention.layer_norm_rms_epsilon", 6); b.put(0.75f);
  b.key(key + "rope.freq_base", 6); b.put(10000.0f);
  b.key("tokenizer.ggml.tokens", 9); b.put<std::uint32_t>(8); b.put<std::uint64_t>(3);
  b.string("a"); b.string("b"); b.string("c");
  struct Tensor { std::string name; std::vector<std::uint64_t> dims; std::vector<float> values; };
  const std::vector<float> identity{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
  std::vector<Tensor> tensors{
    {"token_embd.weight", {4,3}, {1,0,0,0, 0,1,0,0, 0,0,1,0}},
    {"output_norm.weight", {4}, {1,1,1,1}},
    {"blk.0.attn_norm.weight", {4}, {1,1,1,1}},
    {"blk.0.attn_q.weight", {4,4}, std::vector<float>(16,0)},
    {"blk.0.attn_k.weight", {4,2}, std::vector<float>(8,0)},
    {"blk.0.attn_v.weight", {4,2}, {1,0,0,0, 0,1,0,0}},
    {"blk.0.attn_output.weight", {4,4}, identity},
    {"blk.0.ffn_norm.weight", {4}, {1,1,1,1}},
    {"blk.0.ffn_gate.weight", {4,4}, std::vector<float>(16,0)},
    {"blk.0.ffn_up.weight", {4,4}, std::vector<float>(16,0)},
    {"blk.0.ffn_down.weight", {4,4}, std::vector<float>(16,0)}};
  if (!tied) tensors.push_back({"output.weight", {4,3}, {1,0,0,0, 0,1,0,0, 0,0,1,0}});
  if (qwen) {
    tensors.push_back({"blk.0.attn_q.bias", {4}, {1,0,0,1}});
    tensors.push_back({"blk.0.attn_k.bias", {2}, {0,1}});
    tensors.push_back({"blk.0.attn_v.bias", {2}, {1,0}});
    tensors[3].values = identity;
    tensors[4].values = {1,0,0,0, 0,1,0,0};
  }
  std::uint64_t offset = 0;
  for (const auto& t : tensors) {
    b.tensor(t.name, t.dims, half ? 1 : 0, offset);
    offset += t.values.size() * (half ? 2u : 4u); offset = (offset + 31) & ~std::uint64_t{31};
  }
  b.align();
  for (const auto& t : tensors) {
    for (float x : t.values) {
      if (half) b.put<std::uint16_t>(x == 0 ? 0 : 0x3c00);
      else b.put(x);
    }
    b.align();
  }
  return b;
}
}
NEURON_TEST(gqa_causal_attention_residual_and_f16_weights) {
  // Queries/keys are zero, so attention averages past values. Both query
  // heads share the one KV head. FFN is zero. Norm epsilon is exactly 0.75.
  for (const bool half : {false, true}) for (const bool tied : {false, true}) {
    fixture::File f(tiny_model(half, tied)); neuron::Model m(f.path);
    auto logits = m.step(0);
    // embedding [1,0,0,0] normalizes to itself. Attention [1,0,1,0]
    // gives residual [2,0,1,0], whose final mean square + epsilon is 2.
    CHECK_LT(std::abs(logits[0] - 2 / std::sqrt(2.0f)), 1e-6f);
    CHECK_EQ(logits[1], 0.0f); CHECK_LT(std::abs(logits[2] - 1 / std::sqrt(2.0f)), 1e-6f);
    logits = m.step(1);
    // At position 1: attention [0.5,0.5,0.5,0.5], residual
    // [0.5,1.5,0.5,0.5], mean square + epsilon = 1.5.
    CHECK_LT(std::abs(logits[0] - 0.5f / std::sqrt(1.5f)), 1e-6f);
    CHECK_LT(std::abs(logits[1] - 1.5f / std::sqrt(1.5f)), 1e-6f);
    CHECK_LT(std::abs(logits[2] - 0.5f / std::sqrt(1.5f)), 1e-6f);
    const std::array<std::int32_t, 2> prompt{0,1};
    const auto full = m.forward(prompt); CHECK(full == logits); CHECK_EQ(m.position(), 2u);
  }
}
NEURON_TEST(context_token_and_shape_errors) {
  fixture::File f(tiny_model(false)); neuron::Model m(f.path);
  CHECK_THROWS_AS(m.step(-1), std::runtime_error); CHECK_EQ(m.position(), 0u);
  CHECK_THROWS_AS(m.step(3), std::runtime_error);
  for (int i = 0; i < 4; ++i) m.step(0, false);
  CHECK_THROWS_AS(m.step(0), std::runtime_error);
  const std::array<std::int32_t, 5> too_long{}; CHECK_THROWS_AS(m.forward(too_long), std::runtime_error);
  CHECK_THROWS_AS(m.forward({}), std::runtime_error);
  m.reset(); CHECK_EQ(m.position(), 0u);
  neuron::Model override(f.path, 8); for (int i = 0; i < 8; ++i) override.step(0, false);
  auto broken = tiny_model(false);
  // Find the metadata dimension value using the fixture's encoded key.
  fixture::Bytes key; key.string("llama.embedding_length");
  const auto at = std::search(broken.data.begin(), broken.data.end(), key.data.begin(), key.data.end());
  REQUIRE(at != broken.data.end());
  broken.patch<std::uint32_t>(static_cast<std::size_t>(at - broken.data.begin()) + key.data.size() + 4, 6);
  fixture::File wrong(broken); CHECK_THROWS_AS(neuron::Model{wrong.path}, std::runtime_error);
}

NEURON_TEST(qwen2_metadata_and_qkv_bias) {
  fixture::File f(tiny_model(false, true, true)); neuron::Model m(f.path);
  CHECK(m.config().rope_neox); CHECK_EQ(m.config().heads, 2u); CHECK_EQ(m.config().kv_heads, 1u);
  const auto logits = m.step(0);
  // Single-position attention ignores Q/K. V = normalized embedding + bias
  // = [2,0], broadcast to both heads. Residual = [3,0,2,0].
  CHECK_LT(std::abs(logits[0] - 1.5f), 1e-6f);
  CHECK_LT(std::abs(logits[2] - 1.0f), 1e-6f);
}
NEURON_TEST(cached_generation_equals_full_recomputation_for_32_tokens) {
  for (const bool qwen : {false, true}) {
    fixture::File f(tiny_model(false, false, qwen));
    neuron::Model cached(f.path, 64), recomputed(f.path, 64);
    std::vector<std::int32_t> sequence{0,1,2,0};
    auto logits = cached.forward(sequence);
    for (int i = 0; i < 32; ++i) {
      const auto full = recomputed.forward(sequence);
      CHECK(logits == full);
      const auto next = static_cast<std::int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
      const auto expected = static_cast<std::int32_t>(std::max_element(full.begin(), full.end()) - full.begin());
      CHECK_EQ(next, expected);
      sequence.push_back(next);
      logits = cached.step(next);
      CHECK_EQ(cached.position(), sequence.size());
    }
  }
}
