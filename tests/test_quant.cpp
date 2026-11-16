#define NEURON_TEST_MAIN
#include "neuron/quant.hpp"
#include "neuron/gguf.hpp"
#include "neuron/kernels.hpp"
#include "neuron/model.hpp"
#include "neuron_test.hpp"
#include "gguf_fixture.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>

NEURON_TEST(round_trip_bounds_include_half_scale_rounding_and_q4_clipping) {
  std::mt19937 random(42);
  std::uniform_real_distribution<float> uniform(-1, 1);
  for (const auto type : {neuron::q8_0, neuron::q4_0}) {
    double worst_ratio = 0;
    std::size_t blocks = 0;
    for (float amplitude : {0.0f, 1e-8f, 1e-4f, 1.0f, 1000.0f, 100000.0f}) {
      for (int iteration = 0; iteration < 1000; ++iteration) {
        std::array<float, 32> input{}, decoded{};
        for (float& x : input) x = uniform(random) * amplitude;
        // Equal opposite extrema exercise Q4_0's clipped +8 endpoint.
        if (iteration % 2 == 0) { input[0] = amplitude; input[1] = -amplitude; }
        std::vector<std::byte> bytes(neuron::quant_block_bytes(type));
        neuron::quantize_block(input, type, bytes);
        neuron::dequantize(bytes, type, decoded);
        const float maximum = *std::max_element(input.begin(), input.end(), [](float a, float b) { return std::abs(a) < std::abs(b); });
        const float scale = type == neuron::q8_0 ? std::abs(maximum) / 127 : maximum / -8;
        std::uint16_t bits; std::memcpy(&bits, bytes.data(), 2);
        const double rounding = std::abs(static_cast<double>(scale) - neuron::f16_to_f32(bits));
        const double bound = (type == neuron::q8_0 ? 0.5 : 1.0) * std::abs(scale) +
                             (type == neuron::q8_0 ? 127 : 8) * rounding;
        for (std::size_t j = 0; j < 32; ++j) {
          const double error = std::abs(static_cast<double>(input[j]) - decoded[j]);
          const double tolerance = 2e-7 * amplitude;
          CHECK(error <= bound + tolerance);
          if (bound > 0) worst_ratio = std::max(worst_ratio, error / bound);
        }
        ++blocks;
      }
    }
    std::printf("type=%u blocks=%zu maximum_error/bound=%.9g\n", type, blocks, worst_ratio);
  }
}
NEURON_TEST(quantization_rejects_bad_shapes_and_nonfinite_input) {
  std::array<float, 32> x{}; std::array<std::byte, 34> bytes{};
  CHECK_THROWS_AS(neuron::quant_block_bytes(99), std::invalid_argument);
  CHECK_THROWS_AS(neuron::quantize_block(std::span(x).first(31), neuron::q8_0, bytes), std::invalid_argument);
  CHECK_THROWS_AS(neuron::dequantize(std::span(bytes).first(33), neuron::q8_0, x), std::invalid_argument);
  x[0] = std::numeric_limits<float>::infinity();
  CHECK_THROWS_AS(neuron::quantize_block(x, neuron::q8_0, bytes), std::invalid_argument);
  x[0] = std::numeric_limits<float>::max();
  CHECK_THROWS_AS(neuron::quantize_block(x, neuron::q8_0, bytes), std::invalid_argument);
}
NEURON_TEST(gguf_block_layout_rows_and_truncation) {
  for (const auto type : {neuron::q8_0, neuron::q4_0}) {
    fixture::Bytes b; b.header(1, 0); b.tensor("blocks", {64, 2}, type, 0); b.align();
    const auto header = b.data.size();
    // Four blocks with different exact scales. Payload deliberately bypasses
    // our encoder, checking GGUF nibble order and signed int8 independently.
    for (int block = 0; block < 4; ++block) {
      b.put<std::uint16_t>(static_cast<std::uint16_t>(0x3c00 + block * 0x400));
      if (type == neuron::q8_0) for (int j = 0; j < 32; ++j) b.put<std::int8_t>(static_cast<std::int8_t>(j - 16));
      else for (int j = 0; j < 16; ++j) b.put<std::uint8_t>(static_cast<std::uint8_t>(j | ((15 - j) << 4)));
    }
    fixture::File file(b); neuron::Gguf g(file.path);
    CHECK_EQ(g.tensor("blocks").bytes, 4 * neuron::quant_block_bytes(type));
    const auto x = g.tensor_f32("blocks");
    CHECK_EQ(x.size(), 128u);
    for (std::size_t j = 0; j < x.size(); ++j) {
      const int code = type == neuron::q8_0 ? static_cast<int>(j % 32) - 16 :
                       (j % 32 < 16 ? static_cast<int>(j % 16) - 8 : 7 - static_cast<int>(j % 16));
      CHECK_EQ(x[j], std::ldexp(static_cast<float>(code), static_cast<int>(j / 32)));
    }
    for (std::size_t n = header; n < b.data.size(); ++n) {
      auto truncated = b; truncated.data.resize(n); fixture::File f(truncated);
      CHECK_THROWS_AS(neuron::Gguf{f.path}, std::runtime_error);
    }
    fixture::Bytes bad; bad.header(1, 0); bad.tensor("bad-row", {16, 2}, type, 0); bad.align();
    bad.data.resize(bad.data.size() + 34); fixture::File f(bad);
    CHECK_THROWS_AS(neuron::Gguf{f.path}, std::runtime_error);
  }
}
NEURON_TEST(quantized_matmul_matches_independent_dequantization_batched) {
  for (const auto type : {neuron::q8_0, neuron::q4_0}) {
    std::vector<std::byte> w(6 * neuron::quant_block_bytes(type));
    std::array<float, 192> decoded{}; std::array<float, 128> x{};
    for (std::size_t j = 0; j < x.size(); ++j) x[j] = std::sin(static_cast<float>(j));
    for (std::size_t b = 0; b < 6; ++b) {
      std::array<float, 32> values{};
      for (std::size_t j = 0; j < 32; ++j) values[j] = std::cos(static_cast<float>(32 * b + j));
      neuron::quantize_block(values, type, std::span(w).subspan(b * neuron::quant_block_bytes(type), neuron::quant_block_bytes(type)));
    }
    neuron::dequantize(w, type, decoded);
    std::array<float, 6> expected{}, actual{};
    neuron::scalar::matmul(decoded, x, expected, 3, 64, 2);
    neuron::scalar::matmul_quantized(w, type, x, actual, 3, 64, 2);
    for (std::size_t j = 0; j < actual.size(); ++j) CHECK_LT(std::abs(actual[j] - expected[j]), 2e-6f);
    CHECK_THROWS_AS(neuron::scalar::matmul_quantized(w, type, x, actual, 3, 63, 2), std::invalid_argument);
  }
}
namespace {
fixture::Bytes model_fixture(std::uint32_t type, bool dense, bool tied) {
  fixture::Bytes b; b.header(tied ? 14u : 15u, 9);
  b.text("general.architecture", "qwen2");
  b.uint("qwen2.embedding_length", 32); b.uint("qwen2.feed_forward_length", 64);
  b.uint("qwen2.block_count", 1); b.uint("qwen2.attention.head_count", 4);
  b.uint("qwen2.attention.head_count_kv", 2); b.uint("qwen2.context_length", 64);
  b.key("qwen2.attention.layer_norm_rms_epsilon", 6); b.put(1e-5f);
  b.key("tokenizer.ggml.tokens", 9); b.put<std::uint32_t>(8); b.put<std::uint64_t>(3);
  b.string("a"); b.string("b"); b.string("c");
  struct Tensor { std::string name; std::vector<std::uint64_t> dims; };
  std::vector<Tensor> tensors{
    {"token_embd.weight", {32,3}}, {"output_norm.weight", {32}},
    {"blk.0.attn_norm.weight", {32}}, {"blk.0.attn_q.weight", {32,32}},
    {"blk.0.attn_k.weight", {32,16}}, {"blk.0.attn_v.weight", {32,16}},
    {"blk.0.attn_q.bias", {32}}, {"blk.0.attn_k.bias", {16}}, {"blk.0.attn_v.bias", {16}},
    {"blk.0.attn_output.weight", {32,32}}, {"blk.0.ffn_norm.weight", {32}},
    {"blk.0.ffn_gate.weight", {32,64}}, {"blk.0.ffn_up.weight", {32,64}},
    {"blk.0.ffn_down.weight", {64,32}}};
  if (!tied) tensors.push_back({"output.weight", {32,3}});
  std::vector<std::vector<std::byte>> payloads;
  std::uint64_t offset = 0;
  for (const auto& t : tensors) {
    std::size_t elements = 1; for (auto d : t.dims) elements *= d;
    std::vector<float> values(elements);
    for (std::size_t j = 0; j < elements; ++j) values[j] = t.dims.size() == 1 ? 1.0f : 0.1f * std::sin(static_cast<float>(j + payloads.size()));
    std::vector<std::byte> bytes;
    if (t.dims.size() == 2) {
      bytes.resize(elements / 32 * neuron::quant_block_bytes(type));
      for (std::size_t j = 0; j < elements / 32; ++j)
        neuron::quantize_block(std::span(values).subspan(j * 32, 32), type,
                              std::span(bytes).subspan(j * neuron::quant_block_bytes(type), neuron::quant_block_bytes(type)));
      if (dense) neuron::dequantize(bytes, type, values);
    }
    if (dense || t.dims.size() == 1) {
      bytes.resize(elements * 4); std::memcpy(bytes.data(), values.data(), bytes.size());
    }
    b.tensor(t.name, t.dims, dense || t.dims.size() == 1 ? 0 : type, offset);
    offset = (offset + bytes.size() + 31) & ~std::uint64_t{31}; payloads.push_back(std::move(bytes));
  }
  b.align();
  for (const auto& bytes : payloads) {
    for (auto byte : bytes) b.put(byte);
    b.align();
  }
  return b;
}
}
NEURON_TEST(mapped_quantized_transformer_and_tied_embeddings_match_dense) {
  for (const auto type : {neuron::q8_0, neuron::q4_0}) for (bool tied : {false, true}) {
    fixture::File q(model_fixture(type, false, tied)), f(model_fixture(type, true, tied));
    neuron::Model quantized(q.path), dense(f.path);
    for (int i = 0; i < 32; ++i) {
      const auto a = quantized.step(i % 3), expected = dense.step(i % 3);
      for (std::size_t j = 0; j < a.size(); ++j) CHECK_LT(std::abs(a[j] - expected[j]), 1e-5f);
    }
  }
}
