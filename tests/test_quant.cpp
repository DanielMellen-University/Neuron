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
  const auto width = static_cast<std::uint64_t>(neuron::quant_block_elements(type));
  fixture::Bytes b; b.header(tied ? 14u : 15u, 9);
  b.text("general.architecture", "qwen2");
  b.uint("qwen2.embedding_length", static_cast<std::uint32_t>(width)); b.uint("qwen2.feed_forward_length", static_cast<std::uint32_t>(2 * width));
  b.uint("qwen2.block_count", 1); b.uint("qwen2.attention.head_count", 4);
  b.uint("qwen2.attention.head_count_kv", 2); b.uint("qwen2.context_length", 64);
  b.key("qwen2.attention.layer_norm_rms_epsilon", 6); b.put(1e-5f);
  b.key("tokenizer.ggml.tokens", 9); b.put<std::uint32_t>(8); b.put<std::uint64_t>(3);
  b.string("a"); b.string("b"); b.string("c");
  struct Tensor { std::string name; std::vector<std::uint64_t> dims; };
  std::vector<Tensor> tensors{
    {"token_embd.weight", {width,3}}, {"output_norm.weight", {width}},
    {"blk.0.attn_norm.weight", {width}}, {"blk.0.attn_q.weight", {width,width}},
    {"blk.0.attn_k.weight", {width,width/2}}, {"blk.0.attn_v.weight", {width,width/2}},
    {"blk.0.attn_q.bias", {width}}, {"blk.0.attn_k.bias", {width/2}}, {"blk.0.attn_v.bias", {width/2}},
    {"blk.0.attn_output.weight", {width,width}}, {"blk.0.ffn_norm.weight", {width}},
    {"blk.0.ffn_gate.weight", {width,2*width}}, {"blk.0.ffn_up.weight", {width,2*width}},
    {"blk.0.ffn_down.weight", {2*width,width}}};
  if (!tied) tensors.push_back({"output.weight", {width,3}});
  std::vector<std::vector<std::byte>> payloads;
  std::uint64_t offset = 0;
  for (const auto& t : tensors) {
    std::size_t elements = 1; for (auto d : t.dims) elements *= d;
    std::vector<float> values(elements);
    for (std::size_t j = 0; j < elements; ++j) values[j] = t.dims.size() == 1 ? 1.0f : 0.1f * std::sin(static_cast<float>(j + payloads.size()));
    std::vector<std::byte> bytes;
    if (t.dims.size() == 2) {
      const auto block = neuron::quant_block_elements(type);
      bytes.resize(elements / block * neuron::quant_block_bytes(type));
      for (std::size_t j = 0; j < elements / block; ++j)
        neuron::quantize_block(std::span(values).subspan(j * block, block), type,
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
  for (const auto type : {neuron::q8_0, neuron::q4_0, neuron::q4_k, neuron::q6_k}) for (bool tied : {false, true}) {
    fixture::File q(model_fixture(type, false, tied)), f(model_fixture(type, true, tied));
    neuron::Model quantized(q.path), dense(f.path);
    for (int i = 0; i < 32; ++i) {
      const auto a = quantized.step(i % 3), expected = dense.step(i % 3);
      for (std::size_t j = 0; j < a.size(); ++j) CHECK_LT(std::abs(a[j] - expected[j]), 1e-5f);
    }
  }
}

NEURON_TEST(k_round_trip_bounds_include_both_scale_levels) {
  std::mt19937 random(43);
  std::uniform_real_distribution<float> uniform(-1, 1);
  for (auto type : {neuron::q4_k, neuron::q6_k}) {
    double worst = 0;
    for (float amplitude : {0.0f, 1e-8f, 1e-4f, 1.0f, 1000.0f, 100000.0f}) {
      for (int iteration = 0; iteration < 1000; ++iteration) {
        std::array<float, 256> input{}, decoded{};
        for (auto& x : input) x = uniform(random) * amplitude;
        if (iteration % 2 == 0) { input[0] = amplitude; input[1] = -amplitude; }
        // Include positive-only, negative-only and zero subblocks.
        if (iteration % 3 == 0) for (std::size_t i = 32; i < 64; ++i) input[i] = std::abs(input[i]);
        if (iteration % 3 == 1) for (std::size_t i = 64; i < 96; ++i) input[i] = -std::abs(input[i]);
        if (iteration % 5 == 0) std::fill_n(input.begin() + 96, 32, 0.0f);
        std::vector<std::byte> bytes(neuron::quant_block_bytes(type));
        neuron::quantize_block(input, type, bytes); neuron::dequantize(bytes, type, decoded);
        const std::size_t width = type == neuron::q4_k ? 32 : 16, groups = 256 / width;
        std::array<double, 16> scales{}, minima{};
        double max_scale = 0, max_min = 0;
        for (std::size_t g = 0; g < groups; ++g) {
          const auto values = std::span(input).subspan(g * width, width);
          if (type == neuron::q4_k) {
            const double lo = std::min(0.0f, *std::min_element(values.begin(), values.end()));
            const double hi = std::max(0.0f, *std::max_element(values.begin(), values.end()));
            scales[g] = (hi - lo) / 15; minima[g] = -lo;
          } else {
            const double maximum = *std::max_element(values.begin(), values.end(),
                [](float a, float b) { return std::abs(a) < std::abs(b); });
            scales[g] = maximum / -32;
          }
          max_scale = std::max(max_scale, std::abs(scales[g])); max_min = std::max(max_min, minima[g]);
        }
        const double levels = type == neuron::q4_k ? 63 : 127;
        const double d = max_scale / levels, dm = max_min / 63;
        std::uint16_t h, hm = 0;
        std::memcpy(&h, bytes.data() + (type == neuron::q4_k ? 0 : 208), 2);
        if (type == neuron::q4_k) std::memcpy(&hm, bytes.data() + 2, 2);
        const double scale_error = d / 2 + levels * std::abs(d - neuron::f16_to_f32(h));
        const double min_error = dm / 2 + 63 * std::abs(dm - neuron::f16_to_f32(hm));
        for (std::size_t i = 0; i < 256; ++i) {
          const double bound = type == neuron::q4_k ? scales[i / width] / 2 + 15 * scale_error + min_error :
              std::abs(scales[i / width]) + 32 * scale_error;
          const double error = std::abs(static_cast<double>(input[i]) - decoded[i]);
          CHECK(error <= bound + 4e-7 * amplitude);
          if (bound > 0) worst = std::max(worst, error / bound);
        }
      }
    }
    std::printf("type=%u blocks=6000 maximum_error/bound=%.9g\n", type, worst);
  }
}
NEURON_TEST(k_gguf_independent_layout_and_every_payload_truncation) {
  for (auto type : {neuron::q4_k, neuron::q6_k}) {
    const auto size = neuron::quant_block_bytes(type);
    fixture::Bytes b; b.header(1, 0); b.tensor("blocks", {512, 2}, type, 0); b.align();
    const auto header = b.data.size();
    std::array<float, 1024> expected{};
    for (std::size_t block = 0; block < 4; ++block) {
      std::vector<std::byte> bytes(size);
      const std::uint16_t half = static_cast<std::uint16_t>(0x3c00 + block * 0x400);
      if (type == neuron::q4_k) {
        std::memcpy(bytes.data(), &half, 2); std::memcpy(bytes.data() + 2, &half, 2);
        // Every scale bit, offset bit and nibble position varies. Pack by
        // bit position rather than invoking Neuron's reference encoder.
        for (unsigned g = 0; g < 8; ++g) {
          const unsigned sc = 7 * g + 3, mn = 63 - 8 * g;
          for (unsigned bit = 0; bit < 6; ++bit) {
            const unsigned si = g < 4 ? 4 + g : bit < 4 ? 8 + g : g;
            const unsigned mi = g < 4 ? 8 + g : bit < 4 ? 8 + g : 4 + g;
            const unsigned sb = g < 4 || bit < 4 ? bit : bit + 2;
            const unsigned mb = g < 4 ? bit : bit < 4 ? bit + 4 : bit + 2;
            bytes[si] |= static_cast<std::byte>(((sc >> bit) & 1u) << sb);
            bytes[mi] |= static_cast<std::byte>(((mn >> bit) & 1u) << mb);
          }
          for (unsigned j = 0; j < 32; ++j) {
            const unsigned q = (j + 3 * g) % 16;
            bytes[16 + (g / 2) * 32 + j] |= static_cast<std::byte>(q << (g % 2 * 4));
            expected[block * 256 + g * 32 + j] = std::ldexp(static_cast<float>(static_cast<int>(sc * q) - static_cast<int>(mn)), static_cast<int>(block));
          }
        }
      } else {
        std::memcpy(bytes.data() + 208, &half, 2);
        for (unsigned i = 0; i < 256; ++i) {
          const unsigned q = (13 * i + i / 16) % 64, g = i / 16;
          const auto sc = static_cast<std::int8_t>(static_cast<int>(g) * 17 - 128);
          std::memcpy(bytes.data() + 192 + g, &sc, 1);
          const unsigned half_index = i % 128, quarter = half_index / 32, lane = half_index % 32;
          bytes[(i / 128) * 64 + (quarter % 2) * 32 + lane] |= static_cast<std::byte>((q & 15u) << (quarter / 2 * 4));
          bytes[128 + i / 128 * 32 + lane] |= static_cast<std::byte>((q >> 4) << (quarter * 2));
          expected[block * 256 + i] = std::ldexp(static_cast<float>(static_cast<int>(sc) * (static_cast<int>(q) - 32)), static_cast<int>(block));
        }
      }
      for (auto v : bytes) b.put(v);
    }
    fixture::File file(b); neuron::Gguf gguf(file.path);
    CHECK_EQ(gguf.tensor("blocks").bytes, 4 * size);
    const auto actual = gguf.tensor_f32("blocks");
    for (std::size_t i = 0; i < actual.size(); ++i) CHECK_EQ(actual[i], expected[i]);
    std::array<float, 1024> x{}; for (std::size_t i = 0; i < x.size(); ++i) x[i] = std::sin(static_cast<float>(i));
    std::array<float, 4> a{}, e{};
    neuron::scalar::matmul(expected, x, e, 2, 512, 2);
    neuron::scalar::matmul_quantized(gguf.tensor_data(gguf.tensor("blocks")), type, x, a, 2, 512, 2);
    for (std::size_t i = 0; i < a.size(); ++i) CHECK_EQ(a[i], e[i]);
    for (std::size_t n = header; n < b.data.size(); ++n) {
      auto truncated = b; truncated.data.resize(n); fixture::File f(truncated);
      CHECK_THROWS_AS(neuron::Gguf{f.path}, std::runtime_error);
    }
    fixture::Bytes bad; bad.header(1, 0); bad.tensor("bad", {128, 2}, type, 0); bad.align();
    bad.data.resize(bad.data.size() + size); fixture::File f(bad);
    CHECK_THROWS_AS(neuron::Gguf{f.path}, std::runtime_error);
    std::array<float, 256> values{}; std::vector<std::byte> bytes(size);
    CHECK_THROWS_AS(neuron::dequantize(std::span(bytes).first(size - 1), type, values), std::invalid_argument);
    CHECK_THROWS_AS(neuron::quantize_block(std::span(values).first(32), type, bytes), std::invalid_argument);
    CHECK_THROWS_AS(neuron::dot_k_block(bytes, type, std::span(values).first(255)), std::invalid_argument);
    values[0] = std::numeric_limits<float>::infinity();
    CHECK_THROWS_AS(neuron::quantize_block(values, type, bytes), std::invalid_argument);
  }
}
