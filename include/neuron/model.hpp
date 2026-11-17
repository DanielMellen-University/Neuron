#pragma once

#include "neuron/gguf.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace neuron {
struct ModelConfig {
  std::size_t width, hidden, layers, heads, kv_heads, head_size, rotary_dims, vocabulary, context;
  float rms_epsilon, rope_base;
  bool rope_neox;
};

// Dense llama or Qwen2 transformer. F32/Q8_0/Q4_0 stay mapped; F16 weights are converted
// once on load. Activations and the single-sequence K/V state use FP32.
class Model {
 public:
  explicit Model(const std::string& path, std::size_t context_override = 0);
  const ModelConfig& config() const { return config_; }
  const Gguf& file() const { return file_; }
  void reset();
  // Appends one token. Returns logits unless output is false (prompt prefix).
  std::vector<float> step(std::int32_t token, bool output = true);
  // Reuses model storage. The view is valid until the next step/forward call.
  std::span<const float> step_view(std::int32_t token, bool output = true);
  // Starts a fresh sequence and returns the final position's logits.
  std::vector<float> forward(std::span<const std::int32_t> tokens);
  std::span<const float> forward_view(std::span<const std::int32_t> tokens);
  std::size_t position() const { return position_; }
 private:
  struct Weight {
    std::vector<float> converted;
    std::span<const float> data;
    std::span<const std::byte> blocks;
    std::uint32_t type = 0;
    void multiply(std::span<const float> input, std::span<float> output,
                  std::size_t rows, std::size_t cols) const;
    void row(std::size_t index, std::span<float> output) const;
  };
  struct Layer {
    Weight attn_norm, q, k, v, q_bias, k_bias, v_bias, attn_output, ffn_norm, gate, up, down;
    std::vector<float> keys, values;
  };
  Weight load(const std::string& name, std::initializer_list<std::uint64_t> dimensions);
  Gguf file_;
  ModelConfig config_{};
  Weight embedding_, output_norm_, output_;
  std::vector<Layer> layers_;
  std::vector<float> x_, normalized_, q_, k_, v_, attention_, projected_, gate_, up_, scores_, logits_;
  std::vector<float> rope_cos_, rope_sin_;
  std::size_t position_ = 0;
};
}  // namespace neuron
