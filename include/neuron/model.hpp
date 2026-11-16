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
};

// Dense llama transformer. F32 weights stay mapped; F16 weights are converted
// once on load. Activations and the single-sequence K/V state use FP32.
class Model {
 public:
  explicit Model(const std::string& path, std::size_t context_override = 0);
  const ModelConfig& config() const { return config_; }
  const Gguf& file() const { return file_; }
  void reset();
  // Appends one token. Returns logits unless output is false (prompt prefix).
  std::vector<float> step(std::int32_t token, bool output = true);
  // Starts a fresh sequence and returns the final position's logits.
  std::vector<float> forward(std::span<const std::int32_t> tokens);
  std::size_t position() const { return position_; }
 private:
  struct Weight {
    std::vector<float> converted;
    std::span<const float> data;
  };
  struct Layer {
    Weight attn_norm, q, k, v, attn_output, ffn_norm, gate, up, down;
    std::vector<float> keys, values;
  };
  Weight load(const std::string& name, std::initializer_list<std::uint64_t> dimensions);
  Gguf file_;
  ModelConfig config_{};
  Weight embedding_, output_norm_, output_;
  std::vector<Layer> layers_;
  std::size_t position_ = 0;
};
}  // namespace neuron
