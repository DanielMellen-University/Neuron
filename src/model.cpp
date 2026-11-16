#include "neuron/model.hpp"
#include "neuron/kernels.hpp"
#include "neuron/quant.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace neuron {
namespace {
[[noreturn]] void fail(const std::string& message) { throw std::runtime_error("model: " + message); }
std::size_t dimension(const Gguf& file, const std::string& key) {
  const auto n = file.at(key).uint();
  if (n == 0 || n > 1048576) fail("invalid dimension: " + key);
  return static_cast<std::size_t>(n);
}
float positive_float(const Gguf& file, const std::string& key, float fallback) {
  const auto* v = file.find(key);
  const auto result = v ? static_cast<float>(v->number()) : fallback;
  if (!std::isfinite(result) || result <= 0) fail("invalid parameter: " + key);
  return result;
}
}  // namespace
void Model::Weight::multiply(std::span<const float> input, std::span<float> output,
                             std::size_t rows, std::size_t cols) const {
  if (blocks.empty()) scalar::matmul(data, input, output, rows, cols);
  else scalar::matmul_quantized(blocks, type, input, output, rows, cols);
}
void Model::Weight::row(std::size_t index, std::span<float> output) const {
  if (blocks.empty()) std::copy_n(data.begin() + static_cast<std::ptrdiff_t>(index * output.size()), output.size(), output.begin());
  else {
    const auto bytes = output.size() / 32 * quant_block_bytes(type);
    dequantize(blocks.subspan(index * bytes, bytes), type, output);
  }
}
Model::Weight Model::load(const std::string& name, std::initializer_list<std::uint64_t> dimensions) {
  const auto& t = file_.tensor(name);
  if (t.dimensions != std::vector<std::uint64_t>(dimensions)) fail("wrong shape for " + name);
  const auto bytes = file_.tensor_data(t);
  Weight result;
  result.type = t.type;
  if (t.type > 1 && dimensions.size() == 2) result.blocks = bytes;
  else if (t.type == 0 && reinterpret_cast<std::uintptr_t>(bytes.data()) % alignof(float) == 0) {
    result.data = {reinterpret_cast<const float*>(bytes.data()), t.bytes / sizeof(float)};
  } else {
    result.converted = file_.tensor_f32(name);
    result.data = result.converted;
  }
  return result;
}
Model::Model(const std::string& path, std::size_t context_override) : file_(path) {
  const auto architecture = file_.at("general.architecture").string();
  if (architecture != "llama" && architecture != "qwen2") fail("only dense llama and qwen2 models are supported");
  const auto key = architecture + ".";
  if (const auto* scaling = file_.find(key + "rope.scaling.type"))
    if (scaling->string() != "none") fail("RoPE scaling is not supported");
  auto& c = config_;
  c.rope_neox = architecture == "qwen2";
  c.width = dimension(file_, key + "embedding_length");
  c.hidden = dimension(file_, key + "feed_forward_length");
  c.layers = dimension(file_, key + "block_count");
  c.heads = dimension(file_, key + "attention.head_count");
  c.kv_heads = file_.find(key + "attention.head_count_kv") ? dimension(file_, key + "attention.head_count_kv") : c.heads;
  if (c.width % c.heads != 0 || c.heads % c.kv_heads != 0) fail("incompatible attention head counts");
  c.head_size = c.width / c.heads;
  c.rotary_dims = file_.find(key + "rope.dimension_count") ? dimension(file_, key + "rope.dimension_count") : c.head_size;
  if (c.rotary_dims > c.head_size || c.rotary_dims % 2 != 0) fail("invalid RoPE dimension count");
  c.context = dimension(file_, key + "context_length");
  if (context_override > 1048576) fail("context override exceeds resource limit");
  if (context_override != 0) c.context = context_override;
  c.vocabulary = file_.at("tokenizer.ggml.tokens").array().size();
  if (c.vocabulary == 0 || c.vocabulary > static_cast<std::size_t>(INT32_MAX)) fail("invalid vocabulary size");
  c.rms_epsilon = positive_float(file_, key + "attention.layer_norm_rms_epsilon", 1e-5f);
  c.rope_base = positive_float(file_, key + "rope.freq_base", 10000);
  const auto kv_width = c.kv_heads * c.head_size;
  embedding_ = load("token_embd.weight", {c.width, c.vocabulary});
  output_norm_ = load("output_norm.weight", {c.width});
  const bool tied = std::none_of(file_.tensors().begin(), file_.tensors().end(), [](const auto& t) { return t.name == "output.weight"; });
  if (tied) { output_.data = embedding_.data; output_.blocks = embedding_.blocks; output_.type = embedding_.type; }
  else output_ = load("output.weight", {c.width, c.vocabulary});
  for (std::size_t i = 0; i < c.layers; ++i) {
    const auto prefix = "blk." + std::to_string(i) + ".";
    Layer l;
    l.attn_norm = load(prefix + "attn_norm.weight", {c.width});
    l.q = load(prefix + "attn_q.weight", {c.width, c.width});
    l.k = load(prefix + "attn_k.weight", {c.width, kv_width});
    l.v = load(prefix + "attn_v.weight", {c.width, kv_width});
    if (architecture == "qwen2") {
      l.q_bias = load(prefix + "attn_q.bias", {c.width});
      l.k_bias = load(prefix + "attn_k.bias", {kv_width});
      l.v_bias = load(prefix + "attn_v.bias", {kv_width});
    }
    l.attn_output = load(prefix + "attn_output.weight", {c.width, c.width});
    l.ffn_norm = load(prefix + "ffn_norm.weight", {c.width});
    l.gate = load(prefix + "ffn_gate.weight", {c.width, c.hidden});
    l.up = load(prefix + "ffn_up.weight", {c.width, c.hidden});
    l.down = load(prefix + "ffn_down.weight", {c.hidden, c.width});
    layers_.push_back(std::move(l));
  }
}
void Model::reset() {
  position_ = 0;
  for (auto& l : layers_) { l.keys.clear(); l.values.clear(); }
}
std::vector<float> Model::step(std::int32_t token, bool output) {
  const auto& c = config_;
  if (token < 0 || static_cast<std::size_t>(token) >= c.vocabulary) fail("token id out of range");
  if (position_ >= c.context) fail("context limit exceeded (" + std::to_string(c.context) + " tokens)");
  const auto kv_width = c.kv_heads * c.head_size;
  std::vector<float> x(c.width);
  embedding_.row(static_cast<std::size_t>(token), x);
  std::vector<float> normalized(c.width), q(c.width), k(kv_width), v(kv_width), attention(c.width), projected(c.width);
  std::vector<float> gate(c.hidden), up(c.hidden), scores(position_ + 1);
  const float scale = 1.0f / std::sqrt(static_cast<float>(c.head_size));
  for (auto& l : layers_) {
    scalar::rmsnorm(x, l.attn_norm.data, normalized, c.rms_epsilon);
    l.q.multiply(normalized, q, c.width, c.width);
    l.k.multiply(normalized, k, kv_width, c.width);
    l.v.multiply(normalized, v, kv_width, c.width);
    if (!l.q_bias.data.empty()) {
      for (std::size_t d = 0; d < c.width; ++d) q[d] += l.q_bias.data[d];
      for (std::size_t d = 0; d < kv_width; ++d) { k[d] += l.k_bias.data[d]; v[d] += l.v_bias.data[d]; }
    }
    scalar::rope(q, c.head_size, c.rotary_dims, position_, c.rope_base, c.rope_neox);
    scalar::rope(k, c.head_size, c.rotary_dims, position_, c.rope_base, c.rope_neox);
    l.keys.insert(l.keys.end(), k.begin(), k.end()); l.values.insert(l.values.end(), v.begin(), v.end());
    for (std::size_t h = 0; h < c.heads; ++h) {
      const auto group = h / (c.heads / c.kv_heads);
      for (std::size_t t = 0; t <= position_; ++t) {
        double dot = 0;
        for (std::size_t d = 0; d < c.head_size; ++d)
          dot += static_cast<double>(q[h * c.head_size + d]) * l.keys[t * kv_width + group * c.head_size + d];
        scores[t] = static_cast<float>(dot) * scale;
      }
      scalar::softmax(scores);
      for (std::size_t d = 0; d < c.head_size; ++d) {
        double sum = 0;
        for (std::size_t t = 0; t <= position_; ++t)
          sum += static_cast<double>(scores[t]) * l.values[t * kv_width + group * c.head_size + d];
        attention[h * c.head_size + d] = static_cast<float>(sum);
      }
    }
    l.attn_output.multiply(attention, projected, c.width, c.width);
    for (std::size_t d = 0; d < c.width; ++d) x[d] += projected[d];
    scalar::rmsnorm(x, l.ffn_norm.data, normalized, c.rms_epsilon);
    l.gate.multiply(normalized, gate, c.hidden, c.width);
    l.up.multiply(normalized, up, c.hidden, c.width);
    for (std::size_t d = 0; d < c.hidden; ++d) gate[d] = (gate[d] / (1.0f + std::exp(-gate[d]))) * up[d];
    l.down.multiply(gate, projected, c.width, c.hidden);
    for (std::size_t d = 0; d < c.width; ++d) x[d] += projected[d];
  }
  ++position_;
  if (!output) return {};
  scalar::rmsnorm(x, output_norm_.data, normalized, c.rms_epsilon);
  std::vector<float> logits(c.vocabulary);
  output_.multiply(normalized, logits, c.vocabulary, c.width);
  return logits;
}
std::vector<float> Model::forward(std::span<const std::int32_t> tokens) {
  if (tokens.empty()) fail("cannot evaluate an empty token sequence");
  if (tokens.size() > config_.context) fail("prompt exceeds context limit");
  for (auto token : tokens)
    if (token < 0 || static_cast<std::size_t>(token) >= config_.vocabulary) fail("token id out of range");
  reset();
  std::vector<float> result;
  for (std::size_t i = 0; i < tokens.size(); ++i) result = step(tokens[i], i + 1 == tokens.size());
  return result;
}
}  // namespace neuron
