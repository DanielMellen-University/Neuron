// External numerical diagnostic. Linked with llama.cpp only for local checks.
#include "neuron/model.hpp"
#include "neuron/kernels.hpp"
#include "neuron/quant.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-quants.h"
#include "llama.h"
#include "json.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace r2 {
std::filesystem::path directory;
std::size_t position = 0;
void save(const std::string& name, std::span<const float> values) {
  std::ofstream out(directory / (name + ".f32"), std::ios::binary);
  out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size_bytes()));
  if (!out) throw std::runtime_error("trace write failed");
}
std::vector<float> read(const std::string& name) {
  const auto path = directory / (name + ".f32");
  std::ifstream in(path, std::ios::binary);
  if (!in || std::filesystem::file_size(path) % 4 != 0) throw std::runtime_error("invalid trace input");
  std::vector<float> result(std::filesystem::file_size(path) / 4);
  in.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(result.size() * 4));
  if (!in) throw std::runtime_error("trace read failed");
  return result;
}
struct Callback { bool active = false; };
void tensor(ggml_tensor* t, const std::string& name) {
  if (t->type != GGML_TYPE_F32) throw std::runtime_error("trace requires F32 tensors");
  std::vector<std::byte> raw(ggml_nbytes(t));
  ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
  std::vector<float> values;
  for (std::int64_t i3 = 0; i3 < t->ne[3]; ++i3)
    for (std::int64_t i2 = 0; i2 < t->ne[2]; ++i2)
      for (std::int64_t i1 = 0; i1 < t->ne[1]; ++i1)
        for (std::int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
          float v;
          std::memcpy(&v, raw.data() + static_cast<std::size_t>(i0) * t->nb[0] + static_cast<std::size_t>(i1) * t->nb[1] +
                      static_cast<std::size_t>(i2) * t->nb[2] + static_cast<std::size_t>(i3) * t->nb[3], 4);
          values.push_back(v);
        }
  save(name, values);
  std::ofstream meta(directory / (name + ".json"));
  meta << nlohmann::json{{"shape", {t->ne[0], t->ne[1], t->ne[2], t->ne[3]}}, {"op", ggml_op_name(t->op)}}.dump();
}
bool callback(ggml_tensor* t, bool ask, void* user) {
  if (!static_cast<Callback*>(user)->active || t->type != GGML_TYPE_F32) return false;
  std::string name = ggml_get_name(t);
  if (t->op == GGML_OP_MUL_MAT) {
    const std::string weight = ggml_get_name(t->src[0]);
    if (weight.starts_with("blk.") && weight.ends_with(".attn_output.weight"))
      name = "attn_out-" + weight.substr(4, weight.find('.', 4) - 4);
  }
  const auto dash = name.find_last_of('-');
  const auto op = dash == std::string::npos ? name : name.substr(0, dash);
  const std::vector<std::string> wanted{"embd", "attn_norm", "Qcur", "Kcur", "Vcur", "kq", "kq_soft_max",
    "kqv", "kqv_out", "ffn_inp", "ffn_norm", "ffn_gate", "ffn_up", "ffn_swiglu", "ffn_out", "l_out",
    "attn_out", "result_norm", "result_output"};
  if (std::find(wanted.begin(), wanted.end(), op) == wanted.end()) return false;
  if (ask) return true;
  tensor(t, name);
  if ((op == "Qcur" || op == "Kcur") && t->op == GGML_OP_ROPE)
    tensor(t->src[0], op.substr(0, 1) + "pre" + name.substr(dash));
  if (op == "kq") tensor(t->src[0], "Kcache" + name.substr(dash));
  if (op == "kqv") tensor(t->src[0], "Vcache" + name.substr(dash));
  return true;
}
double error(std::span<const float> a, std::span<const float> b) {
  if (a.size() != b.size()) throw std::runtime_error("operation shape differs");
  double worst = 0;
  for (std::size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::abs(static_cast<double>(a[i]) - b[i]));
  return worst;
}
nlohmann::json operations(const neuron::Gguf& file, std::size_t layers) {
  nlohmann::json rows = nlohmann::json::array();
  const auto width = static_cast<std::size_t>(file.at("qwen2.embedding_length").uint());
  const auto heads = static_cast<std::size_t>(file.at("qwen2.attention.head_count").uint());
  const auto head_size = width / heads;
  const auto kv_heads = static_cast<std::size_t>(file.at("qwen2.attention.head_count_kv").uint());
  const auto epsilon = static_cast<float>(file.at("qwen2.attention.layer_norm_rms_epsilon").number());
  const auto base = static_cast<float>(file.at("qwen2.rope.freq_base").number());
  for (std::size_t layer = 0; layer < layers; ++layer) {
    const auto suffix = "-" + std::to_string(layer), prefix = "blk." + std::to_string(layer) + ".";
    auto norm = [&](const std::string& name, const std::string& input) {
      const auto x = read(input), w = file.tensor_f32(prefix + name + ".weight");
      std::vector<float> y(x.size());
      neuron::scalar::rmsnorm(x, w, y, epsilon);
      rows.push_back({{"op", name + suffix}, {"max_error_on_shared_input", error(y, read(name + suffix))}});
    };
    norm("attn_norm", layer == 0 ? "embd" : "l_out-" + std::to_string(layer - 1));
    norm("ffn_norm", "ffn_inp" + suffix);
    auto gate = read("ffn_gate" + suffix);
    const auto up = read("ffn_up" + suffix);
    if (gate.size() != up.size()) throw std::runtime_error("invalid SwiGLU shape");
    for (std::size_t i = 0; i < gate.size(); ++i)
      gate[i] = (gate[i] / (1.0f + std::exp(-gate[i]))) * up[i];
    rows.push_back({{"op", "swiglu" + suffix}, {"max_error_on_shared_input", error(gate, read("ffn_swiglu" + suffix))}});
    for (const auto& [input, projection, output] : {
        std::array<std::string, 3>{layer == 0 ? "embd" : "l_out-" + std::to_string(layer - 1), "attn_out" + suffix, "ffn_inp" + suffix},
        std::array<std::string, 3>{"ffn_inp" + suffix, "ffn_out" + suffix, "l_out" + suffix}}) {
      auto x = read(input);
      const auto y = read(projection);
      if (x.size() != y.size()) throw std::runtime_error("invalid residual shape");
      for (std::size_t i = 0; i < x.size(); ++i) x[i] += y[i];
      rows.push_back({{"op", "residual_" + output}, {"max_error_on_shared_input", error(x, read(output))}});
    }
    for (const auto& letter : {std::string("Q"), std::string("K")}) {
      auto x = read(letter + "pre" + suffix);
      neuron::scalar::rope(x, head_size, head_size, position, base, true);
      rows.push_back({{"op", letter + "rope" + suffix}, {"max_error_on_shared_input", error(x, read(letter + "cur" + suffix))}});
    }
    auto raw = read("kq" + suffix), probs = read("kq_soft_max" + suffix);
    if (raw.size() != probs.size() || raw.size() % heads != 0) throw std::runtime_error("invalid attention shape");
    const auto stride = raw.size() / heads;
    const auto keys = read("Kcache" + suffix), vals = read("Vcache" + suffix), query = read("Qcur" + suffix);
    const bool external = std::filesystem::exists(directory / ("Kcache" + suffix + ".json"));
    const auto cache_stride = external ? keys.size() / (kv_heads * head_size) : position + 1;
    const auto attended = read("kqv_out" + suffix);
    double score_error = 0, value_error = 0;
    for (std::size_t head = 0; head < heads; ++head) {
      const auto group = head / (heads / kv_heads);
      for (std::size_t t = 0; t <= position; ++t) {
        double sum = 0;
        for (std::size_t d = 0; d < head_size; ++d) {
          const auto index = external ? (group * cache_stride + t) * head_size + d : (t * kv_heads + group) * head_size + d;
          sum += static_cast<double>(query[head * head_size + d]) * keys[index];
        }
        score_error = std::max(score_error, std::abs(static_cast<float>(sum) - static_cast<double>(raw[head * stride + t])));
      }
      for (std::size_t d = 0; d < head_size; ++d) {
        double sum = 0;
        for (std::size_t t = 0; t <= position; ++t) {
          const auto index = external ? (group * head_size + d) * cache_stride + t : (t * kv_heads + group) * head_size + d;
          sum += static_cast<double>(probs[head * stride + t]) * vals[index];
        }
        value_error = std::max(value_error, std::abs(static_cast<float>(sum) - static_cast<double>(attended[head * head_size + d])));
      }
    }
    rows.push_back({{"op", "attention_scores" + suffix}, {"max_error_on_shared_input", score_error}});
    rows.push_back({{"op", "attention_values" + suffix}, {"max_error_on_shared_input", value_error}});
    double softmax_error = 0;
    for (std::size_t head = 0; head < heads; ++head) {
      auto row = std::span(raw).subspan(head * stride, position + 1);
      for (auto& value : row) value *= 1.0f / std::sqrt(static_cast<float>(head_size));
      neuron::scalar::softmax(row);
      softmax_error = std::max(softmax_error, error(row, std::span(probs).subspan(head * stride, position + 1)));
    }
    rows.push_back({{"op", "softmax" + suffix}, {"max_error_on_shared_input", softmax_error}});
  }
  const auto x = read("l_out-" + std::to_string(layers - 1)), w = file.tensor_f32("output_norm.weight");
  std::vector<float> y(x.size());
  neuron::scalar::rmsnorm(x, w, y, epsilon);
  rows.push_back({{"op", "result_norm"}, {"max_error_on_shared_input", error(y, read("result_norm"))}});
  return rows;
}
// Compare both CPU products on identical saved Neuron activations. Decode a
// whole row independently with GGML before computing the double reference.
nlohmann::json products(const neuron::Gguf& file, std::size_t layers) {
  nlohmann::json results = nlohmann::json::array();
  for (std::size_t layer = 0; layer <= layers; ++layer) {
    const auto suffix = "-" + std::to_string(layer);
    const auto prefix = "blk." + std::to_string(layer) + ".";
    std::vector<std::pair<std::string, std::string>> ops = layer == layers ?
      std::vector<std::pair<std::string, std::string>>{{"token_embd.weight", "result_norm"}} :
      std::vector<std::pair<std::string, std::string>>{{prefix + "attn_q.weight", "attn_norm" + suffix},
        {prefix + "attn_k.weight", "attn_norm" + suffix}, {prefix + "attn_v.weight", "attn_norm" + suffix},
        {prefix + "attn_output.weight", "kqv_out" + suffix}, {prefix + "ffn_gate.weight", "ffn_norm" + suffix},
        {prefix + "ffn_up.weight", "ffn_norm" + suffix}, {prefix + "ffn_down.weight", "ffn_swiglu" + suffix}};
    for (const auto& [weight, input] : ops) {
      const auto& t = file.tensor(weight);
      const auto x = read(input);
      const auto n = static_cast<std::size_t>(t.dimensions[0]);
      const auto rows = static_cast<std::size_t>(t.dimensions[1]);
      if (x.size() != n) throw std::runtime_error("product input shape differs");
      const auto type = static_cast<ggml_type>(t.type);
      const auto* cpu = ggml_get_type_traits_cpu(type);
      const auto* traits = ggml_get_type_traits(type);
      const auto stride = ggml_row_size(type, static_cast<std::int64_t>(n));
      const auto raw = file.tensor_data(t);
      const auto act_type = cpu->vec_dot_type;
      std::vector<std::byte> packed(ggml_row_size(act_type, static_cast<std::int64_t>(n)));
      if (act_type == GGML_TYPE_F32) std::memcpy(packed.data(), x.data(), n * 4);
      else ggml_get_type_traits_cpu(act_type)->from_float(x.data(), packed.data(), static_cast<std::int64_t>(n));
      std::vector<float> unpacked(n);
      if (act_type == GGML_TYPE_F32) unpacked = x;
      else if (act_type == GGML_TYPE_Q8_K)
        dequantize_row_q8_K(reinterpret_cast<const block_q8_K*>(packed.data()), unpacked.data(), static_cast<std::int64_t>(n));
      else ggml_get_type_traits(act_type)->to_float(packed.data(), unpacked.data(), static_cast<std::int64_t>(n));
      double activation_error = 0;
      for (std::size_t j = 0; j < n; ++j) activation_error = std::max(activation_error, std::abs(static_cast<double>(x[j]) - unpacked[j]));
      double weight_error = 0, neuron_error = 0, llama_error = 0, llama_quantized_error = 0;
      const std::vector<std::size_t> sample{0, rows / 4, rows / 2, rows * 3 / 4, rows - 1};
      for (auto row : sample) {
        const auto bytes = raw.subspan(row * stride, stride);
        std::vector<float> decoded(n), own(n);
        if (type == GGML_TYPE_F32) std::memcpy(decoded.data(), bytes.data(), n * 4);
        else traits->to_float(bytes.data(), decoded.data(), static_cast<std::int64_t>(n));
        if (t.type > 1) neuron::dequantize(bytes, t.type, own);
        else if (t.type == 1) for (std::size_t j = 0; j < n; ++j) {
          std::uint16_t h; std::memcpy(&h, bytes.data() + j * 2, 2); own[j] = neuron::f16_to_f32(h);
        } else own = decoded;
        double reference = 0, quantized_reference = 0;
        for (std::size_t j = 0; j < n; ++j) {
          weight_error = std::max(weight_error, std::abs(static_cast<double>(decoded[j]) - own[j]));
          reference += static_cast<double>(decoded[j]) * x[j];
          quantized_reference += static_cast<double>(decoded[j]) * unpacked[j];
        }
        float actual = 0, llama = 0;
        if (t.type > 1) neuron::kernels::matmul_quantized(bytes, t.type, x, {&actual, 1}, 1, n);
        else neuron::kernels::matmul(own, x, {&actual, 1}, 1, n);
        cpu->vec_dot(static_cast<int>(n), &llama, 0, bytes.data(), 0, packed.data(), 0, 1);
        neuron_error = std::max(neuron_error, std::abs(actual - reference));
        llama_error = std::max(llama_error, std::abs(llama - reference));
        llama_quantized_error = std::max(llama_quantized_error, std::abs(llama - quantized_reference));
      }
      results.push_back({{"weight", weight}, {"rows", sample}, {"activation_type", ggml_type_name(act_type)},
        {"activation_block", ggml_blck_size(act_type)}, {"activation_max_error", activation_error},
        {"weight_max_error", weight_error}, {"neuron_dot_max_error", neuron_error},
        {"llama_dot_max_error", llama_error}, {"llama_dot_vs_quantized_input_max_error", llama_quantized_error}});
    }
  }
  return results;
}
}  // namespace r2

// This is an instrumented copy generated by trace_r2.py. The production library
// and executable are never modified or linked against llama.cpp.
#include "r2_model.cpp"

int main(int argc, char** argv) {
  try {
    if (argc != 5) throw std::runtime_error("usage: r2-trace neuron|llama|products MODEL IDS DIRECTORY");
    r2::directory = argv[4];
    std::filesystem::create_directories(r2::directory);
    std::ifstream in(argv[3]);
    std::vector<std::int32_t> tokens;
    std::int32_t token;
    while (in >> token) tokens.push_back(token);
    if (!in.eof() || tokens.empty() || tokens.size() > 256) throw std::runtime_error("invalid input IDs");
    r2::position = tokens.size() - 1;
    llama_backend_init();
    neuron::kernels::set_threads(1); neuron::kernels::set_isa("auto");
    const std::string mode = argv[1];
    if (mode == "products") {
      neuron::Gguf file(argv[2]);
      const auto layers = static_cast<std::size_t>(file.at("qwen2.block_count").uint());
      std::size_t half_mismatches = 0, checked = 0;
      for (std::uint32_t bits = 0; bits < 65536; ++bits) {
        if ((bits & 0x7fffu) > 0x7c00u) continue;
        ++checked;
        if (std::bit_cast<std::uint32_t>(neuron::f16_to_f32(static_cast<std::uint16_t>(bits))) !=
            std::bit_cast<std::uint32_t>(ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(bits)))) ++half_mismatches;
      }
      std::cout << nlohmann::json{{"products", r2::products(file, layers)}, {"operations", r2::operations(file, layers)},
        {"f16_checked", checked}, {"f16_mismatches", half_mismatches}}.dump(2) << '\n';
      return 0;
    }
    std::vector<float> logits;
    if (mode == "neuron") {
      neuron::Model model(argv[2], 256);
      logits = model.forward(tokens);
    } else if (mode == "llama") {
      auto mp = llama_model_default_params(); mp.n_gpu_layers = 0;
      std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[2], mp), llama_model_free);
      if (!model) throw std::runtime_error("model load failed");
      r2::Callback callback;
      auto cp = llama_context_default_params();
      cp.n_ctx = 256; cp.n_batch = cp.n_ubatch = 1; cp.n_threads = cp.n_threads_batch = 1;
      cp.type_k = cp.type_v = GGML_TYPE_F32; cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
      cp.cb_eval = r2::callback; cp.cb_eval_user_data = &callback;
      std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
      if (!ctx) throw std::runtime_error("context creation failed");
      for (std::size_t i = 0; i < tokens.size(); ++i) {
        callback.active = i == r2::position;
        auto batch = llama_batch_get_one(&tokens[i], 1);
        if (llama_decode(ctx.get(), batch) != 0) throw std::runtime_error("decode failed");
      }
      const auto* result = llama_get_logits_ith(ctx.get(), -1);
      const auto size = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
      logits.assign(result, result + size);
    } else throw std::runtime_error("unknown engine");
    const auto first = static_cast<std::size_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
    std::cout << nlohmann::json{{"argmax", first}, {"logit", logits[first]}, {"position", r2::position}}.dump() << '\n';
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
}
