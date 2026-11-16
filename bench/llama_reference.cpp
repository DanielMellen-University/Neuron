// Local validation adapter for the pinned external llama.cpp build.
// This file is never linked into Neuron or its model-free tests.
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc < 3) throw std::runtime_error("usage: llama-reference MODEL TEXT [--logits FILE] [--no-add-special] [--no-parse-special] [--cache-f16]");
    bool add_special = true, parse_special = true, cache_f16 = false;
    std::string output;
    for (int i = 3; i < argc; ++i) {
      const std::string option = argv[i];
      if (option == "--no-add-special") add_special = false;
      else if (option == "--no-parse-special") parse_special = false;
      else if (option == "--cache-f16") cache_f16 = true;
      else if (option == "--logits" && i + 1 < argc) output = argv[++i];
      else throw std::runtime_error("unknown option: " + option);
    }
    llama_backend_init();
    llama_model_kv_override overrides[2]{};
    overrides[0].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
    std::strcpy(overrides[0].key, "llama.context_length"); overrides[0].val_i64 = 2048;
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 0; mp.kv_overrides = overrides;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
    if (!model) throw std::runtime_error("model load failed");
    const auto* vocab = llama_model_get_vocab(model.get());
    const std::string text = argv[2];
    std::vector<llama_token> tokens(text.size() + 16);
    const auto n = llama_tokenize(vocab, text.data(), static_cast<std::int32_t>(text.size()), tokens.data(),
                                   static_cast<std::int32_t>(tokens.size()), add_special, parse_special);
    if (n < 0) throw std::runtime_error("tokenize buffer too small");
    tokens.resize(static_cast<std::size_t>(n));
    std::printf("[");
    for (std::size_t i = 0; i < tokens.size(); ++i) std::printf("%s%d", i ? "," : "", tokens[i]);
    std::printf("]\n");
    if (!output.empty()) {
      if (tokens.empty() || tokens.size() > 512) throw std::runtime_error("logits require 1 to 512 prompt tokens");
      auto cp = llama_context_default_params();
      cp.n_ctx = 2048; cp.n_batch = 512; cp.n_ubatch = 512;
      cp.n_threads = 1; cp.n_threads_batch = 1;
      cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
      cp.type_k = cp.type_v = cache_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
      std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
      if (!ctx) throw std::runtime_error("context creation failed");
      auto batch = llama_batch_get_one(tokens.data(), n);
      if (llama_decode(ctx.get(), batch) != 0) throw std::runtime_error("decode failed");
      const float* logits = llama_get_logits_ith(ctx.get(), -1);
      if (!logits) throw std::runtime_error("missing logits");
      std::ofstream stream(output, std::ios::binary);
      stream.write(reinterpret_cast<const char*>(logits), static_cast<std::streamsize>(llama_vocab_n_tokens(vocab)) * 4);
      if (!stream) throw std::runtime_error("logit write failed");
    }
    model.reset(); llama_backend_free();
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "llama-reference: %s\n", error.what()); return 2;
  }
}
