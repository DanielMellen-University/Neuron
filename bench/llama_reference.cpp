// Local validation adapter for the pinned external llama.cpp build.
// This file is never linked into Neuron or its model-free tests.
#include "llama.h"
#include "chat.h"
#include "json.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc < 3) throw std::runtime_error("usage: llama-reference MODEL TEXT [--logits FILE] [--generate N] [--perplexity CONTEXT --chunks N] [--chat FILE --render FILE] [--no-add-special] [--no-parse-special] [--cache-f16]");
    bool add_special = true, parse_special = true, cache_f16 = false;
    std::string output, chat, rendered, prompt_file;
    int generate = 0, ppl_context = 0, chunks = 0;
    for (int i = 3; i < argc; ++i) {
      const std::string option = argv[i];
      if (option == "--no-add-special") add_special = false;
      else if (option == "--no-parse-special") parse_special = false;
      else if (option == "--cache-f16") cache_f16 = true;
      else if (option == "--logits" && i + 1 < argc) output = argv[++i];
      else if (option == "--prompt-file" && i + 1 < argc) prompt_file = argv[++i];
      else if (option == "--chat" && i + 1 < argc) chat = argv[++i];
      else if (option == "--render" && i + 1 < argc) rendered = argv[++i];
      else if (option == "--perplexity" && i + 1 < argc) ppl_context = std::stoi(argv[++i]);
      else if (option == "--chunks" && i + 1 < argc) chunks = std::stoi(argv[++i]);
      else if (option == "--generate" && i + 1 < argc) generate = std::stoi(argv[++i]);
      else throw std::runtime_error("unknown option: " + option);
    }
    if (generate < 0 || generate > 2048) throw std::runtime_error("invalid generation count");
    llama_backend_init();
    llama_model_kv_override overrides[2]{};
    overrides[0].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
    std::strcpy(overrides[0].key, "llama.context_length"); overrides[0].val_i64 = 2048;
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 0; mp.kv_overrides = overrides;
    mp.vocab_only = output.empty() && generate == 0 && ppl_context == 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
    if (!model) throw std::runtime_error("model load failed");
    const auto* vocab = llama_model_get_vocab(model.get());
    std::string text = argv[2];
    if (!prompt_file.empty()) {
      std::ifstream stream(prompt_file, std::ios::binary);
      if (!stream) throw std::runtime_error("cannot read prompt file");
      text.assign(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
    }
    if (!chat.empty()) {
      std::ifstream stream(chat);
      const auto messages = nlohmann::json::parse(stream);
      common_chat_templates_inputs inputs;
      inputs.use_jinja = true; inputs.add_generation_prompt = true;
      for (const auto& message : messages) {
        common_chat_msg msg;
        msg.role = message.at("role").get<std::string>(); msg.content = message.at("content").get<std::string>();
        inputs.messages.push_back(msg);
      }
      auto templates = common_chat_templates_init(model.get(), "");
      text = common_chat_templates_apply(templates.get(), inputs).prompt.str();
    }
    if (!rendered.empty()) {
      std::ofstream stream(rendered, std::ios::binary); stream << text;
      if (!stream) throw std::runtime_error("render write failed");
    }
    std::vector<llama_token> tokens(text.size() + 16);
    const auto n = llama_tokenize(vocab, text.data(), static_cast<std::int32_t>(text.size()), tokens.data(),
                                   static_cast<std::int32_t>(tokens.size()), add_special, parse_special);
    if (n < 0) throw std::runtime_error("tokenize buffer too small");
    tokens.resize(static_cast<std::size_t>(n));
    nlohmann::json result;
    result["prompt_tokens"] = tokens;
    result["generated_tokens"] = nlohmann::json::array();
    result["top2"] = nlohmann::json::array();
    std::string generated_text;
    if (ppl_context != 0) {
      if (ppl_context < 2 || ppl_context % 2 != 0 || chunks < 0)
        throw std::runtime_error("perplexity needs an even context >= 2 and nonnegative chunks");
      const auto context = static_cast<std::size_t>(ppl_context);
      const auto available = tokens.size() / context;
      const auto count = chunks == 0 ? available : static_cast<std::size_t>(chunks);
      if (count == 0 || count > available) throw std::runtime_error("too few complete chunks");
      auto cp = llama_context_default_params();
      cp.n_ctx = static_cast<std::uint32_t>(context); cp.n_batch = 1; cp.n_ubatch = 1;
      cp.n_threads = cp.n_threads_batch = 1;
      cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
      cp.type_k = cp.type_v = GGML_TYPE_F32;
      std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
      if (!ctx) throw std::runtime_error("context creation failed");
      const auto start = std::chrono::steady_clock::now();
      double nll = 0;
      std::size_t scored = 0;
      for (std::size_t chunk = 0; chunk < count; ++chunk) {
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        for (std::size_t i = 0; i + 1 < context; ++i) {
          auto batch = llama_batch_get_one(tokens.data() + chunk * context + i, 1);
          if (llama_decode(ctx.get(), batch) != 0) throw std::runtime_error("decode failed");
          if (i + 1 < context / 2) continue;
          const auto* logits = llama_get_logits_ith(ctx.get(), -1);
          const auto size = llama_vocab_n_tokens(vocab);
          const double maximum = *std::max_element(logits, logits + size);
          double sum = 0;
          for (int j = 0; j < size; ++j) sum += std::exp(static_cast<double>(logits[j]) - maximum);
          nll += std::log(sum) + maximum - logits[tokens[chunk * context + i + 1]];
          ++scored;
        }
      }
      const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      nlohmann::json report{{"context", context}, {"chunks", count}, {"scored_tokens", scored},
                            {"nll", nll}, {"perplexity", std::exp(nll / static_cast<double>(scored))},
                            {"seconds", elapsed}, {"tokens", tokens}};
      std::printf("%s\n", report.dump().c_str());
      return 0;
    }
    if (!output.empty() || generate > 0) {
      if (tokens.empty() || tokens.size() + static_cast<std::size_t>(generate) > 2048)
        throw std::runtime_error("prompt plus generation exceeds context");
      auto cp = llama_context_default_params();
      cp.n_ctx = 2048; cp.n_batch = 512; cp.n_ubatch = 512;
      cp.n_threads = 1; cp.n_threads_batch = 1;
      cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
      cp.type_k = cp.type_v = cache_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
      std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
      if (!ctx) throw std::runtime_error("context creation failed");
      for (std::size_t at = 0; at < tokens.size(); at += 512) {
        auto batch = llama_batch_get_one(tokens.data() + at, static_cast<std::int32_t>(std::min<std::size_t>(512, tokens.size() - at)));
        if (llama_decode(ctx.get(), batch) != 0) throw std::runtime_error("decode failed");
      }
      const auto size = llama_vocab_n_tokens(vocab);
      const float* logits = llama_get_logits_ith(ctx.get(), -1);
      if (!logits) throw std::runtime_error("missing logits");
      if (!output.empty()) {
        std::ofstream stream(output, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(logits), static_cast<std::streamsize>(size) * 4);
        if (!stream) throw std::runtime_error("logit write failed");
      }
      for (int i = 0; i < generate; ++i) {
        const auto first = static_cast<llama_token>(std::max_element(logits, logits + size) - logits);
        llama_token second = first == 0 ? 1 : 0;
        for (llama_token t = 0; t < size; ++t) if (t != first && logits[t] > logits[second]) second = t;
        result["top2"].push_back({{"first", first}, {"second", second}, {"gap", logits[first] - logits[second]}});
        if (llama_vocab_is_eog(vocab, first)) break;
        result["generated_tokens"].push_back(first);
        char piece[1024]; const auto length = llama_token_to_piece(vocab, first, piece, sizeof(piece), 0, false);
        if (length < 0) throw std::runtime_error("piece buffer too small");
        generated_text.append(piece, static_cast<std::size_t>(length));
        if (i + 1 < generate) {
          auto token = first; auto batch = llama_batch_get_one(&token, 1);
          if (llama_decode(ctx.get(), batch) != 0) throw std::runtime_error("decode failed");
          logits = llama_get_logits_ith(ctx.get(), -1);
        }
      }
    }
    result["generated_text"] = generated_text;
    std::printf("%s\n", (generate > 0 ? result : nlohmann::json(tokens)).dump().c_str());
    model.reset(); llama_backend_free();
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "llama-reference: %s\n", error.what()); return 2;
  }
}
