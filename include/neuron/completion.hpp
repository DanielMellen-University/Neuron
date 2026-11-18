#pragma once

#include "neuron/chat.hpp"
#include "neuron/model.hpp"
#include "neuron/sampling.hpp"
#include "neuron/tokenizer.hpp"

#include <functional>
#include <string_view>
#include <vector>

namespace neuron {
struct Completion {
  std::string text;
  std::size_t tokens = 0;
  std::string finish_reason = "length";
};
// Per-request settings. The default is greedy decoding with no stop strings.
struct GenerationOptions {
  std::size_t max_tokens = 128;
  SamplingOptions sampling;
  std::vector<std::string> stop;
};
using TextSink = std::function<bool(std::string_view)>;
using Cancelled = std::function<bool()>;

// prepare validates context before any HTTP response is sent. A backend is used
// by one caller at a time. Tests can supply a model-free backend.
class ChatBackend {
 public:
  virtual ~ChatBackend() = default;
  virtual std::size_t prepare(std::span<const ChatMessage> messages, const GenerationOptions& options) = 0;
  virtual Completion generate(const TextSink& sink, const Cancelled& cancelled) = 0;
};

class ModelChat final : public ChatBackend {
 public:
  ModelChat(const std::string& path, std::size_t context);
  std::size_t prepare(std::span<const ChatMessage> messages, const GenerationOptions& options) override;
  Completion generate(const TextSink& sink, const Cancelled& cancelled) override;
 private:
  Model model_;
  Tokenizer tokenizer_;
  bool smollm_ = false;
  std::vector<std::int32_t> prompt_;
  GenerationOptions options_;
  bool prepared_ = false;
};

void terminal_chat(ChatBackend& backend, const GenerationOptions& options, const std::string* system = nullptr);
}  // namespace neuron
