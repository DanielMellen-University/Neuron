#pragma once

#include "neuron/chat.hpp"
#include "neuron/model.hpp"
#include "neuron/tokenizer.hpp"

#include <functional>
#include <string_view>

namespace neuron {
struct Completion {
  std::string text;
  std::size_t tokens = 0;
  std::string finish_reason = "length";
};
using TextSink = std::function<bool(std::string_view)>;
using Cancelled = std::function<bool()>;

// prepare validates context before any HTTP response is sent. A backend is used
// by one caller at a time. Tests can supply a model-free backend.
class ChatBackend {
 public:
  virtual ~ChatBackend() = default;
  virtual std::size_t prepare(std::span<const ChatMessage> messages, std::size_t max_tokens) = 0;
  virtual Completion generate(const TextSink& sink, const Cancelled& cancelled) = 0;
};

class ModelChat final : public ChatBackend {
 public:
  ModelChat(const std::string& path, std::size_t context);
  std::size_t prepare(std::span<const ChatMessage> messages, std::size_t max_tokens) override;
  Completion generate(const TextSink& sink, const Cancelled& cancelled) override;
 private:
  Model model_;
  Tokenizer tokenizer_;
  bool smollm_ = false;
  std::vector<std::int32_t> prompt_;
  std::size_t max_tokens_ = 0;
};

void terminal_chat(ChatBackend& backend, std::size_t max_tokens, const std::string* system = nullptr);
}  // namespace neuron
