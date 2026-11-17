#include "neuron/completion.hpp"
#include "utf8.hpp"

#include <algorithm>
#include <stdexcept>

namespace neuron {
ModelChat::ModelChat(const std::string& path, std::size_t context)
    : model_(path, context), tokenizer_(model_.file()) {
  const auto* pre = model_.file().find("tokenizer.ggml.pre");
  smollm_ = pre && pre->string() == "smollm";
  if (!smollm_ && model_.file().at("general.architecture").string() != "qwen2")
    throw std::invalid_argument("chat supports Qwen2 and SmolLM2 Instruct ChatML models");
}
std::size_t ModelChat::prepare(std::span<const ChatMessage> messages, std::size_t max_tokens) {
  std::vector<ChatMessage> history(messages.begin(), messages.end());
  if (smollm_ && (history.empty() || history.front().role != "system"))
    history.insert(history.begin(), {"system", "You are a helpful AI assistant named SmolLM, trained by Hugging Face"});
  auto tokens = tokenizer_.encode(qwen2_chat(history));
  if (max_tokens == 0 || tokens.size() > model_.config().context ||
      max_tokens > model_.config().context - tokens.size())
    throw std::length_error("prompt plus requested generation exceeds context limit; reset or shorten the conversation");
  prompt_ = std::move(tokens);
  max_tokens_ = max_tokens;
  return prompt_.size();
}
Completion ModelChat::generate(const TextSink& sink, const Cancelled& cancelled) {
  if (prompt_.empty() || max_tokens_ == 0) throw std::logic_error("prepare chat before generation");
  Completion result;
  model_.reset();
  std::vector<float> logits;
  for (std::size_t i = 0; i < prompt_.size(); ++i) {
    if (cancelled()) { result.finish_reason = "cancelled"; return result; }
    logits = model_.step(prompt_[i], i + 1 == prompt_.size());
  }
  std::string pending;
  auto emit = [&](bool final) {
    const auto piece = detail::take_utf8(pending, final);
    result.text += piece;
    return piece.empty() || sink(piece);
  };
  for (std::size_t i = 0; i < max_tokens_; ++i) {
    if (cancelled()) { result.finish_reason = "cancelled"; break; }
    const auto next = static_cast<std::int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
    ++result.tokens;
    if (tokenizer_.is_eog(next)) { result.finish_reason = "stop"; break; }
    pending += tokenizer_.piece(next);
    if (!emit(false)) { result.finish_reason = "cancelled"; return result; }
    if (i + 1 < max_tokens_) logits = model_.step(next);
  }
  if (!emit(true)) result.finish_reason = "cancelled";
  return result;
}
}  // namespace neuron
