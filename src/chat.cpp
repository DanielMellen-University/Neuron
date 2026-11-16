#include "neuron/chat.hpp"

#include <stdexcept>

namespace neuron {
std::string qwen2_chat(std::span<const ChatMessage> messages, bool generation_prompt) {
  std::string result;
  if (messages.empty() || messages.front().role != "system")
    result = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n";
  for (const auto& message : messages) {
    if (message.role != "system" && message.role != "user" && message.role != "assistant")
      throw std::invalid_argument("chat: expected system, user or assistant role");
    result += "<|im_start|>" + message.role + "\n" + message.content + "<|im_end|>\n";
  }
  if (generation_prompt) result += "<|im_start|>assistant\n";
  return result;
}
}  // namespace neuron
