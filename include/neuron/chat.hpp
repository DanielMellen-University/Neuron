#pragma once

#include <span>
#include <string>

namespace neuron {
struct ChatMessage { std::string role, content; };
// Text-only Qwen2.5 ChatML, including the GGUF template's default system text.
std::string qwen2_chat(std::span<const ChatMessage> messages, bool generation_prompt = true);
}  // namespace neuron
