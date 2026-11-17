#define NEURON_TEST_MAIN
#include "neuron/chat.hpp"
#include "neuron_test.hpp"
#include "../src/utf8.hpp"

#include <array>
#include <stdexcept>

NEURON_TEST(default_system_history_and_assistant_generation_prompt) {
  const std::array<neuron::ChatMessage, 3> messages{{{"user","Hi"}, {"assistant","Hello"}, {"user","Next"}}};
  CHECK_EQ(neuron::qwen2_chat(messages), std::string(
      "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
      "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello<|im_end|>\n"
      "<|im_start|>user\nNext<|im_end|>\n<|im_start|>assistant\n"));
}
NEURON_TEST(explicit_empty_system_and_no_generation_prompt) {
  const std::array<neuron::ChatMessage, 2> messages{{{"system",""}, {"user"," leading\ntrailing "}}};
  CHECK_EQ(neuron::qwen2_chat(messages, false), std::string(
      "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n leading\ntrailing <|im_end|>\n"));
  CHECK(neuron::qwen2_chat({}).find("You are Qwen") != std::string::npos);
  const std::array<neuron::ChatMessage, 1> invalid{{{"tool","x"}}};
  CHECK_THROWS_AS(neuron::qwen2_chat(invalid), std::invalid_argument);
}
NEURON_TEST(streamed_utf8_handles_split_tokens_and_malformed_bytes) {
  std::string pending = "hello \xf0\x9f";
  CHECK_EQ(neuron::detail::take_utf8(pending, false), std::string("hello "));
  CHECK_EQ(pending, std::string("\xf0\x9f"));
  pending += "\x98\x80";
  CHECK_EQ(neuron::detail::take_utf8(pending, false), std::string("\xf0\x9f\x98\x80"));
  pending = "\xed\xa0\x80\xff\xe2";
  CHECK(neuron::detail::valid_utf8(neuron::detail::take_utf8(pending, true)));
  CHECK(pending.empty());
}
