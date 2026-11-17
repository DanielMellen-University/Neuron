#include "neuron/completion.hpp"
#include "neuron/server.hpp"

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>

namespace {
class FakeBackend final : public neuron::ChatBackend {
 public:
  std::size_t prepare(std::span<const neuron::ChatMessage> messages, std::size_t max_tokens) override {
    std::size_t count = 0;
    for (const auto& message : messages) count += message.content.size();
    if (count > 256 || max_tokens > 256 - count) throw std::length_error("context limit is 256 tokens");
    history_ = messages.size();
    prefill_ = messages.back().content == "prefill";
    slow_ = messages.back().content == "slow";
    fail_ = messages.back().content == "fail";
    return count;
  }
  neuron::Completion generate(const neuron::TextSink& sink, const neuron::Cancelled& cancelled) override {
    if (prefill_) for (int i = 0; i < 200 && !cancelled(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    if (fail_) throw std::runtime_error("synthetic failure");
    neuron::Completion result;
    for (const auto& piece : {"history=" + std::to_string(history_), std::string(": hello "), std::string("\xe2\x98\x83\n")}) {
      if (cancelled()) { result.finish_reason = "cancelled"; return result; }
      result.text += piece; ++result.tokens;
      if (!sink(piece)) { result.finish_reason = "cancelled"; return result; }
      const auto delay = slow_ ? 1000 : 20;
      for (int i = 0; i < delay / 5 && !cancelled(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    result.finish_reason = "stop";
    return result;
  }
 private:
  std::size_t history_ = 0;
  bool prefill_ = false, slow_ = false, fail_ = false;
};
}
int main(int argc, char** argv) {
  FakeBackend backend;
  if (argc == 2 && std::string_view(argv[1]) == "chat") {
    neuron::terminal_chat(backend, 16);
  } else {
    neuron::HttpServer server(backend, "fake-model", "127.0.0.1", 0);
    std::printf("%u\n", static_cast<unsigned int>(server.port()));
    std::fflush(stdout);
    server.run();
  }
}
