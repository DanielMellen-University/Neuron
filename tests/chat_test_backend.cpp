#include "neuron/completion.hpp"
#include "neuron/server.hpp"
#include "neuron/stop.hpp"

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
class FakeBackend final : public neuron::ChatBackend {
 public:
  std::size_t prepare(std::span<const neuron::ChatMessage> messages, const neuron::GenerationOptions& options) override {
    const auto max_tokens = options.max_tokens;
    std::size_t count = 0;
    for (const auto& message : messages) count += message.content.size();
    if (count > 256 || max_tokens > 256 - count) throw std::length_error("context limit is 256 tokens");
    history_ = messages.size();
    prefill_ = messages.back().content == "prefill";
    slow_ = messages.back().content == "slow";
    fail_ = messages.back().content == "fail";
    echo_ = messages.back().content == "options";
    options_ = options;
    return count;
  }
  neuron::Completion generate(const neuron::TextSink& sink, const neuron::Cancelled& cancelled) override {
    if (prefill_) for (int i = 0; i < 200 && !cancelled(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    if (fail_) throw std::runtime_error("synthetic failure");
    neuron::Completion result;
    // "options" replies with the settings the server parsed, so tests can check them.
    std::vector<std::string> pieces{"history=" + std::to_string(history_), ": hello ", "\xe2\x98\x83\n"};
    if (echo_) {
      const auto& s = options_.sampling;
      char text[256];
      std::snprintf(text, sizeof(text), "temperature=%g top_p=%g top_k=%zu min_p=%g repeat_penalty=%g seed=%llu max=%zu stops=%zu",
          static_cast<double>(s.temperature), static_cast<double>(s.top_p), s.top_k, static_cast<double>(s.min_p),
          static_cast<double>(s.repeat_penalty), static_cast<unsigned long long>(s.seed), options_.max_tokens, options_.stop.size());
      pieces = {text};
    }
    neuron::StopMatcher stop(options_.stop);
    for (const auto& raw : pieces) {
      if (cancelled()) { result.finish_reason = "cancelled"; return result; }
      const auto piece = stop.push(raw);
      result.text += piece; ++result.tokens;
      if (!sink(piece)) { result.finish_reason = "cancelled"; return result; }
      if (stop.stopped()) break;
      const auto delay = slow_ ? 1000 : 20;
      for (int i = 0; i < delay / 5 && !cancelled(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    const auto rest = stop.finish();
    if (!rest.empty()) { result.text += rest; if (!sink(rest)) result.finish_reason = "cancelled"; }
    if (result.finish_reason != "cancelled") result.finish_reason = "stop";
    return result;
  }
 private:
  std::size_t history_ = 0;
  bool prefill_ = false, slow_ = false, fail_ = false, echo_ = false;
  neuron::GenerationOptions options_;
};
}
int main(int argc, char** argv) {
  FakeBackend backend;
  if (argc == 2 && std::string_view(argv[1]) == "chat") {
    neuron::GenerationOptions options;
    options.max_tokens = 16;
    neuron::terminal_chat(backend, options);
  } else {
    neuron::HttpServer server(backend, "fake-model", "127.0.0.1", 0);
    // A fixed seed stream, so tests can check which seeds requests get.
    neuron::GenerationOptions defaults;
    defaults.sampling.seed = 1;
    server.set_defaults(defaults);
    std::printf("%u\n", static_cast<unsigned int>(server.port()));
    std::fflush(stdout);
    server.run();
  }
}
