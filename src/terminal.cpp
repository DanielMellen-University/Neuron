#include "neuron/completion.hpp"

#include <csignal>
#include <cstdio>
#include <iostream>
#include <stdexcept>

namespace neuron {
namespace {
volatile std::sig_atomic_t interrupted = 0;
extern "C" void interrupt_reply(int) { interrupted = 1; }
class InterruptHandler {
 public:
  InterruptHandler() {
    struct sigaction action{};
    action.sa_handler = interrupt_reply;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGINT, &action, &previous_) != 0) throw std::runtime_error("cannot install SIGINT handler");
  }
  ~InterruptHandler() { sigaction(SIGINT, &previous_, nullptr); }
 private:
  struct sigaction previous_{};
};
}
void terminal_chat(ChatBackend& backend, const GenerationOptions& options, const std::string* system) {
  const InterruptHandler handler;
  std::vector<ChatMessage> history;
  auto reset = [&] {
    history.clear();
    if (system) history.push_back({"system", *system});
  };
  reset();
  std::cout << "Chat: /reset clears history, /exit quits, Ctrl-C stops a reply.\n";
  std::string line;
  for (;;) {
    interrupted = 0;
    std::cout << "you> " << std::flush;
    if (!std::getline(std::cin, line)) {
      if (interrupted) { std::cin.clear(); std::cout << '\n'; continue; }
      break;
    }
    if (line == "/exit") break;
    if (line == "/reset") { reset(); std::cout << "History cleared.\n"; continue; }
    if (line.empty()) continue;
    history.push_back({"user", line});
    try {
      backend.prepare(history, options);
      std::cout << "assistant> " << std::flush;
      const auto reply = backend.generate([](std::string_view piece) {
        std::cout << piece << std::flush; return true;
      }, [] { return interrupted != 0; });
      std::cout << '\n';
      if (reply.finish_reason == "cancelled") std::cout << "[reply stopped]\n";
      if (reply.text.empty()) history.pop_back();
      else history.push_back({"assistant", reply.text});
    } catch (const std::exception& e) {
      history.pop_back();
      std::cerr << "chat: " << e.what() << '\n';
    }
  }
}
}  // namespace neuron
