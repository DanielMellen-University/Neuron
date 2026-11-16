#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "neuron/model.hpp"
#include "neuron/tokenizer.hpp"
#include "neuron/version.hpp"

namespace {
void print_usage() {
  std::printf(
      "usage: neuron [--version] [--help]\n"
      "       neuron run -m MODEL -p PROMPT [-n 128] --greedy [--json] [--ctx-size N]\n"
      "       neuron tokenize -m MODEL -p PROMPT [--no-add-special] [--no-parse-special]\n"
      "       neuron logits -m MODEL -p PROMPT --output FILE [--ctx-size N]\n"
      "\n"
      "Dense llama GGUF models with F32/F16 weights and llama vocabularies.\n"
      "--json emits prompt/generated token IDs and timings. logits writes raw F32.\n"
      "--ctx-size defaults to 2048, matching the M1 reference context.\n");
}
std::size_t count(std::string_view text) {
  std::size_t value = 0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || value == 0)
    throw std::invalid_argument("expected a positive integer");
  return value;
}
void ids(const std::vector<std::int32_t>& values) {
  std::printf("[");
  for (std::size_t i = 0; i < values.size(); ++i) std::printf("%s%d", i == 0 ? "" : ",", values[i]);
  std::printf("]");
}
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
int execute(int argc, char** argv) {
  if (argc == 2 && std::strcmp(argv[1], "--version") == 0) {
    std::printf("neuron %s (%s)\n", neuron::version_string().data(), neuron::compiled_isa().data()); return 0;
  }
  if (argc == 1 || (argc == 2 && (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0))) {
    print_usage(); return 0;
  }
  const std::string command = argv[1];
  if (command != "run" && command != "tokenize" && command != "logits") throw std::invalid_argument("unknown command: " + command);
  std::string path, prompt, output;
  bool has_prompt = false, greedy = false, json = false, add_special = true, parse_special = true;
  std::size_t predict = 128, context = 2048;
  for (int i = 2; i < argc; ++i) {
    const std::string_view option = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument("missing value for " + std::string(option));
      return argv[++i];
    };
    if (option == "-m") path = value();
    else if (option == "-p") { prompt = value(); has_prompt = true; }
    else if (option == "-n" && command == "run") predict = count(value());
    else if (option == "--ctx-size" && command != "tokenize") context = count(value());
    else if (option == "--output" && command == "logits") output = value();
    else if (option == "--greedy" && command == "run") greedy = true;
    else if (option == "--json" && command == "run") json = true;
    else if (option == "--no-add-special" && command == "tokenize") add_special = false;
    else if (option == "--no-parse-special" && command == "tokenize") parse_special = false;
    else throw std::invalid_argument("unknown option: " + std::string(option));
  }
  if (path.empty() || !has_prompt) throw std::invalid_argument("-m MODEL and -p PROMPT are required");
  if (command == "tokenize") {
    neuron::Gguf file(path); neuron::Tokenizer tokenizer(file);
    ids(tokenizer.encode(prompt, add_special, parse_special)); std::printf("\n"); return 0;
  }
  if (command == "run" && !greedy) throw std::invalid_argument("run requires --greedy");
  if (command == "logits" && output.empty()) throw std::invalid_argument("logits requires --output FILE");
  neuron::Model model(path, context); neuron::Tokenizer tokenizer(model.file());
  const auto tokens = tokenizer.encode(prompt);
  if (command == "run" && (tokens.size() > model.config().context || predict > model.config().context - tokens.size()))
    throw std::invalid_argument("prompt plus requested generation exceeds context limit; use --ctx-size to override explicitly");
  const auto start = Clock::now(); auto logits = model.forward(tokens); const double prompt_seconds = seconds(start);
  if (command == "logits") {
    std::ofstream stream(output, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(logits.data()), static_cast<std::streamsize>(logits.size() * sizeof(float)));
    if (!stream) throw std::runtime_error("cannot write logits: " + output);
    return 0;
  }
  std::vector<std::int32_t> generated;
  const auto generation_start = Clock::now();
  for (std::size_t i = 0; i < predict; ++i) {
    const auto next = static_cast<std::int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
    if (next == tokenizer.eos()) break;
    generated.push_back(next);
    if (!json) { const auto piece = tokenizer.piece(next); std::fwrite(piece.data(), 1, piece.size(), stdout); std::fflush(stdout); }
    if (i + 1 < predict) logits = model.step(next);
  }
  const double generation_seconds = seconds(generation_start);
  if (json) {
    std::printf("{\"prompt_tokens\":"); ids(tokens); std::printf(",\"generated_tokens\":"); ids(generated);
    const double steps = static_cast<double>(generated.empty() ? 0 : generated.size() - 1);
    std::printf(",\"timings\":{\"prompt_n\":%zu,\"prompt_ms\":%.9g,\"predicted_n\":%zu,\"predicted_ms\":%.9g,\"prompt_per_second\":%.9g,\"predicted_per_second\":%.9g}}\n",
                tokens.size(), prompt_seconds * 1000, generated.size(), generation_seconds * 1000,
                static_cast<double>(tokens.size()) / prompt_seconds, steps / generation_seconds);
  } else std::printf("\n");
  return 0;
}
}  // namespace
int main(int argc, char** argv) {
  try { return execute(argc, argv); }
  catch (const std::exception& e) { std::fprintf(stderr, "neuron: %s\n", e.what()); return 2; }
}
