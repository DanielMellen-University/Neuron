#include <algorithm>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

#include "neuron/model.hpp"
#include "neuron/kernels.hpp"
#include "neuron/perplexity.hpp"
#include "neuron/chat.hpp"
#include "neuron/completion.hpp"
#include "neuron/server.hpp"
#include "neuron/tokenizer.hpp"
#include "neuron/version.hpp"

namespace {
volatile std::sig_atomic_t server_stopped = 0;
extern "C" void stop_server(int) { server_stopped = 1; }
void print_usage() {
  std::printf(
      "usage: neuron [--version] [--help]\n"
      "       neuron run -m MODEL -p PROMPT [-n 128] --greedy [--json] [--ctx-size N]\n"
      "       neuron tokenize -m MODEL -p PROMPT [--no-add-special] [--no-parse-special]\n"
      "       neuron logits -m MODEL -p PROMPT --output FILE [--ctx-size N]\n"
      "\n"
      "       neuron run -m MODEL --chat -p PROMPT [--system TEXT] -n N --greedy\n"
      "       neuron tokenize|template -m MODEL --chat [--message ROLE TEXT]... [-p TEXT]\n"
      "\n"
      "       neuron perplexity -m MODEL --prompt-file FILE --ctx-size N [--chunks N]\n"
      "       neuron chat -m MODEL [-n 128] [--system TEXT] [--ctx-size N]\n"
      "       neuron serve -m MODEL [--host 127.0.0.1] [--port 8080] [--ctx-size N]\n"
      "\nDense llama and Qwen2 GGUF models with F32/F16/Q8_0/Q4_0/Q4_K/Q6_K weights.\n"
      "--prompt-file FILE reads prompt bytes instead of -p.\n"
      "--chat applies Qwen2.5 ChatML; repeated --message flags supply history.\n"
      "--system replaces the default system text; -p adds a final user message.\n"
      "--recompute checks generation by evaluating the full sequence at every step.\n"
      "--teacher-tokens FILE replays whitespace-separated IDs while recording argmax.\n"
      "--json emits prompt/generated token IDs and timings. logits writes raw F32.\n"
      "--threads N sets total matmul threads (default NEURON_THREADS or 1).\n"
      "--isa auto|scalar|avx2|avx512|vnni overrides NEURON_ISA.\n"
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
    std::printf("neuron %s (%s; selected %s; threads %zu)\n", neuron::version_string().data(),
        neuron::compiled_isa().data(), neuron::kernels::selected_isa().data(), neuron::kernels::threads()); return 0;
  }
  if (argc == 1 || (argc == 2 && (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0))) {
    print_usage(); return 0;
  }
  const std::string command = argv[1];
  if (command != "run" && command != "tokenize" && command != "logits" && command != "template" && command != "perplexity" && command != "chat" && command != "serve") throw std::invalid_argument("unknown command: " + command);
  std::string path, prompt, output, teacher_file;
  bool has_prompt = false, greedy = false, json = false, add_special = true, parse_special = true;
  bool chat = false, recompute = false, has_system = false;
  std::string system;
  std::vector<neuron::ChatMessage> messages;
  std::size_t predict = 128, context = 2048, chunks = 0;
  std::string host = "127.0.0.1";
  std::size_t port = 8080;
  for (int i = 2; i < argc; ++i) {
    const std::string_view option = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument("missing value for " + std::string(option));
      return argv[++i];
    };
    if (option == "--threads") neuron::kernels::set_threads(count(value()));
    else if (option == "--isa") neuron::kernels::set_isa(value());
    else if (option == "-m") path = value();
    else if (option == "--host" && command == "serve") host = value();
    else if (option == "--port" && command == "serve") { port = count(value()); if (port > 65535) throw std::invalid_argument("port must be in 1..65535"); }
    else if (option == "-p") { prompt = value(); has_prompt = true; }
    else if (option == "--prompt-file") {
      const auto prompt_path = value();
      std::ifstream stream(prompt_path, std::ios::binary);
      if (!stream) throw std::invalid_argument("cannot read prompt file: " + prompt_path);
      prompt.assign(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()); has_prompt = true;
    }
    else if (option == "-n" && (command == "run" || command == "chat")) predict = count(value());
    else if (option == "--chunks" && command == "perplexity") chunks = count(value());
    else if (option == "--chat" && command != "perplexity") chat = true;
    else if (option == "--system" && command != "perplexity") { system = value(); has_system = true; chat = true; }
    else if (option == "--message" && command != "perplexity") { const auto role = value(); messages.push_back({role, value()}); chat = true; }
    else if (option == "--teacher-tokens" && command == "run") teacher_file = value();
    else if (option == "--recompute" && command == "run") recompute = true;
    else if (option == "--ctx-size" && command != "tokenize" && command != "template") context = count(value());
    else if (option == "--output" && command == "logits") output = value();
    else if (option == "--greedy" && command == "run") greedy = true;
    else if (option == "--json" && command == "run") json = true;
    else if (option == "--no-add-special" && command == "tokenize") add_special = false;
    else if (option == "--no-parse-special" && command == "tokenize") parse_special = false;
    else throw std::invalid_argument("unknown option: " + std::string(option));
  }
  if (command == "chat" || command == "serve") {
    if (path.empty()) throw std::invalid_argument("-m MODEL is required");
    neuron::ModelChat backend(path, context);
    if (command == "chat") neuron::terminal_chat(backend, predict, has_system ? &system : nullptr);
    else {
      neuron::HttpServer server(backend, std::filesystem::path(path).filename().string(), host, static_cast<std::uint16_t>(port));
      struct sigaction action{};
      action.sa_handler = stop_server;
      sigemptyset(&action.sa_mask);
      if (sigaction(SIGINT, &action, nullptr) != 0 || sigaction(SIGTERM, &action, nullptr) != 0)
        throw std::runtime_error("cannot install server signal handlers");
      std::printf("Neuron listening on http://%s:%u (model: %s)\n", host.c_str(),
          static_cast<unsigned int>(server.port()), std::filesystem::path(path).filename().c_str());
      std::fflush(stdout);
      server.run([] { return server_stopped != 0; });
    }
    return 0;
  }
  if (path.empty() || (!has_prompt && messages.empty()))
    throw std::invalid_argument("-m MODEL and a prompt (-p or --message) are required");
  if (command == "template" && !chat) throw std::invalid_argument("template requires --chat");
  // Validate and tokenize before loading weights or allocating a model cache.
  neuron::Gguf file(path);
  neuron::Tokenizer tokenizer(file);
  if (chat) {
    if (file.at("general.architecture").string() != "qwen2") throw std::invalid_argument("--chat requires a Qwen2 model");
    if (has_system) {
      if (!messages.empty() && messages.front().role == "system")
        throw std::invalid_argument("use either --system or a leading system message");
      messages.insert(messages.begin(), {"system", system});
    }
    if (has_prompt) messages.push_back({"user", prompt});
    prompt = neuron::qwen2_chat(messages);
  }
  if (command == "template") { std::fwrite(prompt.data(), 1, prompt.size(), stdout); return 0; }
  const auto tokens = tokenizer.encode(prompt, add_special, parse_special);
  if (command == "tokenize") { ids(tokens); std::printf("\n"); return 0; }
  if (command == "run" && !greedy) throw std::invalid_argument("run requires --greedy");
  if (command == "logits" && output.empty()) throw std::invalid_argument("logits requires --output FILE");
  if (command == "perplexity") {
    neuron::Model model(path, context);
    const auto start = Clock::now();
    const auto result = neuron::perplexity(model, tokens, chunks);
    std::printf("{\"context\":%zu,\"chunks\":%zu,\"scored_tokens\":%zu,\"nll\":%.17g,\"perplexity\":%.17g,\"seconds\":%.9g,\"tokens\":",
                context, result.chunks, result.scored_tokens, result.negative_log_likelihood, result.value(), seconds(start));
    ids(tokens); std::printf("}\n");
    return 0;
  }
  if (tokens.size() > context || (command == "run" && predict > context - tokens.size()))
    throw std::invalid_argument("prompt plus requested generation exceeds context limit; use --ctx-size to override explicitly");
  std::vector<std::int32_t> teacher;
  if (!teacher_file.empty()) {
    std::ifstream stream(teacher_file);
    if (!stream) throw std::invalid_argument("cannot read teacher token file");
    std::string word;
    while (stream >> word) {
      std::int32_t token = 0;
      const auto parsed = std::from_chars(word.data(), word.data() + word.size(), token);
      if (parsed.ec != std::errc{} || parsed.ptr != word.data() + word.size() || token < 0 ||
          static_cast<std::size_t>(token) >= file.at("tokenizer.ggml.tokens").array().size())
        throw std::invalid_argument("teacher token outside vocabulary");
      teacher.push_back(token);
    }
    if (!stream.eof() || teacher.size() != predict || !json || recompute)
      throw std::invalid_argument("teacher tokens require exactly -n IDs, --json and cached generation");
  }
  neuron::Model model(path, context);
  const auto start = Clock::now(); auto logits = model.forward_view(tokens); const double prompt_seconds = seconds(start);
  if (command == "logits") {
    std::ofstream stream(output, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(logits.data()), static_cast<std::streamsize>(logits.size() * sizeof(float)));
    if (!stream) throw std::runtime_error("cannot write logits: " + output);
    return 0;
  }
  struct TopTwo { std::int32_t first, second; float gap; };
  std::vector<TopTwo> top2;
  std::vector<std::int32_t> generated, sequence = tokens;
  top2.reserve(predict); generated.reserve(predict); sequence.reserve(context);
  const auto generation_start = Clock::now();
  for (std::size_t i = 0; i < predict; ++i) {
    const auto next = static_cast<std::int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
    std::int32_t second = next;
    for (std::size_t t = 0; t < logits.size(); ++t)
      if (static_cast<std::int32_t>(t) != next && (second == next || logits[t] > logits[static_cast<std::size_t>(second)])) second = static_cast<std::int32_t>(t);
    top2.push_back({next, second, logits[static_cast<std::size_t>(next)] - logits[static_cast<std::size_t>(second)]});
    if (teacher.empty() && tokenizer.is_eog(next)) break;
    generated.push_back(next);
    if (!json) { const auto piece = tokenizer.piece(next); std::fwrite(piece.data(), 1, piece.size(), stdout); std::fflush(stdout); }
    if (i + 1 < predict) {
      if (recompute) { sequence.push_back(next); logits = model.forward_view(sequence); }
      else logits = model.step_view(teacher.empty() ? next : teacher[i]);
    }
  }
  const double generation_seconds = seconds(generation_start);
  if (json) {
    std::printf("{\"isa\":\"%s\",\"threads\":%zu,\"prompt_tokens\":",
        neuron::kernels::selected_isa().data(), neuron::kernels::threads()); ids(tokens); std::printf(",\"generated_tokens\":"); ids(generated);
    std::printf(",\"top2\":[");
    for (std::size_t i = 0; i < top2.size(); ++i)
      std::printf("%s{\"first\":%d,\"second\":%d,\"gap\":%.9g}", i == 0 ? "" : ",", top2[i].first, top2[i].second, static_cast<double>(top2[i].gap));
    std::printf("]");
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
