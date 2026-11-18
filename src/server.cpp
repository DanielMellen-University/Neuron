#include "neuron/server.hpp"
#include "neuron/completion.hpp"
#include "json.hpp"
#include "web_chat.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <map>
#include <stdexcept>
#include <utility>

namespace neuron {
namespace {
using json::quote;
struct HttpError : std::runtime_error {
  int status;
  std::string code, param;
  HttpError(int status_value, const std::string& message, std::string code_value = "invalid_request_error", std::string param_value = {})
      : std::runtime_error(message), status(status_value), code(std::move(code_value)), param(std::move(param_value)) {}
};
struct Disconnected {};
struct Socket {
  int fd;
  ~Socket() { if (fd >= 0) close(fd); }
};
void write_all(int fd, std::string_view text) {
  while (!text.empty()) {
    const auto n = send(fd, text.data(), text.size(), MSG_NOSIGNAL);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) throw Disconnected{};
    text.remove_prefix(static_cast<std::size_t>(n));
  }
}
std::string error_json(const HttpError& e) {
  return "{\"error\":{\"message\":" + quote(e.what()) + ",\"type\":" +
      quote(e.status == 500 ? "server_error" : "invalid_request_error") + ",\"param\":" +
      (e.param.empty() ? "null" : quote(e.param)) + ",\"code\":" + quote(e.code) + "}}";
}
void response(int fd, int status, std::string_view content, std::string_view type = "application/json") {
  const std::string reason = status == 200 ? "OK" : status == 400 ? "Bad Request" : status == 404 ? "Not Found" :
      status == 405 ? "Method Not Allowed" : status == 408 ? "Request Timeout" : status == 413 ? "Content Too Large" :
      status == 415 ? "Unsupported Media Type" : status == 431 ? "Request Header Fields Too Large" : "Internal Server Error";
  write_all(fd, "HTTP/1.1 " + std::to_string(status) + " " + reason + "\r\nContent-Type: " + std::string(type) +
      "\r\nContent-Length: " + std::to_string(content.size()) +
      "\r\nConnection: close\r\nX-Content-Type-Options: nosniff\r\n\r\n");
  write_all(fd, content);
}
std::string lower(std::string text) {
  for (auto& c : text) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + ('a' - 'A'));
  return text;
}
std::string trim(std::string_view text) {
  const auto start = text.find_first_not_of(" \t");
  if (start == std::string_view::npos) return {};
  return std::string(text.substr(start, text.find_last_not_of(" \t") - start + 1));
}
struct Request { std::string method, path, body; std::map<std::string, std::string> headers; };
Request read_request(int fd, const std::function<bool()>& stopped) {
  constexpr std::size_t max_header = 16384, max_body = 1048576;
  std::string bytes;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  auto read_more = [&] {
    for (;;) {
      if (stopped()) throw Disconnected{};
      if (std::chrono::steady_clock::now() >= deadline) throw HttpError(408, "request timed out");
      pollfd p{fd, POLLIN, 0};
      const auto ready = poll(&p, 1, 100);
      if (ready < 0 && errno == EINTR) continue;
      if (ready < 0) throw Disconnected{};
      if (ready == 0) continue;
      char buffer[4096];
      const auto n = recv(fd, buffer, sizeof(buffer), 0);
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) throw HttpError(400, "incomplete HTTP request");
      bytes.append(buffer, static_cast<std::size_t>(n));
      return;
    }
  };
  std::size_t end = 0;
  for (;;) {
    end = bytes.find("\r\n\r\n");
    if (end != std::string::npos) break;
    if (bytes.size() >= max_header) throw HttpError(431, "HTTP headers exceed 16 KiB");
    read_more();
  }
  if (end + 4 > max_header) throw HttpError(431, "HTTP headers exceed 16 KiB");
  const auto line_end = bytes.find("\r\n");
  const auto first = bytes.find(' '), second = first == std::string::npos ? first : bytes.find(' ', first + 1);
  if (first == std::string::npos || second == std::string::npos || second >= line_end ||
      bytes.substr(second + 1, line_end - second - 1) != "HTTP/1.1") throw HttpError(400, "expected HTTP/1.1 request line");
  Request request{bytes.substr(0, first), bytes.substr(first + 1, second - first - 1), {}, {}};
  if (request.path.empty() || request.path.front() != '/') throw HttpError(400, "invalid request target");
  for (std::size_t pos = line_end + 2; pos < end;) {
    const auto next = bytes.find("\r\n", pos), colon = bytes.find(':', pos);
    if (colon == std::string::npos || colon >= next || colon == pos) throw HttpError(400, "invalid HTTP header");
    auto name = lower(bytes.substr(pos, colon - pos));
    if (name.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789!#$%&'*+-.^_`|~") != std::string::npos)
      throw HttpError(400, "invalid HTTP header name");
    auto value = trim(std::string_view(bytes).substr(colon + 1, next - colon - 1));
    for (const char c : value) if ((static_cast<unsigned char>(c) < 32 && c != '\t') || c == 127) throw HttpError(400, "invalid HTTP header value");
    if (!request.headers.emplace(name, value).second) throw HttpError(400, "duplicate HTTP header");
    pos = next + 2;
  }
  if (!request.headers.contains("host")) throw HttpError(400, "Host header is required");
  if (request.headers.contains("transfer-encoding")) throw HttpError(400, "Transfer-Encoding is not supported; send Content-Length");
  if (request.headers.contains("expect")) throw HttpError(400, "Expect is not supported");
  std::size_t length = 0;
  if (const auto it = request.headers.find("content-length"); it != request.headers.end()) {
    const auto& text = it->second;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), length);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || text.empty()) throw HttpError(400, "invalid Content-Length");
  } else if (request.method == "POST") throw HttpError(400, "Content-Length is required");
  if (length > max_body) throw HttpError(413, "request body exceeds 1 MiB");
  const auto body_start = end + 4;
  while (bytes.size() - body_start < length) read_more();
  request.body = bytes.substr(body_start, length);
  return request;
}
const std::string& string_value(const json::Value* value, const std::string& name) {
  if (!value || !std::holds_alternative<std::string>(value->data)) throw HttpError(400, name + " must be a string", "invalid_request_error", name);
  return std::get<std::string>(value->data);
}
double number_value(const json::Value& value, const std::string& name) {
  if (!std::holds_alternative<double>(value.data)) throw HttpError(400, name + " must be a number");
  return std::get<double>(value.data);
}
bool bool_value(const json::Value& value, const std::string& name) {
  if (!std::holds_alternative<bool>(value.data)) throw HttpError(400, name + " must be a boolean");
  return std::get<bool>(value.data);
}
std::string shown(double value) {
  char text[32];
  std::snprintf(text, sizeof(text), "%.17g", value);
  return text;
}
double bounded(const json::Value& value, const std::string& name, double low, double high) {
  const double n = number_value(value, name);
  if (!(n >= low && n <= high)) throw HttpError(400, name + " must be in " + shown(low) + ".." + shown(high), "invalid_request_error", name);
  return n;
}
std::vector<std::string> stop_value(const json::Value& value) {
  if (std::holds_alternative<std::nullptr_t>(value.data)) return {};
  std::vector<std::string> stops;
  if (std::holds_alternative<std::string>(value.data)) stops.push_back(std::get<std::string>(value.data));
  else if (std::holds_alternative<json::Value::Array>(value.data)) {
    for (const auto& item : std::get<json::Value::Array>(value.data)) stops.push_back(string_value(&item, "stop"));
    if (stops.size() > 4) throw HttpError(400, "stop accepts at most 4 strings", "invalid_request_error", "stop");
  } else throw HttpError(400, "stop must be a string or an array of strings", "invalid_request_error", "stop");
  for (const auto& stop : stops) if (stop.empty()) throw HttpError(400, "stop strings must not be empty", "invalid_request_error", "stop");
  return stops;
}
bool disconnected(int fd) {
  char c;
  const auto n = recv(fd, &c, 1, MSG_PEEK | MSG_DONTWAIT);
  // n == 0 is only a half-close (the peer may still read the reply), so it is not a disconnect.
  return n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR;
}
}
HttpServer::HttpServer(ChatBackend& backend, std::string model, const std::string& host, std::uint16_t port)
    : backend_(backend), model_(std::move(model)) {
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(port);
  if (inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1) throw std::invalid_argument("--host requires an IPv4 address");
  const Socket pending{socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0)};
  if (pending.fd < 0) throw std::runtime_error("cannot create HTTP socket");
  const int reuse = 1;
  setsockopt(pending.fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
  if (bind(pending.fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0 || listen(pending.fd, 16) != 0)
    throw std::runtime_error("cannot bind/listen on " + host + ":" + std::to_string(port));
  socklen_t size = sizeof(address);
  if (getsockname(pending.fd, reinterpret_cast<sockaddr*>(&address), &size) != 0) throw std::runtime_error("cannot read HTTP port");
  port_ = ntohs(address.sin_port);
  socket_ = dup(pending.fd);
  if (socket_ < 0) throw std::runtime_error("cannot retain HTTP socket");
}
HttpServer::~HttpServer() { if (socket_ >= 0) close(socket_); }
void HttpServer::set_defaults(const GenerationOptions& defaults) {
  defaults.sampling.validate();
  defaults_ = defaults;
  seeds_ = Random(defaults.sampling.seed);
}
void HttpServer::run(const std::function<bool()>& stopped) {
  while (!stopped()) {
    pollfd p{socket_, POLLIN, 0};
    const auto ready = poll(&p, 1, 100);
    if (ready < 0 && errno == EINTR) continue;
    if (ready < 0) throw std::runtime_error("HTTP accept poll failed");
    if (ready == 0) continue;
    const Socket client{accept4(socket_, nullptr, nullptr, SOCK_CLOEXEC)};
    if (client.fd < 0 && errno == EINTR) continue;
    if (client.fd < 0) throw std::runtime_error("HTTP accept failed");
    const timeval timeout{10, 0};
    setsockopt(client.fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    handle(client.fd, stopped);
    // Drain already queued input after an early error so close does not reset
    // the connection before the client reads the HTTP error response.
    shutdown(client.fd, SHUT_WR);
    char discard[4096];
    for (std::size_t drained = 0; drained < 65536;) {
      const auto n = recv(client.fd, discard, sizeof(discard), MSG_DONTWAIT);
      if (n <= 0) break;
      drained += static_cast<std::size_t>(n);
    }
  }
}
void HttpServer::handle(int client, const std::function<bool()>& stopped) {
  bool streaming = false;
  try {
    const auto request = read_request(client, stopped);
    if (request.path == "/" || request.path == "/v1/models") {
      if (request.method != "GET") throw HttpError(405, "use GET for this resource");
      if (request.path == "/") response(client, 200, web_chat, "text/html; charset=utf-8");
      else response(client, 200, "{\"object\":\"list\",\"data\":[{\"id\":" + quote(model_) +
          ",\"object\":\"model\",\"created\":0,\"owned_by\":\"neuron\"}]}");
      return;
    }
    if (request.path != "/v1/chat/completions") throw HttpError(404, "unknown resource", "not_found");
    if (request.method != "POST") throw HttpError(405, "use POST for chat completions");
    const auto ct = request.headers.find("content-type");
    if (ct == request.headers.end() || lower(trim(std::string_view(ct->second).substr(0, ct->second.find(';')))) != "application/json")
      throw HttpError(415, "Content-Type must be application/json");
    json::Value root;
    try { root = json::parse(request.body); }
    catch (const std::invalid_argument&) { throw HttpError(400, "invalid JSON", "invalid_json"); }
    if (!std::holds_alternative<json::Value::Object>(root.data)) throw HttpError(400, "request must be a JSON object");
    if (string_value(root.find("model"), "model") != model_) throw HttpError(404, "unknown model; use GET /v1/models", "model_not_found", "model");
    const auto* list = root.find("messages");
    if (!list || !std::holds_alternative<json::Value::Array>(list->data) || std::get<json::Value::Array>(list->data).empty())
      throw HttpError(400, "messages must be a nonempty array");
    std::vector<ChatMessage> messages;
    for (const auto& item : std::get<json::Value::Array>(list->data)) {
      const auto role = string_value(item.find("role"), "role");
      if (role != "system" && role != "user" && role != "assistant") throw HttpError(400, "unsupported message role");
      messages.push_back({role, string_value(item.find("content"), "content")});
    }
    GenerationOptions generation = defaults_;
    auto& max_tokens = generation.max_tokens;
    const auto* maximum = root.find("max_completion_tokens");
    if (maximum && root.find("max_tokens")) throw HttpError(400, "use only one maximum token field");
    if (!maximum) maximum = root.find("max_tokens");
    if (maximum) {
      const double n = number_value(*maximum, "max_tokens");
      if (n < 1 || n > 1048576 || std::floor(n) != n) throw HttpError(400, "max_tokens must be an integer in 1..1048576");
      max_tokens = static_cast<std::size_t>(n);
    }
    for (const auto& [name, expected] : std::map<std::string, double>{{"n", 1}, {"presence_penalty", 0}, {"frequency_penalty", 0}})
      if (const auto* v = root.find(name); v && number_value(*v, name) != expected) throw HttpError(400, "only " + name + "=" + shown(expected) + " is supported");
    // OpenAI fields, plus the llama.cpp server extensions top_k, min_p and repeat_penalty.
    auto& sampling = generation.sampling;
    auto present = [&](const char* name) {
      const auto* v = root.find(name);
      return v && !std::holds_alternative<std::nullptr_t>(v->data) ? v : nullptr;
    };
    if (const auto* v = present("temperature")) sampling.temperature = static_cast<float>(bounded(*v, "temperature", 0, 2));
    if (const auto* v = present("top_p")) {
      sampling.top_p = static_cast<float>(bounded(*v, "top_p", 0, 1));
      if (sampling.top_p == 0) throw HttpError(400, "top_p must be above 0", "invalid_request_error", "top_p");
    }
    if (const auto* v = present("min_p")) {
      sampling.min_p = static_cast<float>(bounded(*v, "min_p", 0, 1));
      if (sampling.min_p == 1) throw HttpError(400, "min_p must be below 1", "invalid_request_error", "min_p");
    }
    if (const auto* v = present("top_k")) {
      const double k = bounded(*v, "top_k", 0, 1048576);
      if (std::floor(k) != k) throw HttpError(400, "top_k must be an integer", "invalid_request_error", "top_k");
      sampling.top_k = static_cast<std::size_t>(k);
    }
    if (const auto* v = present("repeat_penalty")) {
      sampling.repeat_penalty = static_cast<float>(bounded(*v, "repeat_penalty", 0, 100));
      if (sampling.repeat_penalty == 0) throw HttpError(400, "repeat_penalty must be above 0", "invalid_request_error", "repeat_penalty");
    }
    // Draw from the seed stream for every request, so the sequence of seeds
    // depends only on the server's starting seed and request order.
    const auto drawn = seeds_.next();
    if (const auto* v = present("seed")) {
      const double seed = bounded(*v, "seed", 0, 9007199254740992.0);
      if (std::floor(seed) != seed) throw HttpError(400, "seed must be an integer", "invalid_request_error", "seed");
      sampling.seed = static_cast<std::uint64_t>(seed);
    } else sampling.seed = drawn;
    if (const auto* v = root.find("stop")) generation.stop = stop_value(*v);
    for (const auto* name : {"tools", "tool_choice", "functions", "function_call", "response_format", "logit_bias"})
      if (const auto* v = root.find(name); v && !std::holds_alternative<std::nullptr_t>(v->data)) throw HttpError(400, std::string(name) + " is not supported");
    if (const auto* v = root.find("logprobs"); v && bool_value(*v, "logprobs")) throw HttpError(400, "logprobs is not supported");
    bool stream = false, include_usage = false;
    if (const auto* v = root.find("stream")) stream = bool_value(*v, "stream");
    if (const auto* options = root.find("stream_options")) {
      if (!std::holds_alternative<json::Value::Object>(options->data)) throw HttpError(400, "stream_options must be an object");
      if (const auto* v = options->find("include_usage")) include_usage = bool_value(*v, "include_usage");
    }
    std::size_t prompt_tokens = 0;
    try { prompt_tokens = backend_.prepare(messages, generation); }
    catch (const std::length_error& e) { throw HttpError(400, e.what(), "context_length_exceeded", "messages"); }
    const auto id = "chatcmpl-neuron-" + std::to_string(std::time(nullptr)) + "-" + std::to_string(++sequence_);
    const std::string base = "{\"id\":" + quote(id) + ",\"created\":" + std::to_string(std::time(nullptr)) + ",\"model\":" + quote(model_);
    const auto cancelled = [&] { return stopped() || disconnected(client); };
    auto usage = [&](std::size_t tokens) {
      return "{\"prompt_tokens\":" + std::to_string(prompt_tokens) + ",\"completion_tokens\":" + std::to_string(tokens) +
          ",\"total_tokens\":" + std::to_string(prompt_tokens + tokens) + "}";
    };
    if (!stream) {
      const auto result = backend_.generate([](std::string_view) { return true; }, cancelled);
      if (cancelled()) return;
      response(client, 200, base + ",\"object\":\"chat.completion\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":" +
          quote(result.text) + "},\"finish_reason\":" + quote(result.finish_reason) + "}],\"usage\":" + usage(result.tokens) + "}");
      return;
    }
    write_all(client, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nX-Accel-Buffering: no\r\n\r\n");
    streaming = true;
    const auto chunk = [&](const std::string& delta, const std::string& finish) {
      write_all(client, "data: " + base + ",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":" +
          delta + ",\"finish_reason\":" + finish + "}]}\n\n");
    };
    chunk("{\"role\":\"assistant\",\"content\":\"\"}", "null");
    const auto result = backend_.generate([&](std::string_view piece) {
      chunk("{\"content\":" + quote(piece) + "}", "null"); return !cancelled();
    }, cancelled);
    if (cancelled()) return;
    chunk("{}", quote(result.finish_reason));
    if (include_usage) write_all(client, "data: " + base + ",\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":" + usage(result.tokens) + "}\n\n");
    write_all(client, "data: [DONE]\n\n");
  } catch (const Disconnected&) {
  } catch (const HttpError& e) {
    try { response(client, e.status, error_json(e)); } catch (const Disconnected&) {}
  } catch (const std::exception&) {
    const auto error = error_json(HttpError(500, "inference failed", "internal_error"));
    try {
      if (streaming) write_all(client, "data: " + error + "\n\ndata: [DONE]\n\n");
      else response(client, 500, error);
    } catch (const Disconnected&) {}
  }
}
}  // namespace neuron
