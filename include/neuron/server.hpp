#pragma once

#include "neuron/completion.hpp"
#include "neuron/sampling.hpp"

#include <cstdint>
#include <functional>
#include <string>

namespace neuron {
// The accept loop serializes all model access. Each connection has one request.
class HttpServer {
 public:
  HttpServer(ChatBackend& backend, std::string model, const std::string& host = "127.0.0.1",
             std::uint16_t port = 8080);
  ~HttpServer();
  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;
  std::uint16_t port() const { return port_; }
  // Settings for requests that leave a field out. The default is greedy. The
  // sampling seed starts the stream of per-request seeds used when a sampled
  // request has no seed of its own, so a fixed one makes a run repeatable.
  void set_defaults(const GenerationOptions& defaults);
  void run(const std::function<bool()>& stopped = [] { return false; });
 private:
  void handle(int client, const std::function<bool()>& stopped);
  ChatBackend& backend_;
  std::string model_;
  int socket_ = -1;
  std::uint16_t port_ = 0;
  std::uint64_t sequence_ = 0;
  GenerationOptions defaults_;
  Random seeds_{0};
};
}  // namespace neuron
