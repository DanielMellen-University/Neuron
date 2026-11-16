#include "neuron/model.hpp"
#include <chrono>
#include <cstdio>
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  neuron::Model model(argv[1], 2048);
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < 2048; ++i) {
    model.step(16, i == 2047);
    if ((i + 1) % 32 == 0) {
      const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      std::fprintf(stderr, "positions=%d elapsed=%.3f\n", i + 1, elapsed);
    }
  }
  std::printf("context=%zu position=%zu\n", model.config().context, model.position());
}
