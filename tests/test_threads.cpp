#include "neuron/kernels.hpp"
#include "neuron/quant.hpp"

#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

int main() {
  try {
    std::mt19937 random(5012);
    std::uniform_real_distribution<float> uniform(-1, 1);
    std::size_t comparisons = 0;
    for (const auto* isa : {"scalar", "avx2", "avx512", "vnni"}) {
      if (!neuron::kernels::supported(isa)) { std::printf("SKIP %s threads\n", isa); continue; }
      neuron::kernels::set_isa(isa);
      for (auto type : {0u, neuron::q8_0, neuron::q4_0, neuron::q4_k, neuron::q6_k}) {
        for (std::size_t rows : {3u, 257u}) {
          const std::size_t cols = type == 0 ? 513 : 3 * neuron::quant_block_elements(type), batch = 2;
          std::vector<float> w(rows * cols), x(batch * cols), reference(batch * rows), actual(batch * rows);
          for (auto& v : w) v = uniform(random);
          for (auto& v : x) v = uniform(random);
          std::vector<std::byte> packed;
          if (type != 0) {
            const auto block = neuron::quant_block_elements(type), bytes = neuron::quant_block_bytes(type);
            packed.resize(rows * cols / block * bytes);
            for (std::size_t i = 0; i < rows * cols / block; ++i)
              neuron::quantize_block(std::span(w).subspan(i * block, block), type, std::span(packed).subspan(i * bytes, bytes));
          }
          auto multiply = [&](auto& output) {
            if (type == 0) neuron::kernels::matmul(w, x, output, rows, cols, batch);
            else neuron::kernels::matmul_quantized(packed, type, x, output, rows, cols, batch);
          };
          neuron::kernels::set_threads(1);
          multiply(reference);
          for (std::size_t threads : {2u, 4u, 8u}) {
            neuron::kernels::set_threads(threads);
            // Repeated jobs detect stale work and incomplete barriers.
            for (int repeat = 0; repeat < 3; ++repeat) {
              multiply(actual);
              if (actual != reference) throw std::runtime_error("threaded result is not bit-identical");
              comparisons += actual.size();
            }
          }
        }
      }
      std::printf("%s: all 1/2/4/8-thread outputs bit-identical\n", isa);
    }
    neuron::kernels::set_threads(1);
    bool rejected = false;
    try { neuron::kernels::set_threads(0); } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("zero threads accepted");
    std::printf("%zu threaded comparisons passed\n", comparisons);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
