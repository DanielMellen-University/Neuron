#include "neuron/kernels.hpp"
#include "neuron/quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  try {
    const std::string isa = argc > 1 ? argv[1] : "avx2";
    if (!neuron::kernels::supported(isa)) {
      std::printf("SKIP: %s unavailable on this CPU/build\n", isa.c_str());
      return 77;
    }
    neuron::kernels::set_isa(isa);
    std::mt19937 random(5401);
    std::uniform_real_distribution<float> uniform(-1, 1);
    double worst = 0;
    std::size_t comparisons = 0;
    auto compare = [&](const auto& a, const auto& b) {
      for (std::size_t i = 0; i < a.size(); ++i) {
        const double error = std::abs(static_cast<double>(a[i]) - b[i]);
        worst = std::max(worst, error);
        if (!std::isfinite(b[i]) || error > 1e-5 * (1 + std::abs(static_cast<double>(a[i]))))
          throw std::runtime_error("SIMD differs from scalar");
        ++comparisons;
      }
    };
    for (std::size_t cols : {1u, 3u, 7u, 8u, 15u, 16u, 17u, 31u, 33u, 129u, 1536u}) {
      for (std::size_t rows : {1u, 3u, 19u}) {
        std::vector<float> w(rows * cols), x(3 * cols), a(3 * rows), b(3 * rows);
        for (auto& v : w) v = uniform(random);
        for (auto& v : x) v = uniform(random);
        neuron::scalar::matmul(w, x, a, rows, cols, 3);
        neuron::kernels::matmul(w, x, b, rows, cols, 3);
        compare(a, b);
        const auto dot = neuron::kernels::dot(std::span(w).first(cols), std::span(x).first(cols));
        double ref = 0;
        for (std::size_t i = 0; i < cols; ++i) ref += static_cast<double>(w[i]) * x[i];
        if (std::abs(dot - ref) > 1e-10 * (1 + std::abs(ref)))
          throw std::runtime_error("dot tail differs from scalar");
      }
    }
    for (auto type : {neuron::q8_0, neuron::q4_0, neuron::q4_k, neuron::q6_k}) {
      const auto block = neuron::quant_block_elements(type), bytes = neuron::quant_block_bytes(type);
      for (std::size_t blocks : {1u, 3u, 7u}) {
        const std::size_t cols = block * blocks, rows = 19, batch = 3;
        std::vector<float> w(rows * cols), x(batch * cols), a(batch * rows), b(batch * rows);
        std::vector<std::byte> packed(rows * blocks * bytes + 1);
        for (auto& v : w) v = uniform(random);
        for (auto& v : x) v = uniform(random);
        // Unaligned blocks and an odd row count exercise load and row tails.
        auto weights = std::span(packed).subspan(1);
        for (std::size_t i = 0; i < rows * blocks; ++i)
          neuron::quantize_block(std::span(w).subspan(i * block, block), type, weights.subspan(i * bytes, bytes));
        neuron::scalar::matmul_quantized(weights, type, x, a, rows, cols, batch);
        neuron::kernels::matmul_quantized(weights, type, x, b, rows, cols, batch);
        compare(a, b);
        // Zero activations, cancellation and highly unequal magnitudes.
        for (std::size_t i = 0; i < x.size(); ++i) x[i] = i % 5 == 0 ? 0 : std::ldexp(uniform(random), -static_cast<int>(i % 40));
        neuron::scalar::matmul_quantized(weights, type, x, a, rows, cols, batch);
        neuron::kernels::matmul_quantized(weights, type, x, b, rows, cols, batch);
        compare(a, b);
      }
    }
    bool rejected = false;
    try { neuron::kernels::set_isa("invalid"); } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("unknown ISA accepted");
    std::printf("%s: %zu scalar comparisons passed, max absolute error %.9g\n", isa.c_str(), comparisons, worst);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
