#include "neuron/kernels.hpp"
#include "neuron/quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
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
        // Arbitrary code and scale bytes cover patterns the reference encoder
        // does not normally emit, including signed -128 Q8/Q6 values.
        for (auto& v : weights) v = static_cast<std::byte>(random() & 255u);
        for (std::size_t i = 0; i < rows * blocks; ++i) {
          const std::uint16_t d = 0x2001, m = 0x1e31;
          auto* p = weights.data() + i * bytes;
          std::memcpy(p + (type == neuron::q6_k ? 208 : 0), &d, 2);
          if (type == neuron::q4_k) std::memcpy(p + 2, &m, 2);
        }
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
    for (std::size_t n : {0u, 1u, 15u, 16u, 17u, 31u, 32u, 63u, 64u, 65u, 4097u}) {
      std::vector<std::uint8_t> a(n);
      std::vector<std::int8_t> b(n);
      for (std::size_t i = 0; i < n; ++i) {
        a[i] = static_cast<std::uint8_t>(random() % 256);
        b[i] = static_cast<std::int8_t>(static_cast<int>(random() % 256) - 128);
      }
      if (neuron::scalar::dot_i8(a, b) != neuron::kernels::dot_i8(a, b))
        throw std::runtime_error("int8 dot differs from scalar");
      std::fill(a.begin(), a.end(), 255);
      std::fill(b.begin(), b.end(), -128);
      if (neuron::scalar::dot_i8(a, b) != neuron::kernels::dot_i8(a, b))
        throw std::runtime_error("int8 extremes differ from scalar");
      comparisons += 2;
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
