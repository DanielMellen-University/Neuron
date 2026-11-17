#include "simd.hpp"

#include <algorithm>
#include <cmath>

namespace neuron::detail {
bool prepare_k_activations(std::span<const float> x, std::vector<KActivation>& out) {
  out.resize(x.size() / 256);
  for (std::size_t b = 0; b < out.size(); ++b) {
    auto& a = out[b];
    a.sums.fill(0);
    float maximum = 0;
    for (std::size_t j = 0; j < 256; ++j) {
      const float v = x[b * 256 + j];
      if (!std::isfinite(v)) return false;
      maximum = std::max(maximum, std::abs(v));
      a.sums[j / 32] += v;
    }
    int exponent = 0;
    std::frexp(maximum, &exponent);
    a.scale = std::ldexp(1.0, exponent - 31);
    for (std::size_t j = 0; j < 256; ++j) {
      const auto value = static_cast<std::int32_t>(std::llround(static_cast<double>(x[b * 256 + j]) / a.scale));
      const auto bits = std::bit_cast<std::uint32_t>(value);
      a.low[j] = static_cast<std::int16_t>(static_cast<int>(bits & 65535u) - 32768);
      a.high[j] = static_cast<std::int16_t>(value >> 16);
    }
  }
  return true;
}
}  // namespace neuron::detail
