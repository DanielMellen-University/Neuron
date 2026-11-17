#include "simd.hpp"

#include <algorithm>
#include <cmath>

namespace neuron::detail {
bool prepare_k_activations(std::span<const float> x, std::vector<KActivation>& out) {
  out.resize(x.size() / 256);
  for (std::size_t b = 0; b < out.size(); ++b) {
    auto& a = out[b];
    a.sums.fill(0);
    a.original = x.data() + b * 256;
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
    // Values with FP32 spacing at least the packing scale are exact.
    // Count only smaller nonzero values, each bounded by half a step.
    const auto exact_min = std::bit_cast<std::uint32_t>(std::ldexp(1.0f, exponent - 8));
    const auto small_limit = exact_min == 0 ? 0 : exact_min - 1;
    std::size_t small_values = 0;
    for (std::size_t j = 0; j < 256; ++j) {
      const auto magnitude = std::bit_cast<std::uint32_t>(x[b * 256 + j]) & 0x7fffffffu;
      small_values += magnitude - 1u < small_limit;
      const auto value = static_cast<std::int32_t>(std::llround(static_cast<double>(x[b * 256 + j]) / a.scale));
      const auto bits = std::bit_cast<std::uint32_t>(value);
      a.low[j] = static_cast<std::int16_t>(static_cast<int>(bits & 65535u) - 32768);
      a.high[j] = static_cast<std::int16_t>(value >> 16);
    }
    // Precompute the largest safe F16 block scale, so each output row only
    // needs an integer comparison. 4096 bounds both K coefficient formats;
    // Q4 minima already use sums of the original activations.
    const double rounding_bound = a.scale * 0.5 * static_cast<double>(small_values);
    const double limit = rounding_bound == 0 ? 65504 : 1e-5 / (4096 * rounding_bound);
    if (limit >= 65504) a.max_scale = 0x7bff;
    else if (limit < 0x1p-14) a.max_scale = static_cast<std::uint16_t>(std::ldexp(limit, 24));
    else {
      const auto bits = std::bit_cast<std::uint32_t>(static_cast<float>(limit));
      a.max_scale = static_cast<std::uint16_t>((bits >> 13) - 0x1c000u);
      if (f16_to_f32(static_cast<std::uint16_t>(a.max_scale)) > limit) --a.max_scale;
    }
  }
  return true;
}
}  // namespace neuron::detail
