#pragma once
#include "simd.hpp"
#include "neuron/quant.hpp"

namespace neuron::detail {
inline int k_scale(const std::byte* w, std::size_t group, bool minimum = false) {
  const auto* s = w + 4;
  auto b = [s](std::size_t i) { return std::to_integer<unsigned>(s[i]); };
  if (minimum) return static_cast<int>(group < 4 ? b(group + 4) & 63u :
      (b(group + 4) >> 4) | ((b(group) >> 6) << 4));
  return static_cast<int>(group < 4 ? b(group) & 63u :
      (b(group + 4) & 15u) | ((b(group - 4) >> 6) << 4));
}
inline int k_signed_scale(const std::byte* w, std::size_t group) {
  std::int8_t value;
  std::memcpy(&value, w + 192 + group, 1);
  return value;
}
inline double k_offset(const std::byte* w, const KActivation& x) {
  const float m = load_half(w + 2);
  double sum = 0;
  for (std::size_t g = 0; g < 8; ++g)
    sum += static_cast<double>(m * static_cast<float>(k_scale(w, g, true))) * x.sums[g];
  return sum;
}
}  // namespace neuron::detail
