#pragma once
#include "simd.hpp"
#include "neuron/quant.hpp"

namespace neuron::detail {
inline bool k_needs_direct(const std::byte* w, std::uint32_t type, const KActivation& x) {
  if (type != q4_k) {
    std::uint16_t d;
    std::memcpy(&d, w + 208, 2);
    return (d & 0x7fffu) > x.max_scale;
  }
  std::uint32_t halves;
  std::memcpy(&halves, w, 4);
  const unsigned d = halves & 0x7fffu, m = (halves >> 16) & 0x7fffu;
  // F16 significands have at most 11 bits. Products by 63*15 and 63
  // are exact in FP32. At a common quantum, their difference also fits
  // 24 bits for these exponent gaps: 2047*(945*8+63) < 2^24 and
  // 2047*(945+63*64) < 2^24. Subnormal halves use exponent 1's quantum,
  // bringing the actual gap closer to zero than the encoded gap.
  // Comparing the ordered half encodings avoids extracting both exponents;
  // this slightly narrower interval still guarantees gaps within [-6, 3].
  return d > x.max_scale || d + 6u * 1024u - m > 9u * 1024u;
}
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
