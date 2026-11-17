#pragma once
#include <cstddef>
#include <cstdint>
#include <array>
#include <span>
#include <vector>
#include <bit>
#include <cstring>
#include "neuron/gguf.hpp"

namespace neuron::detail {
// The normal F16 case is exact bit expansion. Keep subnormal/Inf/NaN
// handling in the shared converter without a call for every weight group.
inline float load_half(const std::byte* p) {
  std::uint16_t h;
  std::memcpy(&h, p, sizeof(h));
  if ((h & 0x7c00u) == 0 || (h & 0x7c00u) == 0x7c00u) return f16_to_f32(h);
  return std::bit_cast<float>(((static_cast<std::uint32_t>(h) & 0x8000u) << 16) |
      (((static_cast<std::uint32_t>(h) & 0x7fffu) + 0x1c000u) << 13));
}
using Dot = double (*)(const float*, const float*, std::size_t);
using QuantDot = double (*)(const std::byte*, std::uint32_t, const float*, std::size_t);
struct alignas(64) KActivation {
  std::array<std::int16_t, 256> low{}, high{};
  std::array<double, 8> sums{};
  double scale = 0;
  std::uint32_t max_scale = 0;
  const float* original = nullptr;
};
// Signed 31-bit fixed point, split into two signed word planes. For a
// finite block, absolute rounding error is <= 2^(frexp(max_abs)-32).
// Retain the input (which must outlive the packed data) for guarded direct dots.
bool prepare_k_activations(std::span<const float> x, std::vector<KActivation>& out);
using KQuantDot = double (*)(const std::byte*, std::uint32_t, const KActivation*, std::size_t);
double kquant_avx2(const std::byte* w, std::uint32_t type, const KActivation* x, std::size_t n);
double kquant_avx512(const std::byte* w, std::uint32_t type, const KActivation* x, std::size_t n);
double kquant_vnni(const std::byte* w, std::uint32_t type, const KActivation* x, std::size_t n);
double dot_avx2(const float* a, const float* b, std::size_t n);
double quant_avx2(const std::byte* w, std::uint32_t type, const float* x, std::size_t n);
double dot_avx512(const float* a, const float* b, std::size_t n);
double quant_avx512(const std::byte* w, std::uint32_t type, const float* x, std::size_t n);
struct Activation {
  std::array<std::array<std::uint8_t, 16>, 4> planes{};
  double scale = 0, sum = 0;
};
std::vector<Activation> prepare_activations(std::span<const float> x);
double quant_vnni(const std::byte* w, std::uint32_t type, const Activation* x, std::size_t n);
std::int64_t dot_i8_avx2(const std::uint8_t* a, const std::int8_t* b, std::size_t n);
std::int64_t dot_i8_avx512(const std::uint8_t* a, const std::int8_t* b, std::size_t n);
std::int64_t dot_i8_vnni(const std::uint8_t* a, const std::int8_t* b, std::size_t n);
}  // namespace neuron::detail
