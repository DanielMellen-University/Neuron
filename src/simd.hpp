#pragma once
#include <cstddef>
#include <cstdint>
#include <array>
#include <span>
#include <vector>

namespace neuron::detail {
using Dot = double (*)(const float*, const float*, std::size_t);
using QuantDot = double (*)(const std::byte*, std::uint32_t, const float*, std::size_t);
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
