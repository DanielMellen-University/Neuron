#pragma once
#include <cstddef>
#include <cstdint>

namespace neuron::detail {
using Dot = double (*)(const float*, const float*, std::size_t);
using QuantDot = double (*)(const std::byte*, std::uint32_t, const float*, std::size_t);
double dot_avx2(const float* a, const float* b, std::size_t n);
double quant_avx2(const std::byte* w, std::uint32_t type, const float* x, std::size_t n);
double dot_avx512(const float* a, const float* b, std::size_t n);
double quant_avx512(const std::byte* w, std::uint32_t type, const float* x, std::size_t n);
}  // namespace neuron::detail
