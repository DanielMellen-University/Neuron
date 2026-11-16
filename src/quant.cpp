#include "neuron/quant.hpp"
#include "neuron/gguf.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace neuron {
namespace {
// Round a finite scale to nearest F16, ties to even. Reject overflow.
std::uint16_t half(float value) {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  const auto sign = (bits >> 16) & 0x8000u;
  const auto magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x477fe000u) throw std::invalid_argument("quantize: scale exceeds F16 range");
  if (magnitude < 0x33000000u) return static_cast<std::uint16_t>(sign);
  const auto exponent = magnitude >> 23;
  if (exponent < 113) {
    const auto mantissa = (magnitude & 0x7fffffu) | 0x800000u;
    const auto shift = 126u - exponent;
    const auto rounded = (mantissa + ((1u << (shift - 1)) - 1) + ((mantissa >> shift) & 1u)) >> shift;
    return static_cast<std::uint16_t>(sign | rounded);
  }
  const auto rounded = magnitude + 0xfffu + ((magnitude >> 13) & 1u);
  return static_cast<std::uint16_t>(sign | ((rounded - 0x38000000u) >> 13));
}
}
std::size_t quant_block_bytes(std::uint32_t type) {
  if (type == q8_0) return 34;
  if (type == q4_0) return 18;
  throw std::invalid_argument("quantization: expected Q8_0 or Q4_0");
}
void dequantize(std::span<const std::byte> blocks, std::uint32_t type, std::span<float> output) {
  const auto size = quant_block_bytes(type);
  if (output.size() % 32 != 0 || blocks.size() / size != output.size() / 32 || blocks.size() % size != 0)
    throw std::invalid_argument("dequantize: invalid block shape");
  for (std::size_t b = 0; b < output.size() / 32; ++b) {
    const auto* block = blocks.data() + b * size;
    std::uint16_t bits;
    std::memcpy(&bits, block, 2);
    const float scale = f16_to_f32(bits);
    for (std::size_t j = 0; j < 32; ++j) {
      int code;
      if (type == q8_0) {
        std::int8_t value;
        std::memcpy(&value, block + 2 + j, 1);
        code = value;
      } else {
        const auto packed = std::to_integer<unsigned>(block[2 + j % 16]);
        code = static_cast<int>((packed >> (j < 16 ? 0 : 4)) & 15u) - 8;
      }
      output[b * 32 + j] = scale * static_cast<float>(code);
    }
  }
}
void quantize_block(std::span<const float> input, std::uint32_t type, std::span<std::byte> output) {
  if (input.size() != 32 || output.size() != quant_block_bytes(type))
    throw std::invalid_argument("quantize: expected one block of 32 weights");
  float maximum = 0;
  for (float x : input) {
    if (!std::isfinite(x)) throw std::invalid_argument("quantize: nonfinite weight");
    if (std::abs(x) > std::abs(maximum)) maximum = x;
  }
  const float scale = type == q8_0 ? std::abs(maximum) / 127 : maximum / -8;
  const auto bits = half(scale);
  std::memcpy(output.data(), &bits, 2);
  std::fill(output.begin() + 2, output.end(), std::byte{0});
  for (std::size_t j = 0; j < 32; ++j) {
    const float scaled = scale == 0 ? 0 : input[j] / scale;
    if (type == q8_0) {
      const auto code = static_cast<std::int8_t>(std::clamp(std::round(scaled), -127.0f, 127.0f));
      std::memcpy(output.data() + 2 + j, &code, 1);
    } else {
      const auto code = static_cast<unsigned>(std::clamp(std::floor(scaled + 8.5f), 0.0f, 15.0f));
      output[2 + j % 16] |= static_cast<std::byte>(code << (j < 16 ? 0 : 4));
    }
  }
}
}  // namespace neuron
