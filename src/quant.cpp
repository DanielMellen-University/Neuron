#include "neuron/quant.hpp"
#include "neuron/gguf.hpp"

#include <algorithm>
#include <array>
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
float load_half(const std::byte* p) {
  std::uint16_t bits;
  std::memcpy(&bits, p, 2);
  return f16_to_f32(bits);
}
unsigned byte(const std::byte* p, std::size_t j) { return std::to_integer<unsigned>(p[j]); }
int signed_byte(const std::byte* p, std::size_t j) {
  std::int8_t x;
  std::memcpy(&x, p + j, 1);
  return x;
}
// Emit in logical weight order. Products and offsets round as in GGML's
// FP32 dequantizer before the scalar dot product accumulates them.
template <typename Emit>
void decode_k(const std::byte* p, std::uint32_t type, Emit emit) {
  if (type == q4_k) {
    const float d = load_half(p), m = load_half(p + 2);
    const auto* scales = p + 4;
    for (std::size_t g = 0; g < 8; ++g) {
      const auto sc = g < 4 ? byte(scales, g) & 63u :
          (byte(scales, g + 4) & 15u) | ((byte(scales, g - 4) >> 6) << 4);
      const auto mn = g < 4 ? byte(scales, g + 4) & 63u :
          (byte(scales, g + 4) >> 4) | ((byte(scales, g) >> 6) << 4);
      const float scale = d * static_cast<float>(sc), offset = m * static_cast<float>(mn);
      for (std::size_t j = 0; j < 32; ++j) {
        const auto q = (byte(p + 16, (g / 2) * 32 + j) >> (4 * (g % 2))) & 15u;
        emit(g * 32 + j, scale * static_cast<float>(q) - offset);
      }
    }
  } else {
    const float d = load_half(p + 208);
    for (std::size_t g = 0; g < 16; ++g) {
      const float scale = d * static_cast<float>(signed_byte(p + 192, g));
      for (std::size_t j = 0; j < 16; ++j) {
        const auto i = g * 16 + j, half_index = i % 128, lane = half_index % 32;
        const auto low = byte(p, (i / 128) * 64 + lane + (half_index / 32 % 2) * 32);
        const auto high = byte(p + 128, (i / 128) * 32 + lane);
        const int q = static_cast<int>(((low >> (half_index >= 64 ? 4 : 0)) & 15u) |
                                      (((high >> (2 * (half_index / 32))) & 3u) << 4)) - 32;
        emit(i, scale * static_cast<float>(q));
      }
    }
  }
}
void encode_k(std::span<const float> input, std::uint32_t type, std::span<std::byte> output) {
  std::fill(output.begin(), output.end(), std::byte{0});
  std::array<float, 16> scales{}, minima{};
  const std::size_t group_size = type == q4_k ? 32 : 16, groups = 256 / group_size;
  float max_scale = 0, max_min = 0;
  for (std::size_t g = 0; g < groups; ++g) {
    const auto values = input.subspan(g * group_size, group_size);
    if (type == q4_k) {
      const float lo = std::min(0.0f, *std::min_element(values.begin(), values.end()));
      const float hi = std::max(0.0f, *std::max_element(values.begin(), values.end()));
      scales[g] = (hi - lo) / 15; minima[g] = -lo;
    } else {
      const float maximum = *std::max_element(values.begin(), values.end(),
          [](float a, float b) { return std::abs(a) < std::abs(b); });
      scales[g] = maximum / -32;
    }
    max_scale = std::max(max_scale, std::abs(scales[g])); max_min = std::max(max_min, minima[g]);
  }
  const float d = max_scale / (type == q4_k ? 63 : 127), dm = max_min / 63;
  const auto h = half(d), hm = half(dm);
  std::memcpy(output.data() + (type == q4_k ? 0 : 208), &h, 2);
  if (type == q4_k) std::memcpy(output.data() + 2, &hm, 2);
  for (std::size_t g = 0; g < groups; ++g) {
    const int sc = d == 0 ? 0 : static_cast<int>(std::round(scales[g] / d));
    if (type == q4_k) {
      const auto a = static_cast<unsigned>(std::clamp(sc, 0, 63));
      const auto m = dm == 0 ? 0u : static_cast<unsigned>(std::clamp(std::round(minima[g] / dm), 0.0f, 63.0f));
      if (g < 4) { output[4 + g] |= static_cast<std::byte>(a); output[8 + g] |= static_cast<std::byte>(m); }
      else {
        output[g] |= static_cast<std::byte>((a >> 4) << 6);
        output[4 + g] |= static_cast<std::byte>((m >> 4) << 6);
        output[8 + g] = static_cast<std::byte>((a & 15u) | ((m & 15u) << 4));
      }
    } else {
      const auto a = static_cast<std::int8_t>(std::clamp(sc, -127, 127));
      std::memcpy(output.data() + 192 + g, &a, 1);
    }
    for (std::size_t j = 0; j < group_size; ++j) {
      const auto i = g * group_size + j;
      const float v = scales[g] == 0 ? 0 : (input[i] + minima[g]) / scales[g];
      if (type == q4_k) {
        const auto q = static_cast<unsigned>(std::clamp(std::round(v), 0.0f, 15.0f));
        output[16 + (g / 2) * 32 + j] |= static_cast<std::byte>(q << (4 * (g % 2)));
      } else {
        const auto q = static_cast<unsigned>(static_cast<int>(std::clamp(std::round(v), -32.0f, 31.0f)) + 32);
        const auto k = i % 128, lane = k % 32;
        output[(i / 128) * 64 + lane + (k / 32 % 2) * 32] |= static_cast<std::byte>((q & 15u) << (k >= 64 ? 4 : 0));
        output[128 + (i / 128) * 32 + lane] |= static_cast<std::byte>((q >> 4) << (2 * (k / 32)));
      }
    }
  }
}

}
std::size_t quant_block_bytes(std::uint32_t type) {
  if (type == q8_0) return 34;
  if (type == q4_0) return 18;
  if (type == q4_k) return 144;
  if (type == q6_k) return 210;
  throw std::invalid_argument("quantization: unsupported type");
}
std::size_t quant_block_elements(std::uint32_t type) {
  (void)quant_block_bytes(type);
  return type == q4_k || type == q6_k ? 256 : 32;
}
double dot_k_block(std::span<const std::byte> block, std::uint32_t type, std::span<const float> input) {
  if ((type != q4_k && type != q6_k) || block.size() != quant_block_bytes(type) || input.size() != 256)
    throw std::invalid_argument("K dot: invalid block shape");
  double sum = 0;
  decode_k(block.data(), type, [&](std::size_t i, float v) { sum += static_cast<double>(v) * input[i]; });
  return sum;
}
void dequantize(std::span<const std::byte> blocks, std::uint32_t type, std::span<float> output) {
  const auto size = quant_block_bytes(type);
  const auto elements = quant_block_elements(type);
  if (output.size() % elements != 0 || blocks.size() / size != output.size() / elements || blocks.size() % size != 0)
    throw std::invalid_argument("dequantize: invalid block shape");
  for (std::size_t b = 0; b < output.size() / elements; ++b) {
    const auto* block = blocks.data() + b * size;
    if (elements == 256) {
      decode_k(block, type, [&](std::size_t i, float v) { output[b * elements + i] = v; });
      continue;
    }
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
  if (input.size() != quant_block_elements(type) || output.size() != quant_block_bytes(type))
    throw std::invalid_argument("quantize: expected one complete block");
  float maximum = 0;
  for (float x : input) {
    if (!std::isfinite(x)) throw std::invalid_argument("quantize: nonfinite weight");
    if (std::abs(x) > std::abs(maximum)) maximum = x;
  }
  if (type == q4_k || type == q6_k) { encode_k(input, type, output); return; }
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
