#include "simd.hpp"
#include "neuron/gguf.hpp"
#include "neuron/quant.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <immintrin.h>

#define TARGET __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx512vnni")))
namespace neuron::detail {
std::vector<Activation> prepare_activations(std::span<const float> x) {
  std::vector<Activation> result(x.size() / 16);
  for (std::size_t g = 0; g < result.size(); ++g) {
    auto& a = result[g];
    float maximum = 0;
    for (std::size_t j = 0; j < 16; ++j) {
      const float v = x[g * 16 + j];
      if (!std::isfinite(v)) return {};
      maximum = std::max(maximum, std::abs(v));
      a.sum += v;
    }
    int exponent = 0;
    std::frexp(maximum, &exponent);
    a.scale = std::ldexp(1.0, exponent - 30);
    for (std::size_t j = 0; j < 16; ++j) {
      const auto n = static_cast<std::int32_t>(std::llround(static_cast<double>(x[g * 16 + j]) / a.scale));
      const auto u = std::bit_cast<std::uint32_t>(n) ^ 0x80000000u;
      for (unsigned k = 0; k < 4; ++k) a.planes[k][j] = static_cast<std::uint8_t>((u >> (k * 8)) & 255u);
    }
  }
  return result;
}
namespace {
float half(const std::byte* p) {
  std::uint16_t bits;
  std::memcpy(&bits, p, 2);
  return f16_to_f32(bits);
}
unsigned byte(const std::byte* p, std::size_t i) { return std::to_integer<unsigned>(p[i]); }
int signed_byte(const std::byte* p, std::size_t i) {
  std::int8_t v;
  std::memcpy(&v, p + i, 1);
  return v;
}
TARGET int sum4(__m128i v) {
  v = _mm_add_epi32(v, _mm_srli_si128(v, 8));
  v = _mm_add_epi32(v, _mm_srli_si128(v, 4));
  return _mm_cvtsi128_si32(v);
}
TARGET double activation_dot(__m128i weights, const Activation& x) {
  std::int64_t sum = 0;
  for (unsigned k = 0; k < 4; ++k) {
    const auto values = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x.planes[k].data()));
    const int partial = sum4(_mm_dpbusd_epi32(_mm_setzero_si128(), values, weights));
    sum += static_cast<std::int64_t>(partial) * (std::int64_t{1} << (8 * k));
  }
  const int weight_sum = sum4(_mm_dpbusd_epi32(_mm_setzero_si128(), _mm_set1_epi8(1), weights));
  sum -= static_cast<std::int64_t>(weight_sum) * (std::int64_t{1} << 31);
  return static_cast<double>(sum) * x.scale;
}
}
TARGET std::int64_t dot_i8_vnni(const std::uint8_t* a, const std::int8_t* b, std::size_t n) {
  std::int64_t result = 0;
  std::size_t i = 0;
  for (; i + 64 <= n; i += 64) {
    const auto sums = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_loadu_si512(a + i), _mm512_loadu_si512(b + i));
    result += _mm512_reduce_add_epi32(sums);
  }
  for (; i < n; ++i) result += static_cast<int>(a[i]) * static_cast<int>(b[i]);
  return result;
}
TARGET double quant_vnni(const std::byte* w, std::uint32_t type, const Activation* x, std::size_t n) {
  const std::size_t elements = type == q4_k || type == q6_k ? 256 : 32;
  const std::size_t bytes = type == q4_k ? 144 : type == q6_k ? 210 : type == q8_0 ? 34 : 18;
  const auto mask = _mm_set1_epi8(15);
  double result = 0;
  for (std::size_t c = 0; c < n; c += elements, w += bytes) {
    const float d = half(w + (type == q6_k ? 208 : 0));
    for (std::size_t j = 0; j < elements; j += 16) {
      __m128i codes;
      float scale = d, offset = 0;
      if (type == q8_0) {
        codes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + 2 + j));
      } else if (type == q4_0) {
        codes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + 2));
        if (j >= 16) codes = _mm_srli_epi16(codes, 4);
        codes = _mm_sub_epi8(_mm_and_si128(codes, mask), _mm_set1_epi8(8));
      } else if (type == q4_k) {
        const auto g = j / 32;
        const auto* scales = w + 4;
        const auto sc = g < 4 ? byte(scales, g) & 63u : (byte(scales, g + 4) & 15u) | ((byte(scales, g - 4) >> 6) << 4);
        const auto mn = g < 4 ? byte(scales, g + 4) & 63u : (byte(scales, g + 4) >> 4) | ((byte(scales, g) >> 6) << 4);
        codes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + 16 + (g / 2) * 32 + j % 32));
        if (g % 2 != 0) codes = _mm_srli_epi16(codes, 4);
        codes = _mm_and_si128(codes, mask);
        scale = d * static_cast<float>(sc);
        offset = half(w + 2) * static_cast<float>(mn);
      } else {
        const auto k = j % 128;
        auto low = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + (j / 128) * 64 + k % 32 + (k / 32 % 2) * 32));
        if (k >= 64) low = _mm_srli_epi16(low, 4);
        auto high = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + 128 + (j / 128) * 32 + k % 32));
        high = _mm_srl_epi16(high, _mm_cvtsi32_si128(static_cast<int>(2 * (k / 32))));
        codes = _mm_or_si128(_mm_and_si128(low, mask), _mm_slli_epi16(_mm_and_si128(high, _mm_set1_epi8(3)), 4));
        codes = _mm_sub_epi8(codes, _mm_set1_epi8(32));
        scale = d * static_cast<float>(signed_byte(w + 192, j / 16));
      }
      const auto& a = x[(c + j) / 16];
      result += static_cast<double>(scale) * activation_dot(codes, a) - static_cast<double>(offset) * a.sum;
    }
  }
  return result;
}
}  // namespace neuron::detail
