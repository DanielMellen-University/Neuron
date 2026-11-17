#include "simd.hpp"
#include "neuron/gguf.hpp"
#include "neuron/quant.hpp"

#include <cstring>
#include <immintrin.h>

#define TARGET __attribute__((target("avx2")))
namespace neuron::detail {
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
TARGET __m256i bytes8(const std::byte* p) {
  return _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
}
TARGET double reduce(__m256d a) {
  alignas(32) double values[4];
  _mm256_store_pd(values, a);
  return (values[0] + values[1]) + (values[2] + values[3]);
}
TARGET __m256d products(__m256 w, const float* x) {
  const auto a = _mm256_cvtps_pd(_mm256_castps256_ps128(w));
  const auto b = _mm256_cvtps_pd(_mm256_extractf128_ps(w, 1));
  return _mm256_add_pd(_mm256_mul_pd(a, _mm256_cvtps_pd(_mm_loadu_ps(x))),
                       _mm256_mul_pd(b, _mm256_cvtps_pd(_mm_loadu_ps(x + 4))));
}
}
TARGET double dot_avx2(const float* a, const float* b, std::size_t n) {
  auto sum = _mm256_setzero_pd();
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) sum = _mm256_add_pd(sum, products(_mm256_loadu_ps(a + i), b + i));
  double result = reduce(sum);
  for (; i < n; ++i) result += static_cast<double>(a[i]) * b[i];
  return result;
}
TARGET double quant_avx2(const std::byte* w, std::uint32_t type, const float* x, std::size_t n) {
  const std::size_t elements = type == q4_k || type == q6_k ? 256 : 32;
  const std::size_t bytes = type == q4_k ? 144 : type == q6_k ? 210 : type == q8_0 ? 34 : 18;
  double result = 0;
  for (std::size_t c = 0; c < n; c += elements, w += bytes) {
    auto sum = _mm256_setzero_pd();
    const float d = half(w + (type == q6_k ? 208 : 0));
    const auto mask = _mm256_set1_epi32(15);
    for (std::size_t j = 0; j < elements; j += 8) {
      __m256 values;
      if (type == q8_0) {
        values = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + 2 + j))));
      } else if (type == q4_0) {
        auto codes = bytes8(w + 2 + j % 16);
        if (j >= 16) codes = _mm256_srli_epi32(codes, 4);
        values = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_and_si256(codes, mask), _mm256_set1_epi32(8)));
      } else if (type == q4_k) {
        const auto g = j / 32;
        const auto* scales = w + 4;
        const auto sc = g < 4 ? byte(scales, g) & 63u : (byte(scales, g + 4) & 15u) | ((byte(scales, g - 4) >> 6) << 4);
        const auto mn = g < 4 ? byte(scales, g + 4) & 63u : (byte(scales, g + 4) >> 4) | ((byte(scales, g) >> 6) << 4);
        auto codes = bytes8(w + 16 + (g / 2) * 32 + j % 32);
        if (g % 2 != 0) codes = _mm256_srli_epi32(codes, 4);
        values = _mm256_sub_ps(_mm256_mul_ps(_mm256_set1_ps(d * static_cast<float>(sc)),
            _mm256_cvtepi32_ps(_mm256_and_si256(codes, mask))), _mm256_set1_ps(half(w + 2) * static_cast<float>(mn)));
      } else {
        const auto k = j % 128;
        auto low = bytes8(w + (j / 128) * 64 + k % 32 + (k / 32 % 2) * 32);
        if (k >= 64) low = _mm256_srli_epi32(low, 4);
        auto high = bytes8(w + 128 + (j / 128) * 32 + k % 32);
        high = _mm256_srlv_epi32(high, _mm256_set1_epi32(static_cast<int>(2 * (k / 32))));
        auto codes = _mm256_or_si256(_mm256_and_si256(low, mask), _mm256_slli_epi32(_mm256_and_si256(high, _mm256_set1_epi32(3)), 4));
        values = _mm256_mul_ps(_mm256_set1_ps(d * static_cast<float>(signed_byte(w + 192, j / 16))),
            _mm256_cvtepi32_ps(_mm256_sub_epi32(codes, _mm256_set1_epi32(32))));
      }
      sum = _mm256_add_pd(sum, products(values, x + c + j));
    }
    result += (elements == 32 ? static_cast<double>(d) : 1.0) * reduce(sum);
  }
  return result;
}
}  // namespace neuron::detail
