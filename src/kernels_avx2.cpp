#include "simd.hpp"
#include "neuron/gguf.hpp"
#include "neuron/quant.hpp"

#include <cstring>
#include <immintrin.h>

#define TARGET __attribute__((target("avx2")))
namespace neuron::detail {
namespace {
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
// Specialized K loops extract scales once per group and reduce once per row.
// No activation rounding or expanded weight copy is needed.
TARGET double q4_k_dot(const std::byte* w, const float* x, std::size_t n) {
  auto sum = _mm256_setzero_pd();
  const auto mask = _mm256_set1_epi32(15);
  for (std::size_t c = 0; c < n; c += 256, w += 144) {
    const float d = load_half(w), m = load_half(w + 2);
    const auto* scales = w + 4;
    for (std::size_t g = 0; g < 8; ++g) {
      const auto sc = g < 4 ? byte(scales, g) & 63u :
          (byte(scales, g + 4) & 15u) | ((byte(scales, g - 4) >> 6) << 4);
      const auto mn = g < 4 ? byte(scales, g + 4) & 63u :
          (byte(scales, g + 4) >> 4) | ((byte(scales, g) >> 6) << 4);
      const auto scale = _mm256_set1_ps(d * static_cast<float>(sc));
      const auto offset = _mm256_set1_ps(m * static_cast<float>(mn));
      for (std::size_t j = 0; j < 32; j += 8) {
        auto codes = bytes8(w + 16 + (g / 2) * 32 + j);
        if (g % 2 != 0) codes = _mm256_srli_epi32(codes, 4);
        const auto values = _mm256_sub_ps(_mm256_mul_ps(scale,
            _mm256_cvtepi32_ps(_mm256_and_si256(codes, mask))), offset);
        sum = _mm256_add_pd(sum, products(values, x + c + g * 32 + j));
      }
    }
  }
  return reduce(sum);
}
template <int Quarter>
TARGET void q6_quarter(const std::byte* w, const float* x, float d, __m256d& sum) {
  const auto mask = _mm256_set1_epi32(15);
  for (std::size_t g = 0; g < 2; ++g) {
    const auto scale = _mm256_set1_ps(d * static_cast<float>(signed_byte(w + 192, Quarter * 2 + g)));
    for (std::size_t j = 0; j < 16; j += 8) {
      const auto lane = g * 16 + j;
      auto low = bytes8(w + (Quarter / 4) * 64 + (Quarter % 2) * 32 + lane);
      if constexpr (Quarter % 4 >= 2) low = _mm256_srli_epi32(low, 4);
      const auto high = _mm256_srli_epi32(bytes8(w + 128 + (Quarter / 4) * 32 + lane), 2 * (Quarter % 4));
      auto codes = _mm256_or_si256(_mm256_and_si256(low, mask),
          _mm256_slli_epi32(_mm256_and_si256(high, _mm256_set1_epi32(3)), 4));
      codes = _mm256_sub_epi32(codes, _mm256_set1_epi32(32));
      const auto values = _mm256_mul_ps(scale, _mm256_cvtepi32_ps(codes));
      sum = _mm256_add_pd(sum, products(values, x + Quarter * 32 + lane));
    }
  }
}
TARGET double q6_k_dot(const std::byte* w, const float* x, std::size_t n) {
  auto sum = _mm256_setzero_pd();
  for (std::size_t c = 0; c < n; c += 256, w += 210) {
    const float d = load_half(w + 208);
    q6_quarter<0>(w, x + c, d, sum);
    q6_quarter<1>(w, x + c, d, sum);
    q6_quarter<2>(w, x + c, d, sum);
    q6_quarter<3>(w, x + c, d, sum);
    q6_quarter<4>(w, x + c, d, sum);
    q6_quarter<5>(w, x + c, d, sum);
    q6_quarter<6>(w, x + c, d, sum);
    q6_quarter<7>(w, x + c, d, sum);
  }
  return reduce(sum);
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
TARGET std::int64_t dot_i8_avx2(const std::uint8_t* a, const std::int8_t* b, std::size_t n) {
  std::int64_t result = 0;
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    const auto u = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i)));
    const auto v = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i)));
    alignas(64) std::int32_t sums[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(sums), _mm256_madd_epi16(u, v));
    for (auto value : sums) result += value;
  }
  for (; i < n; ++i) result += static_cast<int>(a[i]) * static_cast<int>(b[i]);
  return result;
}
TARGET double quant_avx2(const std::byte* w, std::uint32_t type, const float* x, std::size_t n) {
  if (type == q4_k) return q4_k_dot(w, x, n);
  if (type == q6_k) return q6_k_dot(w, x, n);
  const std::size_t bytes = type == q8_0 ? 34 : 18;
  double result = 0;
  for (std::size_t c = 0; c < n; c += 32, w += bytes) {
    auto sum = _mm256_setzero_pd();
    const float d = load_half(w);
    for (std::size_t j = 0; j < 32; j += 8) {
      __m256 values;
      if (type == q8_0) {
        values = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + 2 + j))));
      } else {
        auto codes = bytes8(w + 2 + j % 16);
        if (j >= 16) codes = _mm256_srli_epi32(codes, 4);
        values = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_and_si256(codes,
            _mm256_set1_epi32(15)), _mm256_set1_epi32(8)));
      }
      sum = _mm256_add_pd(sum, products(values, x + c + j));
    }
    result += static_cast<double>(d) * reduce(sum);
  }
  return result;
}
}  // namespace neuron::detail
