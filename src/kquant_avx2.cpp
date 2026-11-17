#include "kquant_internal.hpp"
#include <immintrin.h>

#define TARGET __attribute__((target("avx2")))
namespace neuron::detail {
namespace {
TARGET __attribute__((always_inline)) inline __m256i coefficients(const std::byte* w, std::uint32_t type, std::size_t j) {
  const auto mask = _mm_set1_epi8(15);
  if (type == q4_k) {
    auto codes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + 16 + (j / 64) * 32 + j % 32));
    if (j % 64 >= 32) codes = _mm_srli_epi16(codes, 4);
    return _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm_and_si128(codes, mask)),
        _mm256_set1_epi16(static_cast<short>(k_scale(w, j / 32))));
  }
  const auto quarter = j % 128 / 32, lane = j % 32;
  auto low = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + (j / 128) * 64 + (quarter % 2) * 32 + lane));
  if (quarter >= 2) low = _mm_srli_epi16(low, 4);
  auto high = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + 128 + (j / 128) * 32 + lane));
  high = _mm_srl_epi16(high, _mm_cvtsi32_si128(static_cast<int>(quarter * 2)));
  auto codes = _mm_or_si128(_mm_and_si128(low, mask), _mm_slli_epi16(_mm_and_si128(high, _mm_set1_epi8(3)), 4));
  codes = _mm_sub_epi8(codes, _mm_set1_epi8(32));
  return _mm256_mullo_epi16(_mm256_cvtepi8_epi16(codes),
      _mm256_set1_epi16(static_cast<short>(k_signed_scale(w, j / 16))));
}
TARGET __m256d widen_sum(__m256i v) {
  return _mm256_add_pd(_mm256_cvtepi32_pd(_mm256_castsi256_si128(v)),
      _mm256_cvtepi32_pd(_mm256_extracti128_si256(v, 1)));
}
}
template <std::uint32_t Type>
TARGET double kquant_avx2_impl(const std::byte* w, const KActivation* x, std::size_t n) {
  constexpr auto type = Type;
  double result = 0;
  for (std::size_t b = 0; b < n / 256; ++b, w += type == q4_k ? 144 : 210) {
    if (k_needs_direct(w, type, x[b])) [[unlikely]] {
      result += quant_avx2(w, type, x[b].original, 256);
      continue;
    }
    auto sum = _mm256_setzero_pd();
    // At most 16 products enter each int32 lane. Even the Q6 endpoint
    // (-128 * -32) * -32768 reaches exactly INT32_MIN without overflow.
    for (std::size_t h = 0; h < 2; ++h) {
      auto low = _mm256_setzero_si256(), high = low, correction = low;
      #pragma GCC unroll 8
      for (std::size_t j = h * 128; j < (h + 1) * 128; j += 16) {
        const auto codes = coefficients(w, type, j);
        low = _mm256_add_epi32(low, _mm256_madd_epi16(codes,
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[b].low.data() + j))));
        high = _mm256_add_epi32(high, _mm256_madd_epi16(codes,
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[b].high.data() + j))));
        correction = _mm256_add_epi32(correction, _mm256_madd_epi16(codes, _mm256_set1_epi16(1)));
      }
      auto partial = _mm256_add_pd(widen_sum(low), _mm256_mul_pd(widen_sum(high), _mm256_set1_pd(65536)));
      sum = _mm256_add_pd(sum, _mm256_add_pd(partial,
          _mm256_mul_pd(widen_sum(correction), _mm256_set1_pd(32768))));
    }
    alignas(32) double lanes[4];
    _mm256_store_pd(lanes, sum);
    const double scale = static_cast<double>(load_half(w + (type == q6_k ? 208 : 0))) * x[b].scale;
    result += ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) * scale - (type == q4_k ? k_offset(w, x[b]) : 0);
  }
  return result;
}
TARGET double kquant_avx2(const std::byte* w, std::uint32_t type, const KActivation* x, std::size_t n) {
  return type == q4_k ? kquant_avx2_impl<q4_k>(w, x, n) : kquant_avx2_impl<q6_k>(w, x, n);
}
}  // namespace neuron::detail
