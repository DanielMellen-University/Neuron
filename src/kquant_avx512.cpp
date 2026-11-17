#include "kquant_internal.hpp"
#include <immintrin.h>

#define TARGET __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl")))
#define VNNI __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx512vnni")))
namespace neuron::detail {
namespace {
TARGET __attribute__((always_inline)) inline __m512i coefficients(const std::byte* w, std::uint32_t type, std::size_t j) {
  const auto mask = _mm256_set1_epi8(15);
  if (type == q4_k) {
    auto codes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 16 + (j / 64) * 32));
    if (j % 64 != 0) codes = _mm256_srli_epi16(codes, 4);
    return _mm512_mullo_epi16(_mm512_cvtepu8_epi16(_mm256_and_si256(codes, mask)),
        _mm512_set1_epi16(static_cast<short>(k_scale(w, j / 32))));
  }
  const auto quarter = j % 128 / 32;
  auto low = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + (j / 128) * 64 + (quarter % 2) * 32));
  if (quarter >= 2) low = _mm256_srli_epi16(low, 4);
  auto high = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + 128 + (j / 128) * 32));
  high = _mm256_srl_epi16(high, _mm_cvtsi32_si128(static_cast<int>(quarter * 2)));
  auto codes = _mm256_or_si256(_mm256_and_si256(low, mask),
      _mm256_slli_epi16(_mm256_and_si256(high, _mm256_set1_epi8(3)), 4));
  codes = _mm256_sub_epi8(codes, _mm256_set1_epi8(32));
  const auto scales = _mm512_mask_blend_epi16(0xffff0000u,
      _mm512_set1_epi16(static_cast<short>(k_signed_scale(w, j / 16))),
      _mm512_set1_epi16(static_cast<short>(k_signed_scale(w, j / 16 + 1))));
  return _mm512_mullo_epi16(_mm512_cvtepi8_epi16(codes), scales);
}
TARGET __m512d widen_sum(__m512i v) {
  return _mm512_add_pd(_mm512_cvtepi32_pd(_mm512_castsi512_si256(v)),
      _mm512_cvtepi32_pd(_mm512_extracti64x4_epi64(v, 1)));
}
TARGET __attribute__((always_inline)) inline double finish(const std::byte* w, std::uint32_t type, const KActivation& x,
                     __m512i low, __m512i high, __m512i correction) {
  auto result = _mm512_add_pd(widen_sum(low), _mm512_mul_pd(widen_sum(high), _mm512_set1_pd(65536)));
  result = _mm512_add_pd(result, _mm512_mul_pd(widen_sum(correction), _mm512_set1_pd(32768)));
  const double scale = static_cast<double>(load_half(w + (type == q6_k ? 208 : 0))) * x.scale;
  return _mm512_reduce_add_pd(result) * scale - (type == q4_k ? k_offset(w, x) : 0);
}
}
template <std::uint32_t Type>
TARGET double kquant_avx512_impl(const std::byte* w, const KActivation* x, std::size_t n) {
  constexpr auto type = Type;
  double result = 0;
  for (std::size_t b = 0; b < n / 256; ++b, w += type == q4_k ? 144 : 210) {
    if (k_needs_direct(w, type, x[b])) [[unlikely]] {
      result += quant_avx512(w, type, x[b].original, 256);
      continue;
    }
    auto low = _mm512_setzero_si512(), high = low, correction = low;
    #pragma GCC unroll 8
    for (std::size_t j = 0; j < 256; j += 32) {
      const auto codes = coefficients(w, type, j);
      low = _mm512_add_epi32(low, _mm512_madd_epi16(codes, _mm512_loadu_si512(x[b].low.data() + j)));
      high = _mm512_add_epi32(high, _mm512_madd_epi16(codes, _mm512_loadu_si512(x[b].high.data() + j)));
      correction = _mm512_add_epi32(correction, _mm512_madd_epi16(codes, _mm512_set1_epi16(1)));
    }
    result += finish(w, type, x[b], low, high, correction);
  }
  return result;
}
template <std::uint32_t Type>
VNNI double kquant_vnni_impl(const std::byte* w, const KActivation* x, std::size_t n) {
  constexpr auto type = Type;
  double result = 0;
  for (std::size_t b = 0; b < n / 256; ++b, w += type == q4_k ? 144 : 210) {
    if (k_needs_direct(w, type, x[b])) [[unlikely]] {
      result += quant_avx512(w, type, x[b].original, 256);
      continue;
    }
    auto low = _mm512_setzero_si512(), high = low, correction = low;
    #pragma GCC unroll 8
    for (std::size_t j = 0; j < 256; j += 32) {
      const auto codes = coefficients(w, type, j);
      low = _mm512_dpwssd_epi32(low, codes, _mm512_loadu_si512(x[b].low.data() + j));
      high = _mm512_dpwssd_epi32(high, codes, _mm512_loadu_si512(x[b].high.data() + j));
      correction = _mm512_dpwssd_epi32(correction, codes, _mm512_set1_epi16(1));
    }
    result += finish(w, type, x[b], low, high, correction);
  }
  return result;
}
TARGET double kquant_avx512(const std::byte* w, std::uint32_t type, const KActivation* x, std::size_t n) {
  return type == q4_k ? kquant_avx512_impl<q4_k>(w, x, n) : kquant_avx512_impl<q6_k>(w, x, n);
}
VNNI double kquant_vnni(const std::byte* w, std::uint32_t type, const KActivation* x, std::size_t n) {
  return type == q4_k ? kquant_vnni_impl<q4_k>(w, x, n) : kquant_vnni_impl<q6_k>(w, x, n);
}
}  // namespace neuron::detail
