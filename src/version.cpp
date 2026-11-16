#include "neuron/version.hpp"

#include "neuron/config.hpp"

namespace neuron {

Version version() noexcept {
  return {NEURON_VERSION_MAJOR, NEURON_VERSION_MINOR, NEURON_VERSION_PATCH};
}

std::string_view version_string() noexcept { return NEURON_VERSION_STRING; }

std::string_view compiled_isa() noexcept {
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
  return "avx512f avx512vnni avx2 fma";
#elif defined(__AVX512F__)
  return "avx512f avx2 fma";
#elif defined(__AVX2__) && defined(__FMA__)
  return "avx2 fma";
#elif defined(__AVX2__)
  return "avx2";
#else
  return "baseline";
#endif
}

}  // namespace neuron
