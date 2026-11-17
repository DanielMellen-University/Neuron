#include "neuron/version.hpp"

#include "neuron/config.hpp"

namespace neuron {

Version version() noexcept {
  return {NEURON_VERSION_MAJOR, NEURON_VERSION_MINOR, NEURON_VERSION_PATCH};
}

std::string_view version_string() noexcept { return NEURON_VERSION_STRING; }

std::string_view compiled_isa() noexcept {
  return "scalar avx2 avx512 vnni (runtime dispatch)";
}

}  // namespace neuron
