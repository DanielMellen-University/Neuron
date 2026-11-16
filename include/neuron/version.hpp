#pragma once

#include <string_view>

namespace neuron {

struct Version {
  int major;
  int minor;
  int patch;
};

// Version this library was built as.
Version version() noexcept;

// "MAJOR.MINOR.PATCH".
std::string_view version_string() noexcept;

// Instruction sets the library was compiled for, e.g. "avx2 fma".
// "baseline" when no SIMD option was enabled.
std::string_view compiled_isa() noexcept;

}  // namespace neuron
