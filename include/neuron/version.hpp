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

// Compiled kernel variants. Runtime selection also checks CPU/OS support.
std::string_view compiled_isa() noexcept;

}  // namespace neuron
