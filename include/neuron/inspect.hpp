#pragma once

#include "neuron/gguf.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace neuron {
struct InspectOptions {
  bool tensors = true;            // Include the tensor table.
  std::size_t array_items = 8;    // Array elements shown before "... N more".
  std::size_t string_bytes = 96;  // Longer strings are cut, with their full size noted.
};
// Human-readable dump of a GGUF file's header, metadata and tensor table.
std::string describe_gguf(const Gguf& file, const InspectOptions& options = {});
std::string_view gguf_type_name(GgufType type);
// GGML tensor type name such as F16 or Q4_K.
std::string tensor_type_name(std::uint32_t type);
}  // namespace neuron
