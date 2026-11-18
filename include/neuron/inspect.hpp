#pragma once

#include "neuron/gguf.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace neuron {
struct InspectOptions {
  bool tensors = true;            // Include the tensor table.
  std::size_t array_items = 8;    // Array elements shown before "... N more"; 0 shows all.
  std::size_t string_bytes = 96;  // Longer strings are cut, with their full size noted.
};
// Human-readable dump of a GGUF file's header, metadata and tensor table.
// The header line is followed by the parameter count, total size, bits per
// weight and a breakdown by tensor type.
std::string describe_gguf(const Gguf& file, const InspectOptions& options = {});
std::string summarize_tensors(const Gguf& file);
// The same information as one JSON object. Strings are complete (invalid
// UTF-8 becomes U+FFFD) and arrays list up to array_items elements.
std::string describe_gguf_json(const Gguf& file, const InspectOptions& options = {});
// One metadata value with no quoting or truncation: strings as raw bytes,
// arrays one element per line. Throws std::invalid_argument for a missing key.
std::string metadata_value(const Gguf& file, const std::string& key);
std::string_view gguf_type_name(GgufType type);
// GGML tensor type name such as F16 or Q4_K.
std::string tensor_type_name(std::uint32_t type);
}  // namespace neuron
