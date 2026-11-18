#include "neuron/inspect.hpp"
#include "neuron/quant.hpp"
#include "json.hpp"
#include "utf8.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>

namespace neuron {
namespace {
std::string escape(std::string_view text, std::size_t limit) {
  std::string out = "\"";
  // Cut on a UTF-8 boundary so a shortened string stays readable.
  while (limit > 0 && limit < text.size() && (static_cast<unsigned char>(text[limit]) & 0xc0) == 0x80) --limit;
  const auto shown = text.substr(0, limit);
  for (const char c : shown) {
    const auto u = static_cast<unsigned char>(c);
    if (c == '"' || c == '\\') { out += '\\'; out += c; }
    else if (c == '\n') out += "\\n";
    else if (c == '\t') out += "\\t";
    else if (c == '\r') out += "\\r";
    else if (u < 0x20 || u == 0x7f) { char hex[5]; std::snprintf(hex, sizeof(hex), "\\x%02x", u); out += hex; }
    else out += c;
  }
  out += '"';
  if (text.size() > shown.size()) out += "... (" + std::to_string(text.size()) + " bytes)";
  return out;
}
std::string scalar(const GgufValue& value, const InspectOptions& options) {
  switch (value.type) {
    case GgufType::string: return escape(value.string(), options.string_bytes);
    case GgufType::boolean: return value.boolean() ? "true" : "false";
    case GgufType::f32: case GgufType::f64: {
      char text[32]; std::snprintf(text, sizeof(text), "%.9g", value.number()); return text;
    }
    case GgufType::i8: case GgufType::i16: case GgufType::i32: case GgufType::i64:
      return std::to_string(std::get<std::int64_t>(value.data));
    default: return std::to_string(value.uint());
  }
}
std::string shape(const GgufTensor& tensor) {
  std::string out;
  for (const auto dim : tensor.dimensions) out += (out.empty() ? "" : " x ") + std::to_string(dim);
  return out;
}
std::string size_text(std::uint64_t bytes) {
  char text[32];
  if (bytes < 1024) return std::to_string(bytes) + " B";
  const double kib = static_cast<double>(bytes) / 1024;
  if (kib < 1024) std::snprintf(text, sizeof(text), "%.1f KiB", kib);
  else if (kib < 1024 * 1024) std::snprintf(text, sizeof(text), "%.1f MiB", kib / 1024);
  else std::snprintf(text, sizeof(text), "%.2f GiB", kib / (1024 * 1024));
  return text;
}
std::string json_value(const GgufValue& value, const InspectOptions& options) {
  switch (value.type) {
    case GgufType::string: {
      std::string copy = value.string();
      return json::quote(detail::take_utf8(copy, true));
    }
    case GgufType::f32: case GgufType::f64: {
      if (!std::isfinite(value.number())) return "null";
      char text[32]; std::snprintf(text, sizeof(text), "%.9g", value.number()); return text;
    }
    case GgufType::array: {
      const auto& items = value.array();
      const auto shown = options.array_items == 0 ? items.size() : std::min(items.size(), options.array_items);
      std::string out = "{\"element_type\":" + json::quote(gguf_type_name(value.element_type)) +
          ",\"count\":" + std::to_string(items.size()) + ",\"items\":[";
      for (std::size_t i = 0; i < shown; ++i) out += (i == 0 ? "" : ",") + json_value(items[i], options);
      return out + "]}";
    }
    default: return scalar(value, options);
  }
}
std::string pad(std::string text, std::size_t width) {
  if (text.size() < width) text.resize(width, ' ');
  return text + ' ';
}
}  // namespace
std::string_view gguf_type_name(GgufType type) {
  switch (type) {
    case GgufType::u8: return "u8";
    case GgufType::i8: return "i8";
    case GgufType::u16: return "u16";
    case GgufType::i16: return "i16";
    case GgufType::u32: return "u32";
    case GgufType::i32: return "i32";
    case GgufType::f32: return "f32";
    case GgufType::boolean: return "bool";
    case GgufType::string: return "string";
    case GgufType::array: return "array";
    case GgufType::u64: return "u64";
    case GgufType::i64: return "i64";
    case GgufType::f64: return "f64";
  }
  return "unknown";
}
std::string tensor_type_name(std::uint32_t type) {
  switch (type) {
    case 0: return "F32";
    case 1: return "F16";
    case q4_0: return "Q4_0";
    case q8_0: return "Q8_0";
    case q4_k: return "Q4_K";
    case q6_k: return "Q6_K";
    default: return "type " + std::to_string(type);
  }
}
std::string summarize_tensors(const Gguf& file) {
  struct Total { std::size_t tensors = 0; std::uint64_t parameters = 0, bytes = 0; };
  std::map<std::uint32_t, Total> by_type;
  Total all;
  for (const auto& tensor : file.tensors()) {
    std::uint64_t elements = 1;
    for (const auto dim : tensor.dimensions) elements *= dim;
    for (auto* total : {&all, &by_type[tensor.type]}) {
      ++total->tensors; total->parameters += elements; total->bytes += tensor.bytes;
    }
  }
  if (all.tensors == 0) return "no tensors\n";
  char bits[32];
  std::snprintf(bits, sizeof(bits), "%.2f", static_cast<double>(all.bytes) * 8 / static_cast<double>(all.parameters));
  std::string out = std::to_string(all.parameters) + " parameters in " + size_text(all.bytes) + " (" + bits + " bits per weight)\n";
  for (const auto& [type, total] : by_type)
    out += "  " + pad(tensor_type_name(type), 4) + std::to_string(total.tensors) + " tensors, " +
        std::to_string(total.parameters) + " parameters, " + size_text(total.bytes) + "\n";
  return out;
}
std::string metadata_value(const Gguf& file, const std::string& key) {
  const auto* value = file.find(key);
  if (!value) throw std::invalid_argument("no metadata key " + key);
  InspectOptions raw;
  raw.string_bytes = static_cast<std::size_t>(-1);
  auto one = [&](const GgufValue& v) -> std::string {
    if (v.type == GgufType::string) return v.string();
    if (v.type == GgufType::array) return "[" + std::string(gguf_type_name(v.element_type)) + " x " + std::to_string(v.array().size()) + "]";
    return scalar(v, raw);
  };
  if (value->type != GgufType::array) return one(*value);
  std::string out;
  for (const auto& item : value->array()) out += one(item) + "\n";
  return out;
}
std::string describe_gguf_json(const Gguf& file, const InspectOptions& options) {
  std::uint64_t parameters = 0, bytes = 0;
  std::string tensors;
  for (const auto& tensor : file.tensors()) {
    std::uint64_t elements = 1;
    std::string dims;
    for (const auto dim : tensor.dimensions) { elements *= dim; dims += (dims.empty() ? "" : ",") + std::to_string(dim); }
    parameters += elements; bytes += tensor.bytes;
    tensors += std::string(tensors.empty() ? "" : ",") + "{\"name\":" + json::quote(tensor.name) + ",\"type\":" +
        json::quote(tensor_type_name(tensor.type)) + ",\"shape\":[" + dims + "],\"parameters\":" + std::to_string(elements) +
        ",\"bytes\":" + std::to_string(tensor.bytes) + ",\"offset\":" + std::to_string(tensor.offset) + "}";
  }
  std::string out = "{\"version\":" + std::to_string(file.version()) + ",\"alignment\":" + std::to_string(file.alignment()) +
      ",\"data_offset\":" + std::to_string(file.data_offset()) + ",\"parameters\":" + std::to_string(parameters) +
      ",\"tensor_bytes\":" + std::to_string(bytes) + ",\"metadata\":{";
  bool first = true;
  for (const auto& [key, value] : file.metadata()) {
    out += std::string(first ? "" : ",") + json::quote(key) + ":{\"type\":" + json::quote(gguf_type_name(value.type)) +
        ",\"value\":" + json_value(value, options) + "}";
    first = false;
  }
  out += "}";
  if (options.tensors) out += ",\"tensors\":[" + tensors + "]";
  return out + "}\n";
}
std::string describe_gguf(const Gguf& file, const InspectOptions& options) {
  std::string out = "GGUF v" + std::to_string(file.version()) + ", " + std::to_string(file.metadata().size()) +
      " metadata keys, " + std::to_string(file.tensors().size()) + " tensors, alignment " +
      std::to_string(file.alignment()) + ", data at byte " + std::to_string(file.data_offset()) + "\n";
  out += summarize_tensors(file);
  std::size_t key_width = 0;
  for (const auto& [key, value] : file.metadata()) key_width = std::max(key_width, key.size());
  out += "\nmetadata:\n";
  for (const auto& [key, value] : file.metadata()) {
    out += "  " + pad(key, key_width);
    if (value.type != GgufType::array) {
      out += pad(std::string(gguf_type_name(value.type)), 6) + scalar(value, options) + "\n";
      continue;
    }
    const auto& items = value.array();
    out += "[" + std::string(gguf_type_name(value.element_type)) + " x " + std::to_string(items.size()) + "]";
    const auto shown = options.array_items == 0 ? items.size() : std::min(items.size(), options.array_items);
    for (std::size_t i = 0; i < shown; ++i)
      out += (i == 0 ? " " : ", ") + (items[i].type == GgufType::array ? std::string("[...]") : scalar(items[i], options));
    if (items.size() > shown) out += ", ... " + std::to_string(items.size() - shown) + " more";
    out += "\n";
  }
  if (!options.tensors) return out;
  std::size_t name_width = 0, shape_width = 0;
  for (const auto& tensor : file.tensors()) {
    name_width = std::max(name_width, tensor.name.size());
    shape_width = std::max(shape_width, shape(tensor).size());
  }
  out += "\ntensors:\n";
  for (const auto& tensor : file.tensors())
    out += "  " + pad(tensor.name, name_width) + pad(tensor_type_name(tensor.type), 4) + pad(shape(tensor), shape_width) +
        std::to_string(tensor.bytes) + " bytes at " + std::to_string(tensor.offset) + "\n";
  return out;
}
}  // namespace neuron
