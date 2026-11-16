#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace neuron {

enum class GgufType : std::uint32_t {
  u8, i8, u16, i16, u32, i32, f32, boolean, string, array, u64, i64, f64
};

struct GgufValue {
  GgufType type;
  using Array = std::vector<GgufValue>;
  std::variant<std::uint64_t, std::int64_t, double, bool, std::string, Array> data;
  GgufType element_type = GgufType::u8;  // Used when type is array, including empty arrays.
  std::uint64_t uint() const;
  double number() const;
  bool boolean() const;
  const std::string& string() const;
  const Array& array() const;
};

struct GgufTensor {
  std::string name;
  // GGUF order: the first dimension is contiguous.
  std::vector<std::uint64_t> dimensions;
  std::uint32_t type;
  std::size_t offset;
  std::size_t bytes;
};

// Little-endian GGUF v2/v3, with dense F32/F16 tensor storage. The mapping
// outlives every tensor_data span. Invalid input throws std::runtime_error.
class Gguf {
 public:
  explicit Gguf(const std::string& path);
  ~Gguf();
  Gguf(const Gguf&) = delete;
  Gguf& operator=(const Gguf&) = delete;
  const std::map<std::string, GgufValue>& metadata() const { return metadata_; }
  const GgufValue& at(const std::string& key) const;
  const GgufValue* find(const std::string& key) const;
  const std::vector<GgufTensor>& tensors() const { return tensors_; }
  const GgufTensor& tensor(const std::string& name) const;
  std::span<const std::byte> tensor_data(const GgufTensor& tensor) const;
  std::vector<float> tensor_f32(const std::string& name) const;
  std::size_t alignment() const { return alignment_; }

 private:
  struct Mapping;
  std::unique_ptr<Mapping> mapping_;
  std::map<std::string, GgufValue> metadata_;
  std::vector<GgufTensor> tensors_;
  std::size_t alignment_ = 32;
  std::size_t data_offset_ = 0;
};

float f16_to_f32(std::uint16_t bits);

}  // namespace neuron
