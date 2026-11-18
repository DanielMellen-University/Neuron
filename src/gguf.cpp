#include "neuron/gguf.hpp"
#include "neuron/quant.hpp"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <set>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace neuron {
namespace {
[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error("GGUF: " + message);
}
std::size_t checked_add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a) fail("size overflow");
  return a + b;
}
std::size_t checked_mul(std::size_t a, std::size_t b) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) fail("size overflow");
  return a * b;
}
class Reader {
 public:
  explicit Reader(std::span<const std::byte> bytes) : bytes_(bytes) {}
  std::size_t position() const { return position_; }
  std::size_t remaining() const { return bytes_.size() - position_; }
  template <typename T> T read() {
    if (sizeof(T) > remaining()) fail("truncated at byte " + std::to_string(position_));
    T result;
    std::memcpy(&result, bytes_.data() + position_, sizeof(T));
    position_ += sizeof(T);
    return result;
  }
  std::string string() {
    const auto n = read<std::uint64_t>();
    if (n > remaining()) fail("truncated string at byte " + std::to_string(position_));
    const auto size = static_cast<std::size_t>(n);
    std::string result(reinterpret_cast<const char*>(bytes_.data() + position_), size);
    position_ += size;
    return result;
  }
  static std::size_t minimum(GgufType type) {
    switch (type) {
      case GgufType::u8: case GgufType::i8: case GgufType::boolean: return 1;
      case GgufType::u16: case GgufType::i16: return 2;
      case GgufType::u32: case GgufType::i32: case GgufType::f32: return 4;
      case GgufType::string: case GgufType::u64: case GgufType::i64: case GgufType::f64: return 8;
      case GgufType::array: return 12;
    }
    fail("unknown metadata value type");
  }
  GgufType type() {
    const auto t = read<std::uint32_t>();
    if (t > static_cast<std::uint32_t>(GgufType::f64)) fail("unknown metadata value type");
    return static_cast<GgufType>(t);
  }
  GgufValue value(GgufType t, unsigned depth = 0) {
    if (++values_ > 4000000 || depth > 16) fail("metadata resource limit exceeded");
    switch (t) {
      case GgufType::u8: return {t, std::uint64_t(read<std::uint8_t>())};
      case GgufType::i8: return {t, std::int64_t(read<std::int8_t>())};
      case GgufType::u16: return {t, std::uint64_t(read<std::uint16_t>())};
      case GgufType::i16: return {t, std::int64_t(read<std::int16_t>())};
      case GgufType::u32: return {t, std::uint64_t(read<std::uint32_t>())};
      case GgufType::i32: return {t, std::int64_t(read<std::int32_t>())};
      case GgufType::f32: return {t, double(read<float>())};
      case GgufType::u64: return {t, read<std::uint64_t>()};
      case GgufType::i64: return {t, read<std::int64_t>()};
      case GgufType::f64: return {t, read<double>()};
      case GgufType::boolean: {
        const auto b = read<std::uint8_t>();
        if (b > 1) fail("invalid boolean");
        return {t, b != 0};
      }
      case GgufType::string: return {t, string()};
      case GgufType::array: {
        const auto element_type = type();
        const auto n = read<std::uint64_t>();
        if (n > remaining() / minimum(element_type) || n > 4000000 - values_)
          fail("oversized or truncated array");
        GgufValue::Array result;
        result.reserve(static_cast<std::size_t>(n));
        for (std::uint64_t i = 0; i < n; ++i) result.push_back(value(element_type, depth + 1));
        return {t, std::move(result), element_type};
      }
    }
    fail("unknown metadata value type");
  }
 private:
  std::span<const std::byte> bytes_;
  std::size_t position_ = 0;
  std::size_t values_ = 0;
};
}  // namespace

struct Gguf::Mapping {
  void* address = MAP_FAILED;
  std::size_t size = 0;
  explicit Mapping(const std::string& path) {
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) fail("cannot open " + path + ": " + std::strerror(errno));
    struct stat status {};
    if (fstat(fd, &status) != 0 || status.st_size <= 0 ||
        static_cast<std::uintmax_t>(status.st_size) > std::numeric_limits<std::size_t>::max()) {
      close(fd);
      fail("empty file or invalid file size");
    }
    size = static_cast<std::size_t>(status.st_size);
    address = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    const int saved_errno = errno;
    close(fd);
    if (address == MAP_FAILED) fail("mmap failed: " + std::string(std::strerror(saved_errno)));
  }
  ~Mapping() { if (address != MAP_FAILED) munmap(address, size); }
};

std::uint64_t GgufValue::uint() const {
  if (const auto* p = std::get_if<std::uint64_t>(&data)) return *p;
  fail("expected unsigned integer metadata");
}
double GgufValue::number() const {
  if (const auto* p = std::get_if<double>(&data)) return *p;
  fail("expected floating point metadata");
}
bool GgufValue::boolean() const {
  if (const auto* p = std::get_if<bool>(&data)) return *p;
  fail("expected boolean metadata");
}
const std::string& GgufValue::string() const {
  if (const auto* p = std::get_if<std::string>(&data)) return *p;
  fail("expected string metadata");
}
const GgufValue::Array& GgufValue::array() const {
  if (const auto* p = std::get_if<Array>(&data)) return *p;
  fail("expected array metadata");
}

Gguf::Gguf(const std::string& path) : mapping_(std::make_unique<Mapping>(path)) {
  static_assert(std::endian::native == std::endian::little);
  Reader reader({static_cast<const std::byte*>(mapping_->address), mapping_->size});
  if (reader.read<std::uint32_t>() != 0x46554747) fail("bad magic (expected GGUF)");
  const auto version = reader.read<std::uint32_t>();
  if (version != 2 && version != 3) fail("unsupported version " + std::to_string(version));
  version_ = version;
  const auto nt = reader.read<std::uint64_t>();
  const auto nm = reader.read<std::uint64_t>();
  if (nm > reader.remaining() / 12 || nt > reader.remaining() / 32 || nm > 100000 || nt > 100000)
    fail("oversized metadata or tensor count");
  for (std::uint64_t i = 0; i < nm; ++i) {
    auto key = reader.string();
    if (key.empty() || key.find('\0') != std::string::npos) fail("invalid metadata key");
    auto value = reader.value(reader.type());
    if (!metadata_.emplace(std::move(key), std::move(value)).second) fail("duplicate metadata key");
  }
  if (const auto* a = find("general.alignment")) {
    if (a->type != GgufType::u32) fail("general.alignment must be uint32");
    alignment_ = static_cast<std::size_t>(a->uint());
  }
  if (alignment_ == 0 || (alignment_ & (alignment_ - 1)) != 0) fail("invalid alignment");
  std::set<std::string> names;
  for (std::uint64_t i = 0; i < nt; ++i) {
    GgufTensor tensor;
    tensor.name = reader.string();
    if (tensor.name.empty() || tensor.name.find('\0') != std::string::npos || !names.insert(tensor.name).second)
      fail("invalid or duplicate tensor name");
    const auto rank = reader.read<std::uint32_t>();
    if (rank == 0 || rank > 4) fail("invalid tensor rank for " + tensor.name);
    std::size_t elements = 1;
    for (std::uint32_t j = 0; j < rank; ++j) {
      const auto dim = reader.read<std::uint64_t>();
      if (dim == 0 || dim > std::numeric_limits<std::size_t>::max()) fail("invalid tensor dimension");
      elements = checked_mul(elements, static_cast<std::size_t>(dim));
      tensor.dimensions.push_back(dim);
    }
    tensor.type = reader.read<std::uint32_t>();
    if (tensor.type <= 1) tensor.bytes = checked_mul(elements, tensor.type == 0 ? 4 : 2);
    else if (tensor.type == q8_0 || tensor.type == q4_0 || tensor.type == q4_k || tensor.type == q6_k) {
      const auto block = quant_block_elements(tensor.type);
      if (tensor.dimensions.front() % block != 0) fail("quantized row has incomplete block for " + tensor.name);
      tensor.bytes = checked_mul(elements / block, quant_block_bytes(tensor.type));
    } else fail("unsupported tensor type " + std::to_string(tensor.type) + " for " + tensor.name);
    const auto offset = reader.read<std::uint64_t>();
    if (offset > std::numeric_limits<std::size_t>::max() || offset % alignment_ != 0) fail("bad tensor offset");
    tensor.offset = static_cast<std::size_t>(offset);
    tensors_.push_back(std::move(tensor));
  }
  data_offset_ = checked_add(reader.position(), alignment_ - 1) & ~(alignment_ - 1);
  std::vector<std::pair<std::size_t, std::size_t>> ranges;
  for (const auto& tensor : tensors_) {
    const auto begin = checked_add(data_offset_, tensor.offset);
    const auto end = checked_add(begin, tensor.bytes);
    if (end > mapping_->size) fail("truncated tensor data or bad offset for " + tensor.name);
    ranges.emplace_back(begin, end);
  }
  std::sort(ranges.begin(), ranges.end());
  for (std::size_t i = 1; i < ranges.size(); ++i)
    if (ranges[i].first < ranges[i - 1].second) fail("overlapping tensor data");
}
Gguf::~Gguf() = default;
const GgufValue* Gguf::find(const std::string& key) const {
  const auto it = metadata_.find(key);
  return it == metadata_.end() ? nullptr : &it->second;
}
const GgufValue& Gguf::at(const std::string& key) const {
  const auto* value = find(key);
  if (!value) fail("missing metadata: " + key);
  return *value;
}
const GgufTensor& Gguf::tensor(const std::string& name) const {
  for (const auto& t : tensors_) if (t.name == name) return t;
  fail("missing tensor: " + name);
}
std::span<const std::byte> Gguf::tensor_data(const GgufTensor& tensor) const {
  const auto begin = checked_add(data_offset_, tensor.offset);
  if (begin > mapping_->size || tensor.bytes > mapping_->size - begin) fail("invalid tensor span");
  return {static_cast<const std::byte*>(mapping_->address) + begin, tensor.bytes};
}
float f16_to_f32(std::uint16_t bits) {
  const auto sign = static_cast<std::uint32_t>(bits & 0x8000) << 16;
  const auto exponent = (bits >> 10) & 31;
  const auto fraction = static_cast<std::uint32_t>(bits & 1023);
  if (exponent == 0) {
    const float value = std::ldexp(static_cast<float>(fraction), -24);
    return (sign != 0) ? -value : value;
  }
  if (exponent == 31) return std::bit_cast<float>(sign | 0x7f800000u | (fraction << 13));
  return std::bit_cast<float>(sign | (static_cast<std::uint32_t>(exponent + 112) << 23) | (fraction << 13));
}
std::vector<float> Gguf::tensor_f32(const std::string& name) const {
  const auto& t = tensor(name);
  const auto bytes = tensor_data(t);
  const auto elements = t.type <= 1 ? t.bytes / (t.type == 0 ? 4 : 2) : t.bytes / quant_block_bytes(t.type) * quant_block_elements(t.type);
  std::vector<float> result(elements);
  if (t.type > 1) { dequantize(bytes, t.type, result); return result; }
  if (t.type == 0) std::memcpy(result.data(), bytes.data(), t.bytes);
  else for (std::size_t i = 0; i < result.size(); ++i) {
    std::uint16_t bits;
    std::memcpy(&bits, bytes.data() + i * 2, 2);
    result[i] = f16_to_f32(bits);
  }
  return result;
}
}  // namespace neuron
