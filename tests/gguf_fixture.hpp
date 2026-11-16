#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace fixture {
struct Bytes {
  std::vector<char> data;
  template <typename T> std::size_t put(T value) {
    const auto offset = data.size();
    data.resize(offset + sizeof(T));
    std::memcpy(data.data() + offset, &value, sizeof(T));
    return offset;
  }
  template <typename T> void patch(std::size_t offset, T value) {
    if (offset + sizeof(T) > data.size()) throw std::runtime_error("bad fixture patch");
    std::memcpy(data.data() + offset, &value, sizeof(T));
  }
  void string(const std::string& s) {
    put<std::uint64_t>(s.size());
    data.insert(data.end(), s.begin(), s.end());
  }
  void header(std::uint64_t tensors, std::uint64_t metadata, std::uint32_t version = 3) {
    put<std::uint32_t>(0x46554747); put(version); put(tensors); put(metadata);
  }
  void key(const std::string& name, std::uint32_t type) { string(name); put(type); }
  void uint(const std::string& name, std::uint32_t value) { key(name, 4); put(value); }
  void text(const std::string& name, const std::string& value) { key(name, 8); string(value); }
  std::size_t tensor(const std::string& name, const std::vector<std::uint64_t>& dims,
                     std::uint32_t type, std::uint64_t offset) {
    string(name); put<std::uint32_t>(static_cast<std::uint32_t>(dims.size()));
    for (auto dim : dims) put(dim);
    put(type); return put(offset);
  }
  void align(std::size_t alignment = 32) { while (data.size() % alignment != 0) data.push_back(0); }
};
class File {
 public:
  explicit File(const Bytes& bytes) {
    char name[] = "/tmp/neuron-gguf-XXXXXX";
    const int fd = mkstemp(name);
    if (fd < 0) throw std::runtime_error("mkstemp failed");
    close(fd);
    path = name;
    std::ofstream stream(path, std::ios::binary);
    stream.write(bytes.data.data(), static_cast<std::streamsize>(bytes.data.size()));
    if (!stream) throw std::runtime_error("fixture write failed");
  }
  ~File() { std::remove(path.c_str()); }
  File(const File&) = delete;
  File& operator=(const File&) = delete;
  std::string path;
};
}  // namespace fixture
