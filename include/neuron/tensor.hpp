#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>

namespace neuron {

// Shape of a dense, row-major tensor with up to kMaxDims dimensions.
// The last dimension is the fastest-moving one.
class Shape {
 public:
  static constexpr std::size_t kMaxDims = 4;

  Shape() = default;
  // Throws std::invalid_argument for more than kMaxDims dims or a zero-size dim.
  Shape(std::initializer_list<std::int64_t> dims);

  std::size_t rank() const noexcept { return rank_; }
  // Throws std::out_of_range for i >= rank().
  std::int64_t dim(std::size_t i) const;

  // Product of all dims. 1 for a rank-0 (scalar) shape.
  std::int64_t numel() const noexcept;

  // Row-major stride of dimension i, in elements.
  std::int64_t stride(std::size_t i) const;

  // Flat element offset of a multi-index. Throws on a wrong index count or an
  // out-of-range index.
  std::int64_t offset(std::initializer_list<std::int64_t> index) const;

  // "[2, 3, 4]".
  std::string to_string() const;

  friend bool operator==(const Shape& a, const Shape& b) noexcept;

 private:
  std::array<std::int64_t, kMaxDims> dims_{};
  std::size_t rank_ = 0;
};

}  // namespace neuron
