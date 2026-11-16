#include "neuron/tensor.hpp"

#include <stdexcept>

namespace neuron {

Shape::Shape(std::initializer_list<std::int64_t> dims) {
  if (dims.size() > kMaxDims) {
    throw std::invalid_argument("Shape: too many dimensions");
  }
  for (std::int64_t d : dims) {
    if (d <= 0) {
      throw std::invalid_argument("Shape: dimensions must be positive");
    }
    dims_[rank_++] = d;
  }
}

std::int64_t Shape::dim(std::size_t i) const {
  if (i >= rank_) {
    throw std::out_of_range("Shape::dim: index past rank");
  }
  return dims_[i];
}

std::int64_t Shape::numel() const noexcept {
  std::int64_t n = 1;
  for (std::size_t i = 0; i < rank_; ++i) {
    n *= dims_[i];
  }
  return n;
}

std::int64_t Shape::stride(std::size_t i) const {
  if (i >= rank_) {
    throw std::out_of_range("Shape::stride: index past rank");
  }
  std::int64_t s = 1;
  for (std::size_t j = i + 1; j < rank_; ++j) {
    s *= dims_[j];
  }
  return s;
}

std::int64_t Shape::offset(std::initializer_list<std::int64_t> index) const {
  if (index.size() != rank_) {
    throw std::invalid_argument("Shape::offset: index count does not match rank");
  }
  std::int64_t off = 0;
  std::size_t i = 0;
  for (std::int64_t idx : index) {
    if (idx < 0 || idx >= dims_[i]) {
      throw std::out_of_range("Shape::offset: index out of range");
    }
    off += idx * stride(i);
    ++i;
  }
  return off;
}

std::string Shape::to_string() const {
  std::string out = "[";
  for (std::size_t i = 0; i < rank_; ++i) {
    if (i != 0) {
      out += ", ";
    }
    out += std::to_string(dims_[i]);
  }
  out += "]";
  return out;
}

bool operator==(const Shape& a, const Shape& b) noexcept {
  if (a.rank_ != b.rank_) {
    return false;
  }
  for (std::size_t i = 0; i < a.rank_; ++i) {
    if (a.dims_[i] != b.dims_[i]) {
      return false;
    }
  }
  return true;
}

}  // namespace neuron
