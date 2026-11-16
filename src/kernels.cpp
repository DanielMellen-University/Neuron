#include "neuron/kernels.hpp"
#include "neuron/gguf.hpp"
#include "neuron/quant.hpp"
#include <cstring>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace neuron::scalar {
namespace {
void require(bool condition) {
  if (!condition) throw std::invalid_argument("kernel: invalid shape or parameter");
}
bool product_is(std::size_t a, std::size_t b, std::size_t size) {
  return (a == 0 || b <= std::numeric_limits<std::size_t>::max() / a) && a * b == size;
}
}  // namespace
void matmul(std::span<const float> weights, std::span<const float> input,
            std::span<float> output, std::size_t rows, std::size_t cols, std::size_t batch) {
  require(rows > 0 && cols > 0 && batch > 0 && product_is(rows, cols, weights.size()) &&
          product_is(batch, cols, input.size()) && product_is(batch, rows, output.size()));
  for (std::size_t b = 0; b < batch; ++b) {
    for (std::size_t r = 0; r < rows; ++r) {
      double sum = 0;
      for (std::size_t c = 0; c < cols; ++c)
        sum += static_cast<double>(weights[r * cols + c]) * input[b * cols + c];
      output[b * rows + r] = static_cast<float>(sum);
    }
  }
}
void matmul_quantized(std::span<const std::byte> weights, std::uint32_t type,
                      std::span<const float> input, std::span<float> output,
                      std::size_t rows, std::size_t cols, std::size_t batch) {
  const auto size = quant_block_bytes(type), elements = quant_block_elements(type);
  require(rows > 0 && cols > 0 && cols % elements == 0 && batch > 0 &&
          weights.size() % size == 0 && product_is(rows, cols / elements, weights.size() / size) &&
          product_is(batch, cols, input.size()) && product_is(batch, rows, output.size()));
  for (std::size_t b = 0; b < batch; ++b) for (std::size_t r = 0; r < rows; ++r) {
    double sum = 0;
    for (std::size_t c = 0; c < cols; c += elements) {
      const auto* block = weights.data() + (r * (cols / elements) + c / elements) * size;
      if (elements == 256) {
        sum += dot_k_block({block, size}, type, input.subspan(b * cols + c, elements));
        continue;
      }
      std::uint16_t bits;
      std::memcpy(&bits, block, 2);
      const float scale = f16_to_f32(bits);
      double dot = 0;
      if (type == q8_0) {
        for (std::size_t j = 0; j < 32; ++j) {
          std::int8_t code;
          std::memcpy(&code, block + 2 + j, 1);
          dot += static_cast<double>(code) * input[b * cols + c + j];
        }
      } else {
        for (std::size_t j = 0; j < 16; ++j) {
          const auto code = std::to_integer<unsigned>(block[2 + j]);
          dot += static_cast<double>(static_cast<int>(code & 15u) - 8) * input[b * cols + c + j];
          dot += static_cast<double>(static_cast<int>(code >> 4) - 8) * input[b * cols + c + j + 16];
        }
      }
      sum += static_cast<double>(scale) * dot;
    }
    output[b * rows + r] = static_cast<float>(sum);
  }
}
void rmsnorm(std::span<const float> input, std::span<const float> weights,
             std::span<float> output, float epsilon) {
  require(!input.empty() && weights.size() == input.size() && output.size() == input.size() &&
          std::isfinite(epsilon) && epsilon > 0);
  double sum = 0;
  for (float v : input) sum += static_cast<double>(v) * v;
  const float scale = 1.0f / std::sqrt(static_cast<float>(sum / static_cast<double>(input.size())) + epsilon);
  for (std::size_t i = 0; i < input.size(); ++i) output[i] = (input[i] * scale) * weights[i];
}
void softmax(std::span<float> values) {
  require(!values.empty());
  const float maximum = *std::max_element(values.begin(), values.end());
  require(std::isfinite(maximum));
  double sum = 0;
  for (float& v : values) { v = std::exp(v - maximum); sum += v; }
  const float scale = static_cast<float>(1.0 / sum);
  for (float& v : values) v *= scale;
}
void rope(std::span<float> values, std::size_t head_size, std::size_t rotary_dims,
          std::size_t position, float base, bool neox) {
  require(head_size > 0 && values.size() % head_size == 0 && rotary_dims > 0 &&
          rotary_dims <= head_size && rotary_dims % 2 == 0 && std::isfinite(base) && base > 0);
  const float step = std::pow(base, -2.0f / static_cast<float>(rotary_dims));
  for (std::size_t h = 0; h < values.size(); h += head_size) {
    float theta = static_cast<float>(position);
    for (std::size_t i = 0; i < rotary_dims; i += 2) {
      const float c = std::cos(theta), s = std::sin(theta);
      const auto a = h + (neox ? i / 2 : i);
      const auto b = neox ? a + rotary_dims / 2 : a + 1;
      const float x = values[a], y = values[b];
      values[a] = x * c - y * s;
      values[b] = x * s + y * c;
      theta *= step;
    }
  }
}
}  // namespace neuron::scalar
