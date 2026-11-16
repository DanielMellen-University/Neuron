#pragma once

#include <cstddef>
#include <span>

namespace neuron::scalar {
// weights[rows, cols], input[batch, cols], output[batch, rows], row major.
void matmul(std::span<const float> weights, std::span<const float> input,
            std::span<float> output, std::size_t rows, std::size_t cols,
            std::size_t batch = 1);
void rmsnorm(std::span<const float> input, std::span<const float> weights,
             std::span<float> output, float epsilon);
void softmax(std::span<float> values);
// Llama GGUF normal RoPE: adjacent pairs within each head, with optional
// unrotated dimensions at the end. No context scaling.
void rope(std::span<float> values, std::size_t head_size, std::size_t rotary_dims,
          std::size_t position, float base = 10000.0f);
}  // namespace neuron::scalar
