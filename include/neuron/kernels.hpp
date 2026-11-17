#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace neuron::scalar {
// weights[rows, cols], input[batch, cols], output[batch, rows], row major.
void matmul(std::span<const float> weights, std::span<const float> input,
            std::span<float> output, std::size_t rows, std::size_t cols,
            std::size_t batch = 1);
// Mapped quantized rows, FP32 activations, double block accumulators.
void matmul_quantized(std::span<const std::byte> weights, std::uint32_t type,
                      std::span<const float> input, std::span<float> output,
                      std::size_t rows, std::size_t cols, std::size_t batch = 1);
void rmsnorm(std::span<const float> input, std::span<const float> weights,
             std::span<float> output, float epsilon);
void softmax(std::span<float> values);
// Adjacent-pair Llama RoPE or split-half NeoX/Qwen2 RoPE, with optional
// unrotated dimensions at the end. No context scaling.
void rope(std::span<float> values, std::size_t head_size, std::size_t rotary_dims,
          std::size_t position, float base = 10000.0f, bool neox = false);
}  // namespace neuron::scalar

namespace neuron::kernels {
// Runtime dispatch. NEURON_ISA=auto|scalar|avx2|avx512.
// Set configuration before inference, not while another caller is running.
bool supported(std::string_view isa);
void set_isa(std::string_view isa);
std::string_view selected_isa();
double dot(std::span<const float> a, std::span<const float> b);
void matmul(std::span<const float> weights, std::span<const float> input,
            std::span<float> output, std::size_t rows, std::size_t cols,
            std::size_t batch = 1);
void matmul_quantized(std::span<const std::byte> weights, std::uint32_t type,
                      std::span<const float> input, std::span<float> output,
                      std::size_t rows, std::size_t cols, std::size_t batch = 1);
}  // namespace neuron::kernels
