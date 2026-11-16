#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace neuron {
// GGML type IDs. Legacy blocks cover 32 weights; K blocks cover 256.
inline constexpr std::uint32_t q4_0 = 2;
inline constexpr std::uint32_t q8_0 = 8;
inline constexpr std::uint32_t q4_k = 12;
inline constexpr std::uint32_t q6_k = 14;
std::size_t quant_block_elements(std::uint32_t type);
std::size_t quant_block_bytes(std::uint32_t type);
void dequantize(std::span<const std::byte> blocks, std::uint32_t type,
                std::span<float> output);
// Reference block encoder, also used to check the quantization error bounds.
void quantize_block(std::span<const float> input, std::uint32_t type,
                    std::span<std::byte> output);
// One K block, decoded as FP32 weights and accumulated in double precision.
double dot_k_block(std::span<const std::byte> block, std::uint32_t type,
                   std::span<const float> input);
}  // namespace neuron
