#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace neuron {
// GGML type IDs. Each block covers 32 contiguous weights and starts with an
// IEEE F16 scale. Q4_0 packs positions j and j+16 into low/high nibbles.
inline constexpr std::uint32_t q4_0 = 2;
inline constexpr std::uint32_t q8_0 = 8;
std::size_t quant_block_bytes(std::uint32_t type);
void dequantize(std::span<const std::byte> blocks, std::uint32_t type,
                std::span<float> output);
// Reference block encoder, also used to check the quantization error bounds.
void quantize_block(std::span<const float> input, std::uint32_t type,
                    std::span<std::byte> output);
}  // namespace neuron
