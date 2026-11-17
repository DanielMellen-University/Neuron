#pragma once
#include <cstddef>
#include <functional>

namespace neuron::detail {
void parallel_rows(std::size_t rows, const std::function<void(std::size_t, std::size_t)>& work);
}  // namespace neuron::detail
