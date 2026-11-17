#pragma once
#include <cstddef>

namespace neuron::detail {
struct RowTask {
  const void* context;
  void (*call)(const void*, std::size_t, std::size_t);
  void operator()(std::size_t begin, std::size_t end) const { call(context, begin, end); }
};
void parallel_rows_impl(std::size_t rows, RowTask work);
// The barrier completes before work leaves scope, so task erasure needs no
// owning std::function or heap allocation.
template <typename Work>
void parallel_rows(std::size_t rows, const Work& work) {
  parallel_rows_impl(rows, {&work, [](const void* context, std::size_t begin, std::size_t end) {
    (*static_cast<const Work*>(context))(begin, end);
  }});
}
}  // namespace neuron::detail
