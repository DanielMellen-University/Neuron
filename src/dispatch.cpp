#include "neuron/kernels.hpp"
#include "neuron/quant.hpp"
#include "simd.hpp"

#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace neuron::kernels {
namespace {
bool cpu_avx2() {
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2");
}
std::string choose(std::string_view name) {
  if (name == "auto") return cpu_avx2() ? "avx2" : "scalar";
  if (name == "scalar" || (name == "avx2" && cpu_avx2())) return std::string(name);
  throw std::invalid_argument("kernel: unknown or unavailable ISA: " + std::string(name));
}
std::string& selection() {
  static std::string name = choose(std::getenv("NEURON_ISA") ? std::getenv("NEURON_ISA") : "auto");
  return name;
}
bool product_is(std::size_t a, std::size_t b, std::size_t size) {
  return (a == 0 || b <= std::numeric_limits<std::size_t>::max() / a) && a * b == size;
}
void shape(bool ok) {
  if (!ok) throw std::invalid_argument("kernel: invalid shape or parameter");
}
}
bool supported(std::string_view isa) {
  return isa == "auto" || isa == "scalar" || (isa == "avx2" && cpu_avx2());
}
void set_isa(std::string_view isa) { selection() = choose(isa); }
std::string_view selected_isa() { return selection(); }
double dot(std::span<const float> a, std::span<const float> b) {
  shape(a.size() == b.size());
  if (selection() != "scalar") return detail::dot_avx2(a.data(), b.data(), a.size());
  double sum = 0;
  for (std::size_t i = 0; i < a.size(); ++i) sum += static_cast<double>(a[i]) * b[i];
  return sum;
}
void matmul(std::span<const float> w, std::span<const float> x, std::span<float> y,
            std::size_t rows, std::size_t cols, std::size_t batch) {
  if (selection() == "scalar") { scalar::matmul(w, x, y, rows, cols, batch); return; }
  shape(rows > 0 && cols > 0 && batch > 0 && product_is(rows, cols, w.size()) &&
        product_is(batch, cols, x.size()) && product_is(batch, rows, y.size()));
  for (std::size_t b = 0; b < batch; ++b) for (std::size_t r = 0; r < rows; ++r)
    y[b * rows + r] = static_cast<float>(detail::dot_avx2(w.data() + r * cols, x.data() + b * cols, cols));
}
void matmul_quantized(std::span<const std::byte> w, std::uint32_t type,
                      std::span<const float> x, std::span<float> y,
                      std::size_t rows, std::size_t cols, std::size_t batch) {
  if (selection() == "scalar") { scalar::matmul_quantized(w, type, x, y, rows, cols, batch); return; }
  const auto elements = quant_block_elements(type), bytes = quant_block_bytes(type);
  shape(rows > 0 && cols > 0 && cols % elements == 0 && batch > 0 &&
        w.size() % bytes == 0 && product_is(rows, cols / elements, w.size() / bytes) &&
        product_is(batch, cols, x.size()) && product_is(batch, rows, y.size()));
  const auto stride = cols / elements * bytes;
  for (std::size_t b = 0; b < batch; ++b) for (std::size_t r = 0; r < rows; ++r)
    y[b * rows + r] = static_cast<float>(detail::quant_avx2(w.data() + r * stride, type, x.data() + b * cols, cols));
}
}  // namespace neuron::kernels
