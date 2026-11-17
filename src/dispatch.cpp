#include "neuron/kernels.hpp"
#include "neuron/quant.hpp"
#include "simd.hpp"
#include "thread_pool.hpp"

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
bool cpu_avx512() {
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
      __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl");
}
bool cpu_vnni() {
  __builtin_cpu_init();
  return cpu_avx512() && __builtin_cpu_supports("avx512vnni");
}
std::string choose(std::string_view name) {
  if (name == "auto") return cpu_avx512() ? "avx512" : cpu_avx2() ? "avx2" : "scalar";
  if (name == "vnni" && cpu_vnni()) return "vnni";
  if (name == "avx512" && cpu_avx512()) return "avx512";
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
  return (isa == "vnni" && cpu_vnni()) || (isa == "avx512" && cpu_avx512()) || isa == "auto" || isa == "scalar" || (isa == "avx2" && cpu_avx2());
}
void set_isa(std::string_view isa) { selection() = choose(isa); }
std::string_view selected_isa() { return selection(); }
double dot(std::span<const float> a, std::span<const float> b) {
  shape(a.size() == b.size());
  if (selection() == "avx512" || selection() == "vnni") return detail::dot_avx512(a.data(), b.data(), a.size());
  if (selection() == "avx2") return detail::dot_avx2(a.data(), b.data(), a.size());
  double sum = 0;
  for (std::size_t i = 0; i < a.size(); ++i) sum += static_cast<double>(a[i]) * b[i];
  return sum;
}
std::int64_t dot_i8(std::span<const std::uint8_t> a, std::span<const std::int8_t> b) {
  shape(a.size() == b.size());
  if (selection() == "vnni") return detail::dot_i8_vnni(a.data(), b.data(), a.size());
  if (selection() == "avx512") return detail::dot_i8_avx512(a.data(), b.data(), a.size());
  if (selection() == "avx2") return detail::dot_i8_avx2(a.data(), b.data(), a.size());
  return scalar::dot_i8(a, b);
}
void matmul(std::span<const float> w, std::span<const float> x, std::span<float> y,
            std::size_t rows, std::size_t cols, std::size_t batch) {
  shape(rows > 0 && cols > 0 && batch > 0 && product_is(rows, cols, w.size()) &&
        product_is(batch, cols, x.size()) && product_is(batch, rows, y.size()));
  const bool reference = selection() == "scalar";
  const auto kernel = (selection() == "avx512" || selection() == "vnni") ? detail::dot_avx512 : detail::dot_avx2;
  detail::parallel_rows(rows, [&](std::size_t begin, std::size_t end) {
    for (std::size_t b = 0; b < batch; ++b) {
      if (reference) {
        scalar::matmul(w.subspan(begin * cols, (end - begin) * cols), x.subspan(b * cols, cols),
            y.subspan(b * rows + begin, end - begin), end - begin, cols);
      } else {
        for (std::size_t r = begin; r < end; ++r)
          y[b * rows + r] = static_cast<float>(kernel(w.data() + r * cols, x.data() + b * cols, cols));
      }
    }
  });
}
void matmul_quantized(std::span<const std::byte> w, std::uint32_t type,
                      std::span<const float> x, std::span<float> y,
                      std::size_t rows, std::size_t cols, std::size_t batch) {
  const auto elements = quant_block_elements(type), bytes = quant_block_bytes(type);
  shape(rows > 0 && cols > 0 && cols % elements == 0 && batch > 0 &&
        w.size() % bytes == 0 && product_is(rows, cols / elements, w.size() / bytes) &&
        product_is(batch, cols, x.size()) && product_is(batch, rows, y.size()));
  const bool reference = selection() == "scalar";
  const auto kernel = (selection() == "avx512" || selection() == "vnni") ? detail::quant_avx512 : detail::quant_avx2;
  const auto stride = cols / elements * bytes;
  // Pack once for the whole matmul, then share read-only across row workers.
  const auto activations = selection() == "vnni" ? detail::prepare_activations(x) : std::vector<detail::Activation>{};
  detail::parallel_rows(rows, [&](std::size_t begin, std::size_t end) {
    for (std::size_t b = 0; b < batch; ++b) {
      if (reference) {
        scalar::matmul_quantized(w.subspan(begin * stride, (end - begin) * stride), type,
            x.subspan(b * cols, cols), y.subspan(b * rows + begin, end - begin), end - begin, cols);
      } else {
        for (std::size_t r = begin; r < end; ++r) {
          y[b * rows + r] = static_cast<float>(activations.empty() ?
              kernel(w.data() + r * stride, type, x.data() + b * cols, cols) :
              detail::quant_vnni(w.data() + r * stride, type, activations.data() + b * cols / 16, cols));
        }
      }
    }
  });
}
}  // namespace neuron::kernels
