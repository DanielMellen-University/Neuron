// Standalone Qwen M5 decode driver with a count of C++ heap allocations.
// Link against libneuron.a. Define NEURON_PROFILE_LEGACY for pre-R1 headers.
#include "neuron/model.hpp"
#include "neuron/kernels.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>

namespace {
std::atomic<std::size_t> allocations{0};
}
void* operator new(std::size_t size) {
  allocations.fetch_add(1, std::memory_order_relaxed);
  if (auto* p = std::malloc(size == 0 ? 1 : size)) return p;
  throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void* operator new(std::size_t size, std::align_val_t alignment) {
  allocations.fetch_add(1, std::memory_order_relaxed);
  void* p = nullptr;
  if (posix_memalign(&p, static_cast<std::size_t>(alignment), size == 0 ? 1 : size) == 0) return p;
  throw std::bad_alloc();
}
void* operator new[](std::size_t size, std::align_val_t alignment) { return ::operator new(size, alignment); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }

int main(int argc, char** argv) {
  if (argc != 4) { std::fprintf(stderr, "usage: profile_decode MODEL THREADS ISA\n"); return 1; }
  neuron::kernels::set_threads(std::stoul(argv[2]));
  neuron::kernels::set_isa(argv[3]);
  neuron::Model model(argv[1], 256);
  std::array<std::size_t, 32> types{};
  for (const auto& t : model.file().tensors()) ++types.at(t.type);
  const std::array<std::int32_t, 4> prompt{12522, 5193, 264, 882};
#ifdef NEURON_PROFILE_LEGACY
  auto logits = model.forward(prompt);
#else
  auto logits = model.forward_view(prompt);
#endif
  const auto before = allocations.load();
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < 31; ++i) {
    const auto next = static_cast<std::int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
#ifdef NEURON_PROFILE_LEGACY
    logits = model.step(next);
#else
    logits = model.step_view(next);
#endif
  }
  const auto after = allocations.load();
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::printf("{\"decode_steps\":31,\"cpp_heap_allocations\":%zu,\"decode_seconds\":%.9g,\"tensor_types\":{", after - before, seconds);
  bool first = true;
  for (std::size_t i = 0; i < types.size(); ++i) if (types[i] != 0) {
    std::printf("%s\"%zu\":%zu", first ? "" : ",", i, types[i]); first = false;
  }
  std::printf("}}\n");
}
