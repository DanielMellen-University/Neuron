#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace neuron {
class Model;
struct Perplexity {
  std::size_t chunks = 0, scored_tokens = 0;
  double negative_log_likelihood = 0;
  double value() const;
};
double token_nll(std::span<const float> logits, std::int32_t target);
// Independent full chunks of an even context length. Reset K/V between chunks.
// Score targets [context/2, context), using the first half as warmup.
// Ignore an incomplete tail; max_chunks == 0 evaluates all complete chunks.
Perplexity perplexity(Model& model, std::span<const std::int32_t> tokens,
                      std::size_t max_chunks = 0);
}  // namespace neuron
