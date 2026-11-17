#include "neuron/perplexity.hpp"
#include "neuron/model.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace neuron {
double Perplexity::value() const {
  if (scored_tokens == 0) throw std::invalid_argument("perplexity: no scored tokens");
  return std::exp(negative_log_likelihood / static_cast<double>(scored_tokens));
}
double token_nll(std::span<const float> logits, std::int32_t target) {
  if (target < 0 || static_cast<std::size_t>(target) >= logits.size())
    throw std::invalid_argument("perplexity: target out of range");
  for (float x : logits) if (!std::isfinite(x)) throw std::runtime_error("perplexity: nonfinite logits");
  const double maximum = *std::max_element(logits.begin(), logits.end());
  double sum = 0;
  for (float x : logits) sum += std::exp(static_cast<double>(x) - maximum);
  return std::log(sum) + maximum - logits[static_cast<std::size_t>(target)];
}
Perplexity perplexity(Model& model, std::span<const std::int32_t> tokens, std::size_t max_chunks) {
  const auto context = model.config().context;
  if (context < 2 || context % 2 != 0) throw std::invalid_argument("perplexity: context must be even and at least 2");
  Perplexity result;
  result.chunks = tokens.size() / context;
  if (max_chunks != 0) {
    if (max_chunks > result.chunks) throw std::invalid_argument("perplexity: text has too few complete chunks");
    result.chunks = max_chunks;
  }
  if (result.chunks == 0) throw std::invalid_argument("perplexity: text is shorter than context");
  for (std::size_t chunk = 0; chunk < result.chunks; ++chunk) {
    model.reset();
    for (std::size_t i = 0; i + 1 < context; ++i) {
      const bool score = i + 1 >= context / 2;
      const auto logits = model.step_view(tokens[chunk * context + i], score);
      if (score) {
        result.negative_log_likelihood += token_nll(logits, tokens[chunk * context + i + 1]);
        ++result.scored_tokens;
      }
    }
  }
  return result;
}
}  // namespace neuron
