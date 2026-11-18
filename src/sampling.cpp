#include "neuron/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace neuron {
void SamplingOptions::validate() const {
  if (!std::isfinite(temperature) || temperature < 0) throw std::invalid_argument("temperature must be a finite number >= 0");
  if (!(top_p > 0 && top_p <= 1)) throw std::invalid_argument("top_p must be in (0, 1]");
  if (!(min_p >= 0 && min_p < 1)) throw std::invalid_argument("min_p must be in [0, 1)");
  if (!std::isfinite(repeat_penalty) || repeat_penalty <= 0) throw std::invalid_argument("repeat_penalty must be a finite number > 0");
}
std::int32_t argmax(std::span<const float> logits) {
  if (logits.empty()) throw std::invalid_argument("sampling needs at least one logit");
  return static_cast<std::int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}
std::uint64_t Random::next() {
  std::uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
double Random::uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }
Sampler::Sampler(const SamplingOptions& options) : options_(options), random_(options.seed) { options_.validate(); }
void Sampler::accept(std::int32_t token) {
  if (options_.repeat_penalty == 1 || options_.repeat_last_n == 0) return;
  if (recent_.size() < options_.repeat_last_n) { recent_.push_back(token); return; }
  recent_[next_recent_] = token;
  next_recent_ = (next_recent_ + 1) % recent_.size();
}
void Sampler::reset() {
  recent_.clear();
  next_recent_ = 0;
}
std::int32_t Sampler::sample(std::span<const float> logits) {
  if (!recent_.empty()) {
    penalized_.assign(logits.begin(), logits.end());
    // Each distinct token is penalized once, however often it repeats.
    std::vector<std::int32_t> seen(recent_);
    std::sort(seen.begin(), seen.end());
    seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
    for (const auto token : seen) {
      if (token < 0 || static_cast<std::size_t>(token) >= penalized_.size()) continue;
      auto& logit = penalized_[static_cast<std::size_t>(token)];
      logit = logit > 0 ? logit / options_.repeat_penalty : logit * options_.repeat_penalty;
    }
    logits = penalized_;
  }
  const auto best = argmax(logits);
  if (options_.greedy()) return best;
  const auto top = logits[static_cast<std::size_t>(best)];
  if (!std::isfinite(top)) return best;
  // Work in log space relative to the best token, so the top weight is 1.
  const double scale = 1.0 / static_cast<double>(options_.temperature);
  const double cutoff = options_.min_p > 0 ? std::log(static_cast<double>(options_.min_p)) : -INFINITY;
  candidates_.clear();
  for (std::size_t i = 0; i < logits.size(); ++i) {
    const double z = (static_cast<double>(logits[i]) - static_cast<double>(top)) * scale;
    if (!(z > -INFINITY)) continue;  // Masked or NaN logits never win.
    // top_k and min_p both keep a prefix of the sorted list, so their order
    // does not matter and min_p can drop tokens before the sort.
    if (z >= cutoff) candidates_.push_back({static_cast<float>(z), static_cast<std::int32_t>(i)});
  }
  const auto descending = [](const Candidate& a, const Candidate& b) {
    return a.logit != b.logit ? a.logit > b.logit : a.token < b.token;
  };
  auto keep = candidates_.size();
  if (options_.top_k > 0 && options_.top_k < keep) {
    keep = options_.top_k;
    std::partial_sort(candidates_.begin(), candidates_.begin() + static_cast<std::ptrdiff_t>(keep), candidates_.end(), descending);
  } else {
    std::sort(candidates_.begin(), candidates_.end(), descending);
  }
  double total = 0;
  for (std::size_t i = 0; i < keep; ++i) total += std::exp(static_cast<double>(candidates_[i].logit));
  if (options_.top_p < 1) {
    double cumulative = 0;
    for (std::size_t i = 0; i < keep; ++i) {
      cumulative += std::exp(static_cast<double>(candidates_[i].logit)) / total;
      if (cumulative >= static_cast<double>(options_.top_p)) { keep = i + 1; break; }
    }
  }
  double kept = 0;
  for (std::size_t i = 0; i < keep; ++i) kept += std::exp(static_cast<double>(candidates_[i].logit));
  double target = random_.uniform() * kept;
  for (std::size_t i = 0; i < keep; ++i) {
    target -= std::exp(static_cast<double>(candidates_[i].logit));
    if (target < 0) return candidates_[i].token;
  }
  return candidates_[keep - 1].token;
}
}  // namespace neuron
