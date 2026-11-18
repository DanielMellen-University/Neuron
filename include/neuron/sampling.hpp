#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace neuron {
// Token selection settings. A temperature of 0 picks the highest logit, which
// keeps the old greedy behavior. Filters work on the temperature-scaled
// distribution: top_k and min_p first, then top_p on what they keep.
struct SamplingOptions {
  float temperature = 0;
  std::size_t top_k = 0;  // 0 keeps every token.
  float top_p = 1;        // Smallest set whose probability reaches top_p.
  float min_p = 0;        // Drops tokens below min_p times the top probability.
  std::uint64_t seed = 0;
  bool greedy() const { return temperature == 0; }
  // Throws std::invalid_argument naming the bad field.
  void validate() const;
};

// First index of the largest value, matching std::max_element.
std::int32_t argmax(std::span<const float> logits);

// Small portable generator (SplitMix64) so a seed gives the same tokens with
// any standard library.
class Random {
 public:
  explicit Random(std::uint64_t seed) : state_(seed) {}
  std::uint64_t next();
  double uniform();  // [0, 1)
 private:
  std::uint64_t state_;
};

class Sampler {
 public:
  explicit Sampler(const SamplingOptions& options);
  const SamplingOptions& options() const { return options_; }
  std::int32_t sample(std::span<const float> logits);
 private:
  struct Candidate { float logit; std::int32_t token; };
  SamplingOptions options_;
  Random random_;
  std::vector<Candidate> candidates_;
};
}  // namespace neuron
