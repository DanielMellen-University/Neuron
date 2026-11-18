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
  // llama.cpp style: logits of tokens seen in the last repeat_last_n accepted
  // tokens are divided by the penalty when positive and multiplied when
  // negative. 1 turns it off; it also applies when temperature is 0.
  float repeat_penalty = 1;
  std::size_t repeat_last_n = 64;
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
  // Records a prompt or generated token for the repeat penalty window.
  void accept(std::int32_t token);
  void reset();
 private:
  struct Candidate { float logit; std::int32_t token; };
  SamplingOptions options_;
  Random random_;
  std::vector<Candidate> candidates_;
  std::vector<std::int32_t> recent_;  // Ring buffer of the last repeat_last_n tokens.
  std::size_t next_recent_ = 0;
  std::vector<float> penalized_;
};
}  // namespace neuron
