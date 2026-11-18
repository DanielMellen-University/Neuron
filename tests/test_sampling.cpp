#define NEURON_TEST_MAIN
#include "neuron/sampling.hpp"
#include "neuron_test.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace {
std::map<std::int32_t, int> histogram(neuron::SamplingOptions options, std::span<const float> logits, int draws) {
  neuron::Sampler sampler(options);
  std::map<std::int32_t, int> counts;
  for (int i = 0; i < draws; ++i) ++counts[sampler.sample(logits)];
  return counts;
}
}
NEURON_TEST(greedy_matches_first_maximum) {
  const std::array<float, 5> logits{1, 3, -2, 3, 0};
  neuron::Sampler sampler({});
  CHECK_EQ(sampler.sample(logits), 1);
  CHECK_EQ(neuron::argmax(logits), 1);
  CHECK_THROWS_AS(neuron::argmax(std::span<const float>{}), std::invalid_argument);
}
NEURON_TEST(validation_names_the_field) {
  auto rejects = [](neuron::SamplingOptions o) {
    try { neuron::Sampler s(o); } catch (const std::invalid_argument&) { return true; }
    return false;
  };
  neuron::SamplingOptions o;
  o.temperature = -1; CHECK(rejects(o));
  o.temperature = std::numeric_limits<float>::infinity(); CHECK(rejects(o));
  o = {}; o.top_p = 0; CHECK(rejects(o));
  o.top_p = 1.5f; CHECK(rejects(o));
  o = {}; o.min_p = 1; CHECK(rejects(o));
  o.min_p = -0.1f; CHECK(rejects(o));
  o = {}; o.temperature = 0.7f; o.top_p = 0.9f; o.min_p = 0.05f; CHECK(!rejects(o));
}
NEURON_TEST(seed_reproduces_sequence) {
  const std::array<float, 4> logits{0, 0.5f, 1, 0.25f};
  neuron::SamplingOptions o; o.temperature = 1; o.seed = 42;
  neuron::Sampler a(o), b(o);
  std::vector<std::int32_t> first, second;
  for (int i = 0; i < 64; ++i) { first.push_back(a.sample(logits)); second.push_back(b.sample(logits)); }
  CHECK(first == second);
  o.seed = 43; neuron::Sampler c(o);
  std::vector<std::int32_t> other;
  for (int i = 0; i < 64; ++i) other.push_back(c.sample(logits));
  CHECK(first != other);
  // SplitMix64 reference value for seed 0, so seeds mean the same on every platform.
  neuron::Random r(0);
  CHECK_EQ(r.next(), 0xe220a8397b1dcdafULL);
}
NEURON_TEST(temperature_one_follows_softmax) {
  const std::array<float, 2> logits{0, std::log(3.0f)};
  neuron::SamplingOptions o; o.temperature = 1; o.seed = 7;
  const auto counts = histogram(o, logits, 20000);
  const double share = counts.at(1) / 20000.0;
  CHECK_LT(std::abs(share - 0.75), 0.02);
  // A low temperature sharpens toward the best token.
  o.temperature = 0.25f;
  const auto sharp = histogram(o, logits, 20000);
  CHECK_LT(0.98, sharp.at(1) / 20000.0);
}
NEURON_TEST(top_k_keeps_only_best_tokens) {
  const std::array<float, 5> logits{4, 1, 3, 2, 0};
  neuron::SamplingOptions o; o.temperature = 5; o.top_k = 2; o.seed = 1;
  const auto counts = histogram(o, logits, 2000);
  CHECK_EQ(counts.size(), std::size_t{2});
  CHECK(counts.contains(0)); CHECK(counts.contains(2));
  o.top_k = 1;
  CHECK_EQ(histogram(o, logits, 200).size(), std::size_t{1});
}
NEURON_TEST(top_p_keeps_smallest_covering_set) {
  // Probabilities 0.5, 0.3, 0.2 at temperature 1.
  const std::array<float, 3> logits{std::log(0.5f), std::log(0.3f), std::log(0.2f)};
  neuron::SamplingOptions o; o.temperature = 1; o.top_p = 0.75f; o.seed = 3;
  auto counts = histogram(o, logits, 4000);
  CHECK_EQ(counts.size(), std::size_t{2});
  CHECK(!counts.contains(2));
  CHECK_LT(std::abs(counts.at(0) / 4000.0 - 0.625), 0.03);
  o.top_p = 0.4f;
  counts = histogram(o, logits, 200);
  CHECK_EQ(counts.size(), std::size_t{1});
  CHECK(counts.contains(0));
}
NEURON_TEST(min_p_drops_unlikely_tokens) {
  const std::array<float, 4> logits{std::log(0.6f), std::log(0.3f), std::log(0.08f), std::log(0.02f)};
  neuron::SamplingOptions o; o.temperature = 1; o.min_p = 0.1f; o.seed = 5;
  const auto counts = histogram(o, logits, 4000);
  CHECK(counts.contains(0)); CHECK(counts.contains(1)); CHECK(counts.contains(2));
  CHECK(!counts.contains(3));
}
NEURON_TEST(masked_logits_are_never_sampled) {
  const float masked = -std::numeric_limits<float>::infinity();
  const std::array<float, 4> logits{masked, 0, masked, 0};
  neuron::SamplingOptions o; o.temperature = 2; o.seed = 9;
  const auto counts = histogram(o, logits, 500);
  CHECK(!counts.contains(0)); CHECK(!counts.contains(2));
  CHECK_EQ(counts.size(), std::size_t{2});
}
