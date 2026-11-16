#define NEURON_TEST_MAIN
#include "neuron/kernels.hpp"
#include "neuron_test.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace neuron::scalar;
NEURON_TEST(matmul_rectangular_and_batched) {
  const std::array<float, 6> w{1, 2, 3, 4, 5, 6};
  const std::array<float, 6> x{2, -1, 1, 0, 1, 2};
  std::array<float, 4> y{};
  matmul(w, x, y, 2, 3, 2);
  CHECK_EQ(y[0], 3.0f); CHECK_EQ(y[1], 9.0f); CHECK_EQ(y[2], 8.0f); CHECK_EQ(y[3], 17.0f);
  CHECK_THROWS_AS(matmul(w, x, y, 3, 2), std::invalid_argument);
}
NEURON_TEST(rmsnorm_hand_computed_and_in_place) {
  std::array<float, 2> x{1, 2}; const std::array<float, 2> w{2, 3};
  // mean square 2.5 + epsilon 1.5 = 4.
  rmsnorm(x, w, x, 1.5f);
  CHECK_EQ(x[0], 1.0f); CHECK_EQ(x[1], 3.0f);
  std::array<float, 2> zero{}; rmsnorm(zero, w, zero, 1e-5f);
  CHECK_EQ(zero[0], 0.0f); CHECK_EQ(zero[1], 0.0f);
  CHECK_THROWS_AS(rmsnorm(x, w, x, 0), std::invalid_argument);
}
NEURON_TEST(softmax_hand_computed_and_stable) {
  std::array<float, 3> a{0, std::log(2.0f), std::log(3.0f)};
  softmax(a);
  CHECK_LT(std::abs(a[0] - 1.0f / 6), 1e-6f);
  CHECK_LT(std::abs(a[1] - 2.0f / 6), 1e-6f);
  CHECK_LT(std::abs(a[2] - 0.5f), 1e-6f);
  std::array<float, 2> b{10000, 10000}; softmax(b);
  CHECK_EQ(b[0], 0.5f); CHECK_EQ(b[1], 0.5f);
}
NEURON_TEST(rope_adjacent_pairs_heads_and_partial_rotation) {
  std::array<float, 12> x{1, 0, 0, 1, 7, 8, 2, 0, 0, 2, 9, 10};
  rope(x, 6, 4, 1, 16);
  // Pair angles are 1 and 1/sqrt(16) = 0.25 in each head.
  CHECK_LT(std::abs(x[0] - std::cos(1.0f)), 1e-6f);
  CHECK_LT(std::abs(x[1] - std::sin(1.0f)), 1e-6f);
  CHECK_LT(std::abs(x[2] + std::sin(0.25f)), 1e-6f);
  CHECK_LT(std::abs(x[3] - std::cos(0.25f)), 1e-6f);
  CHECK_EQ(x[4], 7.0f); CHECK_EQ(x[5], 8.0f);
  CHECK_LT(std::abs(x[6] - 2 * std::cos(1.0f)), 1e-6f);
  CHECK_LT(std::abs(x[9] - 2 * std::cos(0.25f)), 1e-6f);
  CHECK_EQ(x[10], 9.0f); CHECK_EQ(x[11], 10.0f);
  auto unchanged = x; rope(x, 6, 4, 0); CHECK(x == unchanged);
  CHECK_THROWS_AS(rope(x, 6, 3, 0), std::invalid_argument);
}

NEURON_TEST(rope_neox_split_halves_and_multiple_heads) {
  std::array<float, 12> x{1,0,0,1,7,8, 2,0,0,2,9,10};
  rope(x, 6, 4, 1, 16, true);
  CHECK_LT(std::abs(x[0] - std::cos(1.0f)), 1e-6f);
  CHECK_LT(std::abs(x[2] - std::sin(1.0f)), 1e-6f);
  CHECK_LT(std::abs(x[1] + std::sin(0.25f)), 1e-6f);
  CHECK_LT(std::abs(x[3] - std::cos(0.25f)), 1e-6f);
  CHECK_LT(std::abs(x[6] - 2 * std::cos(1.0f)), 1e-6f);
  CHECK_LT(std::abs(x[9] - 2 * std::cos(0.25f)), 1e-6f);
  CHECK_EQ(x[4], 7.0f); CHECK_EQ(x[10], 9.0f);
}
