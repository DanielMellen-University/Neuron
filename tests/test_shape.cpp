#define NEURON_TEST_MAIN
#include <stdexcept>

#include "neuron/tensor.hpp"
#include "neuron_test.hpp"

using neuron::Shape;

NEURON_TEST(scalar_shape_has_one_element) {
  Shape s;
  CHECK_EQ(s.rank(), 0u);
  CHECK_EQ(s.numel(), 1);
  CHECK_EQ(s.to_string(), std::string("[]"));
}

NEURON_TEST(numel_is_product_of_dims) {
  // A Qwen2.5-1.5B attention score block: heads x query x key.
  Shape s{12, 7, 128};
  CHECK_EQ(s.rank(), 3u);
  CHECK_EQ(s.numel(), 12 * 7 * 128);
}

NEURON_TEST(row_major_strides) {
  Shape s{2, 3, 4};
  CHECK_EQ(s.stride(0), 12);
  CHECK_EQ(s.stride(1), 4);
  CHECK_EQ(s.stride(2), 1);
}

NEURON_TEST(offset_walks_last_dim_fastest) {
  Shape s{2, 3, 4};
  CHECK_EQ(s.offset({0, 0, 0}), 0);
  CHECK_EQ(s.offset({0, 0, 1}), 1);
  CHECK_EQ(s.offset({0, 1, 0}), 4);
  CHECK_EQ(s.offset({1, 2, 3}), s.numel() - 1);
}

NEURON_TEST(offset_rejects_bad_indices) {
  Shape s{2, 3};
  CHECK_THROWS_AS(s.offset({2, 0}), std::out_of_range);
  CHECK_THROWS_AS(s.offset({0, -1}), std::out_of_range);
  CHECK_THROWS_AS(s.offset({0}), std::invalid_argument);
}

NEURON_TEST(constructor_rejects_invalid_shapes) {
  CHECK_THROWS_AS((Shape{2, 0}), std::invalid_argument);
  CHECK_THROWS_AS((Shape{1, 2, 3, 4, 5}), std::invalid_argument);
}

NEURON_TEST(equality_and_formatting) {
  CHECK(Shape({4, 8}) == Shape({4, 8}));
  CHECK(!(Shape({4, 8}) == Shape({8, 4})));
  CHECK(!(Shape({4}) == Shape({4, 1})));
  CHECK_EQ(Shape({2, 3, 4}).to_string(), std::string("[2, 3, 4]"));
  CHECK_THROWS_AS(Shape({2}).dim(1), std::out_of_range);
}
