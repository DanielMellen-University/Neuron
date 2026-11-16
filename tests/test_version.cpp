#define NEURON_TEST_MAIN
#include <string>

#include "neuron/version.hpp"
#include "neuron_test.hpp"

NEURON_TEST(version_string_matches_numbers) {
  const neuron::Version v = neuron::version();
  const std::string expected = std::to_string(v.major) + "." + std::to_string(v.minor) +
                               "." + std::to_string(v.patch);
  CHECK_EQ(std::string(neuron::version_string()), expected);
}

NEURON_TEST(compiled_isa_is_reported) {
  CHECK(!neuron::compiled_isa().empty());
}
