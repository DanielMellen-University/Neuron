#define NEURON_TEST_MAIN
#include "neuron/stop.hpp"
#include "neuron_test.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace {
// Feeds pieces one at a time and returns everything the matcher let through.
std::string stream(neuron::StopMatcher& m, const std::vector<std::string>& pieces) {
  std::string out;
  for (const auto& piece : pieces) {
    out += m.push(piece);
    if (m.stopped()) return out;
  }
  return out + m.finish();
}
}
NEURON_TEST(no_stops_passes_text_through) {
  neuron::StopMatcher m;
  CHECK(m.empty());
  CHECK(m.matched().empty());
  CHECK_EQ(m.push("abc"), std::string("abc"));
  CHECK_EQ(m.finish(), std::string());
  CHECK(!m.stopped());
}
NEURON_TEST(stop_inside_one_piece_truncates) {
  neuron::StopMatcher m({"END"});
  CHECK_EQ(stream(m, {"hello END world"}), std::string("hello "));
  CHECK(m.stopped());
  CHECK_EQ(m.matched(), std::string_view("END"));
  CHECK_EQ(m.push("more"), std::string());
}
NEURON_TEST(stop_split_across_pieces) {
  neuron::StopMatcher m({"\nUser:"});
  std::string seen;
  seen += m.push("Sure.");
  seen += m.push("\nUs");
  CHECK_EQ(seen, std::string("Sure."));  // "\nUs" might start the stop string.
  seen += m.push("er: hi");
  CHECK_EQ(seen, std::string("Sure."));
  CHECK(m.stopped());
}
NEURON_TEST(held_prefix_is_released_when_it_diverges) {
  neuron::StopMatcher m({"###"});
  CHECK_EQ(m.push("a#"), std::string("a"));
  CHECK_EQ(m.push("#"), std::string());
  CHECK_EQ(m.push("b"), std::string("##b"));
  CHECK_EQ(m.push("##"), std::string());
  CHECK_EQ(m.finish(), std::string("##"));
  CHECK(!m.stopped());
}
NEURON_TEST(earliest_and_longest_match_wins) {
  neuron::StopMatcher first({"world", "lo"});
  CHECK_EQ(stream(first, {"hello world"}), std::string("hel"));
  CHECK_EQ(first.matched(), std::string_view("lo"));
  neuron::StopMatcher longer({"ab", "abc"});
  CHECK_EQ(stream(longer, {"xabc"}), std::string("x"));
  CHECK_EQ(longer.matched(), std::string_view("abc"));
}
NEURON_TEST(multibyte_stop_and_validation) {
  neuron::StopMatcher m({"\xe2\x98\x83"});
  CHECK_EQ(stream(m, {"snow", "\xe2", "\x98", "\x83", "man"}), std::string("snow"));
  CHECK(m.stopped());
  CHECK_THROWS_AS(neuron::StopMatcher({"ok", ""}), std::invalid_argument);
}
