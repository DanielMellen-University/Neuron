#define NEURON_TEST_MAIN
#include "../src/unicode.hpp"
#include "neuron_test.hpp"

NEURON_TEST(qwen2_unicode_categories_match_pinned_reference_version) {
  using neuron::unicode::category;
  CHECK_EQ(category('a'), 1u); CHECK_EQ(category('9'), 2u);
  CHECK_EQ(category(0x2ebf0), 1u); CHECK_EQ(category(0x2ee5d), 1u);  // Added in 15.1.
  CHECK_EQ(category(0x1c89), 0u); CHECK_EQ(category(0x10d40), 0u);  // Added after 15.1.
  CHECK_EQ(category(0x301), 0u); CHECK_EQ(category(0x10ffff), 0u);
  CHECK((neuron::unicode::qwen2_words("a\u1c89b") == std::vector<std::string_view>{"a", "\u1c89b"}));
  CHECK((neuron::unicode::qwen2_words("a\U0002ebf0b") == std::vector<std::string_view>{"a\U0002ebf0b"}));
  CHECK((neuron::unicode::qwen2_words("a\U00010d40b") == std::vector<std::string_view>{"a", "\U00010d40b"}));
}
NEURON_TEST(qwen2_whitespace_contractions_and_numeric_boundaries) {
  CHECK((neuron::unicode::qwen2_words("  a\r\n \t12!!\n'RE") ==
      std::vector<std::string_view>{" ", " a", "\r\n", " ", "\t", "1", "2", "!!\n", "'RE"}));
  CHECK(neuron::unicode::space(0x2028)); CHECK(neuron::unicode::space(0xa0));
  CHECK(!neuron::unicode::space(0x200b)); CHECK(!neuron::unicode::space(0x1c));
  for (const std::string& text : std::vector<std::string>{"\xc0\x80", "\xe0\x80\x80", "\xed\xa0\x80", "\xf4\x90\x80\x80", "\xf0\x9f"})
    CHECK_THROWS_AS(neuron::unicode::qwen2_words(text), std::runtime_error);
}
