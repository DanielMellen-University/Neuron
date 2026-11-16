#define NEURON_TEST_MAIN
#include "neuron/tokenizer.hpp"
#include "neuron/gguf.hpp"
#include "neuron_test.hpp"
#include "gguf_fixture.hpp"

#include <limits>

namespace {
neuron::Vocabulary vocab() {
  neuron::Vocabulary v;
  v.tokens = {{"<unk>", 0, 2}, {"<s>", 0, 3}, {"</s>", 0, 3}};
  for (unsigned b = 0; b < 256; ++b) {
    char name[7]; std::snprintf(name, sizeof(name), "<0x%02X>", b);
    v.tokens.push_back({name, 0, 6});
  }
  for (const auto* text : {"▁", "a", "b", "c", "ab", "bc", "▁a", "▁ab", "é", "1"}) v.tokens.push_back({text, 1, 1});
  return v;
}
}
NEURON_TEST(empty_prefix_specials_and_byte_fallback) {
  neuron::Tokenizer t(vocab());
  CHECK(t.encode("") == std::vector<std::int32_t>{1}); CHECK(t.encode("", false).empty());
  CHECK((t.encode("a") == std::vector<std::int32_t>{1, 265}));
  CHECK((t.encode(" a") == std::vector<std::int32_t>{1, 259, 265}));
  CHECK((t.encode("é") == std::vector<std::int32_t>{1, 259, 267}));
  CHECK((t.encode("🐈") == std::vector<std::int32_t>{1, 259, 243, 162, 147, 139}));
  CHECK((t.encode("11") == std::vector<std::int32_t>{1, 259, 268, 268}));
  CHECK((t.encode("<s>a", false) == std::vector<std::int32_t>{1, 265}));
  CHECK(t.encode("<s>", false, false).size() > 1);
  CHECK_EQ(t.decode(t.encode(" a é 🐈 11")), std::string(" a é 🐈 11"));
  const std::string binary("\0\xff", 2);
  CHECK_EQ(t.decode(t.encode(binary)), binary);
  CHECK_THROWS_AS(t.piece(-1), std::runtime_error);
}
NEURON_TEST(score_priority_and_leftmost_tie) {
  auto v = vocab(); v.add_space_prefix = false;
  // ab and bc tie, so the left pair wins. Raising bc's score changes it.
  neuron::Tokenizer first(v); CHECK((first.encode("abc", false) == std::vector<std::int32_t>{263, 262}));
  v.tokens[264].score = 2;
  neuron::Tokenizer second(v); CHECK((second.encode("abc", false) == std::vector<std::int32_t>{260, 264}));
  v.add_eos = true; neuron::Tokenizer eos(v); CHECK((eos.encode("") == std::vector<std::int32_t>{1, 2}));
}
NEURON_TEST(gguf_vocabulary_and_validation) {
  const auto v = vocab(); fixture::Bytes b; b.header(0, 4); b.text("tokenizer.ggml.model", "llama");
  b.key("tokenizer.ggml.tokens", 9); b.put<std::uint32_t>(8); b.put<std::uint64_t>(v.tokens.size());
  for (const auto& token : v.tokens) b.string(token.text);
  b.key("tokenizer.ggml.scores", 9); b.put<std::uint32_t>(6); b.put<std::uint64_t>(v.tokens.size());
  for (const auto& token : v.tokens) b.put(token.score);
  b.key("tokenizer.ggml.token_type", 9); b.put<std::uint32_t>(5); b.put<std::uint64_t>(v.tokens.size());
  for (const auto& token : v.tokens) b.put(token.type);
  fixture::File f(b); neuron::Gguf g(f.path); neuron::Tokenizer loaded(g);
  CHECK(loaded.encode("ab é") == neuron::Tokenizer(v).encode("ab é"));
  auto bad = v; bad.tokens.push_back(bad.tokens.back());
  CHECK_THROWS_AS(neuron::Tokenizer{bad}, std::runtime_error);
  bad = v; bad.tokens[0].score = std::numeric_limits<float>::infinity();
  CHECK_THROWS_AS(neuron::Tokenizer{bad}, std::runtime_error);
  bad = v; bad.bos = -1; CHECK_THROWS_AS(neuron::Tokenizer{bad}, std::runtime_error);
}
