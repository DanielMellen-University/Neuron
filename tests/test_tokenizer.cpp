#define NEURON_TEST_MAIN
#include "neuron/tokenizer.hpp"
#include "neuron/gguf.hpp"
#include "neuron_test.hpp"
#include "gguf_fixture.hpp"

#include <algorithm>
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
  // text is the stored entry; piece is the decoded bytes.
  CHECK_EQ(t.text(259), std::string("\xe2\x96\x81"));
  CHECK_EQ(t.piece(259), std::string(" "));
  CHECK_EQ(t.text(1), std::string("<s>"));
  CHECK(t.piece(1).empty());
  CHECK_EQ(t.text(3 + 0x41), std::string("<0x41>"));
  CHECK_EQ(t.piece(3 + 0x41), std::string("A"));
  CHECK_THROWS_AS(t.text(static_cast<std::int32_t>(t.size())), std::runtime_error);
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

namespace {
neuron::Vocabulary bpe_vocab() {
  neuron::Vocabulary v;
  v.byte_bpe = true; v.add_bos = false; v.add_space_prefix = false;
  v.bos = 0; v.eos = 2;
  v.tokens = {{"<|endoftext|>",0,3}, {"<|im_start|>",0,3}, {"<|im_end|>",0,3}};
  unsigned extra = 256;
  for (unsigned i = 0; i < 256; ++i) {
    const unsigned code = (i >= 33 && i <= 126) || (i >= 161 && i <= 172) || i >= 174 ? i : extra++;
    std::string text;
    if (code < 128) text += static_cast<char>(code);
    else { text += static_cast<char>(0xc0u | (code >> 6)); text += static_cast<char>(0x80u | (code & 63u)); }
    v.tokens.push_back({text, 0, 1});
  }
  v.merges = {"a b", "b c", "ab c", "Ġ a", "Ġ ab", "1 2", "! !", "!! Ċ", "Ġ Ġ", "' S", "Ġ 1"};
  for (const auto& merge : v.merges) {
    const auto at = merge.find(' ');
    v.tokens.push_back({merge.substr(0, at) + merge.substr(at + 1), 0, 1});
  }
  for (const auto* text : {"<|fim_pad|>", "<|fim_repo|>", "<|file_sep|>"}) v.tokens.push_back({text,0,3});
  return v;
}
}
NEURON_TEST(byte_bpe_rank_regex_boundaries_and_decode) {
  auto v = bpe_vocab(); neuron::Tokenizer t(v);
  auto id = [&](const std::string& text) {
    const auto it = std::find_if(v.tokens.begin(), v.tokens.end(), [&](const auto& token) { return token.text == text; });
    return static_cast<std::int32_t>(it - v.tokens.begin());
  };
  // BPE uses merge ranks even when all token scores tie.
  CHECK((t.encode("abc") == std::vector<std::int32_t>{id("abc")}));
  CHECK((t.encode(" ab") == std::vector<std::int32_t>{id("Ġab")}));
  // Qwen2 separates each digit, including a leading space.
  CHECK((t.encode(" 12") == std::vector<std::int32_t>{id("Ġ"), id("1"), id("2")}));
  CHECK((t.encode("!!\n") == std::vector<std::int32_t>{id("!!Ċ")}));
  CHECK((t.encode("'S") == std::vector<std::int32_t>{id("'S")}));
  CHECK((t.encode("  a") == std::vector<std::int32_t>{id("Ġ"), id("Ġa")}));
  CHECK((t.encode("  ") == std::vector<std::int32_t>{id("ĠĠ")}));
  CHECK((t.encode("<|im_start|>abc<|im_end|>") == std::vector<std::int32_t>{1,id("abc"),2}));
  CHECK(t.encode("<|im_start|>", false, false).size() > 1);
  CHECK(t.is_eog(0)); CHECK(t.is_eog(2)); CHECK(!t.is_eog(1)); CHECK(!t.is_eog(-1));
  CHECK(t.is_eog(id("<|fim_pad|>"))); CHECK(t.is_eog(id("<|fim_repo|>"))); CHECK(t.is_eog(id("<|file_sep|>")));
  for (const std::string& text : std::vector<std::string>{"café 世界 🐈", "a\u00a0b\u2003c", "1234567890", "\t \r\n  abc  ", std::string("\0\x01", 2)})
    CHECK_EQ(t.decode(t.encode(text)), text);
  CHECK_THROWS_AS(t.encode(std::string("\xff", 1)), std::runtime_error);
  v.merges.push_back("a b"); CHECK_THROWS_AS(neuron::Tokenizer{v}, std::runtime_error);
  v = bpe_vocab(); v.merges.push_back("a missing"); CHECK_THROWS_AS(neuron::Tokenizer{v}, std::runtime_error);
}
