#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace neuron {
class Gguf;
struct VocabToken {
  std::string text;
  float score = 0;
  std::int32_t type = 1;  // GGUF: normal, unknown, control, user, unused, byte.
};
struct Vocabulary {
  std::vector<VocabToken> tokens;
  std::int32_t bos = 1;
  std::int32_t eos = 2;
  bool add_bos = true;
  bool add_eos = false;
  bool add_space_prefix = true;
  bool byte_bpe = false;  // GPT-2 byte alphabet, with the Qwen2 pre-tokenizer.
  std::vector<std::string> merges;
};

// Llama score-ordered merging and Qwen2 rank-ordered byte BPE.
class Tokenizer {
 public:
  explicit Tokenizer(Vocabulary vocabulary);
  explicit Tokenizer(const Gguf& file);
  std::vector<std::int32_t> encode(std::string_view text, bool add_special = true,
                                 bool parse_special = true) const;
  std::string piece(std::int32_t token) const;
  std::string decode(const std::vector<std::int32_t>& tokens, bool remove_prefix = true) const;
  std::int32_t eos() const { return vocabulary_.eos; }
  bool is_eog(std::int32_t token) const;
  std::size_t size() const { return vocabulary_.tokens.size(); }
 private:
  void encode_fragment(std::string_view text, bool prefix, std::vector<std::int32_t>& out) const;
  void encode_symbols(const std::string& text, std::vector<std::int32_t>& out) const;
  Vocabulary vocabulary_;
  std::unordered_map<std::string, std::size_t> ranks_;
  std::array<std::string, 256> byte_alphabet_{};
  std::unordered_map<std::uint32_t, char> byte_decoder_;
  std::unordered_map<std::string, std::int32_t> ids_;
  std::array<std::int32_t, 256> bytes_{};
  std::vector<std::int32_t> special_;
};
}  // namespace neuron
