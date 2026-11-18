#include "neuron/tokenizer.hpp"
#include "neuron/gguf.hpp"
#include "unicode.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>

namespace neuron {
namespace {
[[noreturn]] void fail(const std::string& message) { throw std::runtime_error("tokenizer: " + message); }
Vocabulary load_vocabulary(const Gguf& file) {
  const auto model = file.at("tokenizer.ggml.model").string();
  if (model != "llama" && model != "gpt2") fail("only llama and Qwen2 byte BPE vocabularies are supported");
  if (model == "gpt2" && file.at("tokenizer.ggml.pre").string() != "qwen2" &&
      file.at("tokenizer.ggml.pre").string() != "smollm") fail("unsupported BPE pre-tokenizer");
  const auto& tokens = file.at("tokenizer.ggml.tokens").array();
  const auto* scores_value = file.find("tokenizer.ggml.scores");
  if (!scores_value && model == "llama") fail("missing llama token scores");
  const auto& types = file.at("tokenizer.ggml.token_type").array();
  if ((scores_value && tokens.size() != scores_value->array().size()) || tokens.size() != types.size()) fail("vocabulary arrays differ in size");
  Vocabulary v;
  v.byte_bpe = model == "gpt2";
  v.smollm = v.byte_bpe && file.at("tokenizer.ggml.pre").string() == "smollm";
  if (v.byte_bpe) for (const auto& merge : file.at("tokenizer.ggml.merges").array()) v.merges.push_back(merge.string());
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    if (types[i].type != GgufType::i32) fail("token types must be int32");
    const auto type = std::get<std::int64_t>(types[i].data);
    if (type < 1 || type > 6) fail("invalid token type");
    v.tokens.push_back({tokens[i].string(), scores_value ? static_cast<float>(scores_value->array()[i].number()) : 0.0f, static_cast<std::int32_t>(type)});
  }
  auto id = [&](const char* key, std::int32_t fallback) {
    const auto* value = file.find(key);
    if (!value) return fallback;
    const auto n = value->uint();
    if (n >= tokens.size()) fail("invalid special token id");
    return static_cast<std::int32_t>(n);
  };
  v.bos = id("tokenizer.ggml.bos_token_id", 1);
  v.eos = id("tokenizer.ggml.eos_token_id", 2);
  auto flag = [&](const char* key, bool fallback) {
    const auto* value = file.find(key); return value ? value->boolean() : fallback;
  };
  v.add_bos = flag("tokenizer.ggml.add_bos_token", !v.byte_bpe);
  v.add_eos = flag("tokenizer.ggml.add_eos_token", false);
  v.add_space_prefix = flag("tokenizer.ggml.add_space_prefix", !v.byte_bpe);
  return v;
}
}  // namespace
Tokenizer::Tokenizer(const Gguf& file) : Tokenizer(load_vocabulary(file)) {}
Tokenizer::Tokenizer(Vocabulary vocabulary) : vocabulary_(std::move(vocabulary)) {
  if (vocabulary_.tokens.empty() || vocabulary_.tokens.size() > static_cast<std::size_t>(INT32_MAX))
    fail("invalid vocabulary size");
  auto valid_id = [&](std::int32_t id) { return id >= 0 && static_cast<std::size_t>(id) < size(); };
  if (!valid_id(vocabulary_.bos) || !valid_id(vocabulary_.eos)) fail("invalid BOS or EOS");
  bytes_.fill(-1);
  for (std::size_t i = 0; i < size(); ++i) {
    const auto& token = vocabulary_.tokens[i];
    const auto id = static_cast<std::int32_t>(i);
    if (token.text.empty() || !std::isfinite(token.score) || token.type < 1 || token.type > 6)
      fail("invalid vocabulary entry");
    if (!ids_.emplace(token.text, id).second) fail("duplicate vocabulary entry");
    if (token.type == 2 || token.type == 3 || token.type == 4) special_.push_back(id);
  }
  for (std::size_t i = 0; i < bytes_.size(); ++i) {
    char name[7]; std::snprintf(name, sizeof(name), "<0x%02X>", static_cast<unsigned>(i));
    const auto it = ids_.find(name);
    if (it != ids_.end() && vocabulary_.tokens[static_cast<std::size_t>(it->second)].type == 6) bytes_[i] = it->second;
  }
  if (vocabulary_.byte_bpe) {
    std::uint32_t extra = 256;
    for (std::size_t i = 0; i < 256; ++i) {
      const auto code = (i >= 33 && i <= 126) || (i >= 161 && i <= 172) || i >= 174
          ? static_cast<std::uint32_t>(i) : extra++;
      byte_alphabet_[i] = unicode::utf8(code);
      byte_decoder_.emplace(code, static_cast<char>(i));
      const auto it = ids_.find(byte_alphabet_[i]);
      if (it == ids_.end()) {
        if (!vocabulary_.smollm) fail("missing BPE byte token");
      } else bytes_[i] = it->second;
    }
    for (std::size_t i = 0; i < vocabulary_.merges.size(); ++i) {
      const auto& merge = vocabulary_.merges[i];
      const auto at = merge.find(' ');
      if (at == std::string::npos || at == 0 || at + 1 == merge.size() || merge.find(' ', at + 1) != std::string::npos)
        fail("invalid BPE merge");
      if (!ranks_.emplace(merge, i).second) fail("duplicate BPE merge");
      if (!ids_.contains(merge.substr(0, at)) || !ids_.contains(merge.substr(at + 1)) ||
          !ids_.contains(merge.substr(0, at) + merge.substr(at + 1))) fail("BPE merge missing vocabulary entry");
    }
  }
  std::sort(special_.begin(), special_.end(), [&](auto a, auto b) {
    const auto na = vocabulary_.tokens[static_cast<std::size_t>(a)].text.size();
    const auto nb = vocabulary_.tokens[static_cast<std::size_t>(b)].text.size();
    return na != nb ? na > nb : a < b;
  });
}
void Tokenizer::encode_fragment(std::string_view raw, bool prefix, std::vector<std::int32_t>& out) const {
  if (vocabulary_.byte_bpe) {
    for (const auto word : unicode::qwen2_words(raw, vocabulary_.smollm)) {
      std::string encoded;
      for (const auto c : word) encoded += byte_alphabet_[static_cast<unsigned char>(c)];
      encode_symbols(encoded, out);
    }
    return;
  }
  std::string text;
  if (prefix && vocabulary_.add_space_prefix) text = "▁";
  for (char c : raw) { if (c == ' ') text += "▁"; else text += c; }
  encode_symbols(text, out);
}
void Tokenizer::encode_symbols(const std::string& text, std::vector<std::int32_t>& out) const {
  struct Symbol { std::size_t offset, length, previous, next; };
  const auto none = std::numeric_limits<std::size_t>::max();
  std::vector<Symbol> symbols;
  for (std::size_t offset = 0; offset < text.size();) {
    const auto c = static_cast<unsigned char>(text[offset]);
    const std::size_t width = c < 0xc0 ? 1 : c < 0xe0 ? 2 : c < 0xf0 ? 3 : 4;
    const auto length = std::min(width, text.size() - offset);
    const auto i = symbols.size();
    symbols.push_back({offset, length, i == 0 ? none : i - 1, i + 1});
    offset += length;
  }
  if (symbols.empty()) return;
  symbols.back().next = none;
  struct Pair { float score; std::size_t left, right, length; };
  auto lower_priority = [](const Pair& a, const Pair& b) {
    return a.score != b.score ? a.score < b.score : a.left > b.left;
  };
  std::priority_queue<Pair, std::vector<Pair>, decltype(lower_priority)> queue(lower_priority);
  auto offer = [&](std::size_t left, std::size_t right) {
    if (left == none || right == none) return;
    const auto length = symbols[left].length + symbols[right].length;
    if (vocabulary_.byte_bpe) {
      const auto pair = text.substr(symbols[left].offset, symbols[left].length) + " " +
          text.substr(symbols[right].offset, symbols[right].length);
      const auto it = ranks_.find(pair);
      if (it != ranks_.end()) queue.push({-static_cast<float>(it->second), left, right, length});
    } else {
      const auto it = ids_.find(text.substr(symbols[left].offset, length));
      if (it != ids_.end()) queue.push({vocabulary_.tokens[static_cast<std::size_t>(it->second)].score, left, right, length});
    }
  };
  for (std::size_t i = 1; i < symbols.size(); ++i) offer(i - 1, i);
  while (!queue.empty()) {
    const auto p = queue.top(); queue.pop();
    auto& left = symbols[p.left]; auto& right = symbols[p.right];
    if (left.length == 0 || right.length == 0 || left.next != p.right || left.length + right.length != p.length) continue;
    left.length = p.length; right.length = 0; left.next = right.next;
    if (left.next != none) symbols[left.next].previous = p.left;
    offer(left.previous, p.left); offer(p.left, left.next);
  }
  for (std::size_t i = 0; i != none; i = symbols[i].next) {
    const auto& symbol = symbols[i];
    const auto it = ids_.find(text.substr(symbol.offset, symbol.length));
    if (it != ids_.end()) out.push_back(it->second);
    else if (vocabulary_.byte_bpe) fail("BPE result missing vocabulary entry");
    else for (std::size_t j = 0; j < symbol.length; ++j) {
      const auto byte = static_cast<unsigned char>(text[symbol.offset + j]);
      if (bytes_[byte] < 0) fail("missing byte fallback token for " + std::to_string(byte));
      out.push_back(bytes_[byte]);
    }
  }
}
std::vector<std::int32_t> Tokenizer::encode(std::string_view text, bool add_special, bool parse_special) const {
  struct Fragment { std::string_view text; std::int32_t id = -1; };
  std::vector<Fragment> fragments;
  if (!text.empty()) fragments.push_back({text});
  // Partition longer special tokens first, before applying score-based merges.
  for (auto id : special_) {
    const auto& token = vocabulary_.tokens[static_cast<std::size_t>(id)];
    if (!parse_special && token.type != 4) continue;
    std::vector<Fragment> split;
    for (const auto& fragment : fragments) {
      if (fragment.id >= 0) { split.push_back(fragment); continue; }
      auto rest = fragment.text;
      std::size_t at;
      while ((at = rest.find(token.text)) != std::string_view::npos) {
        if (at != 0) split.push_back({rest.substr(0, at)});
        split.push_back({{}, id}); rest.remove_prefix(at + token.text.size());
      }
      if (!rest.empty()) split.push_back({rest});
    }
    fragments = std::move(split);
  }
  std::vector<std::int32_t> out;
  if (add_special && vocabulary_.add_bos) out.push_back(vocabulary_.bos);
  bool prefix = true;
  for (const auto& fragment : fragments) {
    if (fragment.id >= 0) { out.push_back(fragment.id); prefix = true; }
    else { encode_fragment(fragment.text, prefix, out); prefix = false; }
  }
  if (add_special && vocabulary_.add_eos) out.push_back(vocabulary_.eos);
  return out;
}
const std::string& Tokenizer::text(std::int32_t id) const {
  if (id < 0 || static_cast<std::size_t>(id) >= size()) fail("token id out of range");
  return vocabulary_.tokens[static_cast<std::size_t>(id)].text;
}
std::string Tokenizer::piece(std::int32_t id) const {
  if (id < 0 || static_cast<std::size_t>(id) >= size()) fail("token id out of range");
  const auto& token = vocabulary_.tokens[static_cast<std::size_t>(id)];
  if (token.type == 3) return {};
  if (token.type == 6) {
    for (std::size_t i = 0; i < bytes_.size(); ++i) if (bytes_[i] == id) return std::string(1, static_cast<char>(i));
    fail("invalid byte token");
  }
  if (vocabulary_.byte_bpe && token.type == 1) {
    std::string result;
    for (std::size_t at = 0; at < token.text.size();) {
      const auto it = byte_decoder_.find(unicode::read(token.text, at));
      if (it == byte_decoder_.end()) fail("invalid BPE byte alphabet");
      result += it->second;
    }
    return result;
  }
  std::string out = token.text;
  std::size_t at = 0;
  while ((at = out.find("▁", at)) != std::string::npos) { out.replace(at, 3, " "); ++at; }
  return out;
}
bool Tokenizer::is_eog(std::int32_t token) const {
  if (token == eos()) return true;
  if (!vocabulary_.byte_bpe || token < 0 || static_cast<std::size_t>(token) >= size()) return false;
  const auto& text = vocabulary_.tokens[static_cast<std::size_t>(token)].text;
  return text == "<|im_end|>" || text == "<|endoftext|>" || text == "<|fim_pad|>" ||
      text == "<|fim_repo|>" || text == "<|file_sep|>";
}
std::string Tokenizer::decode(const std::vector<std::int32_t>& tokens, bool remove_prefix) const {
  std::string out;
  for (auto id : tokens) out += piece(id);
  if (remove_prefix && vocabulary_.add_space_prefix && !out.empty() && out.front() == ' ') out.erase(0, 1);
  return out;
}
}  // namespace neuron
