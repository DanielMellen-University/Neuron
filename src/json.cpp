#include "json.hpp"
#include "utf8.hpp"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace neuron::json {
const Value* Value::find(const std::string& key) const {
  const auto* object = std::get_if<Object>(&data);
  if (!object) return nullptr;
  const auto it = object->find(key);
  return it == object->end() ? nullptr : &it->second;
}
namespace {
[[noreturn]] void fail() { throw std::invalid_argument("invalid JSON"); }
class Parser {
 public:
  explicit Parser(std::string_view text) : text_(text) {}
  Value document() {
    auto result = value(0);
    space();
    if (pos_ != text_.size()) fail();
    return result;
  }
 private:
  void space() { while (pos_ < text_.size() && (text_[pos_] == ' ' || text_[pos_] == '\r' || text_[pos_] == '\n' || text_[pos_] == '\t')) ++pos_; }
  bool take(char c) { if (pos_ < text_.size() && text_[pos_] == c) { ++pos_; return true; } return false; }
  void expect(char c) { if (!take(c)) fail(); }
  std::uint32_t hex() {
    std::uint32_t result = 0;
    for (int i = 0; i < 4; ++i) {
      if (pos_ == text_.size()) fail();
      const char c = text_[pos_++];
      const int n = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
      if (n < 0) fail();
      result = result * 16 + static_cast<std::uint32_t>(n);
    }
    return result;
  }
  std::string string() {
    expect('"');
    std::string out;
    while (pos_ < text_.size()) {
      const char c = text_[pos_++];
      if (c == '"') { if (!detail::valid_utf8(out)) fail(); return out; }
      if (static_cast<unsigned char>(c) < 0x20) fail();
      if (c != '\\') { out += c; continue; }
      if (pos_ == text_.size()) fail();
      switch (text_[pos_++]) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          auto cp = hex();
          if (cp >= 0xd800 && cp <= 0xdbff) {
            expect('\\'); expect('u');
            const auto low = hex();
            if (low < 0xdc00 || low > 0xdfff) fail();
            cp = 0x10000 + ((cp - 0xd800) << 10) + low - 0xdc00;
          } else if (cp >= 0xdc00 && cp <= 0xdfff) fail();
          if (cp < 0x80) out += static_cast<char>(cp);
          else {
            if (cp >= 0x10000) out += static_cast<char>(0xf0u | (cp >> 18));
            if (cp >= 0x800) out += static_cast<char>((cp >= 0x10000 ? 0x80u : 0xe0u) | ((cp >> 12) & 0x3fu));
            out += static_cast<char>((cp >= 0x800 ? 0x80u : 0xc0u) | ((cp >> 6) & 0x3fu));
            out += static_cast<char>(0x80u | (cp & 0x3fu));
          }
          break;
        }
        default: fail();
      }
    }
    fail();
  }
  Value value(std::size_t depth) {
    if (depth > 64) fail();
    space();
    if (pos_ == text_.size()) fail();
    if (text_[pos_] == '"') return {string()};
    if (take('{')) {
      Value::Object object;
      space();
      if (take('}')) return {object};
      do {
        space(); const auto key = string(); space(); expect(':');
        if (!object.emplace(key, value(depth + 1)).second) fail();
        space(); if (take('}')) return {object};
      } while (take(','));
      fail();
    }
    if (take('[')) {
      Value::Array array;
      space();
      if (take(']')) return {array};
      do {
        array.push_back(value(depth + 1));
        space(); if (take(']')) return {array};
      } while (take(','));
      fail();
    }
    for (const auto literal : {std::string_view("true"), std::string_view("false"), std::string_view("null")}) {
      if (text_.substr(pos_, literal.size()) == literal) {
        pos_ += literal.size();
        if (literal == "null") return {};
        return {literal == "true"};
      }
    }
    const auto start = pos_;
    take('-');
    if (!take('0')) {
      if (pos_ == text_.size() || text_[pos_] < '1' || text_[pos_] > '9') fail();
      digits();
    }
    if (take('.')) { const auto before = pos_; digits(); if (before == pos_) fail(); }
    if (take('e') || take('E')) {
      if (!take('+')) take('-');
      const auto before = pos_; digits(); if (before == pos_) fail();
    }
    double number = 0;
    const auto parsed = std::from_chars(text_.data() + start, text_.data() + pos_, number);
    if (parsed.ec != std::errc{} || parsed.ptr != text_.data() + pos_ || !std::isfinite(number)) fail();
    return {number};
  }
  void digits() { while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') ++pos_; }
  std::string_view text_;
  std::size_t pos_ = 0;
};
}
Value parse(std::string_view text) { return Parser(text).document(); }
std::string quote(std::string_view text) {
  constexpr char hex[] = "0123456789abcdef";
  std::string out = "\"";
  for (const char raw : text) {
    const auto c = static_cast<unsigned char>(raw);
    if (c == '"' || c == '\\') { out += '\\'; out += raw; }
    else if (c < 0x20) { out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15]; }
    else out += raw;
  }
  return out + '"';
}
}  // namespace neuron::json
