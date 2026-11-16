#pragma once

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace neuron::unicode {
struct Range { std::uint32_t first, last; unsigned category; };
inline constexpr Range ranges[] = {
#include "unicode_ranges.inc"
};
inline unsigned category(std::uint32_t code) {
  const auto it = std::upper_bound(std::begin(ranges), std::end(ranges), code,
      [](auto value, const Range& range) { return value < range.first; });
  return it != std::begin(ranges) && code <= (it - 1)->last ? (it - 1)->category : 0;
}
inline bool space(std::uint32_t code) {
  return (code >= 9 && code <= 13) || code == 32 || code == 0x85 || code == 0xa0 ||
      code == 0x1680 || (code >= 0x2000 && code <= 0x200a) || code == 0x2028 ||
      code == 0x2029 || code == 0x202f || code == 0x205f || code == 0x3000;
}
inline std::uint32_t read(std::string_view text, std::size_t& at) {
  const auto first = static_cast<unsigned char>(text[at++]);
  if (first < 128) return first;
  const unsigned length = first >= 0xc2 && first <= 0xdf ? 2u :
      first >= 0xe0 && first <= 0xef ? 3u : first >= 0xf0 && first <= 0xf4 ? 4u : 0u;
  if (length == 0 || length - 1 > text.size() - at) throw std::runtime_error("tokenizer: invalid UTF-8");
  std::uint32_t code = first & (0x7fu >> length);
  for (unsigned i = 1; i < length; ++i) {
    const auto byte = static_cast<unsigned char>(text[at++]);
    if ((byte & 0xc0) != 0x80) throw std::runtime_error("tokenizer: invalid UTF-8");
    code = (code << 6) | (byte & 63u);
  }
  if ((length == 2 && code < 0x80) || (length == 3 && code < 0x800) ||
      (length == 4 && code < 0x10000) || (code >= 0xd800 && code <= 0xdfff) || code > 0x10ffff)
    throw std::runtime_error("tokenizer: invalid UTF-8");
  return code;
}
inline std::string utf8(std::uint32_t code) {
  std::string result;
  if (code < 128) result += static_cast<char>(code);
  else if (code < 2048) {
    result += static_cast<char>(0xc0u | (code >> 6)); result += static_cast<char>(0x80u | (code & 63u));
  } else {
    result += static_cast<char>(0xe0u | (code >> 12));
    result += static_cast<char>(0x80u | ((code >> 6) & 63u)); result += static_cast<char>(0x80u | (code & 63u));
  }
  return result;
}
// Ordered alternatives of Qwen2's pre-tokenizer regex. Keeping the scan on
// code points gives \p{L}, \p{N}, and Unicode White_Space fixed semantics.
inline std::vector<std::string_view> qwen2_words(std::string_view text) {
  std::vector<std::uint32_t> codes;
  std::vector<std::size_t> offsets;
  for (std::size_t at = 0; at < text.size();) { offsets.push_back(at); codes.push_back(read(text, at)); }
  offsets.push_back(text.size());
  auto letter = [&](std::size_t i) { return i < codes.size() && category(codes[i]) == 1; };
  auto number = [&](std::size_t i) { return i < codes.size() && category(codes[i]) == 2; };
  auto white = [&](std::size_t i) { return i < codes.size() && space(codes[i]); };
  auto newline = [&](std::size_t i) { return i < codes.size() && (codes[i] == 10 || codes[i] == 13); };
  auto symbol = [&](std::size_t i) { return i < codes.size() && !white(i) && !letter(i) && !number(i); };
  auto lower = [&](std::size_t i) { const auto c = codes[i]; return c >= 'A' && c <= 'Z' ? c + 32 : c; };
  std::vector<std::string_view> words;
  for (std::size_t begin = 0, end = 0; begin < codes.size(); begin = end) {
    end = begin;
    if (codes[begin] == '\'' && begin + 1 < codes.size()) {
      const auto c = lower(begin + 1);
      if (c == 's' || c == 't' || c == 'm' || c == 'd') end = begin + 2;
      else if (begin + 2 < codes.size() &&
          ((c == 'r' && lower(begin + 2) == 'e') || (c == 'v' && lower(begin + 2) == 'e') ||
           (c == 'l' && lower(begin + 2) == 'l'))) end = begin + 3;
    }
    if (end == begin && !newline(begin) && !number(begin) && (letter(begin) || letter(begin + 1))) {
      end = begin + 1;
      while (letter(end)) ++end;
    }
    if (end == begin && number(begin)) end = begin + 1;
    if (end == begin) {
      const auto first = begin + (codes[begin] == 32 ? 1u : 0u);
      if (symbol(first)) {
        end = first + 1;
        while (symbol(end)) ++end;
        while (newline(end)) ++end;
      }
    }
    if (end == begin && white(begin)) {
      std::size_t last_newline = begin;
      while (white(end)) { if (newline(end)) last_newline = end + 1; ++end; }
      if (last_newline != begin) end = last_newline;
      else if (end < codes.size() && end - begin > 1) --end;
    }
    if (end == begin) ++end;
    words.push_back(text.substr(offsets[begin], offsets[end] - offsets[begin]));
  }
  return words;
}
}  // namespace neuron::unicode
