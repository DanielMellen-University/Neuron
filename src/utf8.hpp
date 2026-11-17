#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace neuron::detail {
// Hold incomplete code points between token pieces. Replace malformed output
// with U+FFFD so every JSON string and SSE event remains valid UTF-8.
inline std::string take_utf8(std::string& pending, bool final) {
  std::string out;
  std::size_t i = 0;
  while (i < pending.size()) {
    const auto c = static_cast<unsigned char>(pending[i]);
    std::size_t n = c < 0x80 ? 1 : c >= 0xc2 && c <= 0xdf ? 2 :
        c >= 0xe0 && c <= 0xef ? 3 : c >= 0xf0 && c <= 0xf4 ? 4 : 0;
    if (n != 0 && pending.size() - i < n && !final) break;
    bool valid = n != 0 && pending.size() - i >= n;
    std::uint32_t cp = n == 1 ? c : c & (0x7fu >> n);
    for (std::size_t j = 1; valid && j < n; ++j) {
      const auto d = static_cast<unsigned char>(pending[i + j]);
      valid = (d & 0xc0u) == 0x80u;
      cp = (cp << 6) | (d & 0x3fu);
    }
    valid = valid && (n < 2 || cp >= 0x80) && (n < 3 || cp >= 0x800) &&
        (n < 4 || cp >= 0x10000) && cp <= 0x10ffff && !(cp >= 0xd800 && cp <= 0xdfff);
    if (valid) { out.append(pending, i, n); i += n; }
    else { out += "\xef\xbf\xbd"; ++i; }
  }
  pending.erase(0, i);
  return out;
}
inline bool valid_utf8(std::string_view text) {
  std::string copy(text);
  return take_utf8(copy, true) == text;
}
}  // namespace neuron::detail
