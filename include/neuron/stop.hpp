#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace neuron {
// Finds stop strings in streamed text. Pieces arrive a token at a time, so a
// stop string can span several pieces. push returns only text that cannot be
// the start of a stop string; the rest waits for the next piece or finish.
// Once a stop string is seen, the text before it is returned and the matched
// string and everything after it are dropped.
class StopMatcher {
 public:
  StopMatcher() = default;
  // Throws std::invalid_argument for an empty stop string.
  explicit StopMatcher(std::vector<std::string> stops);
  std::string push(std::string_view text);
  // Releases held text when generation ends without a stop.
  std::string finish();
  bool stopped() const { return match_ != npos; }
  // The stop string that ended the text, or empty before a match.
  std::string_view matched() const { return stopped() ? std::string_view(stops_[match_]) : std::string_view(); }
  bool empty() const { return stops_.empty(); }
 private:
  std::vector<std::string> stops_;
  std::string pending_;
  static constexpr std::size_t npos = static_cast<std::size_t>(-1);
  std::size_t match_ = npos;
};
}  // namespace neuron
