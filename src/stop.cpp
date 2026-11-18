#include "neuron/stop.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace neuron {
StopMatcher::StopMatcher(std::vector<std::string> stops) : stops_(std::move(stops)) {
  for (const auto& stop : stops_) if (stop.empty()) throw std::invalid_argument("stop strings must not be empty");
}
std::string StopMatcher::push(std::string_view text) {
  if (stopped()) return {};
  if (stops_.empty()) return std::string(text);
  pending_.append(text);
  // The earliest match wins. On a tie at one position, the longer string wins.
  std::size_t first = std::string::npos;
  for (std::size_t i = 0; i < stops_.size(); ++i) {
    const auto at = pending_.find(stops_[i]);
    if (at == std::string::npos) continue;
    if (at < first || (at == first && stops_[i].size() > stops_[match_].size())) { first = at; match_ = i; }
  }
  if (stopped()) {
    auto out = pending_.substr(0, first);
    pending_.clear();
    return out;
  }
  // Hold back the longest suffix that is a proper prefix of some stop string.
  std::size_t hold = 0;
  for (const auto& stop : stops_) {
    for (std::size_t n = std::min(stop.size() - 1, pending_.size()); n > hold; --n)
      if (std::string_view(pending_).substr(pending_.size() - n) == std::string_view(stop).substr(0, n)) { hold = n; break; }
  }
  auto out = pending_.substr(0, pending_.size() - hold);
  pending_.erase(0, pending_.size() - hold);
  return out;
}
std::string StopMatcher::finish() {
  if (stopped()) return {};
  return std::exchange(pending_, {});
}
}  // namespace neuron
