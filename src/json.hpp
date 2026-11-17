#pragma once

#include <map>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace neuron::json {
struct Value {
  using Array = std::vector<Value>;
  using Object = std::map<std::string, Value>;
  std::variant<std::nullptr_t, bool, double, std::string, Array, Object> data = nullptr;
  const Value* find(const std::string& key) const;
};
Value parse(std::string_view text);
std::string quote(std::string_view text);
}  // namespace neuron::json
