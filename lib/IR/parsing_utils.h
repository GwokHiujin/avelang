#pragma once

#include <optional>
#include <string>

namespace causalflow::avelang::ir {

// Utility functions for parsing constants from string values

// Parse a string as an integer constant
// Returns the parsed integer value if successful, std::nullopt otherwise
std::optional<long long> ParseConstantInteger(const std::string &value);

// Parse a string as a floating-point constant
// Returns the parsed double value if successful, std::nullopt otherwise
std::optional<double> ParseConstantFloat(const std::string &value);

} // namespace causalflow::avelang::ir