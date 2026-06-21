#pragma once

#include <cstdint>
#include <vector>

namespace cortex {

using Dim = std::int64_t;
using Shape = std::vector<Dim>;

std::int64_t numel(const Shape& shape);
Shape contiguous_strides(const Shape& shape);

}  // namespace cortex
