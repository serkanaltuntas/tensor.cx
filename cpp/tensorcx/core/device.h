#pragma once

#include <string>

namespace tensorcx {

struct Device {
  std::string type;
  int index{0};
};

}  // namespace tensorcx
