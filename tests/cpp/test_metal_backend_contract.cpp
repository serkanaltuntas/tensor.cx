#include <iostream>

#include "cortex/backends/metal/metal_backend.h"
#include "cortex/core/status.h"

int main() {
  const auto status = cortex::metal::contract_smoke_test();
  if (status.code() == cortex::StatusCode::kUnavailable) {
    std::cout << "metal backend contract smoke skipped: " << status.message() << '\n';
    return 77;
  }
  if (!status.ok()) {
    std::cerr << "metal backend contract smoke failed with status "
              << static_cast<int>(status.code()) << ": " << status.message() << '\n';
    return 1;
  }
  return 0;
}
