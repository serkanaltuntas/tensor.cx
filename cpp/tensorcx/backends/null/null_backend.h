#pragma once

#include <string>

#include "tensorcx/core/backend.h"
#include "tensorcx/core/status.h"

namespace tensorcx::null_backend {

class NullBackend final : public Backend {
 public:
  std::string name() const override;
  Status execute(const BackendExecution& execution) override;
};

Status contract_smoke_test();

}  // namespace tensorcx::null_backend
