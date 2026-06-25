#pragma once

#include <string>

#include "cortex/core/backend.h"
#include "cortex/core/status.h"

namespace cortex::null_backend {

class NullBackend final : public Backend {
 public:
  std::string name() const override;
  Status execute(const BackendExecution& execution) override;
};

Status contract_smoke_test();

}  // namespace cortex::null_backend
