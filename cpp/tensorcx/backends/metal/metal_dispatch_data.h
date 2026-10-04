#pragma once

#include <dispatch/dispatch.h>

namespace tensorcx::metal::detail {

class DispatchData final {
 public:
  DispatchData() = default;
  explicit DispatchData(dispatch_data_t data) : data_(data) {}
  DispatchData(const DispatchData&) = delete;
  DispatchData& operator=(const DispatchData&) = delete;

  ~DispatchData() { reset(); }

  void reset(dispatch_data_t data = nullptr) {
    if (data_ == data) {
      return;
    }
    if (data_) {
      dispatch_release(data_);
    }
    data_ = data;
  }

  dispatch_data_t get() const { return data_; }

 private:
  dispatch_data_t data_{nullptr};
};

}  // namespace tensorcx::metal::detail
