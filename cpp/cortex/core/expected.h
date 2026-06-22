#pragma once

#include <utility>
#include <variant>

#include "cortex/core/status.h"

namespace cortex {

template <typename T>
class Expected {
 public:
  Expected(T value) : storage_(std::move(value)) {}
  Expected(Status status) : storage_(std::move(status)) {}

  bool has_value() const { return std::holds_alternative<T>(storage_); }
  explicit operator bool() const { return has_value(); }

  T& value() { return std::get<T>(storage_); }
  const T& value() const { return std::get<T>(storage_); }
  T&& move_value() { return std::move(std::get<T>(storage_)); }

  const Status& status() const { return std::get<Status>(storage_); }

 private:
  std::variant<T, Status> storage_;
};

}  // namespace cortex
