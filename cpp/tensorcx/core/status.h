#pragma once

#include <string>
#include <utility>

namespace tensorcx {

enum class StatusCode {
  kOk = 0,
  kInvalidArgument,
  kUnavailable,
  kInternal,
};

class Status {
 public:
  Status() = default;
  Status(StatusCode code, std::string message)
      : code_(code), message_(std::move(message)) {}

  static Status Ok() { return {}; }

  bool ok() const { return code_ == StatusCode::kOk; }
  StatusCode code() const { return code_; }
  const std::string& message() const { return message_; }

 private:
  StatusCode code_{StatusCode::kOk};
  std::string message_;
};

}  // namespace tensorcx
