#include "tensorcx/core/profiling.h"

#include <atomic>
#include <memory>
#include <stdexcept>

namespace tensorcx {
using Clock = std::chrono::steady_clock;

struct ProfileSession {
  std::uint64_t token{}, dropped{};
  std::size_t limit{};
  Clock::time_point started;
  ProfileScope* top{};
  std::vector<ProfileEvent> events;
};

namespace {
// This trivial TLS pointer remains safe for late backend/TLS destructors.
// The nontrivial holder is initialized only on begin, and clears it first.
thread_local ProfileSession* active{};
struct Holder {
  std::unique_ptr<ProfileSession> session;
  ~Holder() { active = nullptr; session.reset(); }
};
Holder& holder() {
  static thread_local Holder value;
  return value;
}
std::atomic<std::uint64_t> next_token{1};

std::uint64_t nanos(Clock::duration duration) noexcept {
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());
}
}  // namespace

std::uint64_t begin_profile(std::size_t max_events) {
  if (active) throw std::runtime_error("runtime profiles cannot be nested on the same thread");
  if (!max_events || max_events > 100000)
    throw std::invalid_argument("max_events must be an integer in [1, 100000]");
  auto session = std::make_unique<ProfileSession>();
  session->limit = max_events;
  session->events.reserve(max_events);
  session->token = next_token.fetch_add(1, std::memory_order_relaxed);
  if (!session->token) throw std::overflow_error("runtime profile token exhausted");
  session->started = Clock::now();
  holder().session = std::move(session);
  active = holder().session.get();
  return active->token;
}

ProfileResult end_profile(std::uint64_t token) {
  if (!active || active->token != token)
    throw std::runtime_error("runtime profile must end on its owning thread with its original token");
  if (active->top) throw std::runtime_error("cannot end a runtime profile inside a native span");
  const auto elapsed = nanos(Clock::now() - active->started);
  ProfileResult result{elapsed, active->dropped, std::move(active->events)};
  active = nullptr;
  holder().session.reset();
  return result;
}

ProfileScope::ProfileScope(ProfileCategory category, const char* name, const char* backend,
                           std::optional<std::uint64_t> bytes, std::optional<OpKind> operation) noexcept {
  if (!active) return;
  session_ = active;
  parent_ = session_->top;
  started_ = Clock::now();
  session_->top = this;
  if (session_->events.size() == session_->limit) { ++session_->dropped; return; }
  index_ = session_->events.size();
  const auto parent_id = parent_ && parent_->recorded_ ? parent_->index_ + 1 : 0;
  session_->events.push_back(ProfileEvent{index_ + 1, parent_id, nanos(started_ - session_->started),
                                        0, 0, category, name, backend, bytes, operation});
  recorded_ = true;
}

ProfileScope::~ProfileScope() noexcept {
  if (!session_) return;
  const auto elapsed = nanos(Clock::now() - started_);
  session_->top = parent_;
  if (parent_) parent_->children_ns_ += elapsed;
  if (recorded_) {
    auto& event = session_->events[index_];
    event.duration_ns = elapsed;
    event.self_ns = elapsed - children_ns_;
  }
}

void ProfileScope::set_bytes(std::uint64_t bytes) noexcept {
  if (recorded_) session_->events[index_].bytes = bytes;
}

const char* profile_category_name(ProfileCategory category) {
  switch (category) {
    case ProfileCategory::kDispatch: return "dispatch";
    case ProfileCategory::kAllocation: return "allocation";
    case ProfileCategory::kDeallocation: return "deallocation";
    case ProfileCategory::kCopy: return "copy_api";
    case ProfileCategory::kCopyAllocate: return "copy_allocate";
    case ProfileCategory::kCompute: return "compute_sync";
    case ProfileCategory::kMemset: return "memset_api";
    case ProfileCategory::kSynchronize: return "synchronize";
  }
  return "unknown";
}

const char* profile_operation_name(OpKind operation) {
  switch (operation) {
#define TENSORCX_PROFILE_OP(kind, name) case OpKind::kind: return name
    TENSORCX_PROFILE_OP(kFill, "fill"); TENSORCX_PROFILE_OP(kAdd, "add");
    TENSORCX_PROFILE_OP(kMultiply, "multiply"); TENSORCX_PROFILE_OP(kMatmul, "matmul");
    TENSORCX_PROFILE_OP(kSum, "sum"); TENSORCX_PROFILE_OP(kMax, "max");
    TENSORCX_PROFILE_OP(kMean, "mean"); TENSORCX_PROFILE_OP(kExp, "exp");
    TENSORCX_PROFILE_OP(kGelu, "gelu"); TENSORCX_PROFILE_OP(kSilu, "silu");
    TENSORCX_PROFILE_OP(kSoftmax, "softmax"); TENSORCX_PROFILE_OP(kRmsNorm, "rmsnorm");
    TENSORCX_PROFILE_OP(kLayerNorm, "layernorm"); TENSORCX_PROFILE_OP(kSubtract, "subtract");
    TENSORCX_PROFILE_OP(kDivide, "divide"); TENSORCX_PROFILE_OP(kNegate, "negative");
    TENSORCX_PROFILE_OP(kAddScalar, "add_scalar"); TENSORCX_PROFILE_OP(kSubtractScalar, "subtract_scalar");
    TENSORCX_PROFILE_OP(kMultiplyScalar, "multiply_scalar"); TENSORCX_PROFILE_OP(kDivideScalar, "divide_scalar");
    TENSORCX_PROFILE_OP(kCast, "astype"); TENSORCX_PROFILE_OP(kTranspose, "transpose");
    TENSORCX_PROFILE_OP(kSlice, "slice"); TENSORCX_PROFILE_OP(kConcat, "concat");
    TENSORCX_PROFILE_OP(kEqual, "equal"); TENSORCX_PROFILE_OP(kNotEqual, "not_equal");
    TENSORCX_PROFILE_OP(kLess, "less"); TENSORCX_PROFILE_OP(kLessEqual, "less_equal");
    TENSORCX_PROFILE_OP(kGreater, "greater"); TENSORCX_PROFILE_OP(kGreaterEqual, "greater_equal");
    TENSORCX_PROFILE_OP(kLogicalAnd, "logical_and"); TENSORCX_PROFILE_OP(kLogicalOr, "logical_or");
    TENSORCX_PROFILE_OP(kLogicalXor, "logical_xor"); TENSORCX_PROFILE_OP(kLogicalNot, "logical_not");
    TENSORCX_PROFILE_OP(kWhere, "where"); TENSORCX_PROFILE_OP(kAny, "any");
    TENSORCX_PROFILE_OP(kAll, "all"); TENSORCX_PROFILE_OP(kMaskedSelect, "masked_select");
    TENSORCX_PROFILE_OP(kLog, "log"); TENSORCX_PROFILE_OP(kSqrt, "sqrt");
    TENSORCX_PROFILE_OP(kAbs, "abs"); TENSORCX_PROFILE_OP(kMin, "min");
    TENSORCX_PROFILE_OP(kArgmax, "argmax"); TENSORCX_PROFILE_OP(kClip, "clip");
    TENSORCX_PROFILE_OP(kTopK, "topk"); TENSORCX_PROFILE_OP(kLinear, "linear");
    TENSORCX_PROFILE_OP(kAffineRmsNorm, "affine_rmsnorm"); TENSORCX_PROFILE_OP(kAffineLayerNorm, "affine_layernorm");
    TENSORCX_PROFILE_OP(kEmbedding, "embedding"); TENSORCX_PROFILE_OP(kAttention, "attention");
    TENSORCX_PROFILE_OP(kAttentionSoftmax, "attention_softmax");
#undef TENSORCX_PROFILE_OP
  }
  return "unknown";
}
}  // namespace tensorcx
