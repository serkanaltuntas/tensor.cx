// Fake lifecycle/failure acceptance runs without a device. Runtime last-error
// preservation is additionally exercised when real CUDA is available.
#include "tensorcx/backends/cuda/cuda_profiling.h"
#include "tensorcx/backends/cuda/cuda_backend.h"
#include "tensorcx/backends/cuda/cuda_context.h"
#include <cstdlib>
#include <algorithm>

#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tensorcx;
namespace {
void check(bool good, const char* message) {
  if (!good) throw std::runtime_error(message);
}
struct Events {
  using Event = int;
  static inline int next{}, fail_at{}, calls{}, peak{};
  static inline float duration{0.125f};
  static inline std::set<Event> live;
  static inline std::vector<std::string> order;
  static bool call(const char* name) {
    order.emplace_back(name);
    return ++calls != fail_at;
  }
  static bool create(Event& value) {
    if (!call("create")) return false;
    value = ++next;
    live.insert(value);
    peak = std::max(peak, static_cast<int>(live.size()));
    return true;
  }
  static bool record(Event value) { check(live.contains(value), "record after release"); return call("record"); }
  static bool elapsed(float& ms, Event a, Event b) {
    check(live.contains(a) && live.contains(b), "read after release"); ms = duration; return call("elapsed");
  }
  static bool destroy(Event value) {
    check(live.erase(value) == 1, "event double destroy"); return call("destroy");
  }
  static void reset(int failure = 0) {
    check(live.empty(), "event leaked across launches");
    next = calls = peak = 0; fail_at = failure; duration = .125f; order.clear();
  }
};
using Timer = cuda::DeviceTimer<Events>;
void launch(bool success = true) {
  Timer timer("test_kernel");
  Events::order.emplace_back("kernel");
  timer.stop();
  Events::order.emplace_back("synchronize");
  timer.finish(success);
}
void disabled_and_bounded() {
  Events::reset();
  launch();
  check(Events::calls == 0, "inactive timer called device API");
  auto token = begin_profile(8);
  launch();
  auto report = end_profile(token);
  check(Events::calls == 0 && report.device_events.empty(), "host-only timer called device API");
  token = begin_profile(1, true);
  {
    ProfileScope host(ProfileCategory::kCompute, "test_kernel", "cuda", std::nullopt, OpKind::kAdd);
    Events::order.clear(); launch();
    check(Events::order == std::vector<std::string>{"create", "create", "record", "kernel", "record",
          "synchronize", "elapsed", "destroy", "destroy"}, "event ordering or extra wait changed");
    const auto calls = Events::calls;
    launch();
    check(Events::calls == calls, "capacity-exhausted timer called device API");
  }
  report = end_profile(token);
  check(report.device_events.size() == 1 && report.dropped_device_events == 1, "device cap not enforced");
  const auto& event = report.device_events[0];
  check(event.host_event_id == 1 && event.operation == OpKind::kAdd && !event.error &&
        event.elapsed_ms == .125f, "device interval correlation/value incorrect");
  check(Events::live.empty() && Events::peak == 2, "event resource bound incorrect");
}
void failures_and_cleanup() {
  for (int failure = 1; failure <= 7; ++failure) {
    Events::reset(failure);
    auto token = begin_profile(4, true);
    launch();
    auto report = end_profile(token);
    check(report.device_events.size() == 1 && report.device_events[0].error &&
          !report.device_events[0].elapsed_ms, "failed event presented as valid duration");
    check(Events::live.empty(), "failure leaked event handle");
  }
  for (bool finish : {false, true}) {
    Events::reset();
    auto token = begin_profile(4, true);
    { Timer timer("early_return"); if (finish) timer.finish(false); }
    auto report = end_profile(token);
    check(report.device_events[0].error && !report.device_events[0].elapsed_ms &&
          Events::live.empty(), "early return failed cleanup");
    check(Events::calls == 5, "failed launch attempted elapsed read or extra event record");
  }
  Events::reset();
  const auto failed_wait = begin_profile(4, true);
  launch(false);
  const auto failed_result = end_profile(failed_wait);
  check(Events::live.empty() && Events::calls == 6 && failed_result.device_events[0].error &&
        !failed_result.device_events[0].elapsed_ms, "failed completion wait read elapsed or leaked events");
  check(std::find(Events::order.begin(), Events::order.end(), "elapsed") == Events::order.end(),
        "failed completion wait queried a duration");
  for (float duration : {0.0f, -1.0f, std::numeric_limits<float>::infinity(), std::nanf("")}) {
    Events::reset(); Events::duration = duration;
    auto token = begin_profile(4, true); launch(); auto report = end_profile(token);
    check((report.device_events[0].error == nullptr) == (duration == 0), "duration validity incorrect");
    check(Events::live.empty(), "invalid duration leaked events");
  }
  Events::reset();
  auto token = begin_profile(64, true);
  for (int i = 0; i < 64; ++i) launch();
  check(Events::live.empty() && Events::peak == 2, "events accumulated until session exit");
  check(end_profile(token).device_events.size() == 64, "repeated launch results lost");
}
void runtime_error_contract() {
  if (!cuda::available()) {
    if (std::getenv("TENSORCX_REQUIRE_CUDA")) throw std::runtime_error("required CUDA device unavailable");
    std::cout << "Runtime error preservation needs a real CUDA device; fake contracts still run\n";
    return;
  }
  // Driver instrumentation must not consume the Runtime launch-error slot.
  {
    cuda::ContextScope context;
    check(context.ready(), "event test context unavailable");
    (void)cudaGetLastError();
    const auto original = cudaSetDevice(-1);  // Safe invalid argument, no bad GPU work.
    check(original != cudaSuccess, "failed to seed Runtime error");
    const auto token = begin_profile(8, true);
    {
      ProfileScope host(ProfileCategory::kCompute, "error_test", "cuda");
      cuda::DeviceTimer timer("error_test");
      check(cudaPeekAtLastError() == original, "event setup consumed Runtime error");
      check(cuda::finish_profiled_launch(timer) == original, "profiling masked original launch error");
    }
    const auto report = end_profile(token);
    check(report.device_events.size() == 1 && report.device_events[0].error &&
            !report.device_events[0].elapsed_ms, "failed launch produced valid device timing");
    check(cudaGetLastError() == cudaSuccess, "Runtime error was not handled normally");
  }
  std::cout << "Runtime error preservation passed on real CUDA\n";
}
}  // namespace
int main() {
  try {
    disabled_and_bounded(); failures_and_cleanup();
    std::cout << "CUDA event lifecycle/failure contracts passed with fake API\n";
    runtime_error_contract();
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
