#include "thread_pool.hpp"
#include "neuron/kernels.hpp"

#include <algorithm>
#include <charconv>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

namespace neuron {
namespace {
class Pool {
 public:
  explicit Pool(std::size_t count) : count_(count) {
    try {
      for (std::size_t i = 1; i < count; ++i) workers_.emplace_back([this, i] { worker(i); });
    } catch (...) {
      stop();
      throw;
    }
  }
  ~Pool() { stop(); }
  std::size_t count() const { return count_; }
  void run(std::size_t rows, const std::function<void(std::size_t, std::size_t)>& work) {
    if (count_ == 1 || rows < 2) { work(0, rows); return; }
    // Independent callers may submit jobs, but one pool executes one at a time.
    const std::lock_guard submit(submit_mutex_);
    {
      const std::lock_guard lock(mutex_);
      work_ = work;
      rows_ = rows;
      remaining_ = workers_.size();
      error_ = nullptr;
      ++generation_;
    }
    ready_.notify_all();
    execute(0);
    std::unique_lock lock(mutex_);
    done_.wait(lock, [this] { return remaining_ == 0; });
    work_ = {};
    if (error_) std::rethrow_exception(error_);
  }
 private:
  void stop() {
    {
      const std::lock_guard lock(mutex_);
      stopping_ = true;
    }
    ready_.notify_all();
    for (auto& thread : workers_) thread.join();
  }
  void execute(std::size_t index) {
    const auto per = rows_ / count_, extra = rows_ % count_;
    const auto begin = index * per + std::min(index, extra);
    const auto end = begin + per + (index < extra ? 1u : 0u);
    try {
      if (begin < end) work_(begin, end);
    } catch (...) {
      const std::lock_guard lock(mutex_);
      if (!error_) error_ = std::current_exception();
    }
  }
  void worker(std::size_t index) {
    std::size_t previous = 0;
    for (;;) {
      std::unique_lock lock(mutex_);
      ready_.wait(lock, [this, previous] { return stopping_ || generation_ != previous; });
      if (stopping_) return;
      previous = generation_;
      lock.unlock();
      execute(index);
      lock.lock();
      if (--remaining_ == 0) done_.notify_one();
    }
  }
  std::size_t count_, generation_ = 0, remaining_ = 0, rows_ = 0;
  bool stopping_ = false;
  std::mutex mutex_, submit_mutex_;
  std::condition_variable ready_, done_;
  std::vector<std::thread> workers_;
  std::function<void(std::size_t, std::size_t)> work_;
  std::exception_ptr error_;
};
void check_count(std::size_t count) {
  if (count == 0 || count > 256) throw std::invalid_argument("threads must be in 1..256");
}
std::size_t environment_threads() {
  const auto* value = std::getenv("NEURON_THREADS");
  if (!value) return 1;
  const std::string_view text(value);
  std::size_t result = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
    throw std::invalid_argument("invalid NEURON_THREADS");
  check_count(result);
  return result;
}
std::unique_ptr<Pool>& pool(std::size_t initial = 0) {
  static auto value = std::make_unique<Pool>(initial == 0 ? environment_threads() : initial);
  return value;
}
}
namespace kernels {
void set_threads(std::size_t count) {
  check_count(count);
  if (pool(count)->count() != count) pool() = std::make_unique<Pool>(count);
}
std::size_t threads() { return pool()->count(); }
}
namespace detail {
void parallel_rows(std::size_t rows, const std::function<void(std::size_t, std::size_t)>& work) {
  pool()->run(rows, work);
}
}
}  // namespace neuron
