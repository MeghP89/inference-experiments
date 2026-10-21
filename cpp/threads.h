// Persistent worker pool. Workers spin briefly between jobs before sleeping,
// since matmuls arrive back to back, and grab chunks from a shared counter so
// faster cores pick up more of the work
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace gpt2 {

class ThreadPool {
 public:
  explicit ThreadPool(int n_threads) {
    for (int i = 1; i < n_threads; ++i) workers_.emplace_back([this] { work(); });
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(m_);
      stop_ = true;
      generation_++;
    }
    cv_.notify_all();
    for (std::thread& t : workers_) t.join();
  }

  // Calls task(i) for every i in [0, n_tasks) and returns once all are done
  void run(int n_tasks, const std::function<void(int)>& task) {
    if (workers_.empty() || n_tasks == 1) {
      for (int i = 0; i < n_tasks; ++i) task(i);
      return;
    }
    task_ = &task;
    n_tasks_ = n_tasks;
    next_ = 0;
    busy_ = static_cast<int>(workers_.size());
    {
      std::lock_guard<std::mutex> lock(m_);
      generation_++;
    }
    cv_.notify_all();
    drain();
    while (busy_ > 0) std::this_thread::yield();
  }

 private:
  static constexpr int kSpins = 200000;

  void drain() {
    for (int i = next_++; i < n_tasks_; i = next_++) (*task_)(i);
  }

  void work() {
    unsigned seen = 0;
    while (true) {
      for (int s = 0; s < kSpins && generation_ == seen; ++s) {
      }
      if (generation_ == seen) {
        std::unique_lock<std::mutex> lock(m_);
        cv_.wait(lock, [&] { return generation_ != seen; });
      }
      seen = generation_;
      if (stop_) return;
      drain();
      busy_--;
    }
  }

  std::vector<std::thread> workers_;
  std::mutex m_;
  std::condition_variable cv_;
  const std::function<void(int)>* task_ = nullptr;
  int n_tasks_ = 0;
  std::atomic<int> next_{0};
  std::atomic<int> busy_{0};
  std::atomic<unsigned> generation_{0};
  std::atomic<bool> stop_{false};
};

inline int default_threads() {
  const char* env = std::getenv("GPT2_THREADS");
  if (env) return std::atoi(env);
  // One per physical core, since hyperthreads share the same FMA units
  return std::max(1u, std::thread::hardware_concurrency() / 2);
}

inline ThreadPool& pool() {
  static ThreadPool p(default_threads());
  return p;
}

}  // namespace gpt2
