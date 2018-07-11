#include <thread>
#include <functional>
#include <vector>
#include <queue>
#include <mutex>
#include <string>
#include <condition_variable>

// Worker dispatch queues
// https://embeddedartistry.com/blog/2017/2/1/c11-implementing-a-dispatch-queue-using-stdfunction

class wd_queue {
  typedef std::function<void(void)> fp_t;

public:
  wd_queue( size_t thread_cnt = 1 );
  ~wd_queue();

  // dispatch and copy
  void dispatch( const fp_t& op );
  // dispatch and move
  void dispatch( fp_t&& op );

  void sync();

  // Deleted operations
  wd_queue( const wd_queue& rhs ) = delete;
  wd_queue& operator=( const wd_queue& rhs ) = delete;
  wd_queue( wd_queue&& rhs ) = delete;
  wd_queue& operator=( wd_queue&& rhs ) = delete;

private:
  std::mutex lock_;
  std::mutex sync_lock_;

  std::vector<std::thread> threads_;
  std::queue<fp_t> q_;

  std::condition_variable cv_;
  std::condition_variable sync_cv_;

  bool quit_ = false;
  bool sync_ = false;

  void dispatch_thread_handler(void);
};

// Package-wide worker threadpool and mutex to use it
extern std::mutex common_workers_mutex;
extern wd_queue common_workers;
