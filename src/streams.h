#include "common_cuda.h"

#include <thread>
#include <functional>
#include <vector>
#include <queue>
#include <mutex>
#include <string>
#include <condition_variable>
#include <map>

// Stream dispatch queues
// https://embeddedartistry.com/blog/2017/2/1/c11-implementing-a-dispatch-queue-using-stdfunction

// Only a single thread (stream) should have access to a dispatch queue!
// Sync does not take into account multi-access

class sd_queue {
  typedef std::function<void(void)> fp_t;
  typedef std::function<void(void* stream)> fp_cuda_t;

public:
  sd_queue( size_t thread_cnt = 1, bool cuda_streams = false );
  ~sd_queue();

  size_t thread_cnt();

  // dispatch and copy
  void dispatch( const fp_t& op );
  void dispatch( const fp_cuda_t& op );

  // dispatch and move
  void dispatch( fp_t&& op );
  void dispatch( fp_cuda_t&& op );

  // wait on the dispatching thread until queue is empty
  void sync();

  // Deleted operations
  sd_queue( const sd_queue& rhs ) = delete;
  sd_queue& operator=( const sd_queue& rhs ) = delete;
  sd_queue( sd_queue&& rhs ) = delete;
  sd_queue& operator=( sd_queue&& rhs ) = delete;

private:
  std::mutex lock_;

  std::vector<std::thread> threads_;
  std::vector<bool> waiting_;
  std::queue<fp_cuda_t> q_;

#ifndef CUDA_EXCLUDE
  std::vector<cudaStream_t> streams_;
#endif

  std::condition_variable cv_;
  std::condition_variable sync_cv_;

  bool quit_ = false;
  bool sync_ = false;

  bool cuda_streams_;

  void dispatch_thread_handler( int id );
};

extern sd_queue common_workers;
