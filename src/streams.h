#include <thread>
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>

// Stream dispatch queues
class sd_queue{
  typedef std::function<void(void)> fp_t;

public:
  sd_queue();
  ~sd_queue();

  // Dispatch and copy
  void dispatch( const fp_t& op );
  // Dispatch and move
  void dispatch( fp_t&& op );

  void sync();

  // Deleted operations
  sd_queue( const sd_queue& rhs ) = delete;
  sd_queue& operator=( const sd_queue& rhs ) = delete;
  sd_queue( sd_queue&& rhs ) = delete;
  sd_queue& operator=( sd_queue&& rhs ) = delete;

private:
  // std::string name_;
  std::mutex lock_;
  std::mutex sync_lock_;

  std::thread thread_;
  std::queue<fp_t> q_;

  std::condition_variable cv_;
  std::condition_variable sync_cv_;

  bool quit_ = false;
  bool sync_ = false;

  void dispatch_thread_handler(void);
};
