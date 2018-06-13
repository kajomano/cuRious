#include <thread>
#include <functional>
#include <queue>
#include <mutex>
#include <condition_variable>

// https://embeddedartistry.com/blog/2017/2/1/c11-implementing-a-dispatch-queue-using-stdfunction

class thread_dispatch_queue{
  typedef std::function<void(void)> fp_t;

public:
  thread_dispatch_queue();
  ~thread_dispatch_queue();

  // Dispatch and copy
  void dispatch( const fp_t& op );
  // Dispatch and move
  void dispatch( fp_t&& op );

  // Deleted operations
  thread_dispatch_queue( const thread_dispatch_queue& rhs ) = delete;
  thread_dispatch_queue& operator=( const thread_dispatch_queue& rhs ) = delete;
  thread_dispatch_queue( thread_dispatch_queue&& rhs ) = delete;
  thread_dispatch_queue& operator=( thread_dispatch_queue&& rhs ) = delete;

private:
  // std::string name_;
  std::mutex lock_;
  std::thread thread_;
  std::queue<fp_t> q_;
  std::condition_variable cv_;
  bool quit_ = false;

  void dispatch_thread_handler(void);
};

thread_dispatch_queue::thread_dispatch_queue(){
  Rprintf("Creating dispatch queue");
  thread_ = std::thread( std::bind( &thread_dispatch_queue::dispatch_thread_handler, this ) );
}

thread_dispatch_queue::~thread_dispatch_queue()
{
  Rprintf("Destructor: Destroying dispatch threads...\n");

  // Signal to dispatch threads that it's time to wrap up
  quit_ = true;
  cv_.notify_all();

  // Wait for threads to finish before we exit
  if(thread_.joinable())
  {
    Rprintf("Destructor: Joining thread until completion\n");
    thread_.join();
  }
}

void thread_dispatch_queue::dispatch( const fp_t& op )
{
  std::unique_lock<std::mutex> lock( lock_ );
  q_.push( op );

  // Manual unlocking is done before notifying, to avoid waking up
  // the waiting thread only to block again (see notify_one for details)
  lock.unlock();
  cv_.notify_all();
}

void thread_dispatch_queue::dispatch( fp_t&& op )
{
  std::unique_lock<std::mutex> lock(lock_);
  q_.push(std::move(op));

  // Manual unlocking is done before notifying, to avoid waking up
  // the waiting thread only to block again (see notify_one for details)
  lock.unlock();
  cv_.notify_all();
}

void thread_dispatch_queue::dispatch_thread_handler(void)
{
  std::unique_lock<std::mutex> lock(lock_);

  do{
    //Wait until we have data or a quit signal
    cv_.wait(lock, [this]{
      return (q_.size() || quit_);
    });

    //after wait, we own the lock
    if(q_.size() && !quit_)
    {
      auto op = std::move(q_.front());
      q_.pop();

      //unlock now that we're done messing with the queue
      lock.unlock();

      printf( "Processing op" );

      op();

      lock.lock();
    }
  } while (!quit_);
}
