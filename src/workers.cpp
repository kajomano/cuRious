#include "common_R.h"
#include "common_debug.h"
#include "workers.h"

wd_queue::wd_queue( size_t thread_cnt ) : threads_( thread_cnt ){
  for( size_t i = 0; i < threads_.size(); i++ ){
    threads_[i] = std::thread(
      std::bind( &wd_queue::dispatch_thread_handler, this )
    );
  }

  debugPrint( Rprintf( "Worker queue created, threads: %zu\n", thread_cnt ) );
}

wd_queue::~wd_queue(){
  // Signal to dispatch threads that it's time to wrap up
  quit_ = true;
  cv_.notify_all();

  // Wait for threads to finish before we exit
  for( size_t i = 0; i < threads_.size(); i++ ){
    if( threads_[i].joinable() ){
      threads_[i].join();
    }
  }

  debugPrint( Rprintf( "Worker queue destroyed\n" ) );
}

void wd_queue::dispatch( const fp_t& op ){
  std::unique_lock<std::mutex> lock( lock_ );
  q_.push( op );

  // Manual unlocking is done before notifying, to avoid waking up
  // the waiting thread only to block again (see notify_one for details)
  lock.unlock();
  cv_.notify_all();
}

void wd_queue::dispatch( fp_t&& op ){
  std::unique_lock<std::mutex> lock( lock_ );
  q_.push( std::move( op ) );

  // Manual unlocking is done before notifying, to avoid waking up
  // the waiting thread only to block again (see notify_one for details)
  lock.unlock();
  cv_.notify_all();
}

void wd_queue::sync(){
  std::unique_lock<std::mutex> sync_lock( sync_lock_ );

  dispatch( [this]{
    std::unique_lock<std::mutex> sync_lock( sync_lock_ );
    sync_ = true;
    sync_lock.unlock();
    sync_cv_.notify_one();
  } );

  sync_cv_.wait( sync_lock, [this]{
    return sync_;
  } );

  sync_ = false;
  sync_lock.unlock();
}

void wd_queue::dispatch_thread_handler( void ){
  std::unique_lock<std::mutex> lock(lock_);

  do{
    //Wait until we have data or a quit signal
    cv_.wait( lock, [this]{
      return( q_.size() || quit_ );
    });

    //after wait, we own the lock
    if( q_.size() && !quit_ ){
      auto op = std::move( q_.front() );
      q_.pop();

      //unlock now that we're done messing with the queue
      lock.unlock();

      op();

      lock.lock();
    }
  }while( !quit_ );
}


// Package-wide worker threadpool and mutex to use it
std::mutex common_workers_mutex;
wd_queue common_workers( std::thread::hardware_concurrency() );
