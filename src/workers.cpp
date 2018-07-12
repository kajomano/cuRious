#include "common_R.h"
#include "common_debug.h"
#include "workers.h"

#include <cstdio>

wd_queue::wd_queue( size_t thread_cnt ) : threads_( thread_cnt ), waiting_( thread_cnt ){
  for( size_t i = 0; i < threads_.size(); i++ ){
    threads_[i] = std::thread(
      &wd_queue::dispatch_thread_handler,
      this,
      (int) i

    //   [this, i]{
    //   dispatch_thread_handler( (int) i );
    // }
      // std::bind( &wd_queue::dispatch_thread_handler, this )
    );
  }

  // debugPrint( Rprintf( "Worker queue created, threads: %zu\n", thread_cnt ) );
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

  // debugPrint( Rprintf( "Worker queue destroyed\n" ) );
}

size_t wd_queue::thread_cnt(){
  return threads_.size();
}

void wd_queue::dispatch( const fp_t& op ){
  std::unique_lock<std::mutex> lock( lock_ );
  q_.push( op );

  // Manual unlocking is done before notifying, to avoid waking up
  // the waiting thread only to block again (see notify_one for details)
  lock.unlock();
  cv_.notify_all();

  // printf( "<%p> Dispatch\n", (void*)this );
}

void wd_queue::dispatch( fp_t&& op ){
  std::unique_lock<std::mutex> lock( lock_ );
  q_.push( std::move( op ) );

  // Manual unlocking is done before notifying, to avoid waking up
  // the waiting thread only to block again (see notify_one for details)
  lock.unlock();
  cv_.notify_all();

  // printf( "<%p> Dispatch\n", (void*)this );
}

void wd_queue::sync(){
  std::unique_lock<std::mutex> lock( lock_ );
  // printf( "<%p> Syncing\n", (void*)this );

  // notify threads that we are waiting for an empty queue
  sync_ = true;

  // wait until all threads are asleep AND there is an empty queue
  sync_cv_.wait( lock, [this]{
    if( q_.size() ) return false;

    for( auto const& waiting : waiting_ ){
      if( !waiting ){
        // printf( "Sync wake failed\n" );
        return false;
      }
    }

    return true;
  } );

  // turn off syncing
  sync_ = false;

  // printf( "Sync wake success\n" );

  // std::unique_lock<std::mutex> sync_lock( sync_lock_ );
  //
  // dispatch( [this]{
  //   std::unique_lock<std::mutex> sync_lock( sync_lock_ );
  //   sync_ = true;
  //   sync_lock.unlock();
  //   sync_cv_.notify_one();
  // } );
  //

  //

  // sync_lock.unlock();
}

void wd_queue::dispatch_thread_handler( int id ){
  std::unique_lock<std::mutex> lock(lock_);

  // insert an entry into waiting_ with thread_id
  waiting_[id] = false;

  do{
    // set status to waiting
    waiting_[id] = true;

    // printf( "<%p> Worker %d sync %d\n", (void*)this, id, sync_ );

    // if sync_ is underway and we are going into waiting, wake the dispatcher
    if( !q_.size() && sync_ ){
      // printf( "Worker %d going to sleep\n", id );

      sync_cv_.notify_all();
    }

    // Wait until we have data or a quit signal
    cv_.wait( lock, [this]{
      return( q_.size() || quit_ );
    });

    // set status to not waiting
    waiting_[id] = false;

    //after wait, we own the lock
    if( q_.size() ){
      auto op = std::move( q_.front() );
      q_.pop();

      // printf( "Worker %d pop, queue: %zu\n", id, q_.size() );

      //unlock now that we're done messing with the queue
      lock.unlock();

      op();

      lock.lock();
    }
    // Only quit when we have the signal AND there are no more jobs to do in the
    // queue
  }while( !( quit_ && !q_.size() ) );
}

wd_queue common_workers(12);
