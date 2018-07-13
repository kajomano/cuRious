#include "common_R.h"
#include "common_cuda.h"
#include "common_debug.h"

#include "streams.h"

// Stream dispatch queues
sd_queue::sd_queue( size_t thread_cnt, bool cuda_streams ) : threads_( thread_cnt ), waiting_( thread_cnt ), cuda_streams_( cuda_streams ){
#ifdef CUDA_EXCLUDE
  cuda_streams_ = false;
#endif

  for( size_t i = 0; i < threads_.size(); i++ ){

#ifndef CUDA_EXCLUDE
    if( cuda_streams_ ){
      streams_.emplace_back(); // Didnt like the default constructor call
      cudaTry( cudaStreamCreate( &streams_[i] ) );
    }
#endif

    waiting_[i] = false;
    threads_[i] = std::thread( &sd_queue::dispatch_thread_handler, this, (int) i );
  }

  // debugPrint( Rprintf( "Worker queue created, threads: %zu\n", thread_cnt ) );
}

sd_queue::~sd_queue(){
  // Signal to dispatch threads that it's time to wrap up
  quit_ = true;
  cv_.notify_all();

  // Wait for threads to finish before we exit
  for( size_t i = 0; i < threads_.size(); i++ ){
    if( threads_[i].joinable() ){
      threads_[i].join();
    }
  }

#ifndef CUDA_EXCLUDE
  if( cuda_streams_ ){
    for( size_t i = 0; i < streams_.size(); i++ ){
      cudaTry( cudaStreamSynchronize( streams_[i] ) );
      cudaTry( cudaStreamDestroy( streams_[i] ) );
    }
  }
#endif

  // debugPrint( Rprintf( "Worker queue destroyed\n" ) );
}

size_t sd_queue::thread_cnt(){
  return threads_.size();
}

void sd_queue::dispatch( const fp_t& op ){
  std::unique_lock<std::mutex> lock( lock_ );
  q_.push( [=]( void* stream ){
    op();
  });

  // Manual unlocking is done before notifying, to avoid waking up
  // the waiting thread only to block again (see notify_one for details)
  lock.unlock();
  cv_.notify_all();

  // printf( "<%p> Dispatch\n", (void*)this );
}

void sd_queue::dispatch( const fp_cuda_t& op ){
  std::unique_lock<std::mutex> lock( lock_ );
  q_.push( op );

  // Manual unlocking is done before notifying, to avoid waking up
  // the waiting thread only to block again (see notify_one for details)
  lock.unlock();
  cv_.notify_all();

  // printf( "<%p> Dispatch\n", (void*)this );
}

void sd_queue::dispatch( fp_t&& op ){
  std::unique_lock<std::mutex> lock( lock_ );
  q_.push( std::move( [=]( void* stream ){
    op();
  } ) );

  // Manual unlocking is done before notifying, to avoid waking up
  // the waiting thread only to block again (see notify_one for details)
  lock.unlock();
  cv_.notify_all();

  // printf( "<%p> Dispatch\n", (void*)this );
}

void sd_queue::dispatch( fp_cuda_t&& op ){
  std::unique_lock<std::mutex> lock( lock_ );
  q_.push( std::move( op ) );

  // Manual unlocking is done before notifying, to avoid waking up
  // the waiting thread only to block again (see notify_one for details)
  lock.unlock();
  cv_.notify_all();

  // printf( "<%p> Dispatch\n", (void*)this );
}

void sd_queue::sync(){
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

#ifndef CUDA_EXCLUDE
  if( cuda_streams_ ){
    for( size_t i = 0; i < streams_.size(); i++ ){
      cudaTry( cudaStreamSynchronize( streams_[i] ) );
    }
  }
#endif

  // lock unlocks when destroyed
}

void sd_queue::dispatch_thread_handler( int id ){
  std::unique_lock<std::mutex> lock(lock_);

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

#ifndef CUDA_EXCLUDE
      if( cuda_streams_ ){
        op( &streams_[id] );
      }
#endif

      if( !cuda_streams_ ){
        op( NULL );
      }

      lock.lock();

      // printf( "Worker %d done, queue: %zu\n", id, q_.size() );
    }
    // Only quit when we have the signal AND there are no more jobs to do in the
    // queue
  }while( !( quit_ && !q_.size() ) );
}

// TODO ====
// Remove this
sd_queue common_workers(4);

// Stream dispatch queue wrappers ==============================================
void cuR_stream_queue_fin( SEXP queue_r ){
  sd_queue* queue = (sd_queue*)R_ExternalPtrAddr( queue_r );

  // Destroy context and free memory!
  // Clear R object too
  if( queue ){
    debugPrint( Rprintf( "<%p> Finalizing queue\n", (void*)queue ) );

    delete queue;
    R_ClearExternalPtr( queue_r );
  }
}

extern "C"
SEXP cuR_stream_queue_create( SEXP threads_r, SEXP cuda_streams_r ){
  int threads_count = Rf_asInteger( threads_r );
  bool cuda_streams = (bool)Rf_asLogical( cuda_streams_r );

  sd_queue* queue = new sd_queue( threads_count, cuda_streams );
  debugPrint( Rprintf( "<%p> Creating queue\n", (void*)queue ) );

  // Return to R with an external pointer SEXP
  SEXP queue_r = Rf_protect( R_MakeExternalPtr( queue, R_NilValue, R_NilValue ) );
  R_RegisterCFinalizerEx( queue_r, cuR_stream_queue_fin, TRUE );

  Rf_unprotect(1);
  return queue_r;
}

extern "C"
SEXP cuR_stream_queue_destroy( SEXP queue_r ){
  cuR_stream_queue_fin( queue_r );

  return R_NilValue;
}

extern "C"
SEXP cuR_stream_queue_sync( SEXP queue_r ){
  sd_queue* queue = (sd_queue*)R_ExternalPtrAddr( queue_r );
  queue -> sync();

  return R_NilValue;
}
