#include "common_R.h"
#include "common_cuda.h"
#include "common_debug.h"
#include "streams.h"

#ifndef CUDA_EXCLUDE

// Device selection and query ==================================================
extern "C"
SEXP cuR_device_count(){
  int count;
  cudaTry( cudaGetDeviceCount	(	&count ) );

  SEXP count_r = Rf_protect( Rf_ScalarInteger( count ) );
  Rf_unprotect(1);
  return count_r;
}

extern "C"
SEXP cuR_device_get(){
  int dev;
  cudaTry( cudaGetDevice ( &dev ) );

  SEXP dev_r = Rf_protect( Rf_ScalarInteger( dev ) );
  Rf_unprotect(1);
  return dev_r;
}

extern "C"
SEXP cuR_device_set( SEXP dev_r ){
  int dev = Rf_asInteger( dev_r );
  cudaTry( cudaSetDevice ( dev ) );

  return R_NilValue;
}

// extern "C"
// SEXP cuR_device_sync(){
//   cudaTry( cudaDeviceSynchronize() );
//
//   return R_NilValue;
// }

// Stream dispatch queues ======================================================
// https://embeddedartistry.com/blog/2017/2/1/c11-implementing-a-dispatch-queue-using-stdfunction

// TODO ====
// Write a join/sync function

sd_queue::sd_queue(){
  thread_ = std::thread( std::bind( &sd_queue::dispatch_thread_handler, this ) );
}

sd_queue::~sd_queue(){
  // Signal to dispatch threads that it's time to wrap up
  quit_ = true;
  cv_.notify_all();

  // Wait for threads to finish before we exit
  if( thread_.joinable() ){
    thread_.join();
  }
}

void sd_queue::dispatch( const fp_t& op ){
  std::unique_lock<std::mutex> lock( lock_ );
  q_.push( op );

  // Manual unlocking is done before notifying, to avoid waking up
  // the waiting thread only to block again (see notify_one for details)
  lock.unlock();
  cv_.notify_all();
}

void sd_queue::dispatch( fp_t&& op ){
  std::unique_lock<std::mutex> lock( lock_ );
  q_.push( std::move(op) );

  // Manual unlocking is done before notifying, to avoid waking up
  // the waiting thread only to block again (see notify_one for details)
  lock.unlock();
  cv_.notify_all();
}

void sd_queue::dispatch_thread_handler(void){
  std::unique_lock<std::mutex> lock( lock_ );

  do{
    //Wait until we have data or a quit signal
    cv_.wait(lock, [this]{
      return ( q_.size() || quit_ );
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
  } while( !quit_ );
}

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
SEXP cuR_stream_queue_create(){
  sd_queue* queue = new sd_queue;
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

// Streams =====================================================================
void cuR_cuda_stream_fin( SEXP stream_r ){
  cudaStream_t* stream = (cudaStream_t*)R_ExternalPtrAddr( stream_r );

  // Destroy context and free memory!
  // Clear R object too
  if( stream ){
    debugPrint( Rprintf( "<%p> Finalizing stream\n", (void*)stream ) );

    cudaStreamDestroy( *stream );
    delete stream;
    R_ClearExternalPtr( stream_r );
  }
}

extern "C"
SEXP cuR_cuda_stream_create(){
  cudaStream_t* stream = new cudaStream_t;
  debugPrint( Rprintf( "<%p> Creating stream\n", (void*)stream ) );

  cudaTry( cudaStreamCreate( stream ) );

  // Return to R with an external pointer SEXP
  SEXP stream_r = Rf_protect( R_MakeExternalPtr( stream, R_NilValue, R_NilValue ) );
  R_RegisterCFinalizerEx( stream_r, cuR_cuda_stream_fin, TRUE );

  Rf_unprotect(1);
  return stream_r;
}

extern "C"
SEXP cuR_cuda_stream_destroy( SEXP stream_r ){
  cuR_cuda_stream_fin( stream_r );

  return R_NilValue;
}

extern "C"
SEXP cuR_cuda_stream_sync( SEXP stream_r ){
  cudaStream_t* stream = (cudaStream_t*)R_ExternalPtrAddr( stream_r );
  cudaTry( cudaStreamSynchronize( *stream ) );

  return R_NilValue;
}

#endif
