#include "common.h"
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

extern "C"
SEXP cuR_device_sync(){
  cudaTry( cudaDeviceSynchronize() );

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
    delete[] stream;
    R_ClearExternalPtr( stream_r );
  }
}

extern "C"
SEXP cuR_cuda_stream_create(){
  cudaStream_t* stream = new cudaStream_t;
  debugPrint( Rprintf( "<%p> Creating stream\n", (void*)stream ) );

  cudaTry( cudaStreamCreate( stream ) );

  // Return to R with an external pointer SEXP
  SEXP ptr = Rf_protect( R_MakeExternalPtr( stream, R_NilValue, R_NilValue ) );
  R_RegisterCFinalizerEx( ptr, cuR_cuda_stream_fin, TRUE );

  Rf_unprotect(1);
  return ptr;
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
