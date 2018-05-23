#include "common.h"
#ifndef CUDA_EXCLUDE


// Device selection and query ==================================================
extern "C"
SEXP cuR_device_count(){
  int count;
  cudaTry( cudaGetDeviceCount	(	&count ) );

  SEXP ret_r = Rf_protect( Rf_ScalarInteger( count ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_get_device(){
  int dev;
  cudaTry( cudaGetDevice ( &dev ) );

  SEXP ret_r = Rf_protect( Rf_ScalarInteger( dev ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_set_device( SEXP dev_r ){
  int dev = Rf_asInteger( dev_r );
  cudaTry( cudaSetDevice ( dev ) );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}


extern "C"
SEXP cuR_sync_device(){
  cudaTry( cudaDeviceSynchronize() );

  // Return something that is not null
  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

// Streams =====================================================================
void cuR_finalize_cuda_stream( SEXP ptr ){
  cudaStream_t* stream = (cudaStream_t*)R_ExternalPtrAddr( ptr );

  // Destroy context and free memory!
  // Clear R object too
  if( stream ){
    debugPrint( Rprintf( "<%p> Finalizing stream\n", (void*)stream ) );

    cudaStreamDestroy( *stream );
    delete[] stream;
    R_ClearExternalPtr( ptr );
  }
}

extern "C"
SEXP cuR_destroy_cuda_stream( SEXP ptr ){
  cuR_finalize_cuda_stream( ptr );
  return R_NilValue;
}

extern "C"
SEXP cuR_create_cuda_stream(){
  cudaStream_t* stream = new cudaStream_t;
  debugPrint( Rprintf( "<%p> Creating stream\n", (void*)stream ) );

  cudaTry( cudaStreamCreate( stream ) );

  // Return to R with an external pointer SEXP
  SEXP ptr = Rf_protect( R_MakeExternalPtr( stream, R_NilValue, R_NilValue ) );
  R_RegisterCFinalizerEx( ptr, cuR_finalize_cuda_stream, TRUE );

  Rf_unprotect(1);
  return ptr;
}

extern "C"
SEXP cuR_sync_cuda_stream( SEXP stream_r ){
  cudaStream_t* stream = (cudaStream_t*)R_ExternalPtrAddr( stream_r );
  cudaTry( cudaStreamSynchronize( *stream ) );

  // Return something that is not null
  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

#endif
