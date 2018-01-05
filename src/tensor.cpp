#include <cuda.h>
#include <cuda_runtime_api.h>

#define R_NO_REMAP 1
//#define R_NO_REMAP_RMATH 1

#include <R.h>
#include <Rinternals.h>

#include "debug.h"

// Various memory allocations/deallocations
// Tensor - device memory
// Stage  - pinned host memory
// Buffer - non-pinned host memory

void cuR_finalize_tensor( SEXP tens_r ){
  float* tens_dev = (float*)R_ExternalPtrAddr( tens_r );

  if( tens_dev ){
#ifdef DEBUG_PRINTS
    Rprintf( "<%p> Finalizing tensor\n", (void*) tens_dev );
#endif

    // Free memory
    cudaFree( tens_dev );
    R_ClearExternalPtr( tens_r );
  }
}

extern "C"
SEXP cuR_create_tensor( SEXP l_r ){
  float* tens_dev;
  cudaError_t cuda_stat;
  int l = Rf_asInteger( l_r );

  cudaTry( cudaMalloc( (void**)&tens_dev, l*sizeof(float) ) )

#ifdef DEBUG_PRINTS
    Rprintf( "<%p> Creating tensor\n", (void*)tens_dev );
#endif

    // Wrap pointer in a SEXP and register a finalizer
    SEXP tens_r = Rf_protect( R_MakeExternalPtr( tens_dev, R_NilValue, R_NilValue ) );
    R_RegisterCFinalizerEx( tens_r, cuR_finalize_tensor, TRUE );

    Rf_unprotect(1);
    return tens_r;
}

extern "C"
SEXP cuR_destroy_tensor( SEXP tens_r ){
  cuR_finalize_tensor( tens_r );
  return R_NilValue;
}

// ----------------------------------------------------------

void cuR_finalize_stage( SEXP stage_r ){
  float* stage = (float*)R_ExternalPtrAddr( stage_r );

  // Destroy context and free memory!
  // Clear R object too
  if( stage ){
#ifdef DEBUG_PRINTS
    Rprintf( "<%p> Finalizing stage\n", (void*)stage );
#endif

    cudaFreeHost( stage );
    R_ClearExternalPtr( stage_r );
  }
}

extern "C"
SEXP cuR_create_stage( SEXP l_r ){
  float* stage;
  cudaError_t cuda_stat;
  int l = Rf_asInteger( l_r );

  cudaTry( cudaHostAlloc( (void**)&stage, l*sizeof(float), cudaHostAllocDefault) )

#ifdef DEBUG_PRINTS
    Rprintf( "<%p> Creating stage\n", (void*) stage );
#endif

    SEXP ret_r = Rf_protect( R_MakeExternalPtr( stage, R_NilValue, R_NilValue ) );
    R_RegisterCFinalizerEx( ret_r, cuR_finalize_stage, TRUE );

    Rf_unprotect(1);
    return ret_r;
}

extern "C"
SEXP cuR_destroy_stage( SEXP stage_r ){
  cuR_finalize_stage( stage_r );
  return R_NilValue;
}

// ----------------------------------------------------------

void cuR_finalize_buffer( SEXP buffer_r ){
  float* buffer = (float*)R_ExternalPtrAddr( buffer_r );

  if( buffer ){
#ifdef DEBUG_PRINTS
    Rprintf( "<%p> Finalizing buffer\n", (void*)buffer );
#endif

    delete[] buffer;
    R_ClearExternalPtr( buffer_r );
  }
}

extern "C"
SEXP cuR_create_buffer( SEXP l_r ){
  int l = Rf_asInteger( l_r );

  float* buffer = new float[l];

#ifdef DEBUG_PRINTS
  Rprintf( "<%p> Creating buffer\n", (void*)buffer );
#endif

  SEXP ret_r = Rf_protect( R_MakeExternalPtr( buffer, R_NilValue, R_NilValue ) );
  R_RegisterCFinalizerEx( ret_r, cuR_finalize_buffer, TRUE );

  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_destroy_buffer( SEXP buffer_r ){
  cuR_finalize_buffer( buffer_r );
  return R_NilValue;
}

// Pull-push ================================================

extern "C"
SEXP cuR_push_preproc( SEXP data_r, SEXP l_r, SEXP buff_r ){
  // Recover pointers and length
  double* data = REAL( data_r );
  int l        = Rf_asInteger( l_r );
  float* buff  = (float*)R_ExternalPtrAddr( buff_r );

  // Convert to float in host memory
  for( int i = 0; i < l; i++ ){
    buff[i] = (float)data[i];
  }

  // Return something that is not null
  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_push_fetch( SEXP buff_r, SEXP l_r, SEXP tens_r ){
  float* buff  = (float*)R_ExternalPtrAddr( buff_r );
  int l        = Rf_asInteger( l_r );
  float* tens  = (float*)R_ExternalPtrAddr( tens_r );
  cudaError_t cuda_stat;

#ifdef DEBUG_PRINTS
  Rprintf( "<%p> Pushing tensor\n", (void*) tens );
#endif

  cudaTry( cudaMemcpy( tens, buff, l*sizeof(float), cudaMemcpyHostToDevice ) )

  // Return something that is not null
  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_push_fetch_async( SEXP stage_r, SEXP l_r, SEXP tens_r, SEXP stream_r ){
  float* stage         = (float*)R_ExternalPtrAddr( stage_r );
  int l                = Rf_asInteger( l_r );
  float* tens          = (float*)R_ExternalPtrAddr( tens_r );
  cudaStream_t* stream = (cudaStream_t*)R_ExternalPtrAddr( stream_r );
  cudaError_t cuda_stat;

#ifdef DEBUG_PRINTS
  Rprintf( "<%p> Async pushing tensor\n", (void*) tens );
#endif

  cudaTry( cudaMemcpyAsync( tens, stage, l*sizeof(float), cudaMemcpyHostToDevice, *stream ) )

  // Return something that is not null
  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

// ----------------------------------------------------------

extern "C"
SEXP cuR_pull_prefetch( SEXP buff_r, SEXP l_r, SEXP tens_r ){
  float* buff  = (float*)R_ExternalPtrAddr( buff_r );
  int l        = Rf_asInteger( l_r );
  float* tens  = (float*)R_ExternalPtrAddr( tens_r );
  cudaError_t cuda_stat;

#ifdef DEBUG_PRINTS
  Rprintf( "<%p> Pulling tensor\n", (void*)tens );
#endif

  cudaTry( cudaMemcpy( buff, tens, l*sizeof(float), cudaMemcpyDeviceToHost ) )

  // Return something that is not null
  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_pull_prefetch_async( SEXP stage_r, SEXP l_r, SEXP tens_r, SEXP stream_r ){
  // Recover pointers and length
  float* stage         = (float*)R_ExternalPtrAddr( stage_r );
  int l                = Rf_asInteger( l_r );
  float* tens          = (float*)R_ExternalPtrAddr( tens_r );
  cudaStream_t* stream = (cudaStream_t*)R_ExternalPtrAddr( stream_r );
  cudaError_t cuda_stat;

#ifdef DEBUG_PRINTS
  Rprintf( "<%p> Async pulling tensor\n", (void*)tens );
#endif

  cudaTry( cudaMemcpyAsync( stage, tens, l*sizeof(float), cudaMemcpyDeviceToHost, *stream ) );

  // Return something that is not null
  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_pull_proc( SEXP dims_r, SEXP buff_r ){
  // Recover pointer and dims
  int* dims    = INTEGER( dims_r );
  int l        = dims[0]*dims[1];
  float* buff  = (float*)R_ExternalPtrAddr( buff_r );

  // Create the correct R object
  SEXP data_r;
  if( dims[1] == 1 ){
    data_r = Rf_protect( Rf_allocVector( REALSXP, dims[0] ) );
  }else{
    data_r = Rf_protect( Rf_allocMatrix( REALSXP, dims[0], dims[1] ) );
  }

  // Recover pointer to data in SEXP
  double* data = REAL( data_r );

  // Fill the SEXP with data
  for (int i = 0; i < l; i++){
    data[i] = (double)buff[i];
  }

  Rf_unprotect(1);
  return data_r;
}


extern "C"
SEXP cuR_push_tensor( SEXP tens_r, SEXP data_r, SEXP l_r, SEXP stage_r, SEXP stream_r ){
  // Create pointer to the actual data in the tens_r
  double* data_c = REAL( data_r );

  // Tensor length
  int l = Rf_asInteger( l_r );

  // Allocate memory on the host
  float* tens_host;
  if( stage_r == R_NilValue ){
    tens_host = new float[l];
  }else{
    tens_host = (float*)R_ExternalPtrAddr( stage_r );
  }

  // Convert to float in host memory
  for (int i = 0; i < l; i++){
    tens_host[i] = (float)data_c[i];
  }

  // Copy host vector to device
  cudaError_t cuda_stat;
  float* tens_dev = (float*)R_ExternalPtrAddr( tens_r );

  // Do an async copy if both a stage and a stream is suported
  if( stage_r != R_NilValue && stream_r != R_NilValue ){
#ifdef DEBUG_PRINTS
    Rprintf( "<%p> Async copying tensor\n", (void*) tens_dev );
#endif

    cudaStream_t* stream = (cudaStream_t*)R_ExternalPtrAddr( stream_r );
    cudaTry( cudaMemcpyAsync( tens_dev, tens_host, l*sizeof(float), cudaMemcpyHostToDevice, *stream ) )
  }else{
#ifdef DEBUG_PRINTS
    Rprintf( "<%p> Copying tensor\n", (void*) tens_dev );
#endif

    cudaTry( cudaMemcpy(tens_dev, tens_host, l*sizeof(float), cudaMemcpyHostToDevice) )
  }

  // Free the host memory
  if( stage_r == R_NilValue ){
    delete[] tens_host;
  }

  // Return something that is not null
  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_pull_tensor( SEXP tens_r, SEXP dims_r, SEXP stage_r, SEXP stream_r ){
  // Tensor length
  int* dims  = INTEGER( dims_r );
  int l = dims[0]*dims[1];

  // Create pointer to the device memory object
  float* tens_dev = (float*)R_ExternalPtrAddr( tens_r );

  // Allocate host memory and copy back content from device
  float* tens_host;
  if( stage_r == R_NilValue ){
    tens_host = new float[l];
  }else{
    tens_host = (float*)R_ExternalPtrAddr( stage_r );
  }

  cudaError_t cuda_stat;
  cudaTry( cudaMemcpy(tens_host, tens_dev, l*sizeof(float), cudaMemcpyDeviceToHost) );

  // Create the correct R object
  SEXP data_r;
  if( dims[1] == 1 ){
    data_r = Rf_protect( Rf_allocVector( REALSXP, dims[0] ) );
  }else{
    data_r = Rf_protect( Rf_allocMatrix( REALSXP, dims[0], dims[1] ) );
  }

  // Create a pointer to the actual data in the SEXP
  double* data_c = REAL( data_r );

  // Fill the SEXP with data
  for (int i = 0; i < l; i++){
    data_c[i] = (double)tens_host[i];
  }

  // Free host and device memory
  if( stage_r == R_NilValue ){
    delete[] tens_host;
  }

  Rf_unprotect(1);
  return data_r;
}
