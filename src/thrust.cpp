#include "common_R.h"
#include "common_debug.h"
#include "streams.h"
#include "thrust.h"

#include <cstdio>

#ifndef CUDA_EXCLUDE

void cuR_thrust_allocator_fin( SEXP allocator_r ){
  void* allocator = R_ExternalPtrAddr( allocator_r );

  // Destroy context and free memory!
  // Clear R object too
  if( allocator ){
    debugPrint( Rprintf( "<%p> Finalizing allocator\n", allocator_r ) );

    cuR_thrust_allocator_destroy_cu( allocator );
    R_ClearExternalPtr( allocator_r );
  }
}

extern "C"
SEXP cuR_thrust_allocator_destroy( SEXP allocator_r ){
  cuR_thrust_allocator_fin( allocator_r );

  return R_NilValue;
}

extern "C"
SEXP cuR_thrust_allocator_create(){
  void* allocator = cuR_thrust_allocator_create_cu();

  debugPrint( Rprintf( "<%p> Creating allocator\n", allocator ) );

  // Return to R with an external pointer SEXP
  SEXP allocator_r = Rf_protect( R_MakeExternalPtr( allocator, R_NilValue, R_NilValue ) );
  R_RegisterCFinalizerEx( allocator_r, cuR_thrust_allocator_fin, TRUE );

  Rf_unprotect(1);
  return allocator_r;
}

// -----------------------------------------------------------------------------

extern "C"
SEXP cuR_thrust_pow( SEXP A_ptr_r,
                     SEXP B_ptr_r,
                     SEXP dims_r,
                     SEXP A_span_off_r,   // Optional
                     SEXP B_span_off_r,   // Optional
                     SEXP pow_r,
                     SEXP allocator_ptr_r,
                     SEXP queue_ptr_r,    // Optional
                     SEXP stream_ptr_r ){ // Optional

  // Recover allocator, queue and stream
  void* allocator_ptr = R_ExternalPtrAddr( allocator_ptr_r );

  sd_queue* queue_ptr = ( R_NilValue == queue_ptr_r ) ? NULL :
    (sd_queue*) R_ExternalPtrAddr( queue_ptr_r );

  cudaStream_t* stream_ptr = ( R_NilValue == stream_ptr_r ) ? NULL :
    (cudaStream_t*)R_ExternalPtrAddr( stream_ptr_r );

  float* A_ptr    = (float*)R_ExternalPtrAddr( A_ptr_r );
  float* B_ptr    = (float*)R_ExternalPtrAddr( B_ptr_r );
  int*   dims     = INTEGER( dims_r );
  float  pow      = Rf_asReal( pow_r );

  int A_span_off  = ( R_NilValue == A_span_off_r ) ? 0:
    ( Rf_asInteger( A_span_off_r ) - 1 );

  int B_span_off  = ( R_NilValue == B_span_off_r ) ? 0:
    ( Rf_asInteger( B_span_off_r ) - 1 );

  // Offsets
  A_ptr = A_ptr + A_span_off * dims[0];
  B_ptr = B_ptr + B_span_off * dims[0];

  if( queue_ptr ){
    queue_ptr -> dispatch( [=]{
      cuR_thrust_pow_cu( A_ptr, B_ptr, dims, pow, allocator_ptr, stream_ptr );
      cudaStreamQuery(0);
    });
  }else{
    cuR_thrust_pow_cu( A_ptr, B_ptr, dims, pow, allocator_ptr, stream_ptr );
    cudaTry( cudaDeviceSynchronize() );
  }

  return R_NilValue;
}

extern "C"
SEXP cuR_thrust_cmin_pos( SEXP A_ptr_r,
                          SEXP x_ptr_r,
                          SEXP A_dims_r,
                          SEXP A_span_off_r,   // Optional
                          SEXP x_span_off_r,   // Optional
                          SEXP allocator_ptr_r,
                          SEXP queue_ptr_r,    // Optional
                          SEXP stream_ptr_r ){ // Optional

  // Recover allocator, queue and stream
  void* allocator_ptr = R_ExternalPtrAddr( allocator_ptr_r );

  sd_queue* queue_ptr = ( R_NilValue == queue_ptr_r ) ? NULL :
    (sd_queue*) R_ExternalPtrAddr( queue_ptr_r );

  cudaStream_t* stream_ptr = ( R_NilValue == stream_ptr_r ) ? NULL :
    (cudaStream_t*)R_ExternalPtrAddr( stream_ptr_r );

  float* A_ptr    = (float*)R_ExternalPtrAddr( A_ptr_r );
  int*   x_ptr    = (int*)R_ExternalPtrAddr( x_ptr_r );
  int*   A_dims   = INTEGER( A_dims_r );

  int A_span_off  = ( R_NilValue == A_span_off_r ) ? 0:
    ( Rf_asInteger( A_span_off_r ) - 1 );

  int x_span_off  = ( R_NilValue == x_span_off_r ) ? 0:
    ( Rf_asInteger( x_span_off_r ) - 1 );

  // Offsets
  A_ptr = A_ptr + A_span_off * A_dims[0];
  x_ptr = x_ptr + x_span_off;

  if( queue_ptr ){
    queue_ptr -> dispatch( [=]{
      cuR_thrust_cmin_pos_cu( A_ptr, x_ptr, A_dims, allocator_ptr, stream_ptr );
      cudaStreamQuery(0);
    });
  }else{
    cuR_thrust_cmin_pos_cu( A_ptr, x_ptr, A_dims, allocator_ptr, stream_ptr );
    cudaTry( cudaDeviceSynchronize() );
  }

  return R_NilValue;
}

extern "C"
SEXP cuR_thrust_table( SEXP x_ptr_r,
                       SEXP p_ptr_r,
                       SEXP w_ptr_r,
                       SEXP s_ptr_r,
                       SEXP x_dims_r,
                       SEXP w_dims_r,
                       SEXP x_span_off_r,
                       SEXP p_span_off_r,
                       SEXP w_span_off_r,
                       SEXP s_span_off_r,
                       SEXP allocator_ptr_r,
                       SEXP queue_ptr_r,    // Optional
                       SEXP stream_ptr_r ){ // Optional

  // Recover allocator, queue and stream
  void* allocator_ptr = R_ExternalPtrAddr( allocator_ptr_r );

  sd_queue* queue_ptr = ( R_NilValue == queue_ptr_r ) ? NULL :
    (sd_queue*) R_ExternalPtrAddr( queue_ptr_r );

  cudaStream_t* stream_ptr = ( R_NilValue == stream_ptr_r ) ? NULL :
    (cudaStream_t*)R_ExternalPtrAddr( stream_ptr_r );

  int* x_ptr     = (int*)R_ExternalPtrAddr( x_ptr_r );
  int* p_ptr     = (int*)R_ExternalPtrAddr( p_ptr_r );
  int* w_ptr     = (int*)R_ExternalPtrAddr( w_ptr_r );
  int* s_ptr     = (int*)R_ExternalPtrAddr( s_ptr_r );

  int* x_dims    = INTEGER( x_dims_r );
  int* w_dims    = INTEGER( w_dims_r );

  int x_span_off = Rf_asInteger( x_span_off_r ) - 1;
  int p_span_off = Rf_asInteger( p_span_off_r ) - 1;
  int w_span_off = Rf_asInteger( w_span_off_r ) - 1;
  int s_span_off = Rf_asInteger( s_span_off_r ) - 1;

  printf( "Offs: %d\n", w_span_off );

  // Offsets
  x_ptr = x_ptr + x_span_off;
  p_ptr = p_ptr + p_span_off;
  w_ptr = w_ptr + w_span_off;
  s_ptr = s_ptr + s_span_off;

  if( queue_ptr ){
    queue_ptr -> dispatch( [=]{
      cuR_thrust_table_cu( x_ptr, p_ptr, w_ptr, s_ptr, x_dims, w_dims, x_span_off, allocator_ptr, stream_ptr );
      cudaStreamQuery(0);
    });
  }else{
    cuR_thrust_table_cu( x_ptr, p_ptr, w_ptr, s_ptr, x_dims, w_dims, x_span_off, allocator_ptr, stream_ptr );
    cudaTry( cudaDeviceSynchronize() );
  }

  return R_NilValue;
}

#endif
