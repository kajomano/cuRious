#include "common_R.h"
#include "common_debug.h"
#include "thrust.h"

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
                     SEXP stream_ptr_r ){ // Optional

  float* A_ptr    = (float*)R_ExternalPtrAddr( A_ptr_r );
  float* B_ptr    = (float*)R_ExternalPtrAddr( B_ptr_r );
  int*   dims     = INTEGER( dims_r );
  float  pow      = Rf_asReal( pow_r );

  int A_span_off  = ( R_NilValue == A_span_off_r ) ? 0:
    ( Rf_asInteger( A_span_off_r ) - 1 );

  int B_span_off  = ( R_NilValue == B_span_off_r ) ? 0:
    ( Rf_asInteger( B_span_off_r ) - 1 );

  cudaStream_t* stream_ptr = ( R_NilValue == stream_ptr_r ) ? NULL :
    (cudaStream_t*) R_ExternalPtrAddr( stream_ptr_r );

  void* allocator_ptr = R_ExternalPtrAddr( allocator_ptr_r );

  // Offsets
  A_ptr = A_ptr + A_span_off * dims[0];
  B_ptr = B_ptr + B_span_off * dims[0];

  if( stream_ptr ){
    // Launch a new thread so thrust becomes fully async
    // Uses C11 lambda for the WDDM flush
    // and common mutex for parallel access of variables
    // std::thread t([=]{
    //   thrust_mutex.lock();
    //
    //   cuR_thrust_pow_cu( A_ptr, B_ptr, dims, pow, allocator_ptr, stream_ptr );
    //   cudaStreamQuery(0);
    //
    //   thrust_mutex.unlock();
    // });
    // t.detach();
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
                          SEXP stream_ptr_r ){ // Optional

  float* A_ptr    = (float*)R_ExternalPtrAddr( A_ptr_r );
  int*   x_ptr    = (int*)R_ExternalPtrAddr( x_ptr_r );
  int*   A_dims   = INTEGER( A_dims_r );

  int A_span_off  = ( R_NilValue == A_span_off_r ) ? 0:
    ( Rf_asInteger( A_span_off_r ) - 1 );

  int x_span_off  = ( R_NilValue == x_span_off_r ) ? 0:
    ( Rf_asInteger( x_span_off_r ) - 1 );

  cudaStream_t* stream_ptr = ( R_NilValue == stream_ptr_r ) ? NULL :
    (cudaStream_t*) R_ExternalPtrAddr( stream_ptr_r );

  void* allocator_ptr = R_ExternalPtrAddr( allocator_ptr_r );

  // Offsets
  A_ptr = A_ptr + A_span_off * A_dims[0];
  x_ptr = x_ptr + x_span_off;

  if( stream_ptr ){
    // Launch a new thread so thrust becomes fully async
    // Uses C11 lambda for the WDDM flush
    // and common mutex for parallel access of variables
    // q.dispatch( [=]{
    //   cuR_thrust_cmin_pos_cu( A_ptr, x_ptr, A_dims, allocator_ptr, stream_ptr );
    //   cudaStreamQuery(0);
    // });

    // std::thread t([=]{
    //   thrust_mutex.lock();
    //
    //   cuR_thrust_cmin_pos_cu( A_ptr, x_ptr, A_dims, allocator_ptr, stream_ptr );
    //   cudaStreamQuery(0);
    //
    //   thrust_mutex.unlock();
    // });
    // t.detach();
  }else{
    cuR_thrust_cmin_pos_cu( A_ptr, x_ptr, A_dims, allocator_ptr, stream_ptr );
    cudaTry( cudaDeviceSynchronize() );
  }

  return R_NilValue;
}

// extern "C"
// SEXP cuB_thrust_table( SEXP quant_r, SEXP perm_r, SEXP temp_quant_r, SEXP dims_r, SEXP weights_r, SEXP dims_weights_r ) {
//   int* quant        = (int*)R_ExternalPtrAddr( quant_r );
//   int* perm         = (int*)R_ExternalPtrAddr( perm_r );
//   int* temp_quant   = (int*)R_ExternalPtrAddr( temp_quant_r );
//   int* dims         = INTEGER( dims_r );
//
//   int* weights      = (int*)R_ExternalPtrAddr( weights_r );
//   int* dims_weights = INTEGER( dims_weights_r );
//
//   cuB_thrust_table_cu( quant, perm, temp_quant, dims, weights, dims_weights );
//
//   // Return something that is not null
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }
