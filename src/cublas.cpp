#include "common_R.h"
#include "common_cuda.h"
#include "common_debug.h"

#include "streams.h"

#ifndef CUDA_EXCLUDE

// This is a good example how to access the handle in other functions
// Also a good example how to finalize things, this function can be called
// from other functions without the fear of double-finalization
void cuR_cublas_handle_fin( SEXP handle_r ){
  cublasHandle_t* handle = (cublasHandle_t*)R_ExternalPtrAddr( handle_r );

  // Destroy context and free memory!
  // Clear R object too
  if( handle ){
    debugPrint( Rprintf( "<%p> Finalizing handle\n", (void*)handle ) );

    cublasDestroy( *handle );
    delete handle;
    R_ClearExternalPtr( handle_r );
  }
}

extern "C"
SEXP cuR_cublas_handle_destroy( SEXP handle_r ){
  cuR_cublas_handle_fin( handle_r );

  return R_NilValue;
}

extern "C"
SEXP cuR_cublas_handle_create(){
  cublasHandle_t* handle = new cublasHandle_t;

  debugPrint( Rprintf( "<%p> Creating handle\n", (void*)handle ) );

  // Try to create handle
  cublasTry( cublasCreate( handle ) );

  // Return to R with an external pointer SEXP
  SEXP handle_r = Rf_protect( R_MakeExternalPtr( handle, R_NilValue, R_NilValue ) );
  R_RegisterCFinalizerEx( handle_r, cuR_cublas_handle_fin, TRUE );

  Rf_unprotect(1);
  return handle_r;
}

// TODO ====
// cublasSetStream could be called when deploying the handle

// This returns a cublasStatus
cublasStatus_t cuR_cublas_recover_stream( cudaStream_t* stream, cublasHandle_t* handle ){
  // If a stream is given, set it before the calculation
  cublasStatus_t stat = CUBLAS_STATUS_SUCCESS;
  if( stream ){
    debugPrint( Rprintf( "Async cublas call\n" ) );
    stat = cublasSetStream( *handle, *stream );
  }else{
    debugPrint( Rprintf( "Sync cublas call\n" ) );
  }

  return stat;
}

extern "C"
SEXP cuR_cublas_sgemv( SEXP A_ptr_r,
                       SEXP x_ptr_r,
                       SEXP y_ptr_r,
                       SEXP A_dims_r,
                       SEXP A_span_off_r,
                       SEXP x_span_off_r,
                       SEXP y_span_off_r,
                       SEXP A_tp_r,
                       SEXP alpha_r,
                       SEXP beta_r,
                       SEXP handle_ptr_r,
                       SEXP queue_ptr_r,     // Optional
                       SEXP stream_ptr_r ){  // Optional

  // Recover handle, queue and stream
  cublasHandle_t* handle_ptr = (cublasHandle_t*)R_ExternalPtrAddr( handle_ptr_r );

  sd_queue* queue_ptr = ( R_NilValue == queue_ptr_r ) ? NULL :
    (sd_queue*) R_ExternalPtrAddr( queue_ptr_r );

  cudaStream_t* stream_ptr = ( R_NilValue == stream_ptr_r ) ? NULL :
    (cudaStream_t*)R_ExternalPtrAddr( stream_ptr_r );

  // Recover tensors, the A_dims and the scalars
  float* A_ptr = (float*)R_ExternalPtrAddr( A_ptr_r );
  float* x_ptr = (float*)R_ExternalPtrAddr( x_ptr_r );
  float* y_ptr = (float*)R_ExternalPtrAddr( y_ptr_r );
  int* A_dims  = INTEGER( A_dims_r );

  int A_span_off = ( R_NilValue == A_span_off_r ) ? 0 :
    ( Rf_asInteger( A_span_off_r ) - 1 );
  int x_span_off = ( R_NilValue == x_span_off_r ) ? 0 :
    ( Rf_asInteger( x_span_off_r ) - 1 );
  int y_span_off = ( R_NilValue == y_span_off_r ) ? 0 :
    ( Rf_asInteger( y_span_off_r ) - 1 );

  float alpha = (float)Rf_asReal( alpha_r );
  float beta  = (float)Rf_asReal( beta_r );

  // Offsets
  A_ptr = A_ptr + A_span_off * A_dims[0];
  x_ptr = x_ptr + x_span_off;
  y_ptr = y_ptr + y_span_off;

  // Transpose
  cublasOperation_t op_A;
  int m, n;
  if( Rf_asLogical( A_tp_r ) == 1 ){
    op_A = CUBLAS_OP_T;
    m = A_dims[1];
    n = A_dims[0];
  }else{
    op_A = CUBLAS_OP_N;
    m = A_dims[0];
    n = A_dims[1];
  }

  // Handle stream
  cublasTry( cuR_cublas_recover_stream( stream_ptr, handle_ptr ) );

  if( queue_ptr ){
    queue_ptr -> dispatch( [=]{
      cublasSgemv( *handle_ptr, op_A, m, n, &alpha, A_ptr, A_dims[0], x_ptr, 1, &beta, y_ptr, 1 );
      cudaStreamQuery(0);
    });
  }else{
    cublasTry( cublasSgemv( *handle_ptr, op_A, m, n, &alpha, A_ptr, A_dims[0], x_ptr, 1, &beta, y_ptr, 1 ) );
    cudaTry( cudaDeviceSynchronize() );
  }

  return R_NilValue;
}

extern "C"
SEXP cuR_cublas_sger( SEXP x_ptr_r,
                      SEXP y_ptr_r,
                      SEXP A_ptr_r,
                      SEXP A_dims_r,
                      SEXP x_span_off_r,
                      SEXP y_span_off_r,
                      SEXP A_span_off_r,
                      SEXP alpha_r,
                      SEXP handle_ptr_r,
                      SEXP queue_ptr_r,     // Optional
                      SEXP stream_ptr_r ){

  // Recover handle, queue and stream
  cublasHandle_t* handle_ptr = (cublasHandle_t*)R_ExternalPtrAddr( handle_ptr_r );

  sd_queue* queue_ptr = ( R_NilValue == queue_ptr_r ) ? NULL :
    (sd_queue*) R_ExternalPtrAddr( queue_ptr_r );

  cudaStream_t* stream_ptr = ( R_NilValue == stream_ptr_r ) ? NULL :
    (cudaStream_t*)R_ExternalPtrAddr( stream_ptr_r );

  // Recover tensors, the dims and the scalars
  float* x_ptr = (float*)R_ExternalPtrAddr( x_ptr_r );
  float* y_ptr = (float*)R_ExternalPtrAddr( y_ptr_r );
  float* A_ptr = (float*)R_ExternalPtrAddr( A_ptr_r );
  int* A_dims  = INTEGER( A_dims_r );

  int x_span_off = ( R_NilValue == x_span_off_r ) ? 0 :
    ( Rf_asInteger( x_span_off_r ) - 1 );
  int y_span_off = ( R_NilValue == y_span_off_r ) ? 0 :
    ( Rf_asInteger( y_span_off_r ) - 1 );
  int A_span_off = ( R_NilValue == A_span_off_r ) ? 0 :
    ( Rf_asInteger( A_span_off_r ) - 1 );

  float alpha = (float)Rf_asReal( alpha_r );

  // Offsets
  x_ptr = x_ptr + x_span_off;
  y_ptr = y_ptr + y_span_off;
  A_ptr = A_ptr + A_span_off * A_dims[0];

  int m = A_dims[0];
  int n = A_dims[1];

  // Handle stream
  cublasTry( cuR_cublas_recover_stream( stream_ptr, handle_ptr ) );

  if( queue_ptr ){
    queue_ptr -> dispatch( [=]{
      cublasSger( *handle_ptr, m, n, &alpha, x_ptr, 1, y_ptr, 1, A_ptr, m );
      cudaStreamQuery(0);
    });
  }else{
    cublasTry( cublasSger( *handle_ptr, m, n, &alpha, x_ptr, 1, y_ptr, 1, A_ptr, m ) );
    cudaTry( cudaDeviceSynchronize() );
  }

  return R_NilValue;
}

extern "C"
SEXP cuR_cublas_sgemm( SEXP A_ptr_r,
                       SEXP B_ptr_r,
                       SEXP C_ptr_r,
                       SEXP A_dims_r,
                       SEXP B_dims_r,
                       SEXP A_span_off_r,
                       SEXP B_span_off_r,
                       SEXP C_span_off_r,
                       SEXP A_tp_r,
                       SEXP B_tp_r,
                       SEXP alpha_r,
                       SEXP beta_r,
                       SEXP handle_ptr_r,
                       SEXP queue_ptr_r,     // Optional
                       SEXP stream_ptr_r ){

  // Recover handle, queue and stream
  cublasHandle_t* handle_ptr = (cublasHandle_t*)R_ExternalPtrAddr( handle_ptr_r );

  sd_queue* queue_ptr = ( R_NilValue == queue_ptr_r ) ? NULL :
    (sd_queue*) R_ExternalPtrAddr( queue_ptr_r );

  cudaStream_t* stream_ptr = ( R_NilValue == stream_ptr_r ) ? NULL :
    (cudaStream_t*)R_ExternalPtrAddr( stream_ptr_r );

  // Recover tensors, the dims and the scalars
  float* A_ptr = (float*)R_ExternalPtrAddr( A_ptr_r );
  float* B_ptr = (float*)R_ExternalPtrAddr( B_ptr_r );
  float* C_ptr = (float*)R_ExternalPtrAddr( C_ptr_r );
  int* A_dims  = INTEGER( A_dims_r );
  int* B_dims  = INTEGER( B_dims_r );

  int A_span_off = ( R_NilValue == A_span_off_r ) ? 0 :
    ( Rf_asInteger( A_span_off_r ) - 1 );
  int B_span_off = ( R_NilValue == B_span_off_r ) ? 0 :
    ( Rf_asInteger( B_span_off_r ) - 1 );
  int C_span_off = ( R_NilValue == C_span_off_r ) ? 0 :
    ( Rf_asInteger( C_span_off_r ) - 1 );

  float alpha = (float)Rf_asReal( alpha_r );
  float beta  = (float)Rf_asReal( beta_r );

  // Offsets
  A_ptr = A_ptr + A_span_off * A_dims[0];
  B_ptr = B_ptr + B_span_off * B_dims[0];
  C_ptr = C_ptr + C_span_off * A_dims[0];

  // Transposes
  cublasOperation_t op_A, op_B;
  int m, n, k;
  if( Rf_asLogical( A_tp_r ) == 1 ){
    op_A = CUBLAS_OP_T;
    m = A_dims[1];
    k = A_dims[0];
  }else{
    op_A = CUBLAS_OP_N;
    m = A_dims[0];
    k = A_dims[1];
  }

  if( Rf_asLogical( B_tp_r ) == 1 ){
    op_B = CUBLAS_OP_T;
    n = B_dims[0];
  }else{
    op_B = CUBLAS_OP_N;
    n = B_dims[1];
  }

  // Handle stream_ptr
  cublasTry( cuR_cublas_recover_stream( stream_ptr, handle_ptr ) );

  if( queue_ptr ){
    queue_ptr -> dispatch( [=]{
      cublasSgemm( *handle_ptr, op_A, op_B, m, n, k, &alpha, A_ptr, A_dims[0], B_ptr, B_dims[0], &beta, C_ptr, m );
      cudaStreamQuery(0);
    });
  }else{
    cublasTry( cublasSgemm( *handle_ptr, op_A, op_B, m, n, k, &alpha, A_ptr, A_dims[0], B_ptr, B_dims[0], &beta, C_ptr, m ) );
    cudaTry( cudaDeviceSynchronize() );
  }

  return R_NilValue;
}

// extern "C"
// SEXP cuR_cublas_saxpy( SEXP tens_x_r, SEXP tens_y_r, SEXP l_r, SEXP al_r, SEXP handle_r, SEXP stream_r ){
//   // Recover handle
//   cublasHandle_t* handle = (cublasHandle_t*)R_ExternalPtrAddr( handle_r );
//
//   // Recover tensors, the length and the scalar
//   int l = Rf_asInteger( l_r );
//   float* tens_x = (float*)R_ExternalPtrAddr( tens_x_r );
//   float* tens_y = (float*)R_ExternalPtrAddr( tens_y_r );
//   float al = (float)Rf_asReal( al_r );
//
//   // Handle stream
//   cublasTry( cuR_cublas_recover_stream( stream_r, handle ) );;
//
//   // Do the operation, the results go into tens_y
//   cublasTry( cublasSaxpy( *handle, l, &al, tens_x, 1, tens_y, 1 ) );
//
//   // Flush for WDDM
//   cudaStreamQuery(0);
//
//   // Return something that is not null
//   SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
//   Rf_unprotect(1);
//   return ret_r;
// }

#endif
