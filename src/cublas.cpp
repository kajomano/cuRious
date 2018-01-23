#include "common.h"
#ifndef CUDA_EXCLUDE

// This is a good example how to access the handle in other functions
// Also a good example how to finalize things, this function can be called
// from other functions without the fear of double-finalization
void cuR_finalize_cublas_handle( SEXP ptr ){
  cublasHandle_t* handle = (cublasHandle_t*)R_ExternalPtrAddr( ptr );

  // Destroy context and free memory!
  // Clear R object too
  if( handle ){
    debugPrint( Rprintf( "<%p> Finalizing handle\n", (void*)handle ) );

    cublasDestroy( *handle );
    delete[] handle;
    R_ClearExternalPtr( ptr );
  }
}

extern "C"
SEXP cuR_deactivate_cublas_handle( SEXP ptr ){
  cuR_finalize_cublas_handle( ptr );
  return R_NilValue;
}

extern "C"
SEXP cuR_activate_cublas_handle(){
  cublasHandle_t* handle = new cublasHandle_t;

  debugPrint( Rprintf( "<%p> Creating handle\n", (void*)handle ) );

  // Try to create handle
  cublasTry( cublasCreate( handle ) );

  // Return to R with an external pointer SEXP
  SEXP ptr = Rf_protect( R_MakeExternalPtr( handle, R_NilValue, R_NilValue ) );
  R_RegisterCFinalizerEx( ptr, cuR_finalize_cublas_handle, TRUE );

  Rf_unprotect(1);
  return ptr;
}

// This returns a cublasStatus
cublasStatus_t cuR_cublas_recover_stream( SEXP stream_r, cublasHandle_t* handle ){
  // If a stream is given, set it before the calculation
  cublasStatus_t stat = CUBLAS_STATUS_SUCCESS;
  if( stream_r != R_NilValue ){
    debugPrint( Rprintf( "Async cublas call\n" ) );

    cudaStream_t* stream = (cudaStream_t*)R_ExternalPtrAddr( stream_r );
    stat = cublasSetStream( *handle, *stream );
  }else{
    debugPrint( Rprintf( "Sync cublas call\n" ) );
  }

  return stat;
}

extern "C"
SEXP cuR_cublas_sgemm( SEXP tens_A_r, SEXP tens_B_r, SEXP tens_C_r, SEXP dims_A_r, SEXP dims_B_r, SEXP al_r, SEXP be_r, SEXP tp_A_r, SEXP tp_B_r, SEXP handle_r, SEXP stream_r ){
  // Recover handle
  cublasHandle_t* handle = (cublasHandle_t*)R_ExternalPtrAddr( handle_r );

  // Recover tensors, the dims and the scalars
  int* dims_A = INTEGER( dims_A_r );
  int* dims_B = INTEGER( dims_B_r );
  float* tens_A = (float*)R_ExternalPtrAddr( tens_A_r );
  float* tens_B = (float*)R_ExternalPtrAddr( tens_B_r );
  float* tens_C = (float*)R_ExternalPtrAddr( tens_C_r );
  float al = (float)Rf_asReal( al_r );
  float be = (float)Rf_asReal( be_r );

  // Transposes
  cublasOperation_t op_A, op_B;
  int m, n, k;
  if( Rf_asLogical( tp_A_r ) == 1 ){
    op_A = CUBLAS_OP_T;
    m = dims_A[1];
    k = dims_A[0];
  }else{
    op_A = CUBLAS_OP_N;
    m = dims_A[0];
    k = dims_A[1];
  }

  if( Rf_asLogical( tp_B_r ) == 1 ){
    op_B = CUBLAS_OP_T;
    n = dims_B[0];
  }else{
    op_B = CUBLAS_OP_N;
    n = dims_B[1];
  }

  // Handle stream
  cublasTry( cuR_cublas_recover_stream( stream_r, handle ) );

  // Do the op
  cublasTry( cublasSgemm( *handle, op_A, op_B, m, n, k, &al, tens_A, dims_A[0], tens_B, dims_B[0], &be, tens_C, m ) )

  // Flush for WDDM
  cudaStreamQuery(0);

  // Return something that is not null
  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_cublas_saxpy( SEXP tens_x_r, SEXP tens_y_r, SEXP l_r, SEXP al_r, SEXP handle_r, SEXP stream_r ){
  // Recover handle
  cublasHandle_t* handle = (cublasHandle_t*)R_ExternalPtrAddr( handle_r );

  // Recover tensors, the length and the scalar
  int l = Rf_asInteger( l_r );
  float* tens_x = (float*)R_ExternalPtrAddr( tens_x_r );
  float* tens_y = (float*)R_ExternalPtrAddr( tens_y_r );
  float al = (float)Rf_asReal( al_r );

  // Handle stream
  cublasTry( cuR_cublas_recover_stream( stream_r, handle ) );;

  // Do the operation, the results go into tens_y
  cublasTry( cublasSaxpy( *handle, l, &al, tens_x, 1, tens_y, 1 ) );

  // Flush for WDDM
  cudaStreamQuery(0);

  // Return something that is not null
  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

#endif
