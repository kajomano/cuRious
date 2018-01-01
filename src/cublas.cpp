#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#define R_NO_REMAP 1
//#define R_NO_REMAP_RMATH 1

#include <R.h>
#include <Rinternals.h>

#include "debug.h"

// This is a good example how to access the handle in other functions
// Also a good example how to finalize things, this function can be called
// from other functions without the fear of double-finalization
void cuR_finalize_cublas_handle( SEXP ptr ){
  cublasHandle_t* handle = (cublasHandle_t*)R_ExternalPtrAddr( ptr );

  // Destroy context and free memory!
  // Clear R object too
  if( handle ){
#ifdef DEBUG_PRINTS
    Rprintf( "Finalizing handle at <%p>\n", (void*)handle );
#endif

    cublasDestroy( *handle );
    delete[] handle;
    R_ClearExternalPtr( ptr );
  }
}

extern "C"
SEXP cuR_destroy_cublas_handle( SEXP ptr ){
  cublasHandle_t* handle = (cublasHandle_t*)R_ExternalPtrAddr( ptr );

  // Destroy context and free memory!
  // Clear R object too
  if( handle ){
#ifdef DEBUG_PRINTS
    Rprintf( "Finalizing handle at <%p>\n", (void*)handle );
#endif

    cublasStatus_t stat;
    cublasTry( cublasDestroy( *handle ) )
    delete[] handle;
    R_ClearExternalPtr( ptr );
  }

  // Return something that is not null
  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_create_cublas_handle(){
  cublasHandle_t* handle = new cublasHandle_t;
#ifdef DEBUG_PRINTS
  Rprintf( "Creating handle at <%p>\n", (void*)handle );
#endif

  // cublasTry is a macro defined in debug.h, you need to create the variable
  // 'stat' beforehand
  cublasStatus_t stat;
  cublasTry( cublasCreate( handle ) )

  // Return to R with an external pointer SEXP
  SEXP ptr = Rf_protect( R_MakeExternalPtr( handle, R_NilValue, R_NilValue ) );
  R_RegisterCFinalizerEx( ptr, cuR_finalize_cublas_handle, TRUE );

  Rf_unprotect(1);
  return ptr;
}

extern "C"
SEXP cuR_cublas_saxpy( SEXP tens_x_r, SEXP tens_y_r, SEXP l_r, SEXP al_r, SEXP handle_r ){
  // Recover handle
  cublasHandle_t* handle = (cublasHandle_t*)R_ExternalPtrAddr( handle_r );

  // Recover tensors, the length and the scalar
  int l = Rf_asInteger( l_r );
  float* tens_x = (float*)R_ExternalPtrAddr( tens_x_r );
  float* tens_y = (float*)R_ExternalPtrAddr( tens_y_r );
  float al = (float)Rf_asReal( al_r );

  // Do the operation, the results go into tens_y
  cublasStatus_t stat;
  cublasTry( cublasSaxpy( *handle, l, &al, tens_x, 1, tens_y, 1 ) )

  // Return something that is not null
  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

cublasOperation_t cuR_cublas_translate_op_int( int op_int ){
  if( op_int == 2 ){
    return CUBLAS_OP_T;
  }else if( op_int == 3 ){
    return CUBLAS_OP_C;
  }else{
    return CUBLAS_OP_N;
  }
}

// The number of arguments is too damn high!
extern "C"
SEXP cuR_cublas_sgemv( SEXP tens_A_r, SEXP tens_x_r, SEXP tens_y_r, SEXP dims_r, SEXP al_r, SEXP be_r, SEXP op_r, SEXP handle_r ){
  // Recover handle
  cublasHandle_t* handle = (cublasHandle_t*)R_ExternalPtrAddr( handle_r );

  // Recover tensors, the dims and the scalars
  int* dims  = INTEGER( dims_r );
  float* tens_A = (float*)R_ExternalPtrAddr( tens_A_r );
  float* tens_x = (float*)R_ExternalPtrAddr( tens_x_r );
  float* tens_y = (float*)R_ExternalPtrAddr( tens_y_r );
  float al = (float)Rf_asReal( al_r );
  float be = (float)Rf_asReal( be_r );

  // Recover the operation
  cublasOperation_t op = cuR_cublas_translate_op_int( Rf_asInteger( op_r ) );

  // Do the operation, the results go into tens_y
  cublasStatus_t stat;
  cublasTry( cublasSgemv( *handle, op, dims[0], dims[1], &al, tens_A, dims[0], tens_x, 1, &be, tens_y, 1 ) )

  // Return something that is not null
  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

extern "C"
SEXP cuR_cublas_sgemm( SEXP tens_A_r, SEXP tens_B_r, SEXP tens_C_r, SEXP dims_A_r, SEXP dims_B_r, SEXP al_r, SEXP be_r, SEXP op_A_r, SEXP op_B_r, SEXP handle_r ){
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

  // Recover the operation
  cublasOperation_t op_A = cuR_cublas_translate_op_int( Rf_asInteger( op_A_r ) );
  cublasOperation_t op_B = cuR_cublas_translate_op_int( Rf_asInteger( op_B_r ) );

  // Do the operation, the results go into tens_y
  cublasStatus_t stat;
  cublasTry( cublasSgemm( *handle, op_A, op_B, dims_A[0], dims_B[1], dims_A[1], &al, tens_A, dims_A[0], tens_B, dims_A[1], &be, tens_C, dims_A[0] ) )

  // Return something that is not null
  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}
