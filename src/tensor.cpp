#include "common.h"

// Various memory allocations/deallocations
// Level 1 ---------------------------------------------------------------------

void cuR_fin_tensor_1( SEXP tens_r ){
  float* tens = (float*)R_ExternalPtrAddr( tens_r );

  if( tens ){
    debugPrint( Rprintf( "<%p> Finalizing L1 tensor\n", (void*) tens ) );

    delete[] tens;
    R_ClearExternalPtr( tens_r );
  }
}

extern "C"
SEXP cuR_create_tensor_1( SEXP dims_r ){
  int* dims = INTEGER(dims_r);
  int l = dims[0]*dims[1];

  float* tens = new float[l];

  debugPrint( Rprintf( "<%p> Creating L1 tensor\n", (void*)tens ) );

  SEXP tens_r = Rf_protect( R_MakeExternalPtr( tens, R_NilValue, R_NilValue ) );
  R_RegisterCFinalizerEx( tens_r, cuR_fin_tensor_1, TRUE );

  Rf_setAttrib(tens_r, R_ClassSymbol, Rf_mkString("tensor.ptr"));
  Rf_setAttrib(tens_r, Rf_mkString("level"), Rf_ScalarInteger(1));
  Rf_setAttrib(tens_r, Rf_mkString("dim0"), Rf_ScalarInteger(dims[0]));
  Rf_setAttrib(tens_r, Rf_mkString("dim1"), Rf_ScalarInteger(dims[1]));

  Rf_unprotect(1);
  return tens_r;
}

extern "C"
SEXP cuR_destroy_tensor_1( SEXP tens_r ){
  cuR_fin_tensor_1( tens_r );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

// Level 2 ---------------------------------------------------------------------

void cuR_fin_tensor_2( SEXP tens_r ){
  float* tens = (float*)R_ExternalPtrAddr( tens_r );

  // Destroy context and free memory!
  // Clear R object too
  if( tens ){
    debugPrint( Rprintf( "<%p> Finalizing L2 tensor\n", (void*)tens ) );

    cudaDo( cudaFreeHost( tens ) );
    R_ClearExternalPtr( tens_r );
  }
}

extern "C"
SEXP cuR_create_tensor_2( SEXP dims_r ){
  float* tens;
  cudaDo( cudaError_t cuda_stat );
  int* dims = INTEGER(dims_r);
  int l = dims[0]*dims[1];

  cudaTry( cudaHostAlloc( (void**)&tens, l*sizeof(float), cudaHostAllocDefault) )

  debugPrint( Rprintf( "<%p> Creating L2 tensor\n", (void*)tens ) );

  SEXP tens_r = Rf_protect( R_MakeExternalPtr( tens, R_NilValue, R_NilValue ) );
  R_RegisterCFinalizerEx( tens_r, cuR_fin_tensor_2, TRUE );

  Rf_setAttrib(tens_r, R_ClassSymbol, Rf_mkString("tensor.ptr"));
  Rf_setAttrib(tens_r, Rf_mkString("level"), Rf_ScalarInteger(2));
  Rf_setAttrib(tens_r, Rf_mkString("dim0"), Rf_ScalarInteger(dims[0]));
  Rf_setAttrib(tens_r, Rf_mkString("dim1"), Rf_ScalarInteger(dims[1]));

  Rf_unprotect(1);
  return tens_r;
}

extern "C"
SEXP cuR_destroy_tensor_2( SEXP tens_r ){
  cuR_fin_tensor_2( tens_r );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}

// Level 3 ---------------------------------------------------------------------

void cuR_fin_tensor_3( SEXP tens_r ){
  float* tens = (float*)R_ExternalPtrAddr( tens_r );

  if( tens ){
    debugPrint( Rprintf( "<%p> Finalizing L3 tensor\n", (void*)tens ) );

    // Free memory
    cudaDo( cudaFree( tens ) );
    R_ClearExternalPtr( tens_r );
  }
}

extern "C"
SEXP cuR_create_tensor_3( SEXP dims_r ){
  float* tens;
  cudaDo( cudaError_t cuda_stat );
  int* dims = INTEGER(dims_r);
  int l = dims[0]*dims[1];

  cudaTry( cudaMalloc( (void**)&tens, l*sizeof(float) ) );

  debugPrint( Rprintf( "<%p> Creating L3 tensor\n", (void*)tens ) );

  // Wrap pointer in a SEXP and register a finalizer
  SEXP tens_r = Rf_protect( R_MakeExternalPtr( tens, R_NilValue, R_NilValue ) );
  R_RegisterCFinalizerEx( tens_r, cuR_fin_tensor_3, TRUE );

  Rf_setAttrib(tens_r, R_ClassSymbol, Rf_mkString("tensor.ptr"));
  Rf_setAttrib(tens_r, Rf_mkString("level"), Rf_ScalarInteger(3));
  Rf_setAttrib(tens_r, Rf_mkString("dim0"), Rf_ScalarInteger(dims[0]));
  Rf_setAttrib(tens_r, Rf_mkString("dim1"), Rf_ScalarInteger(dims[1]));

  Rf_unprotect(1);
  return tens_r;
}

extern "C"
SEXP cuR_destroy_tensor_3( SEXP tens_r ){
  cuR_fin_tensor_3( tens_r );

  SEXP ret_r = Rf_protect( Rf_ScalarLogical( 1 ) );
  Rf_unprotect(1);
  return ret_r;
}
